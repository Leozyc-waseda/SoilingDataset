/*!@file Neuro/VisualTrackerService.C */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file: Lior Elazary
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/VisualTrackerService.C $
// $Id: VisualTrackerService.C 12962 2010-03-06 02:13:53Z irock $
//

#include "NeovisionII/VisualTrackerService.H"
#include "Image/ColorOps.H"

const ModelOptionCateg MOC_VT = {
  MOC_SORTPRI_2, "Visual Tracker Related Options" };

static const ModelOptionDef OPT_VT_TrackIOR =
  { MODOPT_ARG(int), "VCTrackIOR", &MOC_VT, OPTEXP_CORE,
    "Number of frames to wait before deciding to stop the tracker",
    "vt-track-ior", '\0', "<int>", "-1" };

VisualTrackerI::VisualTrackerI(OptionManager& mgr,
    const std::string& descrName,
    const std::string& tagName ) :
  ModelComponent(mgr, descrName, tagName),
  itsTrackIOR(&OPT_VT_TrackIOR, this, ALLOW_ONLINE_CHANGES)
{
    //Subscribe to the various topics
    IceStorm::TopicPrx topicPrx;
    itsTopicsSubscriptions.push_back(TopicInfo("RetinaTopic", topicPrx));
    itsTopicsSubscriptions.push_back(TopicInfo("PrefrontalCortexTopic", topicPrx));

#ifdef HAVE_OPENCV
    this->points[0] = NULL;
    this->points[1] = NULL;
    this->status = NULL;
    this->track_error = NULL;
    this->pyramid = NULL;
    this->prev_pyramid = NULL;
#endif
    itsInitTracker = true;
    itsInitTargets = true;
    itsTracking = false;
    itsTrackFrames = 0;

}

VisualTrackerI::~VisualTrackerI()
{
  unsubscribeSimEvents();
#ifdef HAVE_OPENCV
  //cvFree(&this->points[0]);
  //cvFree(&this->points[1]);
  //cvFree(&this->status);
  cvReleaseImage(&this->pyramid);
  cvReleaseImage(&this->prev_pyramid);
#endif

}

void VisualTrackerI::initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx)
{
  //Get the IceStorm object
  Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 11111");
  IceStorm::TopicManagerPrx topicManager =
    IceStorm::TopicManagerPrx::checkedCast(obj);

  //Create a VisualTracker Topic
  IceStorm::TopicPrx topic;
  try {
    topic = topicManager->retrieve("VisualTrackerTopic"); //check if the Retina topic exists
  } catch (const IceStorm::NoSuchTopic&) {
    topic = topicManager->create("VisualTrackerTopic"); //The retina topic does not exists, create
  }
  //Make a one way visualCortex message publisher for efficency
  Ice::ObjectPrx pub = topic->getPublisher()->ice_oneway();
  itsEventsPub = SimEvents::EventsPrx::uncheckedCast(pub);

  //Subscribe the SimulationViewer to theTopics
  itsObjectPrx = objectPrx;
  //subscribe  to the topics
  for(uint i=0; i<itsTopicsSubscriptions.size();  i++)
  {
    try {
      IceStorm::QoS qos;
      itsTopicsSubscriptions[i].topicPrx =
        topicManager->retrieve(itsTopicsSubscriptions[i].name.c_str()); //Get the
      itsTopicsSubscriptions[i].topicPrx->subscribeAndGetPublisher(qos, itsObjectPrx); //Subscribe to the retina topic
    } catch (const IceStorm::NoSuchTopic&) {
      LFATAL("Error! No %s topic found!", itsTopicsSubscriptions[i].name.c_str());
    } catch (const char* msg) {
      LINFO("Error %s", msg);
    } catch (const Ice::Exception& e) {
      cerr << e << endl;
    }

  }
}

void VisualTrackerI::unsubscribeSimEvents()
{
  //Unsubscribe from all the topics we are registerd to
  for(uint i=0; i<itsTopicsSubscriptions.size();  i++)
  {
    itsTopicsSubscriptions[i].topicPrx->unsubscribe(itsObjectPrx);
  }
}


//The VC excepts a retina image and posts a saliency map
void VisualTrackerI::evolve(const SimEvents::EventMessagePtr& eMsg,
      const Ice::Current&)
{
  if (eMsg->ice_isA("::SimEvents::VisualTrackerBiasMessage")){ //Set the tracker location
    SimEvents::VisualTrackerBiasMessagePtr msg = SimEvents::VisualTrackerBiasMessagePtr::dynamicCast(eMsg);
    itsTrackLocs.clear();
    for(uint i=0; i<msg->locToTrack.size(); i++)
    {
      itsTrackLocs.push_back(msg->locToTrack[i]);
      Point2D<int> trackLoc = Point2D<int>(msg->locToTrack[i].pos.i, msg->locToTrack[i].pos.j);
      LINFO("Tracking %ix%i", trackLoc.i, trackLoc.j);
      break; //only track one point at this time;
    }
    itsInitTargets = true; //Let the system know we want to reinit
  } else if(eMsg->ice_isA("::SimEvents::RetinaMessage")){
    SimEvents::RetinaMessagePtr msg = SimEvents::RetinaMessagePtr::dynamicCast(eMsg);
    Image<PixRGB<byte> > img = Ice2Image<PixRGB<byte> > (msg->img);
    //We got the first image, initalize the tracker
    if (itsInitTracker)
      initTracker(img.getDims());

    if (itsInitTargets)
      setTargets(luminance(img));

    Point2D<int> targetLoc(-1,-1);
    if (itsTracking)
    {
            targetLoc = trackObjects(luminance(img));
    }

    SimEvents::VisualTrackerMessagePtr bMsg = new SimEvents::VisualTrackerMessage;
    if (targetLoc.isValid()) //If we lost the track then send an empty message
    {
            //return the current track location
            SimEvents::TrackInfo trackInfo;
            trackInfo.pos.i = targetLoc.i;
            trackInfo.pos.j = targetLoc.j;
            bMsg->trackLocs.push_back(trackInfo);
    }
    itsEventsPub->evolve(bMsg);
  }
}

void VisualTrackerI::initTracker(Dims imageDims, int nPoints, int wz )
{
  win_size = 30;

#ifdef HAVE_OPENCV
  MAX_COUNT = nPoints;
  count = 0;
  points[0] = (CvPoint2D32f*)cvAlloc(MAX_COUNT*sizeof(points[0][0]));
  points[1] = (CvPoint2D32f*)cvAlloc(MAX_COUNT*sizeof(points[0][0]));

  prev_grey = Image<byte>(imageDims, ZEROS);
  pyramid = cvCreateImage( cvSize(imageDims.w(), imageDims.h()), 8, 1 );
  prev_pyramid = cvCreateImage( cvSize(imageDims.w(), imageDims.h()), 8, 1 );
  status = (char*)cvAlloc(MAX_COUNT);
  track_error = (float*)cvAlloc(MAX_COUNT);
#endif
  flags = 0;
  itsInitTracker = false;
  itsTrackFrames = 0;

  //Send an init message to indicate that we are not tracking

  SimEvents::VisualTrackerMessagePtr bMsg = new SimEvents::VisualTrackerMessage;
  itsEventsPub->evolve(bMsg);

}

// ######################################################################
void VisualTrackerI::setTargets(const Image<byte>& grey)
{
#ifdef HAVE_OPENCV
  count = MAX_COUNT;

  IplImage* tmp = img2ipl(grey);
  for(uint i=0; i<itsTrackLocs.size(); i++)
  {
    Point2D<int> trackLoc = Point2D<int>(itsTrackLocs[i].pos.i, itsTrackLocs[i].pos.j);
    LINFO("Setting target to %ix%i", trackLoc.i, trackLoc.j);
    points[1][0].x = trackLoc.i;
    points[1][0].y = trackLoc.j;

    cvFindCornerSubPix(tmp, points[1], count,
        cvSize(win_size,win_size), cvSize(-1,-1),
        cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,
          20,0.03));
    cvReleaseImageHeader(&tmp);

    prev_grey = grey;
    break; //only a single point is allowed for now
  }
  flags = 0;
  IplImage *swap_temp;
  CV_SWAP( prev_pyramid, pyramid, swap_temp );
  CV_SWAP( points[0], points[1], swap_points );

#endif

  itsTracking = true;
  itsInitTargets = false;
  itsTrackFrames = 0;

}

// ######################################################################
Point2D<int> VisualTrackerI::trackObjects(const Image<byte>& grey)
{
  Point2D<int> targetLoc(-1,-1);

  //Apply the track IOR
  if (itsTrackIOR.getVal() != -1 &&
      (int)itsTrackFrames >= itsTrackIOR.getVal())
  {
                LINFO("Appling IOR");
          itsTrackFrames = 0;
          itsTracking = false;
          return targetLoc;
  }


#ifdef HAVE_OPENCV
  if (count > 0)
  {
    IplImage* tmp1 = img2ipl(prev_grey);
    IplImage* tmp2 = img2ipl(grey);

    //flags = CV_LKFLOW_INITIAL_GUESSES;

    cvCalcOpticalFlowPyrLK(tmp1, tmp2, prev_pyramid, pyramid,
        points[0], points[1], count,
        cvSize(win_size,win_size), 3, status, track_error,
        cvTermCriteria(CV_TERMCRIT_ITER
          |CV_TERMCRIT_EPS,
          20,0.03), flags);

    flags = CV_LKFLOW_PYR_A_READY | CV_LKFLOW_PYR_B_READY;

    cvReleaseImageHeader(&tmp1);
    cvReleaseImageHeader(&tmp2);


    //show track points
    int k, i;
    for(i = k = 0; i<count; i++)
    {
      if (!status[i])
        continue;

      points[1][k++] = points[1][i];
                        //LINFO("Error %i: %f", i, track_error[i]);
                        if (track_error[i] < 2000)
                        {
                                targetLoc.i = std::min(grey.getWidth()-1, std::max(0, (int)points[0][i].x));
                                targetLoc.j = std::min(grey.getHeight()-1, std::max(0, (int)points[0][i].y));
                                ASSERT(grey.coordsOk(targetLoc));
                        }
    }
    count = k;

  }

  IplImage *swap_temp;
  CV_SWAP( prev_pyramid, pyramid, swap_temp );
  CV_SWAP( points[0], points[1], swap_points );

  prev_grey = grey;
#endif

  itsTrackFrames++;

  return targetLoc;
}




/////////////////////////// The VC Service to init the retina and start as a deamon ///////////////
class VisualTrackerService : public Ice::Service {
  protected:
    virtual bool start(int, char* argv[]);
    virtual bool stop() {
      if (itsMgr)
        delete itsMgr;
      return true;
    }

  private:
    Ice::ObjectAdapterPtr itsAdapter;
    ModelManager *itsMgr;
};

bool VisualTrackerService::start(int argc, char* argv[])
{

        itsMgr = new ModelManager("VisualTrackerService");

  nub::ref<VisualTrackerI> vt(new VisualTrackerI(*itsMgr));
        itsMgr->addSubComponent(vt);

        itsMgr->parseCommandLine((const int)argc, (const char**)argv, "", 0, 0);

  char adapterStr[255];
  sprintf(adapterStr, "default -p %i", BrainObjects::VisualTrackerPort);
        itsAdapter = communicator()->createObjectAdapterWithEndpoints("VisualTrackerAdapter",
      adapterStr);

        Ice::ObjectPtr object = vt.get();
  Ice::ObjectPrx objectPrx = itsAdapter->add(object, communicator()->stringToIdentity("VisualTracker"));
  vt->initSimEvents(communicator(), objectPrx);
        itsAdapter->activate();

  itsMgr->start();

        return true;
}

// ######################################################################
int main(int argc, char** argv) {

  VisualTrackerService svc;
  return svc.main(argc, argv);
}


