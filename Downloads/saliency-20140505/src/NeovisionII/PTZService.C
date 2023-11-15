/*!@file NeovisionII/PTZService.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/PTZService.C $
// $Id: PTZService.C 12962 2010-03-06 02:13:53Z irock $
//

#include "NeovisionII/PTZService.H"
#include "Util/sformat.H"
#include "Image/DrawOps.H"

using namespace std;
using namespace ImageIceMod;

PTZI::PTZI(OptionManager& mgr,
    const std::string& descrName,
    const std::string& tagName) :
  ModelComponent(mgr, descrName, tagName),
  itsZoomIn(false),
  itsZoomOut(false),
  itsInitPtz(false)
{
  //Subscribe to the various topics
  IceStorm::TopicPrx topicPrx;
  itsTopicsSubscriptions.push_back(TopicInfo("VisualTrackerTopic", topicPrx));
  itsTopicsSubscriptions.push_back(TopicInfo("PrefrontalCortexTopic", topicPrx));

  itsCameraCtrl = nub::soft_ref<Visca>(new Visca(mgr));
  addSubComponent(itsCameraCtrl);

}



PTZI::~PTZI()
{
  unsubscribeSimEvents();
}

void PTZI::start2()
{
}


void PTZI::initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx)
{
  //Get the IceStorm object
  Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 11111");
  IceStorm::TopicManagerPrx topicManager =
    IceStorm::TopicManagerPrx::checkedCast(obj);

  //Create a Segmenter Topic
  IceStorm::TopicPrx topic;
  try {
    topic = topicManager->retrieve("CameraCtrlTopic"); //check if the Retina topic exists
  } catch (const IceStorm::NoSuchTopic&) {
    topic = topicManager->create("CameraCtrlTopic"); //The retina topic does not exists, create
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

void PTZI::unsubscribeSimEvents()
{
  //Unsubscribe from all the topics we are registerd to
  for(uint i=0; i<itsTopicsSubscriptions.size();  i++)
  {
    itsTopicsSubscriptions[i].topicPrx->unsubscribe(itsObjectPrx);
  }
}


void PTZI::evolve(const SimEvents::EventMessagePtr& eMsg,
    const Ice::Current&)
{
  if (eMsg->ice_isA("::SimEvents::VisualTrackerMessage")){
    SimEvents::VisualTrackerMessagePtr msg = SimEvents::VisualTrackerMessagePtr::dynamicCast(eMsg);

    if (msg->trackLocs.size() > 0)
      if (itsZoomIn)
        moveCameraToTarget(Point2D<int>(msg->trackLocs[0].pos.i, msg->trackLocs[0].pos.j));

  } else if (eMsg->ice_isA("::SimEvents::CameraCtrlBiasMessage")){
    SimEvents::CameraCtrlBiasMessagePtr msg = SimEvents::CameraCtrlBiasMessagePtr::dynamicCast(eMsg);

    itsZoomIn = msg->zoomIn;
    itsZoomOut = msg->zoomOut;
    itsInitPtz = msg->initPtz;
  }

  if (itsInitPtz)
  {
      sleep(2);
      LINFO("INit pan tilt");
      itsCameraCtrl->zoom(0); //Zoom out
      sleep(3);
      itsCameraCtrl->movePanTilt(0,0);
      itsInitPtz = false;
      sleep(3); //TODO replace with a blocking call

      SimEvents::CameraCtrlMessagePtr ptzMsg = new SimEvents::CameraCtrlMessage;
      ptzMsg->initPtzDone = true;
      ptzMsg->zoomDone = false;
      ptzMsg->pan = 0;
      ptzMsg->tilt = 0;
      itsEventsPub->evolve(ptzMsg);
  }

}


void PTZI::moveCameraToTarget(Point2D<int> targetLoc)
{

  //move camera to target loc
  Point2D<int> locErr = targetLoc - Point2D<int>(320/2, 240/2);

  int currentZoom = itsCameraCtrl->getCurrentZoom();

  Point2D<float> pGain(0.30, 0.30);

  if (currentZoom > 800)
  {
    pGain.i = 0.01; pGain.j = 0.01;
  } else if (currentZoom > 1000)
  {
    pGain.i = 0.05; pGain.j = 0.05;
  }


  //P controller for now
  Point2D<float> u = pGain*locErr;


  LINFO("Target is at: %ix%i zoom=%i (err %ix%i) move=%ix%i",
      targetLoc.i, targetLoc.j, currentZoom,
      locErr.i, locErr.j,
      (int)u.i, (int)u.j);

  if (fabs(locErr.distance(Point2D<int>(0,0))) > 16)
          itsCameraCtrl->movePanTilt((int)u.i, -1*(int)u.j, true); //move relative
  else
  {
          if (currentZoom < 750)
                  itsCameraCtrl->zoom(10, true); //move relative
          else
          {
                  LINFO("Done zooming");
                  sleep(1);
                  itsZoomOut = false;
                  itsZoomIn = false;
                  SimEvents::CameraCtrlMessagePtr ptzMsg = new SimEvents::CameraCtrlMessage;
                  ptzMsg->initPtzDone = false;
                  ptzMsg->zoomDone = true;
                  short int pan=0, tilt = 0;
                  bool gotPanTilt = false;
                  for(int i=0; i<5 && !gotPanTilt; i++)
                          if (itsCameraCtrl->getPanTilt(pan,tilt))
                                  gotPanTilt = true;

                  if (!gotPanTilt)
                  {
                          LINFO("Error getting pan tilt");
                          sleep(10);
                  }

                  ptzMsg->pan = pan;
                  ptzMsg->tilt = tilt;
                  itsEventsPub->evolve(ptzMsg);
          }
  }

}

/////////////////////////// The VC Service to init the retina and start as a deamon ///////////////
class PTZService : public Ice::Service {
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

bool PTZService::start(int argc, char* argv[])
{

  itsMgr = new ModelManager("PTZService");

  nub::ref<PTZI> ptz(new PTZI(*itsMgr));
  itsMgr->addSubComponent(ptz);

  itsMgr->parseCommandLine((const int)argc, (const char**)argv, "", 0, 0);

  char adapterStr[255];
  sprintf(adapterStr, "default -p %i", BrainObjects::PTZPort);
  itsAdapter = communicator()->createObjectAdapterWithEndpoints("PTZAdapter",
      adapterStr);

  Ice::ObjectPtr object = ptz.get();
  Ice::ObjectPrx objectPrx = itsAdapter->add(object, communicator()->stringToIdentity("PTZ"));
  ptz->initSimEvents(communicator(), objectPrx);
  itsAdapter->activate();

  itsMgr->start();

  return true;
}

// ######################################################################
int main(int argc, char** argv) {

  PTZService svc;
  return svc.main(argc, argv);
}
