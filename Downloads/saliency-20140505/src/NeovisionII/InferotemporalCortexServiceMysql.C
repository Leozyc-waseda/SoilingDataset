/*!@file Neuro/InferotemporalCortexService.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/InferotemporalCortexServiceMysql.C $
// $Id: InferotemporalCortexServiceMysql.C 12962 2010-03-06 02:13:53Z irock $
//

#include "NeovisionII/InferotemporalCortexService.H"
#include "Image/ColorOps.H"
#include "Image/DrawOps.H"
#include "Util/sformat.H"

InferotemporalCortexI::InferotemporalCortexI(OptionManager& mgr,
    nub::ref<OutputFrameSeries> ofs,
    const std::string& descrName,
    const std::string& tagName ) :
  ModelComponent(mgr, descrName, tagName),
  itsOfs(ofs)
{
    //Subscribe to the various topics
    IceStorm::TopicPrx topicPrx;
    itsTopicsSubscriptions.push_back(TopicInfo("SegmenterTopic", topicPrx));

    itsDBCon = new mysqlpp::Connection(false);

    //Connect to the database
    const char* db = "ObjRec", *server = "localhost", *user = "neobrain", *pass = "12341234";
    if (!itsDBCon->connect(db, server, user, pass))
        LFATAL("DB connection failed: ");
   // , itsDBCon->error().c_str());
}

InferotemporalCortexI::~InferotemporalCortexI()
{
  unsubscribeSimEvents();
}

void InferotemporalCortexI::initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx)
{
  //Get the IceStorm object
  Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -h ilab21 -p 11111");
  IceStorm::TopicManagerPrx topicManager =
    IceStorm::TopicManagerPrx::checkedCast(obj);

  //Create a InferotemporalCortex Topic
  IceStorm::TopicPrx topic;
  try {
    topic = topicManager->retrieve("InferotemporalCortexTopic"); //check if the Retina topic exists
  } catch (const IceStorm::NoSuchTopic&) {
    topic = topicManager->create("InferotemporalCortexTopic"); //The retina topic does not exists, create
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
    }
  }
}

void InferotemporalCortexI::unsubscribeSimEvents()
{
  //Unsubscribe from all the topics we are registerd to
  for(uint i=0; i<itsTopicsSubscriptions.size();  i++)
  {
    itsTopicsSubscriptions[i].topicPrx->unsubscribe(itsObjectPrx);
  }
}


bool InferotemporalCortexI::initVDB()
{
  itsUseColor = false;
  itsTrainingMode = false;
  itsVDBFile = "objects.vdb";
  itsMaxLabelHistory = 1;
  itsVDB.loadFrom(itsVDBFile);

  return true;
}

//The VC excepts a retina image and posts a saliency map
void InferotemporalCortexI::evolve(const SimEvents::EventMessagePtr& eMsg,
      const Ice::Current&)
{
  if (eMsg->ice_isA("::SimEvents::SegmenterMessage")){
    SimEvents::SegmenterMessagePtr msg = SimEvents::SegmenterMessagePtr::dynamicCast(eMsg);
    for(uint i=0; i<msg->segLocs.size(); i++)
    {
      SimEvents::SegInfo segInfo = msg->segLocs[i];
      Image<PixRGB<byte> > segImg = Ice2Image<PixRGB<byte> >(segInfo.img);
      Rectangle segRect(Point2D<int>(segInfo.rect.tl.i, segInfo.rect.tl.j),
          Dims(segInfo.rect.br.i - segInfo.rect.tl.i,
            segInfo.rect.br.j - segInfo.rect.tl.j));


      ////Rescale the image to provide some scale inveriance
      //Image<PixRGB<byte> > inputImg = rescale(segImg, 256, 256);
      itsImgMutex.lock();
      itsCurrentImg = segImg;
      itsImgMutex.unlock();
    }
  }
}

void InferotemporalCortexI::run()
{

  while(1)
  {
    itsImgMutex.lock();
    Image<PixRGB<byte> > img = itsCurrentImg;
    itsImgMutex.unlock();
    if (!img.initialized())
     continue;

    itsTimer.reset();
    rutz::shared_ptr<VisualObject>
      vo(new VisualObject("PIC", "PIC", img,
            Point2D<int>(-1,-1),
            std::vector<double>(),
            std::vector< rutz::shared_ptr<Keypoint> >(),
            itsUseColor));
    itsTimer.mark();
    LINFO("Keypoint extraction took %f seconds for %i keypoints (%ix%i)",
        itsTimer.user_secs(),
        vo->numKeypoints(),
        img.getWidth(), img.getHeight());


    mysqlpp::Query query = itsDBCon->query();
    //Insert the keypoints into the database if there is not match;
    const uint nkp = vo->numKeypoints();
    for(uint kpIdx=0; kpIdx<nkp; kpIdx++)
    {
      LINFO("Searching for keypoint %i of %i", kpIdx, nkp);
      //Get the feature fector
      char fv[128];
      rutz::shared_ptr<Keypoint> keyPoint = vo->getKeypoint(kpIdx);
      for(uint i=0; i<keyPoint->getFVlength(); i++)
      {
        fv[i] = keyPoint->getFVelement(i);
        //printf("%i ", (unsigned char)fv[i]);
      }
      //printf("\n");

      LINFO("Check NN");
      itsTimer.reset();
      //check if there is a match in the db
      bool foundKeypoint = false;
      double foundDist = -1;
      int foundMatchCount = -1;
      int foundIdx = -1;

      string fill(fv, 128);
      ostringstream strbuf;
      strbuf << "SELECT kid, matchCount, kp_dist(value, \"" << mysqlpp::escape << fill <<"\") as dist from SiftKeypoints order by dist limit 1" << ends;
      query << strbuf.str();

      mysqlpp::Result res = query.store();
      if (res)
      {
        mysqlpp::Row row;
        mysqlpp::Row::size_type rowIdx;
        for (rowIdx = 0; row = res.at(rowIdx); ++rowIdx) {
          foundIdx = row["kid"];
          foundDist = row["dist"];
          foundMatchCount = row["matchCount"];

          if (foundDist < 100000)
            foundKeypoint = true;
        }
      }
      itsTimer.mark();
      LINFO("NN Searched (idx=%i matchCount=%i dist=%f) took %f seconds (found=%i)",
          foundIdx,
          foundMatchCount,
          foundDist,
          itsTimer.user_secs(),
          foundKeypoint);

      LINFO("Service Keypoint");
      itsTimer.reset();
      if (!foundKeypoint)
      {
        LINFO("Keypoint not found adding");
        std::string kpv(fv, 128);
        std::ostringstream strbuf;
        strbuf << "INSERT INTO SiftKeypoints (value) VALUES (\"" << mysqlpp::escape << kpv <<"\")" << ends;
        query.exec(strbuf.str());
        printf("Insert keypoint\n");
      } else {
        //update the frequency;
        std::ostringstream strbuf;
        foundMatchCount++; //incr the number of times this keyp was found
        strbuf << "UPDATE SiftKeypoints set matchCount = " << foundMatchCount << " WHERE kid = " << foundIdx << ends;
        query.exec(strbuf.str());
        printf("Update keypoint\n");
      }
      itsTimer.mark();
      LINFO("Service keypoint took %f seconds",
          itsTimer.user_secs());

      //Show the keypoint
      PixRGB<byte> color;
      if(foundKeypoint)
        color = PixRGB<byte>(0,255,0);
      else
        color = PixRGB<byte>(255,0,0);

      float x = keyPoint->getX();
      float y = keyPoint->getY();
      float s = keyPoint->getS();
      float o = keyPoint->getO();

      Point2D<int> keypLoc(int(x + 0.5F), int(y + 0.5F));

      drawDisk(img, keypLoc, 2, color);
      if (s > 0.0f)
        drawLine(img, keypLoc,
          Point2D<int>(int(x + s * cosf(o) + 0.5F),
            int(y + s * sinf(o) + 0.5F)),
          color);

      itsOfs->writeRGB(img, "KeyPoints", FrameInfo("KeyPoints", SRC_POS));
    }
    LINFO("Continue");


    LINFO("Done.");
  }
}

std::string InferotemporalCortexI::getBestLabel(const size_t mincount)
{
  if (itsRecentLabels.size() == 0)
    return std::string();

  std::map<std::string, size_t> counts;

  size_t bestcount = 0;
  size_t bestpos = 0;

  for (size_t i = 0; i < itsRecentLabels.size(); ++i)
    {
      const size_t c = ++(counts[itsRecentLabels[i]]);

      if (c >= bestcount)
        {
          bestcount = c;
          bestpos = i;
        }
    }

  if (bestcount >= mincount)
    return itsRecentLabels[bestpos];

  return std::string();
}

std::string InferotemporalCortexI::matchObject(Image<PixRGB<byte> > &ima, float &score)
{
  //find object in the database
  std::vector< rutz::shared_ptr<VisualObjectMatch> > matches;
  rutz::shared_ptr<VisualObject>
    vo(new VisualObject("PIC", "PIC", ima,
                        Point2D<int>(-1,-1),
                        std::vector<double>(),
                        std::vector< rutz::shared_ptr<Keypoint> >(),
                        itsUseColor));

  const uint nmatches = itsVDB.getObjectMatches(vo, matches, VOMA_SIMPLE,
                                             100U, //max objs to return
                                             0.5F, //keypoint distance score default 0.5F
                                             0.5F, //affine distance score default 0.5F
                                             1.0F, //minscore  default 1.0F
                                             3U, //min # of keypoint match
                                             100U, //keypoint selection thershold
                                             false //sort by preattentive
                                             );

  score = 0;
  float avgScore = 0, affineAvgDist = 0;
  int nkeyp = 0;
  int objId = -1;
  if (nmatches > 0)
    {
      rutz::shared_ptr<VisualObject> obj; //so we will have a ref to the last matches obj
      rutz::shared_ptr<VisualObjectMatch> vom;
      //for(unsigned int i=0; i< nmatches; i++){
      for (unsigned int i = 0; i < 1; ++i)
        {
          vom = matches[i];
          obj = vom->getVoTest();
          score = vom->getScore();
          nkeyp = vom->size();
          avgScore = vom->getKeypointAvgDist();
          affineAvgDist = vom->getAffineAvgDist();

          objId = atoi(obj->getName().c_str()+3);

          return obj->getName();
          LINFO("### Object match with '%s' score=%f ID:%i",
                obj->getName().c_str(), vom->getScore(), objId);

          //calculate the actual distance (location of keypoints) between
          //keypoints. If the same patch was found, then the distance should
          //be close to 0
          double dist = 0;
          for (int keyp=0; keyp<nkeyp; keyp++)
            {
              const KeypointMatch kpm = vom->getKeypointMatch(keyp);

              float refX = kpm.refkp->getX();
              float refY = kpm.refkp->getY();

              float tstX = kpm.tstkp->getX();
              float tstY = kpm.tstkp->getY();
              dist += (refX-tstX) * (refX-tstX);
              dist += (refY-tstY) * (refY-tstY);
            }

          //   printf("%i:%s %i %f %i %f %f %f\n", objNum, obj->getName().c_str(),
          //       nmatches, score, nkeyp, avgScore, affineAvgDist, sqrt(dist));

          //analizeImage();
        }

    }

  return std::string("nomatch");
}



/////////////////////////// The VC Service to init the retina and start as a deamon ///////////////
class InferotemporalCortexService : public Ice::Service {
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

bool InferotemporalCortexService::start(int argc, char* argv[])
{

        itsMgr = new ModelManager("InferotemporalCortexService");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(*itsMgr));
  itsMgr->addSubComponent(ofs);

  nub::ref<InferotemporalCortexI> itc(new InferotemporalCortexI(*itsMgr, ofs));
        itsMgr->addSubComponent(itc);

        itsMgr->parseCommandLine((const int)argc, (const char**)argv, "", 0, 0);

  char adapterStr[255];
  sprintf(adapterStr, "default -p %i", BrainObjects::InferoTemporalPort);
        itsAdapter = communicator()->createObjectAdapterWithEndpoints("InferotemporalCortexAdapter",
      adapterStr);

        Ice::ObjectPtr object = itc.get();
  Ice::ObjectPrx objectPrx = itsAdapter->add(object, communicator()->stringToIdentity("InferotemporalCortex"));
  itc->initSimEvents(communicator(), objectPrx);
        itsAdapter->activate();

  itsMgr->start();

  IceUtil::ThreadPtr itcThread = itc.get();
  itcThread->start();


        return true;
}

// ######################################################################
int main(int argc, char** argv) {

  InferotemporalCortexService svc;
  return svc.main(argc, argv);
}


