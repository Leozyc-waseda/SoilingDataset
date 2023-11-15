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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/InferotemporalCortexService.C $
// $Id: InferotemporalCortexService.C 14522 2011-02-17 01:52:47Z dparks $
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
    itsTopicsSubscriptions.push_back(TopicInfo("PrefrontalCortexTopic", topicPrx));
    initVDB();
}

InferotemporalCortexI::~InferotemporalCortexI()
{
  LINFO("Un suscribe");
  unsubscribeSimEvents();
}

void InferotemporalCortexI::initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx)
{
  //Get the IceStorm object
  Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 11111");
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
    } catch (const char* msg) {
      LINFO("Error %s", msg);
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
  itsTrainingMode = true;
  itsVDBFile = "objects.vdb";
  itsMaxLabelHistory = 1;
  itsVDB.loadFrom(itsVDBFile);

  return true;
}

//The VC excepts a retina image and posts a saliency map
void InferotemporalCortexI::evolve(const SimEvents::EventMessagePtr& eMsg,
      const Ice::Current&)
{
  if (eMsg->ice_isA("::SimEvents::InfrotemporalCortexBiasMessage")){
    SimEvents::InfrotemporalCortexBiasMessagePtr msg = SimEvents::InfrotemporalCortexBiasMessagePtr::dynamicCast(eMsg);
    Image<PixRGB<byte> > objImg = Ice2Image<PixRGB<byte> >(msg->img);

    //Recognize the image
    float score;
    float rotation = 0;
    std::string objName = matchObject(objImg, score, rotation);

    if (objName == "nomatch")
    {
      itsRecentLabels.resize(0);

      if (itsTrainingMode)
      {
        LINFO("Enter a label for this object:\n");
        std::getline(std::cin, objName);
        LINFO("You typed '%s'", objName.c_str());

        if (objName != "" && objName != "exit")
        {
          rutz::shared_ptr<VisualObject>
            vo(new VisualObject(objName.c_str(), "NULL", objImg,
                  Point2D<int>(-1,-1),
                  std::vector<float>(),
                  std::vector< rutz::shared_ptr<Keypoint> >(),
                  itsUseColor));
          itsVDB.addObject(vo, false);
          itsVDB.saveTo(itsVDBFile);
        } else {
          objName = "nomatch";
        }

        SimEvents::InfrotemporalCortexMessagePtr itMsg = new SimEvents::InfrotemporalCortexMessage;
        itMsg->objectName = objName;
        itMsg->confidence = -1;
        itMsg->rotation = rotation;
        LINFO("Sending message");
        itsEventsPub->evolve(itMsg);

      }
      LINFO("NO match");
    } else {
      const std::string bestObjName = objName;

      //itsRecentLabels.push_back(objName);
      //while (itsRecentLabels.size() > 1)
      //  itsRecentLabels.pop_front();

      //const std::string bestObjName =
      //  getBestLabel(itsRecentLabels, 1);

      LINFO("Found object %s rot=%f", bestObjName.c_str(), rotation);
      if (bestObjName.size() > 0)
      {
        SimEvents::InfrotemporalCortexMessagePtr itMsg = new SimEvents::InfrotemporalCortexMessage;
        itMsg->objectName = bestObjName;
        itMsg->confidence = score;
        itMsg->rotation = rotation;
        itsEventsPub->evolve(itMsg);

      }
    }
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

std::string InferotemporalCortexI::matchObject(Image<PixRGB<byte> > &ima, float &score, float &rotation)
{
  //find object in the database
  std::vector< rutz::shared_ptr<VisualObjectMatch> > matches;
  rutz::shared_ptr<VisualObject>
    vo(new VisualObject("PIC", "NULL", ima,
          Point2D<int>(-1,-1),
          std::vector<float>(),
          std::vector< rutz::shared_ptr<Keypoint> >(),
          itsUseColor));

  Image<PixRGB<byte> > siftImg = vo->getKeypointImage();
  itsOfs->writeRGB(siftImg, "SIFTKeypoints", FrameInfo("SiftKeypoints", SRC_POS));
  itsOfs->updateNext();
  usleep(10000);

  const uint nmatches = itsVDB.getObjectMatches(vo, matches, VOMA_SIMPLE,
      10000U, //max objs to return
      0.5F, //keypoint distance score default 0.5F
      0.5F, //affine distance score default 0.5F
      1.0F, //minscore  default 1.0F
      3U, //min # of keypoint match
      100U, //keypoint selection thershold
      false //sort by preattentive
      );

  score = 0;
  //float avgScore = 0, affineAvgDist = 0;
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
      //avgScore = vom->getKeypointAvgDist();
      //affineAvgDist = vom->getAffineAvgDist();

      objId = atoi(obj->getName().c_str()+3);

      //Get the rotation
      float theta, sx, sy, str;
      SIFTaffine aff = vom->getSIFTaffine();
      aff.decompose(theta, sx, sy, str);
      rotation = theta;


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

  return true;
}

// ######################################################################
int main(int argc, char** argv) {

  InferotemporalCortexService svc;
  return svc.main(argc, argv);
}


