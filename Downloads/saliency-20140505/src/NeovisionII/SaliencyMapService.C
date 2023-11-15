/*!@file Neuro/SaliencyMapService.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/SaliencyMapService.C $
// $Id: SaliencyMapService.C 12962 2010-03-06 02:13:53Z irock $
//

#include "NeovisionII/SaliencyMapService.H"
#include "Image/ColorOps.H"
#include "Image/ShapeOps.H"

SaliencyMapI::SaliencyMapI(OptionManager& mgr,
    const std::string& descrName,
    const std::string& tagName ) :
  ModelComponent(mgr, descrName, tagName)
{
    itsSMap = nub::soft_ref<EnvSaliencyMap>(new EnvSaliencyMap(mgr));
    addSubComponent(itsSMap);

    //Subscribe to the various topics
    IceStorm::TopicPrx topicPrx;
    TopicInfo tInfo("VisualCortexTopic", topicPrx);
    itsTopicsSubscriptions.push_back(tInfo);

    itsTopicsSubscriptions.push_back(TopicInfo("HippocampusTopic", topicPrx));


}

SaliencyMapI::~SaliencyMapI()
{
  unsubscribeSimEvents();
}


void SaliencyMapI::initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx)
{
  //Get the IceStorm object
  Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 11111");
  IceStorm::TopicManagerPrx topicManager =
    IceStorm::TopicManagerPrx::checkedCast(obj);

  //Create a SaliencyMap Topic
  IceStorm::TopicPrx topic;
  try {
    topic = topicManager->retrieve("SaliencyMapTopic"); //check if the Retina topic exists
  } catch (const IceStorm::NoSuchTopic&) {
    topic = topicManager->create("SaliencyMapTopic"); //The retina topic does not exists, create
  }
  //Make a one way SaliencyMap message publisher for efficency
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


void SaliencyMapI::unsubscribeSimEvents()
{
  //Unsubscribe from all the topics we are registerd to
  for(uint i=0; i<itsTopicsSubscriptions.size();  i++)
  {
    itsTopicsSubscriptions[i].topicPrx->unsubscribe(itsObjectPrx);
  }
}



//Get  a visualCortex Output from the visual cortex and compute the top N most salient locations
void SaliencyMapI::evolve(const SimEvents::EventMessagePtr& eMsg,
      const Ice::Current&)
{
  if(eMsg->ice_isA("::SimEvents::VisualCortexMessage")){
    SimEvents::VisualCortexMessagePtr msg = SimEvents::VisualCortexMessagePtr::dynamicCast(eMsg);
    Image<byte> vco = Ice2Image<byte>(msg->vco);


    Point2D<int> scaled_maxpos(-1,-1);

    //Apply the bias
    if (itsBiasImg.initialized())
      itsSMap->setBiasImg(itsBiasImg);

    const EnvSaliencyMap::State smstate = itsSMap->getSalmap(vco, scaled_maxpos);

    //send a Saliency Map message
    SimEvents::SaliencyMapMessagePtr sMapMsg = new SimEvents::SaliencyMapMessage;
    sMapMsg->smap = Image2Ice(smstate.salmap);


    SimEvents::LocInfoSeq mostSalientLocSeq;
    for(uint i=0; i<smstate.nMostSalientLoc.size(); i++)
    {
      const EnvSaliencyMap::LocInfo envLocInfo = smstate.nMostSalientLoc[i];

      SimEvents::LocInfo locInfo;
      locInfo.lowresMaxpos.i = envLocInfo.lowres_maxpos.i;
      locInfo.lowresMaxpos.j = envLocInfo.lowres_maxpos.j;
      locInfo.fullresMaxpos.i = envLocInfo.fullres_maxpos.i;
      locInfo.fullresMaxpos.j = envLocInfo.fullres_maxpos.j;
      locInfo.maxval = envLocInfo.maxval;
      mostSalientLocSeq.push_back(locInfo);
    }
    sMapMsg->nMostSalientLoc = mostSalientLocSeq;

    //Post the saliency map state
    itsEventsPub->evolve(sMapMsg);
  }

  if(eMsg->ice_isA("::SimEvents::SaliencyMapBiasMessage")){
    SimEvents::SaliencyMapBiasMessagePtr msg = SimEvents::SaliencyMapBiasMessagePtr::dynamicCast(eMsg);
    if (msg->biasImg.width > 0)
      itsBiasImg = Ice2Image<byte>(msg->biasImg);
    else
      itsSMap->resetBiasImg();
  }


}

/////////////////////////// The Retina Service to init the retina and start as a deamon ///////////////
class SaliencyMapService : public Ice::Service {
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

bool SaliencyMapService::start(int argc, char* argv[])
{

        itsMgr = new ModelManager("SaliencyMapService");

  nub::ref<SaliencyMapI> smap(new SaliencyMapI(*itsMgr));
        itsMgr->addSubComponent(smap);

  itsMgr->setOptionValString(&OPT_EsmInertiaHalfLife, "60");
  itsMgr->setOptionValString(&OPT_EsmIorStrength, "8.0");


        itsMgr->parseCommandLine((const int)argc, (const char**)argv, "", 0, 0);

  char adapterStr[255];
  sprintf(adapterStr, "default -p %i", BrainObjects::SaliencyMapPort);
        itsAdapter = communicator()->createObjectAdapterWithEndpoints("SaliencyMapAdapter",
      adapterStr);

        Ice::ObjectPtr object = smap.get();
  Ice::ObjectPrx objectPrx = itsAdapter->add(object, communicator()->stringToIdentity("SaliencyMap"));
  smap->initSimEvents(communicator(), objectPrx);
        itsAdapter->activate();

  itsMgr->start();

        return true;
}

// ######################################################################
int main(int argc, char** argv) {

  SaliencyMapService svc;
  return svc.main(argc, argv);
}


