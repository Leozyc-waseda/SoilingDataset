/*!@file Neuro/VisualCortexService.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/VisualCortexService.C $
// $Id: VisualCortexService.C 12962 2010-03-06 02:13:53Z irock $
//

#include "NeovisionII/VisualCortexService.H"
#include "Image/ColorOps.H"
#include "GUI/DebugWin.H"

VisualCortexI::VisualCortexI(OptionManager& mgr,
    const std::string& descrName,
    const std::string& tagName ) :
  ModelComponent(mgr, descrName, tagName)
{
    itsEvc = nub::soft_ref<EnvVisualCortex>(new EnvVisualCortex(mgr));
    addSubComponent(itsEvc);

    //Subscribe to the various topics
    IceStorm::TopicPrx topicPrx;
    TopicInfo tInfo("RetinaTopic", topicPrx);
    itsTopicsSubscriptions.push_back(tInfo);

}

VisualCortexI::~VisualCortexI()
{
  unsubscribeSimEvents();
}

void VisualCortexI::initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx)
{
  //Get the IceStorm object
  Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 11111");
  IceStorm::TopicManagerPrx topicManager =
    IceStorm::TopicManagerPrx::checkedCast(obj);

  //Create a VisualCortex Topic
  IceStorm::TopicPrx topic;
  try {
    topic = topicManager->retrieve("VisualCortexTopic"); //check if the Retina topic exists
  } catch (const IceStorm::NoSuchTopic&) {
    topic = topicManager->create("VisualCortexTopic"); //The retina topic does not exists, create
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

void VisualCortexI::unsubscribeSimEvents()
{
  //Unsubscribe from all the topics we are registerd to
  for(uint i=0; i<itsTopicsSubscriptions.size();  i++)
  {
    LINFO("Unsubscribe from %s", itsTopicsSubscriptions[i].name.c_str());
    itsTopicsSubscriptions[i].topicPrx->unsubscribe(itsObjectPrx);
  }
}


//The VC excepts a retina image and posts a saliency map
void VisualCortexI::evolve(const SimEvents::EventMessagePtr& eMsg,
      const Ice::Current&)
{
  if(eMsg->ice_isA("::SimEvents::RetinaMessage")){
    SimEvents::RetinaMessagePtr msg = SimEvents::RetinaMessagePtr::dynamicCast(eMsg);
    Image<PixRGB<byte> > retinaImg  = Ice2Image<PixRGB<byte> > (msg->img);

    itsEvc->input(retinaImg);
    Image<byte> smap =  itsEvc->getVCXmap();

    //send a visual cortex message
    SimEvents::VisualCortexMessagePtr vcMsg = new SimEvents::VisualCortexMessage;
    vcMsg->vco = Image2Ice(smap);
    try {
      itsEventsPub->evolve(vcMsg);
    } catch(const IceStorm::NoSuchTopic&) {
    }
  }
}

/////////////////////////// The VC Service to init the retina and start as a deamon ///////////////
class VisualCortexService : public Ice::Service {
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

bool VisualCortexService::start(int argc, char* argv[])
{

        itsMgr = new ModelManager("VisualCortexService");

  nub::ref<VisualCortexI> vc(new VisualCortexI(*itsMgr));
        itsMgr->addSubComponent(vc);

        itsMgr->parseCommandLine((const int)argc, (const char**)argv, "", 0, 0);

  char adapterStr[255];
  sprintf(adapterStr, "default -p %i", BrainObjects::VisualCortexPort);
        itsAdapter = communicator()->createObjectAdapterWithEndpoints("VisualCortexAdapter",
      adapterStr);

        Ice::ObjectPtr object = vc.get();
  Ice::ObjectPrx objectPrx = itsAdapter->add(object, communicator()->stringToIdentity("VisualCortex"));
  vc->initSimEvents(communicator(), objectPrx);
        itsAdapter->activate();

  itsMgr->start();

        return true;
}

// ######################################################################
int main(int argc, char** argv) {

  VisualCortexService svc;
  return svc.main(argc, argv);
}


