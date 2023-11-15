/*!@file Neuro/SegmenterService.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/SegmenterService.C $
// $Id: SegmenterService.C 12962 2010-03-06 02:13:53Z irock $
//

#include "NeovisionII/SegmenterService.H"
#include "Image/ColorOps.H"
#include "GUI/DebugWin.H"

SegmenterI::SegmenterI(OptionManager& mgr,
    nub::ref<OutputFrameSeries> ofs,
    const std::string& descrName,
    const std::string& tagName ) :
  ModelComponent(mgr, descrName, tagName),
  itsOfs(ofs)
{
    //Subscribe to the various topics
    IceStorm::TopicPrx topicPrx;
    itsTopicsSubscriptions.push_back(TopicInfo("RetinaTopic", topicPrx));
    itsTopicsSubscriptions.push_back(TopicInfo("PrefrontalCortexTopic", topicPrx));

    nub::ref<EnvSegmenterConfigurator> esec(new EnvSegmenterConfigurator(mgr));
    addSubComponent(esec);

    itsSeg = esec->getSeg();

}

SegmenterI::~SegmenterI()
{
  unsubscribeSimEvents();
}

void SegmenterI::initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx)
{
  //Get the IceStorm object
  Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 11111");
  IceStorm::TopicManagerPrx topicManager =
    IceStorm::TopicManagerPrx::checkedCast(obj);

  //Create a Segmenter Topic
  IceStorm::TopicPrx topic;
  try {
    topic = topicManager->retrieve("SegmenterTopic"); //check if the Retina topic exists
  } catch (const IceStorm::NoSuchTopic&) {
    topic = topicManager->create("SegmenterTopic"); //The retina topic does not exists, create
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
    } catch (const Ice::Exception& e) {
      cerr << e << endl;
    }

  }
}

void SegmenterI::unsubscribeSimEvents()
{
  //Unsubscribe from all the topics we are registerd to
  for(uint i=0; i<itsTopicsSubscriptions.size();  i++)
  {
    itsTopicsSubscriptions[i].topicPrx->unsubscribe(itsObjectPrx);
  }
}


//The VC excepts a retina image and posts a saliency map
void SegmenterI::evolve(const SimEvents::EventMessagePtr& eMsg,
      const Ice::Current&)
{
  if (eMsg->ice_isA("::SimEvents::SegmenterBiasMessage")){ //Set the tracker location
    SimEvents::SegmenterBiasMessagePtr msg = SimEvents::SegmenterBiasMessagePtr::dynamicCast(eMsg);

    itsRegionsToSeg.clear();
    for(uint i=0; i<msg->regionsToSeg.size(); i++)
    {
      itsRegionsToSeg.push_back(msg->regionsToSeg[i]);
    }
  } else if(eMsg->ice_isA("::SimEvents::RetinaMessage")){

    if (itsRegionsToSeg.size() > 0)
    {
      for(uint i=0; i<itsRegionsToSeg.size(); i++)
      {
        SimEvents::RetinaMessagePtr msg = SimEvents::RetinaMessagePtr::dynamicCast(eMsg);
        Image<PixRGB<byte> > retinaImg  = Ice2Image<PixRGB<byte> > (msg->img);

        Image<byte> foamask;
        Image<PixRGB<byte> > segmentdisp;

        //center to segment
        Point2D<int> locToSeg(itsRegionsToSeg[i].loc.i, itsRegionsToSeg[i].loc.j);
        const Rectangle foa = itsSeg->getFoa(retinaImg, locToSeg,
            &foamask, &segmentdisp);


        //return the current track location
        SimEvents::SegmenterMessagePtr sMsg = new SimEvents::SegmenterMessage;
        SimEvents::SegInfo segInfo;
        segInfo.rect.tl.i = foa.left();
        segInfo.rect.tl.j = foa.top();
        segInfo.rect.br.i = foa.rightO();
        segInfo.rect.br.j = foa.bottomO();
        Image<PixRGB<byte> > cropImg = crop(retinaImg, foa);

   //     itsOfs->writeRGB(cropImg, "Segment", FrameInfo("Segement", SRC_POS));

        segInfo.img = Image2Ice(cropImg);
        sMsg->segLocs.push_back(segInfo);
        itsEventsPub->evolve(sMsg);
      }
    } else {
      SimEvents::RetinaMessagePtr msg = SimEvents::RetinaMessagePtr::dynamicCast(eMsg);
      Image<PixRGB<byte> > retinaImg  = Ice2Image<PixRGB<byte> > (msg->img);

      Image<byte> foamask;
      Image<PixRGB<byte> > segmentdisp;

      //center to segment
      Point2D<int> locToSeg(-1,-1);
      const Rectangle foa = itsSeg->getFoa(retinaImg, locToSeg,
          &foamask, &segmentdisp);

      //return the current segment
      SimEvents::SegmenterMessagePtr sMsg = new SimEvents::SegmenterMessage;
      SimEvents::SegInfo segInfo;
      segInfo.rect.tl.i = foa.left();
      segInfo.rect.tl.j = foa.top();
      segInfo.rect.br.i = foa.rightO();
      segInfo.rect.br.j = foa.bottomO();
      Image<PixRGB<byte> > cropImg = crop(retinaImg, foa);
      segInfo.img = Image2Ice(cropImg);
      sMsg->segLocs.push_back(segInfo);
      itsEventsPub->evolve(sMsg);

    }

  }
}

/////////////////////////// The VC Service to init the retina and start as a deamon ///////////////
class SegmenterService : public Ice::Service {
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

bool SegmenterService::start(int argc, char* argv[])
{

  itsMgr = new ModelManager("SegmenterService");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(*itsMgr));
  itsMgr->addSubComponent(ofs);

  nub::ref<SegmenterI> seg(new SegmenterI(*itsMgr, ofs));
  itsMgr->addSubComponent(seg);

  itsMgr->parseCommandLine((const int)argc, (const char**)argv, "", 0, 0);

  char adapterStr[255];
  sprintf(adapterStr, "default -p %i", BrainObjects::SegmenterPort);
  itsAdapter = communicator()->createObjectAdapterWithEndpoints("SegmenterAdapter",
      adapterStr);

  Ice::ObjectPtr object = seg.get();
  Ice::ObjectPrx objectPrx = itsAdapter->add(object, communicator()->stringToIdentity("Segmenter"));
  seg->initSimEvents(communicator(), objectPrx);
  itsAdapter->activate();

  itsMgr->start();

  return true;
}

// ######################################################################
int main(int argc, char** argv) {

  SegmenterService svc;
  return svc.main(argc, argv);
}


