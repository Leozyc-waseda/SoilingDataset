/*!@file NeovisionII/RetinaService.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/RetinaService.C $
// $Id: RetinaService.C 10794 2009-02-08 06:21:09Z itti $
//

#include "NeovisionII/RetinaService.H"

RetinaI::RetinaI(ModelManager& mgr,
    nub::ref<InputFrameSeries> ifs,
    const std::string& descrName,
    const std::string& tagName
    ) :
  ModelComponent(mgr, descrName, tagName),
  itsIfs(ifs)
{
}

void RetinaI::initSimEvents(const Ice::CommunicatorPtr icPtr, const Ice::ObjectPrx objectPrx)
{
  //Get the IceStorm object and create a new Retina topic
  Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 11111");
  IceStorm::TopicManagerPrx topicManager =
    IceStorm::TopicManagerPrx::checkedCast(obj);
  IceStorm::TopicPrx topic;
  try {
    topic = topicManager->retrieve("RetinaTopic"); //check if the Retina topic exists
  } catch (const IceStorm::NoSuchTopic&) {
    topic = topicManager->create("RetinaTopic"); //The retina topic does not exists, create
  }

  //Make a one way retina message publisher for efficency
  Ice::ObjectPrx pub = topic->getPublisher()->ice_oneway();
  itsEventsPub = SimEvents::EventsPrx::uncheckedCast(pub);
}

//We dont get any messages for now
void RetinaI::evolve(const SimEvents::EventMessagePtr& eMsg,
    const Ice::Current&)
{
}

///The run function which runes in a sperate thread
void RetinaI::run()
{
  while(1)
  {
    getFrame();
    if (itsCurrentImg.initialized())
    {
      SimEvents::RetinaMessagePtr retinaMsg = new SimEvents::RetinaMessage;
      retinaMsg->img = Image2Ice(itsCurrentImg);
      itsEventsPub->evolve(retinaMsg);
    }
    usleep(10000);
  }
}

//Read a frame into currentImg
void RetinaI::getFrame()
{
  itsCurrentImg.clear();
  const FrameState is = itsIfs->updateNext();
  if (is == FRAME_COMPLETE) return;
  //grab the images
  GenericFrame input = itsIfs->readFrame();
  if (!input.initialized()) return;
  itsCurrentImg = input.asRgb();
}


ImageIceMod::ImageIce RetinaI::getOutput(const Ice::Current&)
{
  return Image2Ice(itsCurrentImg);
}

/////////////////////////// The Retina Service to init the retina and start as a deamon ///////////////
class RetinaService : public Ice::Service {
  protected:
    virtual bool start(int, char* argv[]);
    virtual bool stop();
  private:
    Ice::ObjectAdapterPtr itsAdapter;
    ModelManager *itsMgr;
};

bool RetinaService::stop()
{
  if (itsMgr)
    delete itsMgr;
  return true;
}

bool RetinaService::start(int argc, char* argv[])
{

        itsMgr = new ModelManager("RetinaService");

        nub::ref<InputFrameSeries> ifs(new InputFrameSeries(*itsMgr));
        itsMgr->addSubComponent(ifs);

  nub::ref<RetinaI> retina(new RetinaI(*itsMgr, ifs));
        itsMgr->addSubComponent(retina);

        itsMgr->parseCommandLine((const int)argc, (const char**)argv, "", 0, 0);

  char adapterStr[255];
  sprintf(adapterStr, "default -p %i", BrainObjects::RetinaPort);
        itsAdapter = communicator()->createObjectAdapterWithEndpoints("RetinaAdapter", adapterStr);

        Ice::ObjectPtr object = retina.get();
  Ice::ObjectPrx objectPrx = itsAdapter->add(object, communicator()->stringToIdentity("Retina"));
  retina->initSimEvents(communicator(), objectPrx);
        itsAdapter->activate();

  itsMgr->start();

  //Start the retina evolve thread
  IceUtil::ThreadPtr retinaThread = retina.get();
  retinaThread->start();

        return true;
}

// ######################################################################
int main(int argc, char** argv) {

  RetinaService svc;
  return svc.main(argc, argv);
}

