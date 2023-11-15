/*!@file Ice/IceStorm/publisher1.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Ice/IceStorm/Publisher1.C $
// $Id: Publisher1.C 10794 2009-02-08 06:21:09Z itti $
//

#include "Ice/IceStorm/Publisher1.H"
#include "Image/ColorOps.H"

Publisher1I::Publisher1I(OptionManager& mgr,
    const std::string& descrName,
    const std::string& tagName ) :
  ModelComponent(mgr, descrName, tagName)
{
}

void Publisher1I::initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx)
{
  //Get the IceStorm object
  Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 10000");
  IceStorm::TopicManagerPrx topicManager =
    IceStorm::TopicManagerPrx::checkedCast(obj);

  //Create a Publisher1 Topic
  IceStorm::TopicPrx topic;
  try {
    topic = topicManager->retrieve("Publisher1Message"); //check if the Retina topic exists
  } catch (const IceStorm::NoSuchTopic&) {
    topic = topicManager->create("Publisher1Message"); //The retina topic does not exists, create
  }
  //Make a one way Publisher1 message publisher for efficency
  Ice::ObjectPrx pub = topic->getPublisher()->ice_oneway();
  itsMessagePub = EventsNS::EventsPrx::uncheckedCast(pub);
}


void Publisher1I::evolve(const EventsNS::EventPtr& e,
    const Ice::Current&)
{
  LINFO("Got message");
  //itsMessage->Publisher1Output(Image2Ice(smap)); //post the saliency map message to the SimEvents
}

void Publisher1I::run()
{
  int i = 0;
  while(1)
  {
    EventsNS::Message1Ptr msg1 = new EventsNS::Message1;
    msg1->id = 10;
    msg1->m = i;
    msg1->msg = "This is message 1";
    LINFO("Sending message 1: %i", i);
    itsMessagePub->evolve(msg1);
    i++;
    usleep(10000);
  }
}

/////////////////////////// The Retina Service to init the retina and start as a deamon ///////////////
class Publisher1Service : public Ice::Service {
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

bool Publisher1Service::start(int argc, char* argv[])
{

        itsMgr = new ModelManager("Publisher1Service");

  nub::ref<Publisher1I> p1(new Publisher1I(*itsMgr));
        itsMgr->addSubComponent(p1);

        itsMgr->parseCommandLine((const int)argc, (const char**)argv, "", 0, 0);

        itsAdapter = communicator()->createObjectAdapterWithEndpoints("Publisher1Adapter", "default -p 20000");

        Ice::ObjectPtr object = p1.get();
  Ice::ObjectPrx objectPrx = itsAdapter->add(object, communicator()->stringToIdentity("Publisher1"));
  p1->initSimEvents(communicator(), objectPrx);
        itsAdapter->activate();

  itsMgr->start();

  //Start the evolve thread
  IceUtil::ThreadPtr p1Thread = p1.get();
  p1Thread->start();

        return true;
}

// ######################################################################
int main(int argc, char** argv) {

  Publisher1Service svc;
  return svc.main(argc, argv);
}


