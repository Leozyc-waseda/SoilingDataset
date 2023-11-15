/*!@file Ice/IceStorm/Subscriber1.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Ice/IceStorm/Subscriber1.C $
// $Id: Subscriber1.C 10794 2009-02-08 06:21:09Z itti $
//

#include "Ice/IceStorm/Subscriber1.H"
#include "Image/ColorOps.H"

Subscriber1I::Subscriber1I(OptionManager& mgr,
    const std::string& descrName,
    const std::string& tagName ) :
  ModelComponent(mgr, descrName, tagName)
{
}

void Subscriber1I::initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx)
{
  //Get the IceStorm object
  Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 10000");
  IceStorm::TopicManagerPrx topicManager =
    IceStorm::TopicManagerPrx::checkedCast(obj);

  //Create a Subscriber1 Topic
  IceStorm::TopicPrx topic;
  try {
    IceStorm::QoS qos;
    //qos["reliability"] = "ordered";
    topic = topicManager->retrieve("Publisher1Message"); //Get the retina topic
    topic->subscribeAndGetPublisher(qos, objectPrx->ice_oneway()); //Subscribe to the retina topic
  } catch (const IceStorm::NoSuchTopic&) {
    LFATAL("Error! No retina topic found!");
  }
}


void Subscriber1I::evolve(const EventsNS::EventPtr& e,
    const Ice::Current&)
{
  LINFO("Got message %i", e->id);

  if (e->ice_isA("::EventsNS::Message1")) {
    EventsNS::Message1Ptr m1 = EventsNS::Message1Ptr::dynamicCast(e);
    LINFO("Message 1: %i '%s'", m1->m, m1->msg.c_str());
  } else if (e->ice_isA("::EventsNS::Message2")) {
    EventsNS::Message2Ptr m2 = EventsNS::Message2Ptr::dynamicCast(e);
    LINFO("Message 2: %i,%i  '%s'", m2->i, m2->j, m2->msg.c_str());
  }
  sleep(1);
}

/////////////////////////// The Retina Service to init the retina and start as a deamon ///////////////
class Subscriber1Service : public Ice::Service {
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

bool Subscriber1Service::start(int argc, char* argv[])
{

        itsMgr = new ModelManager("Subscriber1Service");

  nub::ref<Subscriber1I> s1(new Subscriber1I(*itsMgr));
        itsMgr->addSubComponent(s1);

        itsMgr->parseCommandLine((const int)argc, (const char**)argv, "", 0, 0);

        itsAdapter = communicator()->createObjectAdapterWithEndpoints("Subscriber1Adapter", "default -p 20001");

        Ice::ObjectPtr object = s1.get();
  Ice::ObjectPrx objectPrx = itsAdapter->add(object, communicator()->stringToIdentity("Subscriber1"));
  s1->initSimEvents(communicator(), objectPrx);
        itsAdapter->activate();

  itsMgr->start();

        return true;
}

// ######################################################################
int main(int argc, char** argv) {

  Subscriber1Service svc;
  return svc.main(argc, argv);
}


