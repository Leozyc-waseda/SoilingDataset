/*!@file Ice/IceStormUtils.H Utilities to interact with ice storm */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Ice/SimEventsUtils.H $
// $Id: SimEventsUtils.H 10794 2009-02-08 06:21:09Z itti $
//
//
#ifndef SIMEVENTSUTILS
#define SIMEVENTSUTILS

#include <IceStorm/IceStorm.h>
#include <Ice/Ice.h>

namespace SimEventsUtils
{

  struct TopicInfo
  {
    TopicInfo(const char* _name, IceStorm::TopicPrx& prx) :
      name(_name), topicPrx(prx)
    {}
    std::string name;
    IceStorm::TopicPrx topicPrx;
  };

  void initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx,
      std::vector<TopicInfo>& topicsSubscriptions)
  {
    //Get the IceStorm object
    Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 11111");
    IceStorm::TopicManagerPrx topicManager =
      IceStorm::TopicManagerPrx::checkedCast(obj);


    //subscribe  to the topics
    for(uint i=0; i<topicsSubscriptions.size();  i++)
    {
      try {
        IceStorm::QoS qos;
        topicsSubscriptions[i].topicPrx =
          topicManager->retrieve(topicsSubscriptions[i].name.c_str()); //Get the
        topicsSubscriptions[i].topicPrx->subscribeAndGetPublisher(qos, objectPrx); //Subscribe to the retina topic
      } catch (const IceStorm::NoSuchTopic&) {
        LFATAL("Error! No %s topic found!", topicsSubscriptions[i].name.c_str());
      } catch (const IceStorm::AlreadySubscribed&) {
        LDEBUG("Skipping already subscribed topic");
      }
    }
  }

  Ice::ObjectPrx getPublisher(Ice::CommunicatorPtr icPtr, std::string topicName)
  {
    //Get the IceStorm object
    Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 11111");
    IceStorm::TopicManagerPrx topicManager =
      IceStorm::TopicManagerPrx::checkedCast(obj);

    ////Create a PrefrontalCortex Topic
    IceStorm::TopicPrx topic;
    try {
      topic = topicManager->retrieve(topicName.c_str()); //check if the topic exists
    } catch (const IceStorm::NoSuchTopic&) {
      topic = topicManager->create(topicName.c_str()); //The topic does not exists, create
    }

    //Make a one way visualCortex message publisher for efficency
    return topic->getPublisher()->ice_oneway();
  }

  void createTopic(Ice::CommunicatorPtr icPtr, std::string topicName)
  {
    //Get the IceStorm object
    Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 11111");
    IceStorm::TopicManagerPrx topicManager =
      IceStorm::TopicManagerPrx::checkedCast(obj);

    ////Create a PrefrontalCortex Topic
    IceStorm::TopicPrx topic;
    try {
      topic = topicManager->retrieve(topicName.c_str()); //check if the topic exists
    } catch (const IceStorm::NoSuchTopic&) {
      topic = topicManager->create(topicName.c_str()); //The topic does not exists, create
    }

  }



  void unsubscribeSimEvents(
      std::vector<TopicInfo>& topicsSubscriptions,
      Ice::ObjectPrx objectPrx)
  {
    //Unsubscribe from all the topics we are registerd to
    for(uint i=0; i<topicsSubscriptions.size();  i++)
    {
      topicsSubscriptions[i].topicPrx->unsubscribe(objectPrx);
    }
  }

}

#endif
