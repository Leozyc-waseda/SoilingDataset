/*!@file NeovisionII/PrimaryMotorCortexService.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/PrimaryMotorCortexService.C $
// $Id: PrimaryMotorCortexService.C 12962 2010-03-06 02:13:53Z irock $
//

#include "NeovisionII/PrimaryMotorCortexService.H"
#include "Util/sformat.H"
#include "Image/DrawOps.H"

using namespace std;
using namespace ImageIceMod;

PrimaryMotorCortexI::PrimaryMotorCortexI(OptionManager& mgr,
    const std::string& descrName,
    const std::string& tagName) :
  ModelComponent(mgr, descrName, tagName)
{

  //Subscribe to the various topics
  IceStorm::TopicPrx topicPrx;
  itsTopicsSubscriptions.push_back(TopicInfo("PrefrontalCortexTopic", topicPrx));
  itsTopicsSubscriptions.push_back(TopicInfo("HomeInterfaceTopic", topicPrx));

  itsScorbot = nub::soft_ref<Scorbot>(new Scorbot(mgr));
  addSubComponent(itsScorbot);

  itsHomePosition.base = 10413;
  itsHomePosition.sholder = -9056;
  itsHomePosition.elbow = 11867;
  itsHomePosition.wristRoll = -6707;
  itsHomePosition.wristPitch = 4513;
  itsHomePosition.gripper = -1434;
  itsHomePosition.ex1 = 500;
}



PrimaryMotorCortexI::~PrimaryMotorCortexI()
{
  unsubscribeSimEvents();
}

void PrimaryMotorCortexI::start2()
{
}

void PrimaryMotorCortexI::stop1()
{
  //Set all of the motors to stay where they are, and turn on the control box
  itsScorbot->stopControl();
  itsScorbot->motorsOff();
  LINFO("Done");
}

void PrimaryMotorCortexI::init()
{
  //Set all of the motors to stay where they are, and turn on the control box
  itsScorbot->motorsOn();
  itsScorbot->startControl();
  Scorbot::ArmPos currArmPos = itsScorbot->getArmPos();
  itsScorbot->setArmPos(currArmPos);

  goHome();
  LINFO("!At Home.");

}


void PrimaryMotorCortexI::initSimEvents(Ice::CommunicatorPtr icPtr, Ice::ObjectPrx objectPrx)
{
  //Get the IceStorm object
  Ice::ObjectPrx obj = icPtr->stringToProxy("SimEvents/TopicManager:tcp -p 11111");
  IceStorm::TopicManagerPrx topicManager =
    IceStorm::TopicManagerPrx::checkedCast(obj);

  //Create a Segmenter Topic
  IceStorm::TopicPrx topic;
  try {
    topic = topicManager->retrieve("PrimaryMotorCortexTopic"); //check if the Retina topic exists
  } catch (const IceStorm::NoSuchTopic&) {
    topic = topicManager->create("PrimaryMotorCortexTopic"); //The retina topic does not exists, create
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

void PrimaryMotorCortexI::unsubscribeSimEvents()
{
  //Unsubscribe from all the topics we are registerd to
  for(uint i=0; i<itsTopicsSubscriptions.size();  i++)
  {
    itsTopicsSubscriptions[i].topicPrx->unsubscribe(itsObjectPrx);
  }
}


void PrimaryMotorCortexI::moveObject(Point3D<float> P1)
{
  std::vector<std::string> PositionNames;

  std::vector<Scorbot::ArmPos> armPositions;
  Scorbot::ArmPos tmpArmPos;

  //Move the arm right above the object
  Point3D<float> aboveObjectPoint = P1;
  aboveObjectPoint.z = aboveObjectPoint.z + 100;
  tmpArmPos = itsScorbot->getIKArmPos(aboveObjectPoint);
  armPositions.push_back(tmpArmPos);
  PositionNames.push_back("Above Object");

  //Open the gripper
  tmpArmPos.gripper = 0;
  armPositions.push_back(tmpArmPos);
  PositionNames.push_back("Open Gripper");

  //Move to the actual object
  tmpArmPos = itsScorbot->getIKArmPos(P1);
  armPositions.push_back(tmpArmPos);
  PositionNames.push_back("Object");

  //Slide the object back
  Point3D<float> slidePoint = P1;
  slidePoint.y = slidePoint.y - 1000;
  tmpArmPos = itsScorbot->getIKArmPos(slidePoint);
  armPositions.push_back(tmpArmPos);
  PositionNames.push_back("Slide Object");

  //Raise the arm again
  slidePoint.z = slidePoint.z + 100;
  tmpArmPos = itsScorbot->getIKArmPos(slidePoint);
  armPositions.push_back(tmpArmPos);
  PositionNames.push_back("Above Slid Object");

  //Move the arm back on the slide
  tmpArmPos.ex1 = 1000;
  armPositions.push_back(tmpArmPos);
  PositionNames.push_back("Sliding Back");


  for(uint i=0; i<armPositions.size(); i++)
  {
    LINFO("Going To %s\n", PositionNames.at(i).c_str());
    itsScorbot->setArmPos(armPositions[i]);
    sleep(1);
    int timer = 0;
    while(!itsScorbot->moveDone() && timer < 300)
    {
      usleep(10000);
      timer++;
    }
    if(PositionNames.at(i) == "Object")
      sleep(5);
//    LINFO("Pos %i reached in timer %i", i, timer);
  }


  LINFO("Finished Moving Object - Going Home");
  goHome();
}

void PrimaryMotorCortexI::goHome()
{

  //Move Arm Home
  Scorbot::ArmPos itsHomePosition;
  itsHomePosition.base = 10413;
  itsHomePosition.sholder = -4500;
  itsHomePosition.elbow = 7567;
  itsHomePosition.wristRoll = 8371;
  itsHomePosition.wristPitch = 4811;
  itsHomePosition.gripper = -1434;
  itsHomePosition.ex1 = 500;

  std::vector<Scorbot::ArmPos> armPositions;
  Scorbot::ArmPos tmpArmPos = itsScorbot->getArmPos();

  LINFO("Add positions");
  {
    //Open the Gripper
    tmpArmPos.gripper = 0;
    armPositions.push_back(tmpArmPos);

    //Move the Shoulder and Elbow up
    tmpArmPos.sholder = itsHomePosition.sholder;
    tmpArmPos.elbow   = itsHomePosition.elbow;
    tmpArmPos.wristRoll   = itsHomePosition.wristRoll;
    tmpArmPos.wristPitch   = itsHomePosition.wristPitch;
    armPositions.push_back(tmpArmPos);

    //Move the slide back
    tmpArmPos.ex1 = itsHomePosition.ex1;
    armPositions.push_back(tmpArmPos);

    //Rotate the base
    tmpArmPos.base = itsHomePosition.base;
    armPositions.push_back(tmpArmPos);
  }

  LINFO("Move to positions");

  for(uint i=0; i<armPositions.size(); i++)
  {
    LINFO("Going to Pos %i...", i);
    itsScorbot->setArmPos(armPositions[i]);
    sleep(1);
    int timer = 0;
    while(!itsScorbot->moveDone() && timer < 300)
    {
      usleep(10000);
      timer++;
    }
    LINFO("Pos %i reached in timer %i", i, timer);
  }
}


void PrimaryMotorCortexI::evolve(const SimEvents::EventMessagePtr& eMsg,
    const Ice::Current&)
{
  LINFO("Got message");
  if (eMsg->ice_isA("::SimEvents::PrimaryMotorCortexBiasMessage"))
  {
    SimEvents::PrimaryMotorCortexBiasMessagePtr msg = SimEvents::PrimaryMotorCortexBiasMessagePtr::dynamicCast(eMsg);

    if (msg->moveArm)
    {
      LINFO("Move Object");
      float x = msg->armPos.x;
      float y = msg->armPos.y;
      float z = msg->armPos.z;

      moveObject(Point3D<float>(x,y,z));

    }
  }

}

/////////////////////////// The VC Service to init the retina and start as a deamon ///////////////
class PrimaryMotorCortexService : public Ice::Service {
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

bool PrimaryMotorCortexService::start(int argc, char* argv[])
{

  itsMgr = new ModelManager("PrimaryMotorCortexService");

  nub::ref<PrimaryMotorCortexI> pmc(new PrimaryMotorCortexI(*itsMgr));
  itsMgr->addSubComponent(pmc);

  itsMgr->parseCommandLine((const int)argc, (const char**)argv, "", 0, 0);

  char adapterStr[255];
  sprintf(adapterStr, "default -p %i", BrainObjects::PrimaryMotorCortexPort);
  itsAdapter = communicator()->createObjectAdapterWithEndpoints("PMCAdapter",
      adapterStr);

  Ice::ObjectPtr object = pmc.get();
  Ice::ObjectPrx objectPrx = itsAdapter->add(object, communicator()->stringToIdentity("PMC"));
  pmc->initSimEvents(communicator(), objectPrx);
  itsAdapter->activate();

  itsMgr->start();
  pmc->init();

  return true;
}

// ######################################################################
int main(int argc, char** argv) {

  PrimaryMotorCortexService svc;
  return svc.main(argc, argv);
}
