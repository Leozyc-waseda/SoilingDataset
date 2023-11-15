/*!@file Beosub/BeeBrain/PreFrontalCortex.C
 decision maker for strategy to complete missions                       */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/PreFrontalCortex.C $
// $Id: PreFrontalCortex.C 8623 2007-07-25 17:57:51Z rjpeters $

#include "BeoSub/BeeBrain/PreFrontalCortex.H"

// ######################################################################
PreFrontalCortexAgent::PreFrontalCortexAgent(std::string name) : Agent(name)
{
  itsRunTimer.reset(new Timer(1000000));
  itsCurrentMissionTimer.reset(new Timer(1000000));
}

// ######################################################################
PreFrontalCortexAgent::PreFrontalCortexAgent
(std::string name, rutz::shared_ptr<AgentManagerA> ama) :
  Agent(name),  itsAgentManager(ama)
{
  itsRunTimer.reset(new Timer(1000000));
  itsCurrentMissionTimer.reset(new Timer(1000000));
}

// ######################################################################
PreFrontalCortexAgent::~PreFrontalCortexAgent()
{ }

// ######################################################################
void PreFrontalCortexAgent::start()
{
  populateMissions();
  currentMission = itsMissions[0];
//  rutz::shared_ptr<OceanObject> cross(new OceanObject());
//   cross->setType(OceanObject::CROSS);
//   itsAgentManager->pushCommand((CommandType)(SEARCH_OCEAN_OBJECT), POSITION, cross);
}


// ######################################################################
void PreFrontalCortexAgent::populateMissions()
{
  LINFO("Populating Missions...");

  // go through gate
  rutz::shared_ptr<Mission> gate(new Mission());
  gate->timeForMission = 90;
  gate->missionName = GATE;
  gate->missionState = NOT_STARTED;
  rutz::shared_ptr<OceanObject> cross(new OceanObject(OceanObject::CROSS));
  gate->missionObjects.push_back(cross);
  itsMissions.push_back(gate);

//   //  hit first buoy
//   rutz::shared_ptr<Mission> firstBuoy;
//   firstBuoy->timeForMission = 210;
//   firstBuoy->missionName = HIT_START_BUOY;
//   rutz::shared_ptr<OceanObject> buoy(new OceanObject(OceanObject::BUOY));
//   firstBuoy->missionObjects.push_back(buoy);
//   itsMissions.push_back(firstBuoy);

//   //drop marker in first bin
//   rutz::shared_ptr<Mission> firstBin;
//   firstBin->timeForMission = 150;
//   firstBin->missionName = FIRST_BIN;
//   rutz::shared_ptr<OceanObject> firstPipe(new OceanObject(OceanObject::PIPE));
//   rutz::shared_ptr<OceanObject> bin(new OceanObject(OceanObject::BIN));
//   firstBin->missionObjects.push_back(firstPipe);
//   firstBin->missionObjects.push_back(bin);
//   itsMissions.push_back(firstBin);

//   //hit second buoy and drop marker in second bin
//   rutz::shared_ptr<Mission> secondBin;
//   secondBin->timeForMission = 150;
//   secondBin->missionName = SECOND_BIN;
//   rutz::shared_ptr<OceanObject> secondPipe(new OceanObject(OceanObject::PIPE));
//   rutz::shared_ptr<OceanObject> secondBuoy(new OceanObject(OceanObject::BUOY));
//   rutz::shared_ptr<OceanObject> coverBin(new OceanObject(OceanObject::BIN));
//   secondBin->missionObjects.push_back(secondPipe);
//   secondBin->missionObjects.push_back(secondBuoy);
//   secondBin->missionObjects.push_back(coverBin);
//   itsMissions.push_back(secondBin);

//   //find and recover the cross
//   rutz::shared_ptr<Mission> treasure;
//   treasure->timeForMission = 300;
//   treasure->missionName = GET_TREASURE;
//   rutz::shared_ptr<OceanObject> pinger(new OceanObject(OceanObject::PINGER));
//   rutz::shared_ptr<OceanObject> cross(new OceanObject(OceanObject::CROSS));
//   treasure->missionObjects.push_back(pinger);
//   treasure->missionObjects.push_back(cross);
//   itsMissions.push_back(treasure);

//   // initialize all mission status
//   for(uint i = 0; i < itsMissions.size(); i++)
//     {
//       itsMissions[i]->missionState = NOT_STARTED;
//     }

  stateChanged();
}

// ######################################################################
void PreFrontalCortexAgent::lookForObjects
(std::vector<rutz::shared_ptr<OceanObject> > oceanObjects,
 bool startLooking)
{
  CommandType cmdType = SEARCH_OCEAN_OBJECT_CMD;

  if(startLooking == false)
    {
      cmdType = STOP_SEARCH_OCEAN_OBJECT_CMD;
    }

  for(uint i = 0; i < oceanObjects.size(); i++)
    {
      rutz::shared_ptr<OceanObject> o = oceanObjects[i];
      switch(o->getType())
        {
        case OceanObject::CROSS:
          itsAgentManager->pushCommand
            (cmdType,
             POSITION,
             o);
          itsAgentManager->pushCommand
            (cmdType,
             ORIENTATION,
             o);
          itsAgentManager->pushCommand
            (cmdType,
             MASS,
             o);
          break;
        case OceanObject::BIN:
          itsAgentManager->pushCommand
            (cmdType,
             POSITION,
             o);
          break;
        case OceanObject::BUOY:
          itsAgentManager->pushCommand
            (cmdType,
             POSITION,
             o);
          itsAgentManager->pushCommand
            (cmdType,
             FREQUENCY,
             o);
          break;
        case OceanObject::PIPE:
          itsAgentManager->pushCommand
            (cmdType,
             POSITION,
             o);
          itsAgentManager->pushCommand
            (cmdType,
             ORIENTATION,
             o);
          break;
        case OceanObject::PINGER:
          itsAgentManager->pushCommand
            (cmdType,
             ORIENTATION,
             o);
          itsAgentManager->pushCommand
            (cmdType,
             DISTANCE,
             o);
          break;
        default:
          LERROR("unknown ocean object type");
          break;
        }
    }
}

// ######################################################################
void PreFrontalCortexAgent::msgOceanObjectUpdate()
{
  LINFO("Recieved msgOceanObjectUpdate");
  stateChanged();
}

// ######################################################################
void PreFrontalCortexAgent::msgMovementComplete()
{
  LINFO("Recieved msgMovementComplete");
  stateChanged();
}

// ######################################################################
bool PreFrontalCortexAgent::pickAndExecuteAnAction()
{
  // if we almost run out of time for the overall run
  if(itsRunTimer->get() > AUVSI07_COMPETITION_TIME)
    {
      // move on to another mission
      //currentMission->missionState = FAILED;
      //return true;
      LINFO("FIX ME - Almost out of time");
    }

  // find the next uncompleted mission
  currentMission.reset();
  for(uint i = 0; i < itsMissions.size(); i++)
    if(itsMissions[i]->missionState != COMPLETED)
      {
        currentMission = itsMissions[i]; break;
      }

  if(currentMission.is_invalid())
    {
      //unloadKoolAid();
      return false;
    }

  if(currentMission->missionName == GATE)
    return executeGateMission();
  if(currentMission->missionName == HIT_START_BUOY)
    return executeBuoyMission();
  if(currentMission->missionName == FIRST_BIN)
    return executeFirstBinMission();
  if(currentMission->missionName == SECOND_BIN)
    return executeSecondBinMission();
  if(currentMission->missionName == GET_TREASURE)
    return executeCrossMission();

  return false;
}

// ######################################################################
bool PreFrontalCortexAgent::executeGateMission()
{
  // if current mission timer is > pre-allocated time
  if(currentMission->missionState != NOT_STARTED &&
     itsCurrentMissionTimer->get() > currentMission->timeForMission)
    {
      // move on to another mission
      currentMission->missionState = FAILED;
      return true;
    }

  // start the mission
  if(currentMission->missionState == NOT_STARTED)
    {
      // set the run and current mission
      itsRunTimer->reset();
      itsCurrentMissionTimer->reset();

      LINFO("Starting Mission");

      lookForObjects(currentMission->missionObjects);

      rutz::shared_ptr<ComplexMovement>
        goThroughGate(new ComplexMovement());
      goThroughGate->addMove(&PreMotorComplex::forward, ONCE, 5.0);
      itsPreMotorComplex->run(goThroughGate);
      currentMission->missionState = SEARCHING;
      return true;
    }
  // searching
  else if(currentMission->missionState == SEARCHING)
    {
      LINFO("Searching for cross");

      rutz::shared_ptr<OceanObject> theCross;

      for(uint i = 0; i < currentMission->missionObjects.size(); i++)
        {
          if(currentMission->missionObjects[i]->getType()
             == OceanObject::CROSS)
            {
              theCross = currentMission->missionObjects[i];
              break;
            }
        }

      // center on the cross
      if(theCross.is_valid()
         && theCross->getPosition().isValidY()
         && theCross->getPosition().isValidX())
        {
          currentMission->missionState = CENTERING;
        }

      return true;
    }
  // if currently centering
  else if(currentMission->missionState == CENTERING)
    {
      LINFO("Centering on cross");
      rutz::shared_ptr<OceanObject> theCross;
      for(uint i = 0; i < currentMission->missionObjects.size(); i++)
        {
          if(currentMission->missionObjects[i]->getType()
             == OceanObject::CROSS)
            {
              theCross = currentMission->missionObjects[i];
              break;
            }
        }

      if(theCross.is_valid()
         && abs(theCross->getPosition().x - 160) < 20
         && abs(theCross->getPosition().y - 120) < 20)
        {
          itsPreMotorComplex->msgHalt();

          currentMission->missionState = COMPLETED;
        }
      else if(theCross.is_valid())
        {
          rutz::shared_ptr<ComplexMovement>
            centerOnCross(new ComplexMovement());
          centerOnCross->addMove(&PreMotorComplex::vis_center, REPEAT, theCross->getPositionPtr());
          itsPreMotorComplex->run(centerOnCross);
        }

      return true;
    }
  // mission is completed
  else if(currentMission->missionState == COMPLETED)
    {
      LINFO("Move through gate complete");
      lookForObjects(currentMission->missionObjects, false);
      //               rutz::shared_ptr<ComplexMovement>
      //                 victoryDance(new ComplexMovement());
      //               Angle a(120);
      //               victoryDance->addOnceMove(&PreMotorComplex::dive, 3.0 , Angle(-1.0));
      //               itsPreMotorComplex->run(victoryDance);
      return false;
      //               setNextMission();
    }

  return false;
}

// ######################################################################
bool PreFrontalCortexAgent::executeBuoyMission()
{
  // if current mission timer is > pre-allocated time
  if(currentMission->missionState != NOT_STARTED &&
     itsCurrentMissionTimer->get() > currentMission->timeForMission)
    {
      // move on to another mission
      currentMission->missionState = FAILED;
      return true;
    }

  // start the mission
  if(currentMission->missionState == NOT_STARTED)
    {
      LINFO("Starting Buoy Mission");
//       itsCurrentMissionTimer->reset();

//       lookForObjects(currentMission->missionObjects);
//       rutz::shared_ptr<ComplexMovement>
//         strafe(new ComplexMovement());
//       strafe->addMove(&PreMotorComplex::forward, ONCE, 5.0);
//       itsPreMotorComplex->run(goThroughGate);

       currentMission->missionState = SEARCHING;
      return true;
    }
  else if(currentMission->missionState == SEARCHING)
    {
      return true;
    }
  else if(currentMission->missionState == CENTERING)    {
      return true;
    }
  // mission is completed
  else if(currentMission->missionState == COMPLETED)
    {
      return true;
    }
  return false;
}

// ######################################################################
bool PreFrontalCortexAgent::executeFirstBinMission()
{
  // if current mission timer is > pre-allocated time
  if(currentMission->missionState != NOT_STARTED &&
     itsCurrentMissionTimer->get() > currentMission->timeForMission)
    {
      // move on to another mission
      currentMission->missionState = FAILED;
      return true;
    }

  // start the mission
  if(currentMission->missionState == NOT_STARTED)
    {
      LINFO("Starting First Bin Mission");
      itsCurrentMissionTimer->reset();

      currentMission->missionState = SEARCHING;
      return true;
    }
  else if(currentMission->missionState == SEARCHING)
    {
      return false;
    }
  else if(currentMission->missionState == CENTERING)
    {
      return true;
    }
  // mission is completed
  else if(currentMission->missionState == COMPLETED)
    {
      return true;
    }
  return false;
}

// ######################################################################
bool PreFrontalCortexAgent::executeSecondBinMission()
{
  // if current mission timer is > pre-allocated time
  if(currentMission->missionState != NOT_STARTED &&
     itsCurrentMissionTimer->get() > currentMission->timeForMission)
    {
      // move on to another mission
      currentMission->missionState = FAILED;
      return true;
    }

  // start the mission
  if(currentMission->missionState == NOT_STARTED)
    {
      LINFO("Starting Second Bin Mission");
      itsCurrentMissionTimer->reset();

      currentMission->missionState = SEARCHING;
      return true;
    }
  else if(currentMission->missionState == SEARCHING)
    {
      return false;
    }
  else if(currentMission->missionState == CENTERING)
    {
      return true;
    }
  // mission is completed
  else if(currentMission->missionState == COMPLETED)
    {
      return true;
    }
  return false;
}

// ######################################################################
bool PreFrontalCortexAgent::executeCrossMission()
{
  // if current mission timer is > pre-allocated time
  if(currentMission->missionState != NOT_STARTED &&
     itsCurrentMissionTimer->get() > currentMission->timeForMission)
    {
      // move on to another mission
      currentMission->missionState = FAILED;
      return true;
    }

  // start the mission
  if(currentMission->missionState == NOT_STARTED)
    {
      LINFO("Starting Cross Mission");
      itsCurrentMissionTimer->reset();

      currentMission->missionState = SEARCHING;
      return true;
    }
  else if(currentMission->missionState == SEARCHING)
    {
      return false;
    }
  else if(currentMission->missionState == CENTERING)
    {
      return true;
    }
  // mission is completed
  else if(currentMission->missionState == COMPLETED)
    {
      return true;
    }
  return false;
}

// ######################################################################
// bool PreFrontalCortexAgent::setNextMission()
// {

// }

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

