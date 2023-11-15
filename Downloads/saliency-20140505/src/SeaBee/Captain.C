/*!@file SeaBee/Captain.C
  decides sumbarine's current mission and strategically accomplishes
  submarine goal                                                        */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/Captain.C $
// $Id: Captain.C 10794 2009-02-08 06:21:09Z itti $

#include "SeaBee/Captain.H"

// ######################################################################
CaptainAgent::CaptainAgent(OptionManager& mgr,
                           nub::soft_ref<AgentManager> ama,
                           const std::string& name) :
  SubmarineAgent(mgr, ama, name)
{
   itsIsInitialized = false;
}

// ######################################################################
CaptainAgent::~CaptainAgent()
{ }

// ######################################################################
void CaptainAgent::start()
{
  populateMissions();
  stateChanged();
}


// ######################################################################
void CaptainAgent::msgGoThroughGateComplete()
{
  if(itsCurrentMission->missionName == Mission::GATE &&
     itsCurrentMission->missionState == Mission::IN_PROGRESS)
    {
      itsCurrentMission->missionState = Mission::COMPLETED;
    }

  LINFO("HEY");
  stateChanged();
}

// ######################################################################
void CaptainAgent::populateMissions()
{
  Do("Populating Missions...");

  //add go through gate mission
  addMission(300,
             Mission::NOT_STARTED,
             Mission::GATE);

  //add first bin mission
  addMission(300,
             Mission::NOT_STARTED,
             Mission::FIRST_BIN);

  itsIsInitialized = true;

}

// ######################################################################
void CaptainAgent::addMission(uint missionTime,
                              Mission::MissionState ms,
                              Mission::MissionName mn)
{
  Do("Adding a mission to mission list.");

  rutz::shared_ptr<Mission> theMission(new Mission());
  theMission->timeForMission = missionTime;
  theMission->missionState = ms;
  theMission->missionName = mn;

  itsMissions.push_back(theMission);
}

// ######################################################################
bool CaptainAgent::pickAndExecuteAnAction()
{
  if(itsIsInitialized)
    {
      uint missionListSize = itsMissions.size();

      if(itsCurrentMission->missionState != Mission::IN_PROGRESS)
        {
          for(uint i = 0; i < missionListSize; i++)
            {

              rutz::shared_ptr<Mission> m = itsMissions.at(i);

              if(m->missionState == Mission::NOT_STARTED)
                {
                  m->missionState = Mission::IN_PROGRESS;
                  updateMission(*m);
                  itsAgentManager->updateAgentsMission(*m);
                  return true;
                }
            }
        }
    }

  return false;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

