/*!@file SeaBee/SubmarineAgent.C  base class for submarine agents       */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/SubmarineAgent.C $
// $Id: SubmarineAgent.C 10794 2009-02-08 06:21:09Z itti $
//
//////////////////////////////////////////////////////////////////////////

#include "SubmarineAgent.H"

// ######################################################################
//Constructors
SubmarineAgent::SubmarineAgent(rutz::shared_ptr<AgentManager> ama,
                               const std::string& name) :
  Agent(name),
  ModelComponent(mgr, name, name),
  itsAgentManager(ama),
{
  itsCurrentMission.reset(new Mission());
  itsCurrentMission->missionName = Mission::NONE;
  itsCurrentMission->missionState = Mission::NOT_STARTED;
  pthread_mutex_init(&itsCurrentMissionLock, NULL);
}

// ######################################################################
SubmarineAgent::~SubmarineAgent() { }

// ######################################################################
//Message indicating that the SubmarineAgent should update its mission
void SubmarineAgent::msgUpdateMission(Mission theMission)
{
  updateMission(theMission);
  stateChanged();
}

// ######################################################################
//Updates the SubmarineAgent's current mission
void SubmarineAgent::updateMission(Mission theMission)
{
  //  if(!itsCurrentMission->isEqual(theMission))
  //  {
      std::string doText(sformat("Updating current mission:%d",
                                 theMission.missionName));
      Do(doText);
      pthread_mutex_lock(&itsCurrentMissionLock);
      //      itsCurrentMission.reset(new Mission());
      itsCurrentMission->timeForMission = theMission.timeForMission;
      itsCurrentMission->missionState = theMission.missionState;
      itsCurrentMission->missionName = theMission.missionName;
      pthread_mutex_unlock(&itsCurrentMissionLock);
      //    }
}




// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */




