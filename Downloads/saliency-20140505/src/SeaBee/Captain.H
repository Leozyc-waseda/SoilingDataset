/*!@file SeaBee/Captain.H
  decides sumbarine's current mission and strategically accomplishes
  submarine goals                                                       */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/Captain.H $
// $Id: Captain.H 10794 2009-02-08 06:21:09Z itti $

#ifndef __CAPTAIN_AGENT_H__
#define __CAPTAIN_AGENT_H__

#include "SubmarineAgent.H"
#include "Movement.H"
#include "ForwardVision.H"
#include "SonarListen.H"
#include "OceanObject.H"

#include "Util/Timer.H"

class ForwardVisionAgent;

class CaptainAgent : public SubmarineAgent
{

public:

  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{
  CaptainAgent(OptionManager& mgr,
               nub::soft_ref<AgentManager> ama,
               const std::string& name = "CaptainAgent");

  ~CaptainAgent();
  //@}

  // ######################################################################
  //! @name CaptainAgent member functions
  //@{

  //! Messages
  void msgGoThroughGateComplete();
  void msgMovementComplete();

 //! Scheduler
  bool pickAndExecuteAnAction();

  void start();

  inline void setMovementAgent
  (nub::soft_ref<MovementAgent> pmc);

  inline void setForwardVisionAgent
  (nub::soft_ref<ForwardVisionAgent> fv);
  //@}

private:

  //! Actions
  void populateMissions();
  void addMission(uint missionTime,
                  Mission::MissionState ms,
                  Mission::MissionName mn);

  //  void search(SEARCH_TYPE type, /*Ocean_Object state*/);
  //  void surface();
  //  void killSwitch();

  //! Variables
  nub::soft_ref<ForwardVisionAgent> itsForwardVisionAgent;
  nub::soft_ref<MovementAgent> itsMovementAgent;

  //! timer for the overall run
  rutz::shared_ptr<Timer> itsRunTimer;

  //! timer for the current mission
  rutz::shared_ptr<Timer> itsCurrentMissionTimer;

  //!
  std::vector<rutz::shared_ptr<Mission> > itsMissions;

  bool currentMovementDone;
  bool itsIsInitialized;

  //!
  void lookForObjects(std::vector<rutz::shared_ptr<OceanObject> > oceanObjects,
                      bool startLooking = true);
};

// ######################################################################
inline void CaptainAgent::setMovementAgent
(nub::soft_ref<MovementAgent> pmc)
{
  itsMovementAgent = pmc;
}

// ######################################################################
inline void CaptainAgent::setForwardVisionAgent
(nub::soft_ref<ForwardVisionAgent> fv)
{
  itsForwardVisionAgent = fv;
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

