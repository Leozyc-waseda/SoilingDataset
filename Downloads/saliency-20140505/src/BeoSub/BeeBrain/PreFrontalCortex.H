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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/PreFrontalCortex.H $
// $Id: PreFrontalCortex.H 8623 2007-07-25 17:57:51Z rjpeters $

#ifndef __PREFRONTAL_CORTEX_H__
#define __PREFRONTAL_CORTEX_H__

#include "BeoSub/BeeBrain/PreMotorComplex.H"
#include "BeoSub/BeeBrain/ForwardVision.H"
#include "BeoSub/BeeBrain/SonarListen.H"
#include "BeoSub/BeeBrain/OceanObject.H"
#include "BeoSub/BeeBrain/AgentManagerA.H"
#include "BeoSub/BeeBrain/ComplexMovement.H"

#include "rutz/shared_ptr.h"
#include "Util/Timer.H"

class AgentManagerA;
class ForwardVisionAgent;

class PreFrontalCortexAgent : public Agent
{

public:

  //
  PreFrontalCortexAgent(std::string name = "preFrontalCortexAgent");

  //
  PreFrontalCortexAgent
  (std::string name, rutz::shared_ptr<AgentManagerA> ama);
  ~PreFrontalCortexAgent();
  //Variables

  //Messages
  void msgOceanObjectUpdate();
  void msgMovementComplete();

  void start();

  //Scheduler
  bool pickAndExecuteAnAction();

  //Actions
  //  void search(SEARCH_TYPE type, /*Ocean_Object state*/);
//   void surface();
//   void killSwitch();
//   void populateMissions();

  inline void setPreMotorComplex
  (rutz::shared_ptr<PreMotorComplex> pmc);

  inline void setForwardVisionAgent
  (rutz::shared_ptr<ForwardVisionAgent> fv);

private:

  rutz::shared_ptr<ForwardVisionAgent> itsForwardVisionAgent;
  rutz::shared_ptr<PreMotorComplex> itsPreMotorComplex;

  rutz::shared_ptr<AgentManagerA> itsAgentManager;

  //! timer for the overall run
  rutz::shared_ptr<Timer> itsRunTimer;

  //! timer for the current mission
  rutz::shared_ptr<Timer> itsCurrentMissionTimer;

  //!
  enum MissionState { NOT_STARTED, SEARCHING, CENTERING, COMPLETED, FAILED };

  //!
  enum MissionName { GATE, HIT_START_BUOY, FIRST_BIN, SECOND_BIN, GET_TREASURE };

  //!
  struct Mission {

    uint timeForMission;
    std::vector< rutz::shared_ptr<OceanObject> > missionObjects;
    MissionState missionState;
    MissionName missionName;
  };

  //!
  rutz::shared_ptr<Mission> currentMission;

  //!
  std::vector< rutz::shared_ptr<Mission> > itsMissions;


  bool currentMovementDone;

  //!
  void populateMissions();

  //!
  void lookForObjects(std::vector<rutz::shared_ptr<OceanObject> > oceanObjects,
                      bool startLooking = true);

  bool executeGateMission();
  bool executeBuoyMission();
  bool executeFirstBinMission();
  bool executeSecondBinMission();
  bool executeCrossMission();
};

// ######################################################################
inline void PreFrontalCortexAgent::setPreMotorComplex
(rutz::shared_ptr<PreMotorComplex> pmc)
{
  itsPreMotorComplex = pmc;
}

// ######################################################################
inline void PreFrontalCortexAgent::setForwardVisionAgent
(rutz::shared_ptr<ForwardVisionAgent> fv)
{
  itsForwardVisionAgent = fv;
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

