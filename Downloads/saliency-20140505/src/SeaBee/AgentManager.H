/*!@file SeaBee/AgentManager.H management class for agents on COM-A    */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/AgentManager.H $
// $Id: AgentManager.H 10794 2009-02-08 06:21:09Z itti $
//

#ifndef __SEABEE_AGENT_MANAGER_H__
#define __SEABEE_AGENT_MANAGER_H__

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Component/ModelManager.H"

#include "Raster/Raster.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/CutPaste.H"

#include "Neuro/EnvVisualCortex.H"

#include "GUI/XWinManaged.H"

#include "Globals.H"
#include "AgentManagerCommand.H"

#include "ForwardVision.H"
// #include "Captain.H"
// #include "Movement.H"
#include "SubController.H"
#include "SensorResult.H"
// #include "DownwardVision.H"
// #include "SonarListen.H"

#include "GUI/DebugWin.H"

#include "Util/Timer.H"
#include <pthread.h>

//class CaptainAgent;
//class MovementAgent;
class ForwardVisionAgent;
//class DownwardVisionAgent;
//class SonarListenAgent;

class AgentManager : public ModelComponent
{
public:

  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  AgentManager(nub::soft_ref<SubController> subController,
                nub::soft_ref<EnvVisualCortex> evc,
                ModelManager& mgr,
                const std::string& descrName = "Agent Manager",
                const std::string& tagName = "Agent Manager");

  //! Destructor
  virtual ~AgentManager();

  //@}

  // ######################################################################
  //! @name Access functions
  //@{

  //! set the current image to be processed
  void setCurrentForwardImage(Image<PixRGB<byte> > image, uint fNum);

  void setCurrentDownwardImage(Image<PixRGB<byte> > image, uint fNum);


  inline void setSubController(nub::soft_ref<SubController> subController);

  inline void setForwardWindow(rutz::shared_ptr<XWinManaged> win);

  inline void setDownwardWindow(rutz::shared_ptr<XWinManaged> win);


  void pushCommand(CommandType cmdType,
                   rutz::shared_ptr<Mission> mission);

  //! get the current image to be processed
  inline Image<PixRGB<byte> > getCurrentForwardImage();
  inline Image<PixRGB<byte> > getCurrentDownwardImage();

  //! get the current frame number
  inline uint getCurrentFrameNumber();

  inline void setWindow(rutz::shared_ptr<XWinManaged> win);

  inline void setVisualCortex(nub::ref<EnvVisualCortex> evc);

  inline void drawForwardImage(rutz::shared_ptr< Image<PixRGB<byte> > > ima);
  inline void drawDownwardImage(rutz::shared_ptr< Image<PixRGB<byte> > > ima);

  //! initialize the vector of SensorResults
  void initSensorResults();

  //! set the current object to be processed
  bool updateSensorResult(rutz::shared_ptr<SensorResult> sensorResult);

  //! update the missions of all of agent manager's agents
  void updateAgentsMission(Mission theMission);

  //  rutz::shared_ptr<DownwardVisionAgent> getDownwardVisionAgent();

  //  rutz::shared_ptr<SonarListenAgent> getSonarListenAgent();

  rutz::shared_ptr<ForwardVisionAgent> getForwardVisionAgent();

 //  rutz::shared_ptr<CaptainAgent> getCaptainAgent();

//   rutz::shared_ptr<MovementAgent> getMovementAgent();

  uint getNumCommands();

  rutz::shared_ptr<SensorResult> getSensorResult(SensorResult::SensorResultType type);

  rutz::shared_ptr<AgentManagerCommand> popCommand();

  void startRun();

  int itsMissionGateForwardTime;
  int itsMissionGateDepth;

  const Image<PixRGB<byte> >* getForwardDebugImagePtr();

  //@}

private:

  Image<PixRGB<byte> > itsForwardDebugImage;

  // for display purposes
  rutz::shared_ptr<XWinManaged> itsForwardWindow;
  rutz::shared_ptr<XWinManaged> itsDownwardWindow;
  Image<PixRGB<byte> > itsDisplayImage;
  pthread_mutex_t itsForwardDisplayLock;
  pthread_mutex_t itsDownwardDisplayLock;

  // the current image and lock
  Image<PixRGB<byte> > itsCurrentForwardImage;
  Image<PixRGB<byte> > itsCurrentDownwardImage;
  uint itsFrameNumber;
  rutz::shared_ptr<Timer> itsTimer;
  std::vector<uint64> itsFrameDuration;
  pthread_mutex_t itsCurrentForwardImageLock;
  pthread_mutex_t itsCurrentDownwardImageLock;

//   // downward vision agent
//   rutz::shared_ptr<DownwardVisionAgent> itsDownwardVisionAgent;
//   pthread_t itsDownwardVisionAgentThread;

//   // sonar agent
//   rutz::shared_ptr<SonarListenAgent> itsSonarListenAgent;
//   pthread_t itsSonarListenAgentThread;

  // forward vision agent
  rutz::shared_ptr<ForwardVisionAgent> itsForwardVisionAgent;
//   pthread_t itsForwardVisionAgentThread;

//   // prefrontal cortex agent: decision maker
//   rutz::shared_ptr<CaptainAgent> itsCaptainAgent;
//   pthread_t itsCaptainAgentThread;

//   // premotor complex agent: generate motor commands
//   rutz::shared_ptr<MovementAgent> itsMovementAgent;
//   pthread_t itsMovementAgentThread;

  // collection of all the agent manager's agents
  // std::vector<rutz::shared_ptr<SubmarineAgent> > itsSubmarineAgents;

  // SubController
  nub::soft_ref<SubController> itsSubController;
  pthread_mutex_t itsSubControllerLock;

  // store the ocean objects
  std::vector<rutz::shared_ptr<SensorResult> > itsSensorResults;
  std::vector<rutz::shared_ptr<SensorResult> >::iterator itsSensorResultsItr;

  pthread_mutex_t itsSensorResultsLock;

  // command cue to send to COM_B
  std::list<rutz::shared_ptr<AgentManagerCommand> > itsCommands;
  pthread_mutex_t itsCommandsLock;

  nub::soft_ref<EnvVisualCortex> itsEVC;

};

// ######################################################################
inline void AgentManager::setSubController(nub::soft_ref<SubController> subController)
{
  pthread_mutex_lock(&itsSubControllerLock);
  itsSubController = subController;
  pthread_mutex_unlock(&itsSubControllerLock);
}

// ######################################################################
inline void AgentManager::setForwardWindow(rutz::shared_ptr<XWinManaged> win)
{
  itsForwardWindow = win;
  //itsDisplayImage = ima;
}

// ######################################################################
inline void AgentManager::setDownwardWindow(rutz::shared_ptr<XWinManaged> win)
{
  itsDownwardWindow = win;
  //itsDisplayImage = ima;
}

// ######################################################################
inline void AgentManager::setVisualCortex
(nub::ref<EnvVisualCortex> evc)
{
  itsEVC = evc;
}

// ######################################################################
inline Image<PixRGB<byte> > AgentManager::getCurrentForwardImage()
{
  Image<PixRGB<byte> > image;
  pthread_mutex_lock(&itsCurrentForwardImageLock);
  image = itsCurrentForwardImage;
  pthread_mutex_unlock(&itsCurrentForwardImageLock);
  return image;
}

// ######################################################################
inline Image<PixRGB<byte> > AgentManager::getCurrentDownwardImage()
{
  Image<PixRGB<byte> > image;
  pthread_mutex_lock(&itsCurrentDownwardImageLock);
  image = itsCurrentDownwardImage;
  pthread_mutex_unlock(&itsCurrentDownwardImageLock);
  return image;
}

// ######################################################################
inline uint AgentManager::getCurrentFrameNumber()
{
  uint fNum;
  pthread_mutex_lock(&itsCurrentForwardImageLock);
  fNum = itsFrameNumber;
  pthread_mutex_unlock(&itsCurrentForwardImageLock);
  return fNum;
}

// // ######################################################################
inline void AgentManager::drawForwardImage(rutz::shared_ptr<Image<PixRGB<byte> > > ima)
{
  pthread_mutex_lock(&itsForwardDisplayLock);
  itsForwardWindow->drawImage(*ima, 0, 0);
  pthread_mutex_unlock(&itsForwardDisplayLock);

}

// // ######################################################################
inline void AgentManager::drawDownwardImage(rutz::shared_ptr<Image<PixRGB<byte> > > ima)
{
  pthread_mutex_lock(&itsDownwardDisplayLock);
  itsDownwardWindow->drawImage(*ima, 0, 0);
  pthread_mutex_unlock(&itsDownwardDisplayLock);

}
#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
