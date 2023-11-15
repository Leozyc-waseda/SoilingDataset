/*!@file BeoSub/BeeBrain/AgentManagerA.H management class for agents on COM-A*/

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/AgentManagerA.H $
// $Id: AgentManagerA.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef BEOSUB_BEEBRAIN_AGENT_MANAGER_A_H_DEFINED
#define BEOSUB_BEEBRAIN_AGENT_MANAGER_A_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"

#include "Raster/Raster.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/CutPaste.H"

#include "GUI/XWinManaged.H"

#include "BeoSub/BeeBrain/Globals.H"
#include "BeoSub/BeeBrain/AgentManagerCommand.H"

#include "BeoSub/BeeBrain/ForwardVision.H"
#include "BeoSub/BeeBrain/PreFrontalCortex.H"
#include "BeoSub/BeeBrain/PreMotorComplex.H"

#include "BeoSub/BeeBrain/OceanObject.H"

#include "Util/Timer.H"
#include <pthread.h>

class PreFrontalCortexAgent;
class PreMotorComplex;
class ForwardVisionAgent;

class AgentManagerA : public ModelComponent
{
public:

  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  AgentManagerA(OptionManager& mgr,
                const std::string& descrName = "Agent Manager A",
                const std::string& tagName = "Agent Manager A");

  //! Destructor
  virtual ~AgentManagerA();

  //@}

  // ######################################################################
  //! @name Access functions
  //@{

  //! set the current image to be processed
  void setCurrentImage(Image<PixRGB<byte> > image, uint fNum);

  inline void setWindow(rutz::shared_ptr<XWinManaged> win,
                        Image<PixRGB<byte> > ima);

  void pushCommand(CommandType cmdType,
                  DataTypes dataType,
                  rutz::shared_ptr<OceanObject> oceanObject);

  //! get the current image to be processed
  inline Image<PixRGB<byte> > getCurrentImage();

  //! get the current frame number
  inline uint getCurrentFrameNumber();

  //! set the current object to be processed
  bool updateOceanObject(rutz::shared_ptr<OceanObject> oceanObject,
                         DataTypes oceanObjectDataType);

  rutz::shared_ptr<ForwardVisionAgent> getForwardVisionAgent();

  rutz::shared_ptr<PreFrontalCortexAgent> getPreFrontalCortexAgent();

  rutz::shared_ptr<PreMotorComplex> getPreMotorComplex();

  uint getNumCommands();

  rutz::shared_ptr<AgentManagerCommand> popCommand();

  void drawImage(Image<PixRGB<byte> > ima, Point2D<int> point);

  void startRun();

  //@}

private:

  // for display purposes
  rutz::shared_ptr<XWinManaged> itsWindow;
  Image<PixRGB<byte> > itsDisplayImage;
  pthread_mutex_t itsDisplayLock;

  // the current image and lock
  Image<PixRGB<byte> > itsCurrentImage;
  uint itsFrameNumber;
  rutz::shared_ptr<Timer> itsInputImageTimer;
  std::vector<uint64> itsFrameDuration;
  pthread_mutex_t itsCurrentImageLock;

  // forward vision agent
  rutz::shared_ptr<ForwardVisionAgent> itsForwardVisionAgent;
  pthread_t itsForwardVisionAgentThread;

  // prefrontal cortex agent: decision maker
  rutz::shared_ptr<PreFrontalCortexAgent> itsPreFrontalCortexAgent;
  pthread_t itsPreFrontalCortexAgentThread;

  // premotor complex agent: generate motor commands
  rutz::shared_ptr<PreMotorComplex> itsPreMotorComplex;
  pthread_t itsPreMotorComplexThread;

  // current motor command
  motorCommand itsCurrentMotorCommand;
  pthread_mutex_t itsCurrentMotorCommandLock;

  // store the ocean objects
  std::vector<rutz::shared_ptr<OceanObject> > itsOceanObjects;
  pthread_mutex_t itsOceanObjectsLock;

  // command cue to send to COM_B
  std::list<rutz::shared_ptr<AgentManagerCommand> > itsCommands;
  pthread_mutex_t itsCommandsLock;

};

// ######################################################################
inline void AgentManagerA::setWindow(rutz::shared_ptr<XWinManaged> win,
                      Image<PixRGB<byte> > ima)
{
  itsWindow = win;
  itsDisplayImage = ima;
}

// ######################################################################
inline Image<PixRGB<byte> > AgentManagerA::getCurrentImage()
{
  Image<PixRGB<byte> > image;
  pthread_mutex_lock(&itsCurrentImageLock);
  image = itsCurrentImage;
  pthread_mutex_unlock(&itsCurrentImageLock);
  return image;
}

// ######################################################################
inline uint AgentManagerA::getCurrentFrameNumber()
{
  uint fNum;
  pthread_mutex_lock(&itsCurrentImageLock);
  fNum = itsFrameNumber;
  pthread_mutex_unlock(&itsCurrentImageLock);
  return fNum;
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
