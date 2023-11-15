/*!@file SeaBee/SubController.H  Control motors and pid */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/SubController.H $
// $Id: SubController.H 10794 2009-02-08 06:21:09Z itti $
//

#ifndef SubController_H_DEFINED
#define SubController_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelOptionDef.H"
#include "Component/ModelParam.H"
#include "Component/OptionManager.H"
#include "Devices/V4L2grabber.H"
#include "rutz/shared_ptr.h"
#include "Util/Types.H"
#include "Util/WorkThreadServer.H"
#include "Util/JobWithSemaphore.H"
#include "Controllers/PID.H"
#include "Devices/BeeStemTiny.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Util/Angle.H"
#include "Transport/FrameIstream.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "Media/FrameSeries.H"
#include "SeaBee/BeeStemSim.H"
#include <list>


//NOTE: Do not change these thruster mappings! Just plug in the cables correctly, please.
//      If for some reason these mappings must be changed, then you need to also change the order
//      in which SubController::setThruster and SubController::updatePID sendd their thruster packets
#define THRUSTER_UP_LEFT 0
#define THRUSTER_UP_RIGHT 4
#define THRUSTER_UP_BACK 2
#define THRUSTER_FWD_RIGHT 3
#define THRUSTER_FWD_LEFT 1

#define THRUSTER_UP_LEFT_THRESH 0
#define THRUSTER_UP_RIGHT_THRESH 0
#define THRUSTER_UP_BACK_THRESH 0
#define THRUSTER_FWD_RIGHT_THRESH 0
#define THRUSTER_FWD_LEFT_THRESH 0

// ######################################################################
class SubController : public ModelComponent
{
public:

  enum SensorType  {DEPTH, PITCH, ROLL, HEADING};
  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  SubController(OptionManager& mgr,
             const std::string& descrName = "SubController",
             const std::string& tagName = "SubController",
             const bool simulation = false);

  //! Destructor
  ~SubController();

  void runSimLoop();
  bool setLevel(); //maintain sub at current pitch/roll level

  bool setHeading(int heading); //maintain a heading
  bool setDepth(int depth);  //maintain a depth
  bool setSpeed(int speed);  //maintain a speed
  bool setTurningSpeed(int speed);  //maintain a turning speed

  // althought we will most likely not use these. It's good to have them anyway
  // maybe it will be part of our victory dance
  bool setRoll(int roll); //maintain roll
  bool setPitch(int pitch); //maintain pitch

  void start1();
  void start2();

  int getHeading() { return itsCurrentHeading; }
  int getHeadingErr();
  int getDepthErr();
  int getPitch() { return itsCurrentPitch; }
  int getRoll() { return itsCurrentRoll; }
  int getDepth() { return itsCurrentDepth; }
  int getIntPressure() { return (itsCurrentIntPressure); }

  bool getKillSwitch() { return false; }

  void setThruster(int thruster, int val);

  const int* getIntPressurePtr() { return &itsCurrentIntPressure; }
  const int* getHeadingPtr() { return &itsCurrentHeading; }
  const int* getPitchPtr() { return &itsCurrentPitch; }
  const int* getRollPtr() { return &itsCurrentRoll; }
  const int* getDepthPtr() { return &itsCurrentDepth; }

  const int* getThruster_Up_Left_Ptr() { return &itsCurrentThruster_Up_Left; }
  const int* getThruster_Up_Right_Ptr() { return &itsCurrentThruster_Up_Right; }
  const int* getThruster_Up_Back_Ptr() { return &itsCurrentThruster_Up_Back; }
  const int* getThruster_Fwd_Right_Ptr() { return &itsCurrentThruster_Fwd_Right; }
  const int* getThruster_Fwd_Left_Ptr() { return &itsCurrentThruster_Fwd_Left; }

  void setMotorsOn(bool val) { motorsOn.setVal(val);}
  void setPidOn(bool val) { pidOn.setVal(val); }

  void updateHeading(unsigned int currentHeading);
  void updatePitch(int currentPitch);
  void updateRoll(int currentRoll);
  void updateDepth(unsigned int currentDepth);
  void updatePID();
  void sendHeartBeat();

  void setIntPressure(unsigned int pressure);

  void killMotors();

  void genPIDImage();
  void genSubImage();

  const Image<PixRGB<byte> >* getPIDImagePtr() { return &itsPIDImage; }
  const Image<PixRGB<byte> >* getSubImagePtr() { return &itsSubImage; }

  const Image<PixRGB<byte> > getImage(int camera);
  const Dims peekDims();

  bool isSimMode();



  //@}

  // ######################################################################
  /*! @name member functions */
  //@{


private:
  OModelParam<bool> itsSimulation;
  int itsDesiredPitch;
  int itsDesiredRoll;
  int itsDesiredHeading;
  int itsDesiredDepth;
  int itsDesiredSpeed;
  int itsDesiredTurningSpeed;


  int itsCurrentPitch;
  int itsCurrentRoll;
  int itsCurrentHeading;
  int itsCurrentDepth;
  int itsCurrentSpeed;
  int itsCurrentIntPressure;

  NModelParam<float> speedScale;
  float itsSpeedScale;

  NModelParam<float> depthRatio;
  float itsDepthRatio;

  nub::soft_ref<BeeStemTiny> itsBeeStemTiny;
  nub::soft_ref<BeeStemSim> itsBeeStemSim;

  PID<float> itsPitchPID;
  PID<float> itsRollPID;

  NModelParam<float> headingP;
  NModelParam<float> headingI;
  NModelParam<float> headingD;

  PID<float> itsHeadingPID;

  NModelParam<float> depthP;
  NModelParam<float> depthI;
  NModelParam<float> depthD;

  PID<float> itsDepthPID;

  int itsCurrentThruster_Up_Left;
  int itsCurrentThruster_Up_Right;
  int itsCurrentThruster_Up_Back;
  int itsCurrentThruster_Fwd_Right;
  int itsCurrentThruster_Fwd_Left;


  void paramChanged(ModelParamBase* const param, const bool valueChanged, ParamClient::ChangeStatus* status);

  NModelParam<int> setCurrentThruster_Up_Left;
  NModelParam<int> setCurrentThruster_Up_Right;
  NModelParam<int> setCurrentThruster_Up_Back;
  NModelParam<int> setCurrentThruster_Fwd_Right;
  NModelParam<int> setCurrentThruster_Fwd_Left;

  NModelParam<float> pitchP;
  NModelParam<float> pitchI;
  NModelParam<float> pitchD;

  NModelParam<float> rollP;
  NModelParam<float> rollI;
  NModelParam<float> rollD;


  NModelParam<bool>  motorsOn;
  NModelParam<bool>  pidOn;
  NModelParam<bool>  guiOn;
  NModelParam<bool>  depthPIDDisplay;
  NModelParam<bool>  pitchPIDDisplay;
  NModelParam<bool>  headingPIDDisplay;
  NModelParam<bool>  rollPIDDisplay;


  NModelParam<int>  setDepthValue;
  NModelParam<int>  setPitchValue;
  NModelParam<int>  setRollValue;
  NModelParam<int>  setHeadingValue;



  /// GUI Related
  Image<PixRGB<byte> > itsPIDImage;
  Image<PixRGB<byte> > itsSubImage;

        int itsPrevDepth;
        int itsPrevPrevDepth;
        int itsDepthCount;
        std::list<int> itsAvgDepth;

  rutz::shared_ptr<WorkThreadServer> itsThreadServer;

  nub::soft_ref<V4L2grabber> itsForwardCam;
  nub::soft_ref<V4L2grabber> itsBottomCam;

  void initSensorVals();

};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
