/*!@file ArmControl/RobotArm.H The whole robot arm */

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
// Primary maintainer for this file: Chin-Kai Chang<chinkaic@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/RobotArm.H $
// $Id: RobotArm.H 12957 2010-03-06 00:34:48Z lior $
//

#ifndef ROBOTARM_H_DEFINED
#define ROBOTARM_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Devices/Serial.H"
#include "Controllers/PID.H"
#include "Util/Types.H"
#include "GUI/ViewPort.H"
class RobotArm : public ModelComponent
{
public:
  // ######################################################################
  //servo definition

  /*! @name Constructors, destructors, and run-time "modules"*/

  //! Constructor
  /*! @param mgr our ModelManager (see ModelManager.H)
      @param descrName descriptive name for human usage
      @param tagName name for ParamMap usage */
  RobotArm(OptionManager& mgr, const std::string& descrName = "RobotArm",
        const std::string& tagName = "RobotArm")
    : ModelComponent(mgr, descrName, tagName)
  {

  }

  //! Virtual destructor for safe inheritance.
  virtual ~RobotArm() { LINFO("Not implimented") ;}

  enum MOTOR {BASE, SHOLDER, ELBOW, WRIST1, WRIST2, GRIPPER, EX1, EX2, WRIST_ROLL, WRIST_PITCH};
  enum COMMANDS {SAFE_MODE=9, SET_MOTORS=10, MOTORS_OFF=11,
          GET_ENCODER=12, RESET_ENCODER=13, GET_PWM=14, GET_MS=15};
  //! Set the Motor Speed and Direction by pwm
  virtual bool setMotor(MOTOR m , int pwm = 0){LFATAL("Not implimented"); return false;}

  //! Set the Motor Position by the encoder value
  virtual void setMotorPos(MOTOR m , int pos) { LFATAL("Not implimented"); }

  //! Get the encoder value from motor
  virtual int getEncoder(MOTOR m ) { LFATAL("Not implimented"); return -1;}

  //! Reset all motor  encoder value to 0
  virtual bool resetEncoders() { LFATAL("Not implimented"); return false;}

  //! Reset all motor position to the home place
  virtual void resetMotorPos() { LFATAL("Not implimented"); }

  //! Stop all the motor movement
  virtual bool stopAllMotors() { LFATAL("Not implimented"); return false;}

  //! Do the homeing to the all motor
  virtual void homeMotor() { LFATAL("Not implimented"); }

  virtual void simLoop() { LFATAL("Not implimented") ;}

  virtual Image<PixRGB<byte> > getFrame(int camera)
  { LFATAL("Not implimented") ; return Image<PixRGB<byte> >();}

  //@}

protected:
  //! do some additional config at start time; see ModelComponent.H
  virtual void start2() { LFATAL("Not implimented") ;}

private:
};

#endif

/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
