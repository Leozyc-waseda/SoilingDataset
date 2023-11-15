/*!@file Ice/Scorbot.ice Interfaces to the robot arm */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: $
// $Id: $
//


#ifndef Scorbot_ICE
#define Scorbot_ICE

module Robots {
  
    enum JOINTS {BASE, SHOULDER, ELBOW, WRIST1, WRIST2, GRIPPER, EX1, EX2, WRISTROLL, WRISTPITCH};

    struct ArmPos {
      int base;
      int shoulder;
      int elbow;
      int wrist1;
      int wrist2;
      int gripper;
      int ex1;
      int ex2;
      int wristroll;
      int wristpitch;
      int duration;
    };

    interface ScorbotIce {

      ArmPos getIK(float x, float y, float z);

      //!Set the end effector position in x,y,z
      bool getEFpos(out float x, out float y, out float z);

      //!get the end effector position in x,y,z
      bool setEFPos(float x,float y,float z);

      //! Set the motor pwm 
      bool setMotor(JOINTS joint, int pwm);

      //! Get the current pwm value
      int getPWM(JOINTS j);

      //! Set the joint to a given position
      bool setJointPos(JOINTS joint, int pos);

      //! Get the joint position
      int getJointPos(JOINTS joint);

      //! Get the anguler joint position
      float getEncoderAng(JOINTS joint);

      //! Reset all encoders to 0
      void resetEncoders();

      //! Stop eveything
      void stopAllMotors();

      void setSafety(bool val);

      //! get the movment time
      int getMovementTime();

      //! Home Joint
      void homeMotor(JOINTS joint, int LimitSeekSpeed, int MSJumpSpeed, float MSJumpDelay, int MSSeekSpeed, bool MSStopCondition, bool checkMS);

      //! Home All Joints
      void homeMotors();

      //! Get the microSwitch states
      int getMicroSwitch();

      //! Get the microswitch to a spacific joint
      int getMicroSwitchMotor(JOINTS m);

      //! Set arm position to 
      bool setArmPos(ArmPos pos);

      ArmPos getArmPos();

      void motorsOn();
      void motorsOff();

      //! Shutdown
      void shutdown();


      double enc2ang(int encoderTicks);
      int ang2enc(double degrees);
      double enc2mm(int encoderTicks);
      int mm2enc(double mm);

    };
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
