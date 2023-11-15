/*!@file BeoSub/BeoSubOneBal.H An autonomous submarine with one ballast */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubOneBal.H $
// $Id: BeoSubOneBal.H 6454 2006-04-11 00:47:40Z rjpeters $
//

#ifndef BEOSUBONEBAL_H_DEFINED
#define BEOSUBONEBAL_H_DEFINED

#include "BeoSub/BeoSub.H"
#include "BeoSub/BeoSubSensor.H"
#include "Devices/BeoChip.H"
#include "Controllers/PID.H"
#include "BeoSub/Stepper.H"
#include "Devices/ParPort.H"
class HMR3300;
class BeoSubBallast;
class BeoSubIMU;
class IEEE1394grabber;


//! Definition and access functions for the BeoSub
/*! BeoSub is an autonomous visually-guided submarine. This class
  impoements the highl-evel functionality, relying on low-level
  drivers to handle motor activation, compass reading, etc */
class BeoSubOneBal : public BeoSub
{
public:
  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  BeoSubOneBal(OptionManager& mgr);

  //! Destructor
  ~BeoSubOneBal();

  //@}

  // ######################################################################
  /*! @name mid-level overloads */
  //@{

  //! Advance the sub by some distance in meters
  /*! This is an overload of BeoSub::advanceRel() which ensures we
    turn off the heading PID during the advance */
  virtual void advanceRel(const float relDist, const bool stop = true);

  //! Strafes the sub by some distance in meters
  /*! This is an overload of BeoSub::strafeRel() which ensures we turn
    the heading PID off during the strafe */
  virtual void strafeRel(const float relDist);

  virtual void turnOpen(const Angle openHeading, const bool stop = true);

  //@}

  // ######################################################################
  /*! @name Low-level access functions */
  //@{

  //! Activate the thrusters
  /*! Values should be between -1.0 (full reverse) and 1.0 (full
    forward), 0.0 being no motion */
  void thrust(const float leftval, const float rightval);

  //! Get the current thruster settings
  void getThrusters(float& leftval, float& rightval) const;

  //! Set the front ballast. Value is between 0.0 (empty) and 1.0 (full)
  void setFrontBallast(const float val, const bool blocking = false);

  //! Set the rear ballast. Value is between 0.0 (empty) and 1.0 (full)
  void setRearBallast(const float val, const bool blocking = false);

  //! Set all ballasts
  void setBallasts(const float f, const float r,
                   const bool blocking = false);

  //! Get the front ballast. Value is between 0.0 (empty) and 1.0 (full)
  float getFrontBallast() const;

  //! Get the rear ballast. Value is between 0.0 (empty) and 1.0 (full)
  float getRearBallast() const;

  //! Get all ballasts
  void getBallasts(float& f, float& r) const;

  //! Drop a marker
  virtual void dropMarker(const bool blocking = false);

  //! Grab a an image from one of the cameras
  virtual Image< PixRGB<byte> > grabImage(const enum BeoSubCamera cam) const;

  //@}

  // **********************************************************************
  /*! closed-loop control */
private:
  void updateThrusters();
  void updateRotVelPID(const Angle current);
  Angle PIDRotVel;
  float RotVelCommand;
  float PIDTransVel, TransVelCommand;
public:
  void setRotVel(const Angle desired);
  void setTransVel(const float desired);

  // ######################################################################
  /*! @name PID control */
  //@{

  //! Turn depth PID on/off
  void useDepthPID(const bool useit);

  //! set depth PID pgain
  void setDepthPgain(float p);

  //! set depth PID igain
  void setDepthIgain(float i);

  //! set depth PID dgain
  void setDepthDgain(float d);

  //! Turn heading PID on/off
  void useHeadingPID(const bool useit);

  //! set heading PID pgain
  void setHeadingPgain(float p);

  //! set heading PID igain
  void setHeadingIgain(float i);

  //! set heading PID dgain
  void setHeadingDgain(float d);

  //! Turn pitch PID on/off
  void usePitchPID(const bool useit);

  //! set pitch PID pgain
  void setPitchPgain(float p);

  //! set pitch PID igain
  void setPitchIgain(float i);

  //! set pitch PID dgain
  void setPitchDgain(float d);

  //! Turn pitch PID on/off
  void useRotVelPID(const bool useit);

  //! set pitch PID pgain
  void setRotVelPgain(float p);

  //! set pitch PID igain
  void setRotVelIgain(float i);

  //! set pitch PID dgain
  void setRotVelDgain(float d);


  //@}

  // ######################################################################
  /*! @name Miscellaneous */
  //@{

  //! Use or ignore the kill switch
  void useKillSwitch(const bool useit);
  /* TODO: two different kill switch code! connected to different hardware, though */
  bool isKilled();

  //! Kill Switch Function
  /*! This is called when the kill switch is activated */
  void killSwitch();

  //@}

  // ######################################################################
  /*! @name For internal use by our listeners only -- do not call */
  //@{

  //! Update compass - used by sensors
  void updateCompass(const Angle heading, const Angle pitch,
                     const Angle roll);

  //! Update depth - used by sensors
  void updateDepth(const float depth);

  void updateIMU(const float xa, const float ya, const float za,
                  const Angle xv, const Angle yv, const Angle zv);
  //@}
  nub::soft_ref<BeoSubIMU> itsIMU;           //!< our IMU

protected:
  NModelParam<int> itsLeftThrusterServoNum;  //!< servo num for left thruster
  NModelParam<int> itsRightThrusterServoNum; //!< servo num for right thruster

  nub::soft_ref<HMR3300> itsHMR3300;         //!< compass for heading
  nub::soft_ref<BeoChip> itsBeo;             //!< our beochip
  nub::soft_ref<BeoSubBallast> itsFballast, itsRballast; //!< our 2 ballasts
 // nub::soft_ref<BeoSubIMU> itsIMU;           //!< our IMU
  nub::soft_ref<FrameIstream> itsCameraFront,
    itsCameraDown, itsCameraUp;              //!< our three cameras
  float itsThrustLeft, itsThrustRight;       //!< our current thruster settings

  BeoSubSensor<float> itsDepthSensor; //!< sensor cleaner for depth
  BeoSubSensor<Angle> itsHeadingSensor; //!< sensor cleaner for heading
  BeoSubSensor<Angle> itsPitchSensor; //!< sensor cleaner for pitch
  PID<float> itsDepthPID;    //!< PID for depth stabilization
  PID<Angle> itsHeadingPID;  //!< PID for heading stabilization
  PID<Angle> itsPitchPID;    //!< PID for pitch stabilization
  PID<Angle> itsRotVelPID;    //!< PID for rotational velocity stabilization

  bool itsDepthPIDon, itsHeadingPIDon, itsPitchPIDon, itsRotVelPIDon;
  bool itsKillSwitchUsed;

  float itsDiveSetting, itsPitchSetting;
  double itsLastHeadingTime, itsLastPitchTime, itsLastDepthTime;

  virtual void start1(); //!< get started, before our subcomponents start
  virtual void start2(); //!< get started, after our subcomponents have started
  virtual void stop1();

  /* TODO: put these stuff in proper place - khooyp*/
#ifdef HAVE_LINUX_PARPORT_H
  nub::soft_ref<ParPort> lp0;
  Stepper markerdropper;
#endif
  int killSwitchDebounceCounter;

private:
  //! Dispatch a BeoChip event
  void dispatchBeoChipEvent(const BeoChipEventType t,
                            const int valint,
                            const float valfloat);

  friend class BeoSubListener;
  friend class HMRlistener;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
