/*!@file BeoSub/BeoSubSim.H */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubSim.H $
// $Id: BeoSubSim.H 6454 2006-04-11 00:47:40Z rjpeters $
//

#ifndef BEOSUBSIML_H_DEFINED
#define BEOSUBSIM_H_DEFINED

#include "BeoSub/BeoSub.H"
#include "BeoSub/BeoSubSensor.H"
#include "Devices/BeoChip.H"
#include "Controllers/PID.H"
class HMR3300;
class BeoSubBallast;
class BeoSubIMU;
class FrameIstream;


//! Definition and access functions for the BeoSub
/*! BeoSub is an autonomous visually-guided submarine. This class
  implements the high-level functionality, relying on low-level
  drivers to handle motor activation, compass reading, etc */
class BeoSubSim : public BeoSub
{
public:
  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  BeoSubSim(OptionManager& mgr);

  //! Destructor
  ~BeoSubSim();

  //@}

  // ####################################################################
  /*! @name mid-level overloads */
  //@{
  //! Advance the sub by some distance in meters
  /*! This is an overload of BeoSub::advanceRel() which ensures we
    turn off the heading PID during the advance */
  virtual void advanceRel(const float relDist, const bool stop = true);

  //! Strafe the sub by some distance in meters
  /*! this is an overload of BeoSub::strafeRel() which ensures we turn
    the heading PID off during the strafe */
  virtual void strafeRel(const float relDist);

  void turnAbs(const Angle finalHeading, const bool blocking = false);
  void turnRel(const Angle relHeading, const bool blocking = false);
  void turnOpen(const Angle relHeading, const bool blocking = false);
  void diveAbs(const float finalDepth, const bool blocking = false);
  void diveRel(const float relDepth, const bool blocking = false);
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

  // ##################################################################

  //! Update compass - used by sensors
  void updateCompass(const Angle heading, const Angle pitch,
                     const Angle roll);

  //! Update depth - used by sensors
  void updateDepth(const float depth);

  //! Update position
  void updatePosition(const float z, const float x);

  inline float getCurrentZ() { return itsCurrentZ; }
  inline float getCurrentX() { return itsCurrentX; }
  inline float getRelDist() { return relDistance; }
  inline bool isStrafing() { return isStrafe; }
  inline void setStrafing(const bool s = false) { isStrafe = s; }
  inline void setUpGrabFlag(const bool s = false) { upImageFlag = s; }
  inline bool getUpGrabFlag() { return upImageFlag; }
  inline void setDownGrabFlag(const bool s = false) { downImageFlag = s; }
  inline bool getDownGrabFlag() { return downImageFlag; }
  inline void setFrontGrabFlag(const bool s = false) { frontImageFlag = s; }
  inline bool getFrontGrabFlag() { return frontImageFlag; }

  //! Turn depth PID on/off
  void useDepthPID(const bool useit);

  //! Turn heading PID on/off
  void useHeadingPID(const bool useit);

  //! Turn pitch PID on/off
  void usePitchPID(const bool useit);

protected:
  NModelParam<int> itsLeftThrusterServoNum;  //!< servo num for left thruster
  NModelParam<int> itsRightThrusterServoNum; //!< servo num for right thruster

  nub::soft_ref<HMR3300> itsHMR3300;         //!< compass for heading
  nub::soft_ref<BeoChip> itsBeo;             //!< our beochip
  nub::soft_ref<BeoSubBallast> itsFballast, itsRballast; //!< our 2 ballasts
  nub::soft_ref<BeoSubIMU> itsIMU;           //!< our IMU
  nub::soft_ref<FrameIstream> itsCameraFront,
    itsCameraDown, itsCameraUp;              //!< our three cameras
  float itsThrustLeft, itsThrustRight;       //!< our current thruster settings
  float relDistance;
  bool isStrafe;
  int imageCounter;
  float itsCurrentZ, itsCurrentX;
  bool upImageFlag, frontImageFlag, downImageFlag;

  BeoSubSensor<float> itsDepthSensor; //!< sensor cleaner for depth
  BeoSubSensor<Angle> itsHeadingSensor; //!< sensor cleaner for heading
  BeoSubSensor<Angle> itsPitchSensor; //!< sensor cleaner for pitch
  PID<float> itsDepthPID;    //!< PID for depth stabilization
  PID<Angle> itsHeadingPID;  //!< PID for heading stabilization
  PID<Angle> itsPitchPID;    //!< PID for pitch stabilization

  bool itsDepthPIDon, itsHeadingPIDon, itsPitchPIDon;

  void start1(); //!< get started, before our subcomponents start
  void start2(); //!< get started, after our subcomponents have started


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
