/*!@file BeoSub/BeoSubTwoBal.H An autonomous submarine with two ballasts */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubTwoBal.H $
// $Id: BeoSubTwoBal.H 6990 2006-08-11 18:13:51Z rjpeters $
//

#ifndef BEOSUB_H_DEFINED
#define BEOSUB_H_DEFINED

#include "BeoSub/BeoSub.H"
#include "BeoSub/BeoSubSensor.H"
#include "Devices/BeoChip.H"
class HMR3300;
class BeoSubListener;   // privately defined in BeoSub.C //NOTE: really?
class BeoSubBallast;
class BeoSubIMU;
class IEEE1394grabber;

//! A simple enum to know left from right
enum BeoSubSide { BEOSUBLEFT = 0, BEOSUBRIGHT = 1 };

//! Simple function to print the name (Left, Right) of a side.
const char* beoSubSideName(const BeoSubSide s);

//! Definition and access functions for the BeoSub
/*! BeoSub is an autonomous visually-guided submarine. This class
  impoements the highl-evel functionality, relying on low-level
  drivers to handle motor activation, compass reading, etc */
class BeoSubTwoBal : public BeoSub
{
public:
  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  BeoSubTwoBal(OptionManager& mgr);

  //! Destructor
  ~BeoSubTwoBal();

  //@}

  // ######################################################################
  /*! @name Low-level behaviors */
  //@{
  /*
  //! Acquire and save a bunch of measurements
  void checkpoint(const char *fmt, ...)
    // NOTE: this __attribute__ tells gcc that it should issue
    // printf-style warnings when compiling calls to
    // BeoSub::checkpoint(), treating the 1st argument (fmt) as the
    // format string, and the 2nd and subsequent arguments as the
    // printf-style parameters (SUBNOTE: because this is a member
    // function, there is a hidden 'this' parameter that counts as arg
    // 1, so the listed arguments are counted starting from 2)
    __attribute__((format(__printf__, 2, 3)));
    ;
  */
  //@}

  //protected:
  // ######################################################################
  /*! @name Low-level access functions */
  //@{

  //! Activate the thrusters
  /*! Values should be between -1.0 (full reverse) and 1.0 (full
    forward), 0.0 being no motion */
  void thrust(const float leftval, const float rightval, const bool blocking = false);

  //! Get the current thruster settings
  void getThrusters(float& leftval, float& rightval);

  //! Set the left-front ballast. Value is between 0.0 (empty) and 1.0 (full)
  void setLFballast(const float val, const bool blocking = false);

  //! Set the left-rear ballast. Value is between 0.0 (empty) and 1.0 (full)
  void setLRballast(const float val, const bool blocking = false);

  //! Set the right-front ballast. Value is between 0.0 (empty) and 1.0 (full)
  void setRFballast(const float val, const bool blocking = false);

  //! Set the right-rear ballast. Value is between 0.0 (empty) and 1.0 (full)
  void setRRballast(const float val, const bool blocking = false);

  //! Set all ballasts
  void setBallasts(const float lf, const float lr,
                   const float rf, const float rr,
                   const bool blocking = false);

  //! Get the left-front ballast. Value is between 0.0 (empty) and 1.0 (full)
  float getLFballast();

  //! Get the left-rear ballast. Value is between 0.0 (empty) and 1.0 (full)
  float getLRballast();

  //! Get the right-front ballast. Value is between 0.0 (empty) and 1.0 (full)
  float getRFballast();

  //! Get the right-rear ballast. Value is between 0.0 (empty) and 1.0 (full)
  float getRRballast();

  //! Get all ballasts
  void getBallasts(float& lf, float& lr,
                   float& rf, float& rr);
  //! Drop a marker
  void dropMarker(const bool blocking = false);

  //! Grab a an image from one of the cameras
  Image< PixRGB<byte> > grabImage(const enum BeoSubCamera cam);

  //@}

protected:
  NModelParam<int> itsLeftThrusterServoNum;  //!< servo num for left thruster
  NModelParam<int> itsRightThrusterServoNum; //!< servo num for right thruster

  nub::soft_ref<HMR3300> itsHMR3300;              //!< compass for heading

  nub::soft_ref<BeoChip> itsBeoLeft, itsBeoRight; //!< our left and rt beochips

  rutz::shared_ptr<BeoSubListener> itsBeoLisLeft, itsBeoLisRight;  //!< listeners

  BeoSubSensor<float> itsDepthSensor;         //!< our depth sensor data

  nub::soft_ref<BeoSubBallast> itsLFballast, itsLRballast,
    itsRFballast, itsRRballast; //!< our 4 ballasts

  nub::soft_ref<BeoSubIMU> itsIMU;  //!< our IMU

  nub::soft_ref<IEEE1394grabber> itsCameraFront,
    itsCameraDown, itsCameraUp;  //!< our three cameras

  float itsThrustLeft, itsThrustRight; //!< our current thruster settings

  void start1(); //!< get started, before our subcomponents start
  void start2(); //!< get started, after our subcomponents have started

private:
  //! Dispatch a BeoChip event
  void dispatchBeoChipEvent(const BeoSubSide side,
                            const BeoChipEventType t,
                            const int valint,
                            const float valfloat);
  friend class BeoSubListener;
};


#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
