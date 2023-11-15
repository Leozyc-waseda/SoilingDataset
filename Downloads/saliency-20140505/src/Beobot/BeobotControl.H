/*!@file Beobot/BeobotControl.H controls the movement of the Beobot
  via passed in BeoChip                                                 */
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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotControl.H $
// $Id: BeobotControl.H 6795 2006-06-29 20:45:32Z rjpeters $
//
#ifndef BEOBOT_BEOBOTCONTROL_H_DEFINED
#define BEOBOT_BEOBOTCONTROL_H_DEFINED

#include "Beobot/BeobotAction.H"
#include "Beobot/BeobotConfig.H"
#include "Devices/BeoChip.H"
#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Util/Timer.H"
#include "Util/Types.H" // for byte definition
#include <pthread.h>

/*Beobot/BeobotControl.H controls the movement of the Beobot
  via passed in BeoChip

  Input:  geographical speed and direction
  Output: motor control for the next time stamp that will go toward
          achieving that goal

          - robot safety (tilting, etc) is being considered
            -> speed ramping is implicitly applied

          - robot can't turn all that well when not moving
            -> robot will move forward while turning
*/

//! Speed ramping options
enum SpeedRampType {
  SPEED_RAMP_LINEAR = 0, //!< use linear ramping
  SPEED_RAMP_SIGMOID = 1 //!< use sigmoid ramping
};

class BeobotControl : public ModelComponent
{
public:

  //! General constructor
  BeobotControl(nub::soft_ref<BeoChip> beoChip,
                OptionManager& mgr,
                const std::string& descrName = "Beobot Controller",
                const std::string& tagName = "BeobotControl");

  //! Constructor
  ~BeobotControl();

  //! Move Beobot
  //BeobotAction move(float vel, float angle);

  //! calibrate the speed SSC to new values
  /*! The speed SSC is calibrated to the values defined in
    a car.conf file during construction.  This function allows
    the SSC to be recalibrated to new min/neutral/max values.
  */
  void calibrateSpeed( const byte neutralval, const byte minval,
                       const byte maxval );

  //! calibrate the steering SSC to new values
  /*! The steering SSC is calibrated to the values defined in
    a car.conf file during construction.  This function allows
    the SSC to be recalibrated to new min/neutral/max values.
  */
  void calibrateSteer( const byte neutralval, const byte minval,
                       const byte maxval );

  //! get speed value [-1.0 .. 1.0]
  float getSpeed() const;

  //! get gear value [-1.0 .. 1.0]
  float getGear() const;

  //! get steer value [-1.0 .. 1.0]
  float getSteer() const;

  //! set speed value: [-1.0 (full reverse) to 1.0 (full forward)]
  bool setSpeed(const float newspeed);

  //! set gear: -1.0 (neutral) 0.0 (first gear, slow), 1.0 (second gear, fast)
  bool setGear(const float newgear);

  //! set steer angle: [-1.0 (full left) to 1.0 (full right)]
  bool setSteer(const float newsteer);

  //! linear ramp of speed from current speed to new speed
  /*! Speed will reach newspeed after t milliseconds */
  bool toSpeedLinear( const float newspeed, const int t );

  //! sigmoid ramp of speed from current speed to new speed
  /*! Speed will reach newspeed after t milliseconds */
  bool toSpeedSigmoid( const float newspeed, const int t );

  //! speed ramping function
  /*! This function is used for gradually increasing the speed with a certain
    behavior over a certain period of time. The ramping will be done in a
    separate thread so this function will return immediately. It is not
    necessary to wait until the new speed has been reached before calling this
    function to modify the ramping parameters again. The ramping function will
    be stopped if abortRamp() is called, or else if the thread continues other
    calls to setSpeed() may seem to have no effect.
    @param newspeed new speed to be reached after ramping
    @param t time in which new speed will be reached (milliseconds)
    @param behavior specifies behavior of ramping function
  */
  void rampSpeed( const float newspeed, const int t, SpeedRampType behavior );

  //! aborts speed ramping
  /*! Used to stop speed ramping thread so that main program has priority for
    setting speed. */
  void abortRamp( void );

private:
  nub::soft_ref<BeoChip> itsBeoChip;  //!< our servo controller
  BeobotConfig itsBeobotConfig;       //!< config params for car

  pthread_mutex_t setSpeedMutex;      //!< mutex for setting speed

  float startSpeed;                   //!< speed at beginning of ramping
  float desiredSpeed;                 //!< desired speed setting
  int desiredTime;                    //!< time to reach desired speed
  Timer speedRampTimer;               //!< timer to track ramping time
  SpeedRampType speedRampType;        //!< type of speed ramping behavior
  pthread_t speedRampThread;          //!< thread for speed ramping functions
  pthread_mutex_t speedRampMutex;     //!< mutex for speed ramping variables
  bool speedRampThreadCreated;        //!< tells if thread is created

  //! routine for speed ramp thread
  friend void *speedRampFunc( void *ptr );

  //! stop the car as we terminate execution
  void stop1();
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
