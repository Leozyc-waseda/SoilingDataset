/*!@file Beobot/BeobotControl.C control the movement of the Beobot
  (via BeoChip)                                                         */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotControl.C $
// $Id: BeobotControl.C 7063 2006-08-29 18:26:55Z rjpeters $

#include "Beobot/BeobotControl.H"
#include "Component/OptionManager.H"
#include "Util/Types.H"
#include "Util/log.H"
#include <math.h>

#define MAX_SERVO_MOVE .10

// ######################################################################
void *speedRampFunc( void *ptr )
{
  // number of times we set speed each second
  const int signalsPerSecond = 50;
  const uint64 delayPerSignal = 1000000 / signalsPerSecond;

  float startSpeed;
  float desiredSpeed;
  int desiredTime;
  int currentTime;
  SpeedRampType type;

  bool threadOK;
  Timer tmr( 1000000 );

  // cast our void pointer back to CarControl
  BeobotControl *bc = reinterpret_cast<BeobotControl *>(ptr);

  // loop while thread is supposed to be active and ramping is not yet done
  do
    {
      // start timing
      tmr.reset();

      // lock our speed ramping related variables
      pthread_mutex_lock( &(bc->speedRampMutex) );

      startSpeed = bc->startSpeed;
      desiredSpeed = bc->desiredSpeed;
      desiredTime = bc->desiredTime;
      currentTime = bc->speedRampTimer.get();
      type = bc->speedRampType;
      threadOK = bc->speedRampThreadCreated;

      // done reading shared variables
      pthread_mutex_unlock( &(bc->speedRampMutex) );

      if( threadOK )
        {
          // if time is up, set and return
          if( currentTime >= desiredTime )
            {
              LDEBUG( "Time's up!  Will exit speed ramping thread..." );
              break;
            }

          // otherwise, do ramping
          else
            {
              switch( type )
                {
                case SPEED_RAMP_LINEAR:
                  bc->setSpeed( startSpeed + ( desiredSpeed - startSpeed ) /
                                desiredTime * currentTime );
                  break;
                case SPEED_RAMP_SIGMOID:
                  bc->setSpeed( startSpeed + ( desiredSpeed - startSpeed ) *
                                ( 1.0f / ( 1.0f +
                                           exp( -10.0f / desiredTime *
                                                currentTime + 5.0f ) ) ) );
                  break;
                default:
                  // user did not specify a valid ramping type
                  LERROR( "Invalid speed ramping type: %d", type );
                  break;
                }
            }
        }

      // wait until time is up before next signal
      while( tmr.get() < delayPerSignal )
        {
          // do nothing
        }
    }
  while( threadOK );

  LDEBUG( "Speed ramping thread finished" );
  pthread_mutex_lock( &(bc->speedRampMutex ) );
  bc->speedRampThreadCreated = false;
  pthread_mutex_unlock( &(bc->speedRampMutex ) );

  return 0;
}

// ######################################################################
BeobotControl::BeobotControl(nub::soft_ref<BeoChip> beoChip,
                             OptionManager& mgr, const std::string& descrName,
                             const std::string& tagName) :
  ModelComponent(mgr, descrName, tagName),
  itsBeoChip(beoChip),
  speedRampTimer( 1000 )
{
  // BeoChip is calibrated in the constructor

  // use the raw value to initialize the servo values
  // keep the gear at the lowest speed/highest torque
  itsBeoChip->setServoRaw(itsBeobotConfig.speedServoNum,
                          itsBeobotConfig.speedNeutralVal);
  itsBeoChip->setServoRaw(itsBeobotConfig.steerServoNum,
                          itsBeobotConfig.steerNeutralVal);
  itsBeoChip->setServoRaw(itsBeobotConfig.gearServoNum,
                          itsBeobotConfig.gearMinVal);

  // not using speed ramping thread yet
  speedRampThreadCreated = false;
}

// ######################################################################
BeobotControl::~BeobotControl(){ }

// ######################################################################
float BeobotControl::getSpeed() const
{ return itsBeoChip->getServo(itsBeobotConfig.speedServoNum); }

// ######################################################################
float BeobotControl::getSteer() const
{ return itsBeoChip->getServo(itsBeobotConfig.steerServoNum); }

// ######################################################################
float BeobotControl::getGear() const
{ return itsBeoChip->getServo(itsBeobotConfig.gearServoNum ); }

// ######################################################################
bool BeobotControl::setSpeed(const float newspeed)
{
  // check if newspeed is within limit
  float procspeed = newspeed;
  if(newspeed > 1.0f) procspeed = 1.0f;
  else if(newspeed < -1.0f) procspeed = -1.0f;

  pthread_mutex_lock( &setSpeedMutex );

  // make sure the speed assignment does not go beyond the jump limit
  float currspeed = getSpeed();
  if(procspeed > currspeed + MAX_SERVO_MOVE)
    procspeed = currspeed + MAX_SERVO_MOVE;
  else if(procspeed < currspeed - MAX_SERVO_MOVE)
    procspeed = currspeed - MAX_SERVO_MOVE;

  bool setOK =
    itsBeoChip->setServo(itsBeobotConfig.speedServoNum, procspeed);
  pthread_mutex_unlock( &setSpeedMutex );
  return setOK;
}

// ######################################################################
bool BeobotControl::setSteer(const float newsteer)
{
  // check if newsteer is within limit
  float procsteer = newsteer;
  if(newsteer > 1.0f) procsteer = 1.0f;
  else if(newsteer < -1.0f) procsteer = -1.0f;

  // make sure the steer assignment does not go beyond the jump limit
  float currsteer = getSteer();
  if(procsteer > currsteer + MAX_SERVO_MOVE)
    procsteer = currsteer + MAX_SERVO_MOVE;
  else if(procsteer < currsteer - MAX_SERVO_MOVE)
    procsteer = currsteer - MAX_SERVO_MOVE;

  return itsBeoChip->setServo(itsBeobotConfig.steerServoNum, procsteer);
}

// ######################################################################
bool BeobotControl::setGear(const float newgear)
{
  // check if newgear is within limit
  float procgear = newgear;
  if(newgear > 1.0f) procgear = 1.0f;
  else if(newgear < -1.0f) procgear = -1.0f;

  // make sure the gear assignment does not go beyond the jump limit
  float currgear = getGear();
  if(procgear > currgear + MAX_SERVO_MOVE)
    procgear = currgear + MAX_SERVO_MOVE;
  else if(procgear < currgear - MAX_SERVO_MOVE)
    procgear = currgear - MAX_SERVO_MOVE;

  return itsBeoChip->setServo(itsBeobotConfig.gearServoNum, procgear);
}

// ######################################################################
void BeobotControl::stop1()
{
  // end ramping process:
  pthread_cancel( speedRampThread );

  // let's stop the car:
  setSpeed(0.0F);
  setSteer(0.0F);
  setGear (0.0F);
}

// ######################################################################
bool BeobotControl::toSpeedLinear( const float newspeed, const int t )
{
  float current = getSpeed();

  // calculate how many times setSpeed() can be called in t milliseconds
  int calls = (int)( t / 1000.0 * 30 + 0.5 ); // 30 calls per second
  if( calls <= 1 )
  {
    return setSpeed( newspeed );
  }
  // calculate appropriate delta values
  else
  {
    bool speedChangeOk = true;
    float delta = ( newspeed - current ) / calls;
    for( int i = 0; i < calls; i++ )
    {
      speedChangeOk = speedChangeOk && setSpeed( current + delta * i );
      Timer tim( 1000000 );
      while( tim.get() < 33333 )
        {
          // couldn't get usleep() to work, this seems to do the job
        }
    }
    return speedChangeOk;
  }
}

// ######################################################################
bool BeobotControl::toSpeedSigmoid( const float newspeed, const int t )
{
  float current = getSpeed();

  // calculate how many times setSpeed() can be called in t milliseconds
  int calls = (int)( t / 1000.0 * 30 + 0.5 ); // 30 calls per second
  if( calls <= 1 )
  {
    return setSpeed( newspeed );
  }
  // calculate appropriate delta values
  else
  {
    bool speedChangeOk = true;
    for( int i = 0; i < calls; i++ )
    {
      speedChangeOk = speedChangeOk &&
        setSpeed( current + ( newspeed - current ) /
                  ( 1 + exp( -11.0825 / calls * ( i - calls/2 ) ) ) );
      Timer tim( 1000000 );
      while( tim.get() < 33333 )
        {
          // couldn't get usleep() to work, this seems to do the job
        }
    }
    return speedChangeOk;
  }

  return true;
}

// ######################################################################
void BeobotControl::rampSpeed( const float newspeed, const int t,
                            SpeedRampType behavior )
{
  // lock our speed ramping related variables
  pthread_mutex_lock( &speedRampMutex );

  startSpeed = getSpeed();
  desiredSpeed = newspeed;
  desiredTime = t;
  speedRampType = behavior;
  speedRampTimer.reset();

  // modification is now complete
  pthread_mutex_unlock( &speedRampMutex );

  // create thread if not already existing
  if( !speedRampThreadCreated )
    {
      speedRampThreadCreated = true;
      pthread_create( &speedRampThread, NULL, speedRampFunc,
                      reinterpret_cast<void *>(this) );
    }
}

// ######################################################################
void BeobotControl::abortRamp( void )
{
  if( speedRampThreadCreated )
    {
      speedRampThreadCreated = false;
      pthread_cancel( speedRampThread );
    }
}

// ######################################################################
// BeobotAction BeobotControl::move(float vel, float angle)
// {
//   // use neural network to get the appropriate values
//   return BeobotAction(0.0,0.0,0);
// }

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
