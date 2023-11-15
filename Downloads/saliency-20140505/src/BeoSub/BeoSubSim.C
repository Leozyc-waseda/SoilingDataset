/*!@file BeoSub/BeoSubSim.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubSim.C $
// $Id: BeoSubSim.C 7880 2007-02-09 02:34:07Z itti $
//

#include "BeoSub/BeoSubSim.H"
#include "Devices/HMR3300.H"
#include "BeoSub/BeoSubBallast.H"
#include "BeoSub/BeoSubIMU.H"
#include "Util/MathFunctions.H"
#include "Devices/IEEE1394grabber.H"

// ######################################################################
//! Class definition for BeoSubListener
/*! This is the listener class that is attached to each BeoChip in the
  ballast tube of the submarine. This class is just a
  pass-through to the function dispatchBeoChipEvent() of class
  BeoSubOneBal. */
/*class BeoSubListener : public BeoChipListener
{
public:
  //! Constructor
  BeoSubListener(BeoSubSim *sub) : itsBeoSub(sub)  { }

  //! destructor
  virtual ~BeoSubListener()  { }

  //! get an event
  virtual void event(const BeoChipEventType t, const int valint,
                     const float valfloat)
  {
    //LDEBUG("Event: %d val = %d, fval = %f", int(t), valint, valfloat);
    itsBeoSub->dispatchBeoChipEvent(t, valint, valfloat);
  }

private:
  BeoSubSim *itsBeoSub;        //!< pointer to our master
};*/

// ######################################################################
// here a listener for the compass:
/*class HMRlistener : public HMR3300Listener {
public:
  //! Constructor
  HMRlistener(BeoSubOneBal *sub) : itsBeoSub(sub) { }

  //! Destructor
  virtual ~HMRlistener() {};

  //! New data was received
  virtual void newData(const Angle heading, const Angle pitch,
                       const Angle roll)
  {
    //LDEBUG("<Heading=%f Pitch=%f Roll=%f>", heading.getVal(),
    //       pitch.getVal(), roll.getVal());
    itsBeoSub->updateCompass(heading, pitch, roll);
  }

private:
  BeoSubSim *itsBeoSub;
  };*/


// ######################################################################
BeoSubSim::BeoSubSim(OptionManager& mgr) :
  BeoSub(mgr),
  itsLeftThrusterServoNum("BeoSubLeftThrusterServoNum", this, 3),
  itsRightThrusterServoNum("BeoSubRightThrusterServoNum", this, 2),
  itsThrustLeft(0),
  itsThrustRight(0),
  relDistance(0),
  isStrafe(false),
  imageCounter(0),
  itsCurrentZ(0),
  itsCurrentX(0),
  upImageFlag(false),
  frontImageFlag(false),
  downImageFlag(false),
  itsDepthSensor(10),
  itsHeadingSensor(10),
  itsPitchSensor(10),
  itsDepthPID(0.1F, 0.001F, 0.1F, 1.5F, 1.5F),
  itsHeadingPID(0.1F, 0.001F, 0.1F, Angle(170.0), Angle(170.0)),
  itsPitchPID(0.1F, 0.001F, 0.1F, Angle(170.0), Angle(170.0)),
  itsDepthPIDon(false),
  itsHeadingPIDon(false),
  itsPitchPIDon(false)

{

}

// ######################################################################
BeoSubSim::~BeoSubSim()
{  }

// ######################################################################
void BeoSubSim::start1()
{
  // select firewire subchannels for our cameras:
  /*itsCameraDown->setModelParamVal("FrameGrabberSubChan", int(BEOSUBCAMDOWN));
  itsCameraFront->setModelParamVal("FrameGrabberSubChan", int(BEOSUBCAMFRONT));
  itsCameraUp->setModelParamVal("FrameGrabberSubChan", int(BEOSUBCAMUP));*/
  BeoSub::start1();
}

// ######################################################################
void BeoSubSim::start2()
{
  // turn on the analog port capture for the pressure sensor, do not
  // capture pulses:
  /*itsBeo->captureAnalog(0, true);
  itsBeo->captureAnalog(1, false);
  itsBeo->capturePulse(0, false);
  itsBeo->capturePulse(1, false);

  // set all servos to neutral:
  for (int i = 0; i < 8; i ++) itsBeo->setServo(i, 0.0F);

  // turn on KBD capture for our ballasts:
  itsBeo->captureKeyboard(true);
  itsBeo->debounceKeyboard(false);
  */
  // initialize all ballasts at once:
  CLINFO("filling ballasts...");
  setBallasts(1.0F, 1.0F, false); sleep(4);
  CLINFO("emptying ballasts...");
  setBallasts(0.0F, 0.0F, true);

  // turn the PIDs on:
  itsHeadingPIDon = true;
  itsPitchPIDon = true;
}

// ######################################################################
void BeoSubSim::updateCompass(const Angle heading, const Angle pitch,
                                 const Angle roll)
{
  // note the time at which the data was received:
  double t = itsMasterClock.getSecs();

  // inject these values into our current attitude:
  pthread_mutex_lock(&itsLock);
  itsCurrentAttitude.heading = heading;
  itsCurrentAttitude.pitch = pitch;
  //itsCurrentAttitude.roll = roll;
  itsCurrentAttitude.compassTime = t;

  // update our heading and pitch sensors:
  //itsHeadingSensor.newMeasurement(heading);
  //itsPitchSensor.newMeasurement(pitch);

  pthread_mutex_unlock(&itsLock);
}


// ######################################################################
void BeoSubSim::updateDepth(const float depth)
{
  // note the time at which the data was received:
  double t = itsMasterClock.getSecs();

  // inject these values into our current attitude:
  pthread_mutex_lock(&itsLock);
  itsCurrentAttitude.depth = depth;
  itsCurrentAttitude.pressureTime = t;

  // update our depth sensor:
  itsDepthSensor.newMeasurement(depth);

  pthread_mutex_unlock(&itsLock);
}

void BeoSubSim::updatePosition(const float z, const float x) {
  itsCurrentZ = z;
  itsCurrentX = x;
}
// ######################################################################
void BeoSubSim::thrust(const float leftval, const float rightval)
{
  itsThrustLeft = leftval;
  itsThrustRight = rightval;
}

// ######################################################################
void BeoSubSim::getThrusters(float& leftval, float& rightval) const
{ leftval = itsThrustLeft; rightval = itsThrustRight; }

// ######################################################################
void BeoSubSim::setFrontBallast(const float val, const bool blocking)
{  }

// ######################################################################
void BeoSubSim::setRearBallast(const float val, const bool blocking)
{  }

// ######################################################################
void BeoSubSim::setBallasts(const float f, const float r,
                               const bool blocking)
{  }

// ######################################################################
float BeoSubSim::getFrontBallast() const
{ return 0;}

// ######################################################################
float BeoSubSim::getRearBallast() const
{ return 0; }

// ######################################################################
void BeoSubSim::getBallasts(float& f, float& r) const
{  }

// ######################################################################
void BeoSubSim::dropMarker(const bool blocking)
{
  LFATAL("unimplemented!");
}

// ######################################################################
void BeoSubSim::dispatchBeoChipEvent(const BeoChipEventType t,
                                  const int valint,
                                  const float valfloat)
{

}
// problems
// ######################################################################
Image< PixRGB<byte> > BeoSubSim::grabImage(const enum BeoSubCamera cam) const
{
  if(cam == BEOSUBCAMFRONT) {
    const_cast<BeoSubSim *>(this)-> upImageFlag = true;
     const_cast<BeoSubSim *>(this)-> imageCounter++;
    sleep(1);
    return Raster::ReadRGB(sformat("upimage%d.ppm", imageCounter));
  }
  if(cam == BEOSUBCAMDOWN) {
    const_cast<BeoSubSim *>(this)-> downImageFlag = true;
     const_cast<BeoSubSim *>(this)-> imageCounter++;
    sleep(1);
    return Raster::ReadRGB(sformat("downimage%d.ppm", imageCounter));
  }
  if(cam == BEOSUBCAMUP) {
    const_cast<BeoSubSim *>(this)-> frontImageFlag = true;
    const_cast<BeoSubSim *>(this)-> imageCounter++;
    sleep(5);

    return Raster::ReadRGB(sformat("upimage%d.ppm", imageCounter));
  }
  //do a wait
  //do a read
  LERROR("Wrong camera %d -- RETURNING EMPTY IMAGE", int(cam));
  return Image< PixRGB<byte> >();
  //return grab(cam);
}

//######################################################################
void BeoSubSim::advanceRel(const float relDist, const bool stop) {
  float value = relDistance = relDist;
  if(relDist > 1.0F) value = 1.0F;
  if(relDist < -1.0F) value = -1.0F;
  thrust(value, value);
  usleep((int) (100000));
}

void BeoSubSim::strafeRel(const float relDist) {
  relDistance = relDist;
  isStrafe = true;
  usleep((int) (100000));
}

void BeoSubSim::turnAbs(const Angle finalHeading, const bool blocking) {
  itsTargetAttitude.heading = finalHeading;
}

void BeoSubSim::turnRel(const Angle relHeading, const bool blocking) {

  itsTargetAttitude.heading = itsCurrentAttitude.heading + relHeading;
  sleep(1);
}

void BeoSubSim::turnOpen(const Angle relHeading, const bool blocking) {

  itsTargetAttitude.heading = itsCurrentAttitude.heading + relHeading;
  sleep(1);
}

void BeoSubSim::diveAbs(const float finalDepth, const bool blocking) {
  itsTargetAttitude.depth = finalDepth;
}

void BeoSubSim::diveRel(const float relDepth, const bool blocking) {
    itsTargetAttitude.depth = itsCurrentAttitude.depth + relDepth;
}

// ######################################################################
void BeoSubSim::useDepthPID(const bool useit)
{ itsDepthPIDon = useit; }

// ######################################################################
void BeoSubSim::useHeadingPID(const bool useit)
{ itsHeadingPIDon = useit; }

// ######################################################################
void BeoSubSim::usePitchPID(const bool useit)
{ itsPitchPIDon = useit; }


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
