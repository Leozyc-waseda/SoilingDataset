/*!@file BeoSub/BeoSubOneBal.C An autonomous submarine with one ballast*/

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubOneBal.C $
// $Id: BeoSubOneBal.C 7880 2007-02-09 02:34:07Z itti $
//

#include "BeoSub/BeoSubOneBal.H"
#include "Devices/HMR3300.H"
#include "BeoSub/BeoSubBallast.H"
#include "BeoSub/BeoSubIMU.H"
#include "Util/MathFunctions.H"
#include "Devices/FrameGrabberFactory.H"

// ######################################################################
//! Class definition for BeoSubListener
/*! This is the listener class that is attached to each BeoChip in the
  ballast tube of the submarine. This class is just a
  pass-through to the function dispatchBeoChipEvent() of class
  BeoSubOneBal. */
class BeoSubListener : public BeoChipListener
{
public:
  //! Constructor
  BeoSubListener(BeoSubOneBal *sub) : itsBeoSub(sub)  { }

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
  BeoSubOneBal *itsBeoSub;        //!< pointer to our master
};

// ######################################################################
// here a listener for the compass:
class HMRlistener : public HMR3300Listener {
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
  BeoSubOneBal *itsBeoSub;
};

// here a listener for the compass:
class IMUListener : public BeoSubIMUListener {
public:
  //! Constructor
  IMUListener(BeoSubOneBal *sub) : itsBeoSub(sub) { }
  //! Destructor
  virtual ~IMUListener() {};

  //! New data was received
  virtual void newData(const float xa, const float ya, const float za,
                       const Angle xv, const Angle yv, const Angle zv)
  {
          itsBeoSub->updateIMU(xa, ya, za, xv, yv, zv);
  }
private:
  BeoSubOneBal *itsBeoSub;
};


/*
// ######################################################################
// here a listener for the compass:
class IMUlistener : public BeoSubIMUListener {
public:
  //! Destructor
  virtual ~HMRlistener() {};

  //! New data was received
  virtual void newData(const float xa, const float ya, const float za,
                       const Angle xv, const Angle yv, const Angle zv)
  {
  }

private:
  BeoSubOneBal *itsBeoSub;
};
*/



// ######################################################################
BeoSubOneBal::BeoSubOneBal(OptionManager& mgr) :
  BeoSub(mgr),
  itsIMU(new BeoSubIMU(mgr)),
  itsLeftThrusterServoNum("BeoSubLeftThrusterServoNum", this, 3),
  itsRightThrusterServoNum("BeoSubRightThrusterServoNum", this, 2),
  itsHMR3300(new HMR3300(mgr)),
  itsBeo(new BeoChip(mgr, "BeoChip", "BeoChip")),
  itsFballast(new BeoSubBallast(mgr, "Front Ballast", "FrontBallast")),
  itsRballast(new BeoSubBallast(mgr, "Rear Ballast", "RearBallast")),
  itsCameraFront(makeIEEE1394grabber(mgr, "Front Camera", "FrontCamera")),
  itsCameraDown(makeIEEE1394grabber(mgr, "Down Camera", "DownCamera")),
  itsCameraUp(makeIEEE1394grabber(mgr, "Up Camera", "UpCamera")),
  itsDepthSensor(30, 0.99999F),
  itsHeadingSensor(3, 0.99999F),
  itsPitchSensor(3, 0.99999F),
  itsDepthPID(0.001F, 0.0F, 0.1F, 3.0F, 3.0F),
  itsHeadingPID(0.002F, 0.0F, 0.03F, Angle(170.0), Angle(170.0)),
  itsPitchPID(0.001F, 0.0F, 0.1F, Angle(170.0), Angle(170.0)),
  itsRotVelPID(0.3, 0.0001, 0, -500, 500),
  itsDepthPIDon(false),
  itsHeadingPIDon(false),
  itsPitchPIDon(false),
  itsRotVelPIDon(false),
  itsKillSwitchUsed(false),
  itsDiveSetting(0.0F),
  itsPitchSetting(0.0F),
  itsLastHeadingTime(0.0),
  itsLastPitchTime(0.0),
  itsLastDepthTime(0.0),
#ifdef HAVE_LINUX_PARPORT_H
  lp0(new ParPort(mgr)),
  markerdropper(lp0),
#endif
  killSwitchDebounceCounter(0)

{
  // register our babies as subcomponents:
  addSubComponent(itsHMR3300);
  addSubComponent(itsIMU);
  addSubComponent(itsBeo);
  addSubComponent(itsCameraDown);
  addSubComponent(itsCameraFront);
  addSubComponent(itsCameraUp);
  addSubComponent(itsFballast);
  addSubComponent(itsRballast);
#ifdef HAVE_LINUX_PARPORT_H
  addSubComponent(lp0);
#endif

  // reset the BeoChip:
  LINFO("Resetting the BeoChip...");
  itsBeo->reset(MC_RECURSE);
  usleep(200000);

  // connect our listener to our beochip:
  rutz::shared_ptr<BeoChipListener> b(new BeoSubListener(this));
  itsBeo->setListener(b);

  // connect our listener to our compass:
  rutz::shared_ptr<HMR3300Listener> bl(new HMRlistener(this));
  itsHMR3300->setListener(bl);

  rutz::shared_ptr<BeoSubIMUListener> imu(new IMUListener(this));
  itsIMU->setListener(imu);

  // hook up BeoChips to Ballasts:
  itsFballast->setBeoChip(itsBeo);
  itsRballast->setBeoChip(itsBeo);

  // disable RTS/CTS flow control on our BeoChips:
  itsBeo->setModelParamVal("BeoChipUseRTSCTS", false);

  // select default serial ports for everybody:
  itsHMR3300->setModelParamString("HMR3300SerialPortDevName",
                                  "/dev/ttyS0", MC_RECURSE);
  itsBeo->setModelParamString("BeoChipDeviceName", "/dev/ttyS2");
  itsIMU->setModelParamString("IMUSerialPortDevName", "/dev/ttyS3",
                              MC_RECURSE);

  // set the I/O for our ballasts:
  itsRballast->setModelParamVal("RearBallastOutRed", 1);
  itsRballast->setModelParamVal("RearBallastOutWhite", 0);
  itsRballast->setModelParamVal("RearBallastInYellow", 1);
  itsRballast->setModelParamVal("RearBallastInWhite", 0);

  itsFballast->setModelParamVal("FrontBallastOutRed", 3);
  itsFballast->setModelParamVal("FrontBallastOutWhite", 2);
  itsFballast->setModelParamVal("FrontBallastInYellow", 3);
  itsFballast->setModelParamVal("FrontBallastInWhite", 2);
  //mgr.loadConfig("camconfig.pmap");
}

// ######################################################################
BeoSubOneBal::~BeoSubOneBal()
{  }

// ######################################################################
void BeoSubOneBal::start1()
{
  // select firewire subchannels for our cameras:
  itsCameraDown->setModelParamVal("FrameGrabberSubChan", int(BEOSUBCAMDOWN));
  itsCameraFront->setModelParamVal("FrameGrabberSubChan", int(BEOSUBCAMFRONT));
  itsCameraUp->setModelParamVal("FrameGrabberSubChan", int(BEOSUBCAMUP));

  //NOTE that the stuff below is certainly necessary, but DOES NOT WORK in current form. FIX!
  /*
  itsCameraFront->setModelParamVal("FrameGrabberExposure", 511);
  itsCameraFront->setModelParamVal("FramGrabberGain", 87);
  itsCameraFront->setModelParamVal("FrameGrabberNbuf", 10);
  itsCameraFront->setModelParamVal("FrameGrabberSaturation", 90);
  itsCameraFront->setModelParamVal("FrameGrabberWhiteBalSU", 95);
  itsCameraFront->setModelParamVal("FrameGrabberWhiteBalRV", 87);

  itsCameraDown->setModelParamVal("FrameGrabberExposure", 511);
  itsCameraDown->setModelParamVal("FramGrabberGain", 87);
  itsCameraDown->setModelParamVal("FrameGrabberNbuf", 10);
  itsCameraDown->setModelParamVal("FrameGrabberSaturation", 90);
  itsCameraDown->setModelParamVal("FrameGrabberWhiteBalSU", 95);
  itsCameraDown->setModelParamVal("FrameGrabberWhiteBalRV", 87);

  itsCameraUp->setModelParamVal("FrameGrabberExposure", 511);
  itsCameraUp->setModelParamVal("FramGrabberGain", 87);
  itsCameraUp->setModelParamVal("FrameGrabberNbuf", 10);
  itsCameraUp->setModelParamVal("FrameGrabberSaturation", 90);
  itsCameraUp->setModelParamVal("FrameGrabberWhiteBalSU", 95);
  itsCameraUp->setModelParamVal("FrameGrabberWhiteBalRV", 87);
  */
  // don't forget to call start1() on our base class!
  BeoSub::start1();
}

// ######################################################################
void BeoSubOneBal::start2()
{
  // turn on the analog port capture for the pressure sensor, do not
  // capture pulses:
  itsBeo->captureAnalog(0, true);
  itsBeo->captureAnalog(1, false);
  itsBeo->capturePulse(0, false);
  itsBeo->capturePulse(1, false);

  // set all servos to neutral:
  for (int i = 0; i < 8; i ++) itsBeo->setServo(i, 0.0F);

  // turn on KBD capture for our ballasts:
  itsBeo->captureKeyboard(true);
  itsBeo->debounceKeyboard(false);

  // initialize all ballasts at once:
  CLINFO("filling ballasts...");
  setBallasts(1.0F, 1.0F, false); sleep(4);
  CLINFO("emptying ballasts...");
  setBallasts(0.0F, 0.0F, true);

  // turn the PIDs on: NOTE: first we should dive some...
  //itsHeadingPIDon = true;
  //setBallasts(0.5F, 0.5F, true);
  //itsPitchPIDon = true;
  itsDiveSetting = 0.5F;

  // display sign of life:
  itsBeo->lcdClear();
  itsBeo->lcdPrintf(3, 0, " iLab and USC rock! ");

  // we currently don't have a start2() in our base class, but if we
  // add one we should call it here.
}

void BeoSubOneBal::stop1() {
        //Turn off all PID's
        useDepthPID(false);
        useHeadingPID(false);
        usePitchPID(false);
        useRotVelPID(false);
        setRotVel(0.0);
        setTransVel(0.0);

        //empty both balasts and  surface!
        setBallasts(0.0F, 0.0F, true);

}

// ######################################################################
void BeoSubOneBal::advanceRel(const float relDist, const bool stop)
{
  float d = relDist * 2.0F;
  // turn our heading PID off as it may interfere with our commands of
  // the thrusters:
  bool hpid = itsHeadingPIDon;
  useHeadingPID(false);

  // just pulse the thrusters:
  if (fabsf(d) < 1.0F)
    {
        setTransVel(d);
      sleep(1);
      if (stop)
        {
        setTransVel(-d);
          usleep(350000);
        }
    }
  else
    {
      float forward = 1.0F, reverse = -1.0F;
      if (d < 0.0F) { forward = -forward; reverse = -reverse; }

      // thrust at full forward:
      setTransVel(forward);

      // for a time that is proportional to distance. Note: we will
      // reach steady-state speed after about 1sec:
      usleep(int(fabsf(d) * 1000000));

      if (stop)
        {
          // now, to stop, thrust full reverse:
          setTransVel(reverse);

          // for one second:
          sleep(1);
        }
    }

  // stop
  setTransVel(0);

  itsGlobalPosition.x += relDist * cos(itsGlobalHeading.getRadians());
  itsGlobalPosition.y += relDist * sin(itsGlobalHeading.getRadians());

  useHeadingPID(hpid);
}

// ######################################################################
void BeoSubOneBal::turnOpen(const Angle openHeading, const bool stop){
  bool hpid = itsHeadingPIDon;
  useHeadingPID(false);

  /* for now, we'll always turn at 15 degrees/sec */
  float value = openHeading.getVal();
  float turnrate;
  if(value > 0) {
    turnrate = 15;
  } else {
    turnrate = -15;
  }

  setRotVel(turnrate);
  usleep((long)((value/turnrate)*1e6));
  setRotVel(0.0);
  itsGlobalHeading += openHeading;
  useHeadingPID(hpid);
}

// ######################################################################
void BeoSubOneBal::strafeRel(const float relDist)
{/*
  bool hpid = itsHeadingPIDon;
  useHeadingPID(false);
  BeoSub::strafeRel(relDist);
  useHeadingPID(hpid);*/
  turnRel(90);
  advanceRel(relDist);
  turnRel(-90);
}

void BeoSubOneBal::updateThrusters() {
        float left = TransVelCommand + RotVelCommand;
        float right = TransVelCommand - RotVelCommand;

        if (left > 1) left = 1;
        else if (left < -1) left = -1;

        if (right > 1) right = 1;
        else if (right < -1) right = -1;

        thrust(left, right);
}

void BeoSubOneBal::updateRotVelPID(const Angle current) {
        if(itsRotVelPIDon) {
                //LINFO("%5.2f %5.2f", (PIDRotVel-current).getVal(), RotVelCommand);
                RotVelCommand = itsRotVelPID.update(PIDRotVel, current).getVal();
                updateThrusters();
        }
}

void BeoSubOneBal::setRotVel(const Angle desired) {
        if(itsRotVelPIDon) {
                PIDRotVel = desired;
        } else {
                RotVelCommand = desired.getVal();
                updateThrusters();
        }
}

void BeoSubOneBal::setTransVel(const float desired) {
        //desiredTransVel = desired;
        TransVelCommand = desired; /* open loop for now */
        updateThrusters();
}

void BeoSubOneBal::updateIMU(const float xa, const float ya, const float za,
                             const Angle xv, const Angle yv, const Angle zv)
{
        //LINFO("%5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f", xa, ya, za, xv.getVal(), yv.getVal(), zv.getVal(), itsCurrentAttitude.heading.getVal(), itsCurrentAttitude.pitch.getVal(), itsCurrentAttitude.roll.getVal());
        updateRotVelPID(zv); /* ignore roll/pitch interaction for now */
}

// ######################################################################
void BeoSubOneBal::updateCompass(const Angle heading, const Angle pitch,
                                 const Angle roll)
{
  // note the time at which the data was received:
  double t = itsMasterClock.getSecs();

  pthread_mutex_lock(&itsLock);

  // update our heading and pitch sensors:
  itsHeadingSensor.newMeasurement(heading);
  itsPitchSensor.newMeasurement(pitch);

  // inject these values into our current attitude:
  itsCurrentAttitude.heading = itsHeadingSensor.getValue();
  itsCurrentAttitude.pitch = itsPitchSensor.getValue();
  itsCurrentAttitude.roll = roll;
  itsCurrentAttitude.compassTime = t;


  // also update our heading PID and thrust accordingly:
  // only update every once in a while (in seconds):
  if (itsHeadingPIDon && t >= itsLastHeadingTime + 0.5)
    {
      // feed the PID controller:
      const float hcmd =
        itsHeadingPID.update(itsTargetAttitude.heading,
                         itsHeadingSensor.getValue()).getVal();
      setRotVel(hcmd);

      // remember when we did all this:
      itsLastHeadingTime = t;

      LINFO("hcmd = %f at t = %f", hcmd, t);
    }

  // also update our pitch PID and adjust ballasts accordingly:
  if (itsPitchPIDon && t >= itsLastPitchTime + 0.5)
    {
      // feed the PID controller:
      const float pcmd =
        itsPitchPID.update(itsTargetAttitude.pitch,
                           itsPitchSensor.getValue()).getVal();

      // update and limit our pitch setting: This setting should be
      // subtracted from the front ballast and added to the rear
      // ballast, relative to level:
      itsPitchSetting += pcmd;
      if (itsPitchSetting > 0.25F) itsPitchSetting = 0.25F;
      if (itsPitchSetting < -0.25F) itsPitchSetting = -0.25F;

      // adjust the ballasts: itsDiveSetting provides the mean, while
      // itsPitchSetting provides the deviation:
      float f = itsDiveSetting - itsPitchSetting;
      float r = itsDiveSetting + itsPitchSetting;
      if (f < 0.0F) f = 0.0F; else if (f > 1.0F) f = 1.0F;
      if (r < 0.0F) r = 0.0F; else if (r > 1.0F) r = 1.0F;
      setBallasts(f, r);

      // remember when we did all this:
      itsLastPitchTime = t;

      LINFO("pcmd = %f at t = %f", pcmd, t);
    }

  pthread_mutex_unlock(&itsLock);
}

// ######################################################################
void BeoSubOneBal::updateDepth(const float depth)
{
  // note the time at which the data was received:
  double t = itsMasterClock.getSecs();

  pthread_mutex_lock(&itsLock);

  // update our depth sensor:
  itsDepthSensor.newMeasurement(depth);

  // inject these values into our current attitude:
  itsCurrentAttitude.depth = itsDepthSensor.getValue();
  itsCurrentAttitude.pressureTime = t;
  const float tdepth = itsTargetAttitude.depth;

  // also update our depth PID and adjust ballasts:
  if (itsDepthPIDon && t >= itsLastDepthTime + 0.5)
    {
      // feed the PID controller:
      const float dcmd = itsDepthPID.update(tdepth, itsDepthSensor.getValue());

      // update and limit our dive setting:
      itsDiveSetting += dcmd;
      if (itsDiveSetting < 0.0F) itsDiveSetting = 0.0F;
      else if (itsDiveSetting > 1.0F) itsDiveSetting = 1.0F;

      // adjust the ballasts: itsDiveSetting provides the mean, while
      // itsPitchSetting provides the deviation:
      float f = itsDiveSetting - itsPitchSetting;
      float r = itsDiveSetting + itsPitchSetting;
      if (f < 0.0F) f = 0.0F; else if (f > 1.0F) f = 1.0F;
      if (r < 0.0F) r = 0.0F; else if (r > 1.0F) r = 1.0F;
      setBallasts(f, r);

      // remember when we did all this:
      itsLastDepthTime = t;

      LINFO("dcmd = %f at t=%f", dcmd, t);
    }

  pthread_mutex_unlock(&itsLock);
}

// ######################################################################
void BeoSubOneBal::thrust(const float leftval, const float rightval)
{
        // note: if the value is out of range, the BeoChip will clamp it.
        // That's why we read it back from the BeoChip before displaying it:
        itsBeo->setServo(itsLeftThrusterServoNum.getVal(), leftval);
        itsThrustLeft = itsBeo->getServo(itsLeftThrusterServoNum.getVal());

        itsBeo->setServo(itsRightThrusterServoNum.getVal(), rightval);
        itsThrustRight = itsBeo->getServo(itsRightThrusterServoNum.getVal());

        //LINFO("Th=%-1.2f/%-1.2f", itsThrustLeft, itsThrustRight);
}

// ######################################################################
void BeoSubOneBal::getThrusters(float& leftval, float& rightval) const
{ leftval = itsThrustLeft; rightval = itsThrustRight; }

// ######################################################################
void BeoSubOneBal::setFrontBallast(const float val, const bool blocking)
{ itsFballast->set(val, blocking); }

// ######################################################################
void BeoSubOneBal::setRearBallast(const float val, const bool blocking)
{ itsRballast->set(val, blocking); }

// ######################################################################
void BeoSubOneBal::setBallasts(const float f, const float r,
                               const bool blocking)
{
  // move all ballasts simultaneously rather than sequentially:
  itsFballast->set(f, false);
  itsRballast->set(r, false);

  if (blocking)
    {
      const double timeout = itsMasterClock.getSecs() + 20.0;
      while(itsMasterClock.getSecs() < timeout)
        {
          if (itsFballast->moving() == false &&
              itsRballast->moving() == false) break;
          usleep(100000);
        }
      if (itsMasterClock.getSecs() >= timeout)
        LERROR("Timeout on blocking setBallasts -- IGNORED");
    }
}

// ######################################################################
float BeoSubOneBal::getFrontBallast() const
{ return itsFballast->get(); }

// ######################################################################
float BeoSubOneBal::getRearBallast() const
{ return itsRballast->get(); }

// ######################################################################
void BeoSubOneBal::getBallasts(float& f, float& r) const
{ f = itsFballast->get(); r = itsRballast->get(); }

// ######################################################################
void BeoSubOneBal::dropMarker(const bool blocking)
{
#ifndef HAVE_LINUX_PARPORT_H
  LFATAL("<linux/parport.h> must be installed to use this function");
#else
  markerdropper.Step(150, 65000);
#endif
}

// ######################################################################
void BeoSubOneBal::killSwitch()
{
  //Turn off all PID's
  useDepthPID(false);
  useHeadingPID(false);
  usePitchPID(false);
  useRotVelPID(false);
  setRotVel(0.0);
  setTransVel(0.0);

  //empty both balasts and  surface!
  setBallasts(0.0F, 0.0F, true);

  LFATAL("Kill switch pulled!");
}

// ######################################################################
void BeoSubOneBal::dispatchBeoChipEvent(const BeoChipEventType t,
                                  const int valint,
                                  const float valfloat)
{
  // what is this event about and whom is it for?
  switch(t)
    {
    case NONE: // ##############################
      LERROR("Unexpected BeoChip NONE event!");
      break;

    case PWM0: // ##############################
      LERROR("Unexpected BeoChip PWM0 event!");
      break;

    case PWM1: // ##############################
      LERROR("Unexpected BeoChip PWM1 event!");
      break;

    case KBD: // ##############################
      LDEBUG("Keyboard: new value %d", valint);
      // current diagram is:
      // #0 = front full/empty switch
      // #1 = front gear encoder
      // #2 = rear full/empty switch
      // #3 = rear gear encoder
      // #4 = kill switch

      // let both ballast units know:
      itsFballast->input(valint);
      itsRballast->input(valint);

      //check if kill switch is activated
      //disabled due to kill-switch/start-switch multi-use - khooyp
      //if (valint & 0x10 != 0 && itsKillSwitchUsed) killSwitch();
      break;

    case ADC0: // ##############################
      {
        //LINFO("ADC0 new value = %d", valint);
        // the ADC0 is connected to the pressure sensor. Convert to
        // meters, with positive going down (so the deeper we are, the
        // larger the positive reading). Then update the depth reading:

        // A reading of 255 corresponds to 2.5V, given our 2.5V
        // reference voltage on the BeoChip's ADC lines:
        float depth = (float(valint) * 2.5F) / 255.0F;  // in volts

        // I here assume that the sensor is an ASDX100A24R: NOTE: a
        // 220nF cap is required on the supply line.

        // the sensor has a 0.5V offset (i.e., outputs 0.5V when
        // pressure is 0):
        depth -= 0.5F;

        // An ASDX100A24R has a resolution of 0.040V/psi:
        depth /= 0.040F; // now in psi

        // convert from PSI to meters: at the surface, we have 14.7
        // psi (= 1 atmosphere); then it's about 0.44 psi/foot, i.e.,
        // 1.443 psi/m:
        depth = (depth - 14.275F) / 1.443F;

        // NOTE: convention in the rest of the code is positive depth
        // the more we go down:
        updateDepth(depth);
      }
      break;

    case ADC1: // ##############################
      LERROR("Unexpected BeoChip ADC1 event!");
      break;

    case RESET: // ##############################
      LERROR("BeoChip RESET!");
      break;

    case ECHOREP: // ##############################
      LINFO("BeoChip Echo reply.");
      break;

    case INOVERFLOW: // ##############################
      LERROR("BeoChip input overflow!");
      break;

    case SERIALERROR: // ##############################
      LERROR("BeoChip serial error!");
      break;

    case OUTOVERFLOW: // ##############################
      LERROR("BeoChip output overflow!");
      break;

    default: // ##############################
      LERROR("BeoChip unknown event %d!", int(t));
      break;
    }

    /* in addition, we also poll the kill switch on the parallel port */
    if(isKilled()) {
      killSwitchDebounceCounter++;
      if(killSwitchDebounceCounter > 50) {
        killSwitch();
      }
      //LINFO("killSwitchDebounce at %i\n", killSwitchDebounceCounter);
    } else {
      if(killSwitchDebounceCounter > 0) {
        LINFO("killSwitchDebounce reset\n");
      }
      killSwitchDebounceCounter = 0;
    }
}

// ######################################################################
Image< PixRGB<byte> >
BeoSubOneBal::grabImage(const enum BeoSubCamera cam) const
{
  if (cam == BEOSUBCAMFRONT) return itsCameraFront->readRGB();
  if (cam == BEOSUBCAMDOWN) return itsCameraDown->readRGB();
  if (cam == BEOSUBCAMUP) return itsCameraUp->readRGB();
  LERROR("Wrong camera %d -- RETURNING EMPTY IMAGE", int(cam));
  return Image< PixRGB<byte> >();
}

// ######################################################################
void BeoSubOneBal::useDepthPID(const bool useit)
{ itsDepthPIDon = useit; }

// ######################################################################
void BeoSubOneBal::useHeadingPID(const bool useit)
{ itsHeadingPIDon = useit; }

// ######################################################################
void BeoSubOneBal::usePitchPID(const bool useit)
{ itsPitchPIDon = useit; }

void BeoSubOneBal::useRotVelPID(const bool useit)
{ itsRotVelPIDon = useit; }

// ######################################################################
void BeoSubOneBal::useKillSwitch(const bool useit)
{ itsKillSwitchUsed = useit; }

// ######################################################################
bool BeoSubOneBal::isKilled()
{
#ifndef HAVE_LINUX_PARPORT_H
  LFATAL("<linux/parport.h> must be installed to use this function");
  return true; // can't happen, but placate compiler
#else
        unsigned char status;
        status = lp0->ReadStatus();
        return (status & PARPORT_STATUS_BUSY) || !(status & PARPORT_STATUS_ACK);
#endif
}

// ######################################################################
void BeoSubOneBal::setDepthPgain(float p)
{ itsDepthPID.setPIDPgain(p); }

// ######################################################################
void BeoSubOneBal::setDepthIgain(float i)
{ itsDepthPID.setPIDIgain(i); }

// ######################################################################
void BeoSubOneBal::setDepthDgain(float d)
{ itsDepthPID.setPIDDgain(d); }

// ######################################################################
void BeoSubOneBal::setHeadingPgain(float p)
{ itsHeadingPID.setPIDPgain(p); }

// ######################################################################
void BeoSubOneBal::setHeadingIgain(float i)
{ itsHeadingPID.setPIDIgain(i); }

// ######################################################################
void BeoSubOneBal::setHeadingDgain(float d)
{ itsHeadingPID.setPIDDgain(d); }

// ######################################################################
void BeoSubOneBal::setPitchPgain(float p)
{ itsPitchPID.setPIDPgain(p); }

// ######################################################################
void BeoSubOneBal::setPitchIgain(float i)
{ itsPitchPID.setPIDIgain(i); }

// ######################################################################
void BeoSubOneBal::setPitchDgain(float d)
{ itsPitchPID.setPIDDgain(d); }

// ######################################################################
void BeoSubOneBal::setRotVelPgain(float p)
{ itsRotVelPID.setPIDPgain(p); }

// ######################################################################
void BeoSubOneBal::setRotVelIgain(float i)
{ itsRotVelPID.setPIDIgain(i); }

// ######################################################################
void BeoSubOneBal::setRotVelDgain(float d)
{ itsRotVelPID.setPIDDgain(d); }
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
