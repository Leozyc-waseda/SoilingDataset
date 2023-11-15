/*!@file BeoSub/SeaBee.C An autonomous submarine with one ballast*/

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/SeaBee.C $
// $Id: SeaBee.C 8521 2007-06-28 17:45:49Z rjpeters $
//

#include "BeoSub/SeaBee.H"
#include "Devices/HMR3300.H"
//#include "BeoSub/BeoSubBallast.H"
#include "BeoSub/BeoSubIMU.H"
#include "Util/MathFunctions.H"
#include "Devices/FrameGrabberFactory.H"

// ######################################################################
//! Class definition for BeoSubListener
/*! This is the listener class that is attached to each BeoChip in the
  ballast tube of the submarine. This class is just a
  pass-through to the function dispatchBeoChipEvent() of class
  SeaBee. */
class BeoSubListener : public BeoChipListener
{
public:
  //! Constructor
  BeoSubListener(SeaBee *sub) : itsBeoSub(sub)  { }

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
  SeaBee *itsBeoSub;        //!< pointer to our master
};




// ######################################################################
SeaBee::SeaBee(OptionManager& mgr) :
  BeoSub(mgr),
  //  itsIMU(new BeoSubIMU(mgr)),
  // itsLeftThrusterServoNum("BeoSubLeftThrusterServoNum", this, 3),
  //itsRightThrusterServoNum("BeoSubRightThrusterServoNum", this, 2),
  //itsHMR3300(new HMR3300(mgr)),
  itsBeo(new BeoChip(mgr, "BeoChip", "BeoChip"))
//  itsFballast(new BeoSubBallast(mgr, "Front Ballast", "FrontBallast")),
//  itsRballast(new BeoSubBallast(mgr, "Rear Ballast", "RearBallast")),
//  itsCameraFront(makeIEEE1394grabber(mgr, "Front Camera", "FrontCamera")),
//  itsCameraDown(makeIEEE1394grabber(mgr, "Down Camera", "DownCamera")),
//  itsCameraUp(makeIEEE1394grabber(mgr, "Up Camera", "UpCamera")),
//   itsDepthSensor(30, 0.99999F),
//   itsHeadingSensor(3, 0.99999F),
//   itsPitchSensor(3, 0.99999F),
//   itsDepthPID(0.001F, 0.0F, 0.1F, 3.0F, 3.0F),
//   itsHeadingPID(0.002F, 0.0F, 0.03F, Angle(170.0), Angle(170.0)),
//   itsPitchPID(0.001F, 0.0F, 0.1F, Angle(170.0), Angle(170.0)),
//   itsRotVelPID(0.3, 0.0001, 0, -500, 500),
//   itsDepthPIDon(false),
//   itsHeadingPIDon(false),
//   itsPitchPIDon(false),
//   itsRotVelPIDon(false),
//   itsKillSwitchUsed(false),
//   itsDiveSetting(0.0F),
//   itsPitchSetting(0.0F),
//   itsLastHeadingTime(0.0),
//   itsLastPitchTime(0.0),
//   itsLastDepthTime(0.0),
// #ifdef HAVE_LINUX_PARPORT_H
//   lp0(new ParPort(mgr)),
//   markerdropper(lp0),
// #endif
//   killSwitchDebounceCounter(0)

//   itsLeftHThrusterServoNum("BeoSubLeftHThrusterServoNum", this, 4),
//   itsRightHThrusterServoNum("BeoSubRightHThrusterServoNum", this, 5),
//   itsBeo(new BeoChip(mgr, "BeoChip", "BeoChip")),
//   itsDiveSetting(0.0F)
{


        //GRAB image from camera to be tested
                //gb->setModelParamVal("FrameGrabberFPS", 15.0F);


// itsCameraFront->setModelParamVal("FrameGrabberSubChan", 0);
// itsCameraFront->setModelParamVal("FrameGrabberBrightness", 128);
// itsCameraFront->setModelParamVal("FrameGrabberHue", 180);
// itsCameraFront->setModelParamVal("FrameGrabberNbuf", 2);

/*
itsCameraDown->setModelParamVal("FrameGrabberSubChan", 0);
itsCameraDown->setModelParamVal("FrameGrabberBrightness", 128);
itsCameraDown->setModelParamVal("FrameGrabberHue", 180);
itsCameraDown->setModelParamVal("FrameGrabberNbuf", 2);

itsCameraUp->setModelParamVal("FrameGrabberSubChan", 0);
itsCameraUp->setModelParamVal("FrameGrabberBrightness", 128);
itsCameraUp->setModelParamVal("FrameGrabberHue", 180);
itsCameraUp->setModelParamVal("FrameGrabberNbuf", 2);
*/

  // register our babies as subcomponents:

//   addSubComponent(itsBeo);

//   // reset the BeoChip:
//   LINFO("Resetting the BeoChip...");
//   itsBeo->reset(MC_RECURSE);
//   usleep(200000);

//   // connect our listener to our beochip:
//   rutz::shared_ptr<BeoChipListener> b(new BeoSubListener(this));
//   itsBeo->setListener(b);

//   // connect our listener to our compass:
//   rutz::shared_ptr<HMR3300Listener> bl(new HMRlistener(this));
//   itsHMR3300->setListener(bl);

//   rutz::shared_ptr<BeoSubIMUListener> imu(new IMUListener(this));
//   itsIMU->setListener(imu);

//   // hook up BeoChips to Ballasts:
// //  itsFballast->setBeoChip(itsBeo);
// //  itsRballast->setBeoChip(itsBeo);

//   // disable RTS/CTS flow control on our BeoChips:
//   itsBeo->setModelParamVal("BeoChipUseRTSCTS", false);

//   // select default serial ports for everybody:
//   itsBeo->setModelParamString("BeoChipDeviceName", "/dev/ttyS1");

  //mgr.loadConfig("camconfig.pmap");
}

// ######################################################################
SeaBee::~SeaBee()
{  }

void SeaBee::test() {
for (int i=0; i < 5; i++) {
        printf("Running Servo %d\n", i);
        itsBeo->setServoRaw(i, 255);
        sleep(2);
        itsBeo->setServoRaw(i, 127);
        sleep(2);
        itsBeo->setServoRaw(i, 0);
        sleep(2);
        itsBeo->setServoRaw(i, 127);
}


Raster::waitForKey();

float x;
while(true) {

        x = itsBeo->getAnalog(0);
        printf("Pressure Sensor value: %f\n", x);
        }
}

// ######################################################################
void SeaBee::start1()
{
  // don't forget to call start1() on our base class!
  BeoSub::start1();
}

// ######################################################################
void SeaBee::start2()
{
  // turn on the analog port capture for the pressure sensor, do not
  // capture pulses:
  /* itsBeo->captureAnalog(0, true);
  itsBeo->captureAnalog(1, false);
  itsBeo->capturePulse(0, false);
  itsBeo->capturePulse(1, false);
  */

  // set all servos to neutral:
  for (int i = 0; i < 8; i ++) itsBeo->setServo(i, 0.0F);

  // turn on KBD capture for our ballasts:
  //  itsBeo->captureKeyboard(true);
  // itsBeo->debounceKeyboard(false);


  // turn the PIDs on: NOTE: first we should dive some...

  itsDiveSetting = 0.5F;

  // display sign of life:
  itsBeo->lcdClear();
  itsBeo->lcdPrintf(3, 0, " iLab and USC rock! ");

  // we currently don't have a start2() in our base class, but if we
  // add one we should call it here.
}

void SeaBee::stop1() {
        //Turn off all PID's

        setTransVel(0.0);

        //empty both balasts and  surface!
 //       setBallasts(0.0F, 0.0F, true);

}

// ######################################################################
void SeaBee::advanceRel(const float relDist, const bool stop)
{
  float d = relDist * 2.0F;
  // turn our heading PID off as it may interfere with our commands of
  // the thrusters:
  //  bool hpid = itsHeadingPIDon;
  // useHeadingPID(false);

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

  // useHeadingPID(hpid);
}

void SeaBee::updateThrusters() {
  float left = TransVelCommand;// + RotVelCommand;
  float right = TransVelCommand;// - RotVelCommand;

        if (left > 1) left = 1;
        else if (left < -1) left = -1;

        if (right > 1) right = 1;
        else if (right < -1) right = -1;

        thrust(left, right);
}


void SeaBee::setTransVel(const float desired) {
        //desiredTransVel = desired;
        TransVelCommand = desired; /* open loop for now */
        updateThrusters();
}


// ######################################################################
void SeaBee::updateDepth(const float depth)
{
  /*
  float left = TransVelCommand;// + RotVelCommand;
  float right = TransVelCommand;// - RotVelCommand;

        if (left > 1) left = 1;
        else if (left < -1) left = -1;

        if (right > 1) right = 1;
        else if (right < -1) right = -1;

        thrust(left, right);
  */


  // note the time at which the data was received:
  /*  double t = itsMasterClock.getSecs();

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
//      setBallasts(f, r);

      // remember when we did all this:
      itsLastDepthTime = t;

      LINFO("dcmd = %f at t=%f", dcmd, t);
    }

  pthread_mutex_unlock(&itsLock);
  */

}

// ######################################################################
void SeaBee::thrust(const float leftval, const float rightval)
{
        // note: if the value is out of range, the BeoChip will clamp it.
        // That's why we read it back from the BeoChip before displaying it:
//         itsBeo->setServo(itsLeftHThrusterServoNum.getVal(), leftval);
//         itsThrustLeftH = itsBeo->getServo(itsLeftHThrusterServoNum.getVal());

//         itsBeo->setServo(itsRightHThrusterServoNum.getVal(), rightval);
//         itsThrustRightH = itsBeo->getServo(itsRightHThrusterServoNum.getVal());

        //LINFO("Th=%-1.2f/%-1.2f", itsThrustLeft, itsThrustRight);
}

void SeaBee::dive(const float leftval, const float rightval)
{
        // note: if the value is out of range, the BeoChip will clamp it.
        // That's why we read it back from the BeoChip before displaying it:
//         itsBeo->setServo(itsLeftVThrusterServoNum.getVal(), leftval);
//         itsThrustLeftV = itsBeo->getServo(itsLeftVThrusterServoNum.getVal());

//         itsBeo->setServo(itsRightVThrusterServoNum.getVal(), rightval);
//         itsThrustRightV = itsBeo->getServo(itsRightVThrusterServoNum.getVal());

        //LINFO("Th=%-1.2f/%-1.2f", itsThrustLeft, itsThrustRight);
}

// ######################################################################
void SeaBee::getThrusters(float& leftval, float& rightval) const
{ leftval = itsThrustLeftH; rightval = itsThrustRightH; }



// ######################################################################
void SeaBee::getDiveThrusters(float& leftval, float& rightval) const
{ leftval = itsThrustLeftV; rightval = itsThrustRightV; }

// ######################################################################
void SeaBee::dispatchBeoChipEvent(const BeoChipEventType t,
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
  //    itsFballast->input(valint);
   //   itsRballast->input(valint);

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


}

void SeaBee::turnOpen(Angle ang, bool b) {


}

void SeaBee::dropMarker(bool b) {

}

Image<PixRGB <byte> > SeaBee::grabImage(BeoSubCamera bsc) const {
  return *(new Image<PixRGB <byte> >);

}
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
