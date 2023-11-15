/*!@file BeoSub/BeoSubTwoBal.C An autonomous submarine with two ballasts*/

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubTwoBal.C $
// $Id: BeoSubTwoBal.C 7880 2007-02-09 02:34:07Z itti $
//

#include "BeoSub/BeoSubTwoBal.H"
#include "Devices/HMR3300.H"
#include "BeoSub/BeoSubBallast.H"
#include "BeoSub/BeoSubIMU.H"
#include "Util/MathFunctions.H"
#include "Devices/IEEE1394grabber.H"
#include "rutz/compat_snprintf.h"

// ######################################################################
//! Class definition for BeoSubListener
/*! This is the listener class that is attached to each BeoChip in the
  left and right ballast tubes of the submarins. This class is just a
  pass-through to the function dispatchBeoChipEvent() of class
  BeoSub. */
class BeoSubListener : public BeoChipListener
{
public:
  //! Constructor
  BeoSubListener(const BeoSubSide side, BeoSub *sub) :
    itsSide(side), itsBeoSub(sub)
  { }

  //! destructor
  virtual ~BeoSubListener()
  { }

  //! get an event
  virtual void event(const BeoChipEventType t, const int valint,
                     const float valfloat)
  {
    LDEBUG("Event: %d val = %d, fval = %f", int(t), valint, valfloat);
    itsBeoSub->dispatchBeoChipEvent(itsSide, t, valint, valfloat);
  }

private:
  const BeoSubSide itsSide; //!< which side of the sub are we on?
  BeoSubTwoBal *itsBeoSub;        //!< pointer to our master
};



// ######################################################################
BeoSubTwoBal::BeoSubTwoBal(OptionManager& mgr) :
  BeoSub(mgr),
  ModelComponent(mgr, descrName, tagName),
  itsLeftThrusterServoNum("BeoSubLeftThrusterServoNum", this, 2),
  itsRightThrusterServoNum("BeoSubRightThrusterServoNum", this, 2),
  itsHMR3300(new HMR3300(mgr)),
  itsBeoLeft(new BeoChip(mgr, "BeoChipLeft", "BeoChipLeft")),
  itsBeoRight(new BeoChip(mgr, "BeoChipRight", "BeoChipRight")),
  itsBeoLisLeft(new BeoSubListener(BEOSUBLEFT, this)),
  itsBeoLisRight(new BeoSubListener(BEOSUBRIGHT, this)),
  itsDepthSensor(),
  itsLFballast(new BeoSubBallast(mgr, "LF Ballast", "LFballast")),
  itsLRballast(new BeoSubBallast(mgr, "LR Ballast", "LRballast")),
  itsRFballast(new BeoSubBallast(mgr, "RF Ballast", "RFballast")),
  itsRRballast(new BeoSubBallast(mgr, "RR Ballast", "RRballast")),
  itsIMU(new BeoSubIMU(mgr)),
  itsCameraFront(new IEEE1394grabber(mgr, "Front Camera", "FrontCamera")),
  itsCameraDown(new IEEE1394grabber(mgr, "Down Camera", "DownCamera")),
  itsCameraUp(new IEEE1394grabber(mgr, "Up Camera", "UpCamera"))
{
  // register our babies as subcomponents:
  addSubComponent(itsHMR3300);
  addSubComponent(itsIMU);
  addSubComponent(itsBeoLeft);
  addSubComponent(itsBeoRight);
  addSubComponent(itsCameraDown);
  addSubComponent(itsCameraFront);
  addSubComponent(itsCameraUp);
  addSubComponent(itsLFballast);
  addSubComponent(itsLRballast);
  addSubComponent(itsRFballast);
  addSubComponent(itsRRballast);

  // connect our listeners to our beochips:
  rutz::shared_ptr<BeoChipListener> bl = itsBeoLisLeft, br = itsBeoLisRight;
  itsBeoLeft->setListener(bl);
  itsBeoRight->setListener(br);

  // hook up BeoChips to Ballasts:
  itsLFballast->setBeoChip(itsBeoLeft);
  itsLRballast->setBeoChip(itsBeoLeft);
  itsRFballast->setBeoChip(itsBeoRight);
  itsRRballast->setBeoChip(itsBeoRight);

  // disable RTS/CTS flow control on our BeoChips:
  itsBeoLeft->setModelParamVal("BeoChipUseRTSCTS", false);
  itsBeoRight->setModelParamVal("BeoChipUseRTSCTS", false);

  // select default serial ports for everybody:
  itsHMR3300->setModelParamString("HMR3300SerialPortDevName",
                                  "/dev/ttyS0", MC_RECURSE);
  itsBeoLeft->setModelParamString("BeoChipDeviceName", "/dev/ttyS2");
  itsBeoRight->setModelParamString("BeoChipDeviceName", "/dev/ttyS1");
  itsIMU->setModelParamString("IMUSerialPortDevName", "/dev/ttyS3",
                              MC_RECURSE);

  // set the I/O for our ballasts:
  itsLRballast->setModelParamVal("LRballastOutRed", 1);
  itsLRballast->setModelParamVal("LRballastOutWhite", 0);
  itsLRballast->setModelParamVal("LRballastInYellow", 1);
  itsLRballast->setModelParamVal("LRballastInWhite", 0);
  itsLFballast->setModelParamVal("LFballastOutRed", 3);
  itsLFballast->setModelParamVal("LFballastOutWhite", 2);
  itsLFballast->setModelParamVal("LFballastInYellow", 3);
  itsLFballast->setModelParamVal("LFballastInWhite", 2);
  itsRRballast->setModelParamVal("RRballastOutRed", 1);
  itsRRballast->setModelParamVal("RRballastOutWhite", 0);
  itsRRballast->setModelParamVal("RRballastInYellow", 1);
  itsRRballast->setModelParamVal("RRballastInWhite", 0);
  itsRFballast->setModelParamVal("RFballastOutRed", 3);
  itsRFballast->setModelParamVal("RFballastOutWhite", 2);
  itsRFballast->setModelParamVal("RFballastInYellow", 3);
  itsRFballast->setModelParamVal("RFballastInWhite", 2);
}

// ######################################################################
BeoSubTwoBal::~BeoSubTwoBal()
{  }

// ######################################################################
void BeoSubTwoBal::start1()
{
  // select firewire subchannels for our cameras:
  itsCameraDown->setModelParamVal("FrameGrabberSubChan", int(BEOSUBCAMDOWN));
  itsCameraFront->setModelParamVal("FrameGrabberSubChan", int(BEOSUBCAMFRONT));
  itsCameraUp->setModelParamVal("FrameGrabberSubChan", int(BEOSUBCAMUP));
}

// ######################################################################
void BeoSubTwoBal::start2()
{
  // turn on the analog port capture for the pressure sensor:
  itsBeoLeft -> captureAnalog(0,true);

  // turn on KBD capture for our ballasts:
  itsBeoLeft->captureKeyboard(true);
  itsBeoRight->captureKeyboard(true);

  // initialize all 4 ballasts at once:
  CLINFO("filling ballasts...");
  setBallasts(1.0F, 1.0F, 1.0F, 1.0F, false); sleep(4);
  CLINFO("emptying ballasts...");
  setBallasts(0.0F, 0.0F, 0.0F, 0.0F, true);
}

// ######################################################################
void BeoSubTwoBal::thrust(const float leftval, const float rightval, const bool blocking)
{
  // note: if the value is out of range, the BeoChip will clamp it.
  // That's why we read it back from the BeoChip before displaying it:
  itsBeoLeft->setServo(itsLeftThrusterServoNum.getVal(), leftval);
  itsThrustLeft = itsBeoLeft->getServo(itsLeftThrusterServoNum.getVal());
  itsBeoLeft->lcdPrintf(0, 0, "Th=%-1.2f", itsThrustLeft);

  itsBeoRight->setServo(itsRightThrusterServoNum.getVal(), rightval);
  itsThrustRight = itsBeoRight->getServo(itsRightThrusterServoNum.getVal());
  itsBeoRight->lcdPrintf(0, 0, "Th=%-1.2f", itsThrustRight);
}

// ######################################################################
void BeoSubTwoBal::getThrusters(float& leftval, float& rightval)
{ leftval = itsThrustLeft; rightval = itsThrustRight; }

// ######################################################################
void BeoSubTwoBal::setLFballast(const float val, const bool blocking)
{ itsLFballast->set(val, blocking); }

// ######################################################################
void BeoSubTwoBal::setLRballast(const float val, const bool blocking)
{ itsLRballast->set(val, blocking); }

// ######################################################################
void BeoSubTwoBal::setRFballast(const float val, const bool blocking)
{ itsRFballast->set(val, blocking); }

// ######################################################################
void BeoSubTwoBal::setRRballast(const float val, const bool blocking)
{ itsRRballast->set(val, blocking); }

// ######################################################################
void BeoSubTwoBal::setBallasts(const float lf, const float lr,
                         const float rf, const float rr, const bool blocking)
{
  // move all four ballasts simultaneously rather than sequentially:
  itsLFballast->set(lf, false);
  itsLRballast->set(lr, false);
  itsRFballast->set(rf, false);
  itsRRballast->set(rr, false);

  if (blocking)
    {
      while(true)
        {
          if (itsLFballast->moving() == false &&
              itsLRballast->moving() == false &&
              itsRFballast->moving() == false &&
              itsRRballast->moving() == false) break;
          usleep(100000);
        }
    }
}

// ######################################################################
float BeoSub::getLFballast()
{ return itsLFballast->get(); }

// ######################################################################
float BeoSub::getLRballast()
{ return itsLRballast->get(); }

// ######################################################################
float BeoSub::getRFballast()
{ return itsRFballast->get(); }

// ######################################################################
float BeoSub::getRRballast()
{ return itsRRballast->get(); }

// ######################################################################
void BeoSub::getBallasts(float& lf, float& lr, float& rf, float& rr)
{
  lf = itsLFballast->get();
  lr = itsLRballast->get();
  rf = itsRFballast->get();
  rr = itsRRballast->get();
}

// ######################################################################
void BeoSub::dropMarker(const bool blocking)
{
  LFATAL("unimplemented!");
}

/*
// ######################################################################
void BeoSub::checkpoint(const char *fmt, ...)
{
  va_list a; va_start(a, fmt); char buf[2048];
  vsnprintf(buf, 2047, fmt, a); va_end(a);
  LDEBUG("===== START CHECKPOINT %d =====", itsCkPt);
  LDEBUG(buf);
  LDEBUG("depth = %f", getDepth());
  LDEBUG("compass = [ %f, %f, %f ]", itsHMR3300->getHeading().getVal(),
         itsHMR3300->getPitch().getVal(), itsHMR3300->getRoll().getVal());
  LDEBUG("===== END CHECKPOINT %d =====", itsCkPt);
  itsCkPt ++;
}
*/

// ######################################################################
void BeoSub::dispatchBeoChipEvent(const BeoSubSide side,
                                  const BeoChipEventType t,
                                  const int valint,
                                  const float valfloat)
{
  // what is this event about and whom is it for?
  switch(t)
    {
    case NONE: // ##############################
      break;

    case PWM0: // ##############################
      /* nobody cares */
      break;

    case PWM1: // ##############################
      /* nobody cares */
      break;

    case KBD: // ##############################
      LDEBUG("%s Keyboard: new value %d", beoSubSideName(side), valint);
      // print keyboard values:
      char kb[6]; kb[5] = '\0';
      for (int i = 0; i < 5; i ++) kb[i] = (valint>>(4-i))&1 ? '1':'0';

      switch(side)
        {
        case BEOSUBLEFT:
          // current diagram is:
          // #0 = front full/empty switch
          // #1 = front gear encoder
          // #2 = rear full/empty switch
          // #3 = rear gear encoder
          // #4 = kill switch

          // let both left ballasts know:
          itsLFballast->input(valint);
          itsLRballast->input(valint);

          itsBeoLeft->lcdPrintf(15, 3, kb);
          break;
        case BEOSUBRIGHT:
          // current diagram is:
          // #0 = front full/empty switch
          // #1 = front gear encoder
          // #2 = rear full/empty switch
          // #3 = rear gear encoder
          // #4 = some other switch

          // let both right ballasts know:
          itsRFballast->input(valint);
          itsRRballast->input(valint);

          itsBeoRight->lcdPrintf(15, 3, kb);
          break;
        }
      break;

    case ADC0: // ##############################
      LDEBUG("%s ADC0 new value = %d", beoSubSideName(side), valint);
      // the left ADC0 is connected to the pressure sensor:
      if (side == BEOSUBLEFT)
        {
          itsDepthSensor.newMeasurement(valfloat);
          itsBeoLeft->lcdPrintf(8, 0, "Pr=%03d", valint);
        }
      else
        LERROR("Getting values from right ADC0 -- turn it off!");
      break;

    case ADC1: // ##############################
      /* anybody? */
      break;

    case RESET: // ##############################
      LERROR("%s BeoChip RESET!", beoSubSideName(side));
      break;

    case ECHOREP: // ##############################
      LINFO("%s BeoChip Echo reply.", beoSubSideName(side));
      break;

    case INOVERFLOW: // ##############################
      LERROR("%s BeoChip input overflow!", beoSubSideName(side));
      break;

    case SERIALERROR: // ##############################
      LERROR("%s BeoChip serial error!", beoSubSideName(side));
      break;

    case OUTOVERFLOW: // ##############################
      LERROR("%s BeoChip output overflow!", beoSubSideName(side));
      break;

    default: // ##############################
      LERROR("%s BeoChip unknown event %d!", beoSubSideName(side), int(t));
      break;
    }
}

// ######################################################################
Image< PixRGB<byte> > BeoSub::grabImage(const enum BeoSubCamera cam)
{
  if (cam == BEOSUBCAMFRONT) return itsCameraFront->readRGB();
  if (cam == BEOSUBCAMDOWN) return itsCameraDown->readRGB();
  if (cam == BEOSUBCAMUP) return itsCameraUp->readRGB();
  LERROR("Wrong camera %d -- RETURNING EMPTY IMAGE", int(cam));
  return Image< PixRGB<byte> >();
}

// ######################################################################
const char* beoSubSideName(const BeoSubSide s)
{
  if (s == BEOSUBLEFT) return "Left";
  if (s == BEOSUBRIGHT) return "Right";
  LERROR("Unknown BeoSubSide value %d", int(s));
  return "Unknown";
}

// ######################################################################
const char* beoSubCameraName(const BeoSubCamera cam)
{
  if (cam == BEOSUBCAMFRONT) return "Front";
  if (cam == BEOSUBCAMDOWN) return "Down";
  if (cam == BEOSUBCAMUP) return "Up";
  LERROR("Unknown BeoSubCamera value %d", int(cam));
  return "Unknown";
}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
