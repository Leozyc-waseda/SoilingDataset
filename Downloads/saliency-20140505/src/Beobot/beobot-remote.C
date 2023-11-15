/*!@file Beobot/beobot-remote.C Drive Beobot with a remote control      */
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/beobot-remote.C $
// $Id: beobot-remote.C 7277 2006-10-18 22:55:24Z beobot $
//

#include "Component/ModelManager.H"
#include "Devices/BeoChip.H"
#include "Devices/DeviceOpts.H"
#include "Util/MathFunctions.H"

#include <cstdlib>

//! Our own little BeoChipListener
class MyBeoChipListener : public BeoChipListener
{
public:
  MyBeoChipListener(nub::soft_ref<BeoChip> bc) :
    itsBeoChip(bc), minp0(9999), maxp0(0), minp1(9999), maxp1(0),
    counter0(0), counter1(0) { }

  virtual ~MyBeoChipListener() { }

  virtual void event(const BeoChipEventType t, const int valint,
                     const float valfloat)
  {
    //LINFO("Event: %d val = %d, fval = %f", int(t), valint, valfloat);
    switch(t)
      {
      case PWM0:
        if (valint < minp0) minp0 = valint;
        else if (valint > maxp0) maxp0 = valint;
        itsBeoChip->setServo(0, valfloat);
        if (++counter0 >= 10)
          {
            itsBeoChip->lcdPrintf(5, 2, "%04d  %04d-%04d",
                                  valint, minp0, maxp0);
            itsBeoChip->lcdPrintf(6, 1, "%03d",
                                  itsBeoChip->getServoRaw(0));
            counter0 = 0;
          }
        break;
      case PWM1:
        if (valint < minp1) minp1 = valint;
        else if (valint > maxp1) maxp1 = valint;
        itsBeoChip->setServo(1, valfloat);
        if (++counter1 >= 10)
          {
            itsBeoChip->lcdPrintf(5, 3, "%04d  %04d-%04d",
                                  valint, minp1, maxp1);
            itsBeoChip->lcdPrintf(17, 1, "%03d",
                                  itsBeoChip->getServoRaw(1));
            counter1 = 0;
          }
        break;
      case RESET: LERROR("BeoChip RESET occurred!"); break;
      case ECHOREP: LINFO("BeoChip Echo reply received."); break;
      case INOVERFLOW: LERROR("BeoChip input overflow!"); break;
      case SERIALERROR: LERROR("BeoChip serial error!"); break;
      case OUTOVERFLOW: LERROR("BeoChip output overflow!"); break;
      default: LERROR("Unknown event %d received!", int(t)); break;
      }
  }

  nub::soft_ref<BeoChip> itsBeoChip;
  int minp0, maxp0, minp1, maxp1;
  int counter0, counter1;
};

//! Drive the Beobot under remote control
/*! This simple test program demonstrate how to capture PWM signals
  from the Beochip, which on the Beobot correspond to steering and
  speed inputs from the remote control. The captured signals are then
  directly fed into the Beobot's steering and speed servos controlled
  by the BeoChip. */
int main(const int argc, const char* argv[])
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("Beobot remote");

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeoChip> b(new BeoChip(manager));
  manager.addSubComponent(b);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<serdev>", 1, 1) == false)
    return(1);

  // let's configure our serial device:
  b->setModelParamVal("BeoChipDeviceName", manager.getExtraArg(0));

  // let's register our listener:
  rutz::shared_ptr<BeoChipListener> lis(new MyBeoChipListener(b));
  b->setListener(lis);

  // let's get all our ModelComponent instances started:
  manager.start();

  // reset the beochip:
  LINFO("Resetting BeoChip...");
  b->resetChip(); sleep(1);

  // keep the gear at the lowest speed/highest torque
  b->setServoRaw(2, 0);

  // calibrate the PWMs:
  b->calibratePulse(0, 934, 665, 1280);
  b->calibratePulse(1, 865, 590, 1320);
  b->capturePulse(0, true);
  b->capturePulse(1, true);

  // let's play with the LCD:
  b->lcdClear();   // 01234567890123456789
  b->lcdPrintf(0, 0, "remote-calib -- 1.00");
  b->lcdPrintf(0, 1, "STEER=XXX  SPEED=XXX");
  b->lcdPrintf(0, 2, "PWM0=0000  0000-0000");
  b->lcdPrintf(0, 3, "PWM1=0000  0000-0000");

  while(true) sleep(1);

  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
