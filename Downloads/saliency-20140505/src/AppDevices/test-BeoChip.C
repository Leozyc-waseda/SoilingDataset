/*!@file AppDevices/test-BeoChip.C test suite for Brian Hudson's BeoChip */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-BeoChip.C $
// $Id: test-BeoChip.C 13708 2010-07-28 16:55:36Z itti $
//

#include "Component/ModelManager.H"
#include "Devices/BeoChip.H"
#include "Devices/DeviceOpts.H"
#include "Util/MathFunctions.H"

#include <cstdlib>
#include <iostream>

//! Our own little BeoChipListener
class MyBeoChipListener : public BeoChipListener
{
public:
  MyBeoChipListener()
  { rawpwm[0] = 0; rawpwm[1] = 0; rawadc[0] = 0; rawadc[1] = 0; kbd = 0x1f; }

  virtual ~MyBeoChipListener()
  { }

  virtual void event(const BeoChipEventType t, const int valint,
                     const float valfloat)
  {
    LDEBUG("Event: %d val = %d, fval = %f", int(t), valint, valfloat);
    switch(t)
      {
      case NONE: break;
      case PWM0: rawpwm[0] = valint; break;
      case PWM1: rawpwm[1] = valint; break;
      case KBD: kbd = valint; break;
      case ADC0: rawadc[0] = valint; break;
      case ADC1: rawadc[1] = valint; break;
      case RESET: LERROR("BeoChip RESET occurred!"); break;
      case ECHOREP: LINFO("BeoChip Echo reply received."); break;
      case INOVERFLOW: LERROR("BeoChip input overflow!"); break;
      case SERIALERROR: LERROR("BeoChip serial error!"); break;
      case OUTOVERFLOW: LERROR("BeoChip output overflow!"); break;
      default: LERROR("Unknown event %d received!", int(t)); break;
      }
  }

  // note: ideally you would want to protect those by a mutex:
  volatile int rawpwm[2];
  volatile int rawadc[2];
  volatile int kbd;
};

//! This program provides basic interfacing to the BeoChip
/*! See the BeoChip class for details. Press both extreme keys
  together on the BeoChip keyboard to quit.  */
int main(const int argc, const char* argv[])
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("BeoChip test program");

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeoChip> b(new BeoChip(manager));
  manager.addSubComponent(b);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<serdev> [all|anim|font|"
                               "flood|rnd|esc|noflow|setServo] servo# servoPos", 1, 4) == false)
    return(1);

  // let's configure our serial device:
  b->setModelParamVal("BeoChipDeviceName", manager.getExtraArg(0));
  std::string task = ""; bool doall = false;
  if (manager.numExtraArgs() > 1)
    {
      task = manager.getExtraArg(1);
      doall = (task.compare("all") == 0);
    }

  // do we want to disable flow control?
  if (task.compare("noflow") == 0)
    {
      b->setModelParamVal("BeoChipUseRTSCTS", false);
      LINFO("Disabled RTS/CTS flow control");
    }

  // let's register our listener:

  rutz::shared_ptr<MyBeoChipListener> lis(new MyBeoChipListener);
  rutz::shared_ptr<BeoChipListener> lis2; lis2.dynCastFrom(lis); // cast down
  b->setListener(lis2);

  // let's get all our ModelComponent instances started:
  manager.start();

  LINFO("Waiting for a bit. Turn BeoChip on if not already on."); sleep(2);
  LINFO("Echo request (should bring an echo reply back)...");
  b->echoRequest(); sleep(1);

  // set servo
  if (task.compare("setServo") == 0)
  {
          if (manager.numExtraArgs() > 3)
          {
                  int servo = manager.getExtraArgAs<int>(2);
                  int val = manager.getExtraArgAs<int>(3);
                  LINFO("Setting servo %i to %i", servo, val);
                  b->setServoRaw(servo, val);
          } else {
                  LINFO("Need servo and position cmd");
          }
          manager.stop();
          exit(0);
  }

  // random junk test:
  if (task.compare("rnd") == 0 || doall)
    {
      LINFO("Starting random junk flood...");
      b->lcdClear();
      b->lcdPrintf(0, 1, "Random junk flood"); sleep(1);
      for (int i = 0; i < 5000; i ++)
        {
          int val = randomUpToIncluding(255);
          if (val != 0xF9) b->writeByte(val);
        }
      b->lcdClear();
      b->lcdPrintf(0, 1, "Random junk over"); sleep(1);
      sleep(30);
      b->lcdPrintf(0, 1, "Reset..."); sleep(1);
      LINFO("Reset BeoChip...");
      b->resetChip(); sleep(1);
    }

  // font test:
  if (task.compare("anim") == 0 || doall)
    {
      b->lcdClear();
      // NOTE: can't send a zero char since it's our end of string marker...
      b->lcdPrintf(0, 0, "%c%c%c%c%c%c%c%c    %c%c%c%c%c%c%c%c", 8, 1, 2, 3, 4,
                   5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
      b->lcdPrintf(0, 3, "%c%c%c%c%c%c%c%c    %c%c%c%c%c%c%c%c", 8, 1, 2, 3, 4,
                   5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
      for (int i = 0; i < 8; i ++)
        {
          LINFO("Loading font %d...", i);
          b->lcdPrintf(0, 1, "Loading LCD font %d ", i);
          b->lcdLoadFont(i);
          sleep(3);
        }
    }

  // animation test:
  if (task.compare("anim") == 0 || doall)
    {
      b->lcdClear(); b->lcdLoadFont(1);
      for (int i = 1; i < 8; i ++)
        {
          LINFO("Starting animation %d...", i);
          b->lcdClear();
          b->lcdPrintf(0, 1, "LCD Animation %d ", i);
          b->lcdSetAnimation(i);
          sleep(10);
        }
    }

  // serial flood test: we send 1000 commands that take more time to
  // execute than the interval at which they come in. The RTS/CTS flow
  // control will then be triggered:
  if (task.compare("flood") == 0 || doall)
    {
      LINFO("Trying an LCD clear flood...");
      b->lcdPrintf(0, 0, "LCD clear flood"); sleep(1);
      for (int i = 0; i < 1000; i ++) b->lcdClear();
      b->lcdPrintf(0, 0, "Clear flood is over."); sleep(5);

      LINFO("Trying an LCD font flood...");
      b->lcdPrintf(0, 0, "LCD font flood"); sleep(1);
      for (int i = 0; i < 1000; i ++) b->lcdLoadFont(i & 7);
      b->lcdPrintf(0, 0, "Font flood is over. "); sleep(15);

      LINFO("Trying an LCD font/clear flood...");
      b->lcdPrintf(0, 0, "LCD font/clear flood"); sleep(1);
      for (int i = 0; i < 1000; i ++) { b->lcdLoadFont(i & 7); b->lcdClear(); }
      b->lcdPrintf(0, 0, "Font/clr flood over "); sleep(15);
    }

  // ESC calibration:
  if (task.compare("esc") == 0 || doall)
    {
      LINFO("Setting all servos to neutral (127)");
      for (int i = 0; i < 8; i ++) b->setServoRaw(i, 127);
      LINFO("Press [RETURN] when ready for full-swing...");
      getchar();
      for (int j = 0; j < 10; j ++)
        {
          LINFO("Setting all servos to full forward (255)");
          for (int i = 0; i < 8; i ++) b->setServoRaw(i, 255);
          usleep(750000);
          LINFO("Setting all servos to full reverse (0)");
          for (int i = 0; i < 8; i ++) b->setServoRaw(i, 0);
          usleep(750000);
        }
      LINFO("Setting all servos to neutral (127)");
      for (int i = 0; i < 8; i ++) b->setServoRaw(i, 127);
      LINFO("Press [RETURN] when done...");
      getchar();
    }

  LINFO("Ok, have a look at the LCD screen now...");

  // load a cool custom font:
  b->lcdLoadFont(1);

  // let's turn everything on:
  b->capturePulse(0, true);
  b->capturePulse(1, true);
  b->captureAnalog(0, true);
  b->captureAnalog(1, true);
  b->debounceKeyboard(true);
  b->captureKeyboard(true);

  // let's play with the LCD:
  LINFO("Starting to play with the servos...");
  b->lcdClear();   // 01234567890123456789
  b->lcdPrintf(0, 0, "test-BeoChip -- 1.00");
  b->lcdPrintf(0, 1, "ADC=000/000 DOU=0000");
  b->lcdPrintf(0, 2, "PWM=0000/0000    X  ");
  b->lcdPrintf(0, 3, "Servos=000  KB=00000");

  // set servo 5 to 127, servo 6 to 0, and servo 7 to 255:
  b->setServoRaw(5, 127);
  b->setServoRaw(6, 0);
  b->setServoRaw(7, 255);

  // let's play with the servos and monitor a bunch of things:
  int ani = 0; int dout = 0; bool keepgoing = true;
  while(keepgoing)
    {
      for (int p = 0; p < 256; p ++)
        {
          // print ADC values:
          b->lcdPrintf(4, 1, "%03d/%03d", lis->rawadc[0], lis->rawadc[1]);

          // set and print dout values:
          char digout[5]; digout[4] = '\0';
          for (int i = 0; i < 4; i ++)
            {
              digout[i] = (dout>>(3-i))&1 ? '1':'0';
              b->setDigitalOut(3-i, (dout>>(3-i))&1?true:false);
            }
          b->lcdPrintf(16, 1, "%s", digout); dout ++; dout &= 15;

          // print PWM values:
          b->lcdPrintf(4, 2, "%04d/%04d", lis->rawpwm[0], lis->rawpwm[1]);

          // print keyboard values:
          char kb[6]; kb[5] = '\0';
          for (int i = 0; i < 5; i ++) kb[i] = (lis->kbd>>(4-i))&1 ? '1':'0';
          b->lcdPrintf(15, 3, "%s", kb);

          // quit if both extreme keys pressed simultaneously:
          if (kb[0] == '0' && kb[4] == '0') { keepgoing = false; break; }

          // print a little animated character, slow:
          if ((dout & 7) == 0)
            { b->lcdPrintf(17, 2, "%c", ani+8); ani ++; ani &= 7; }

          // set the servos:
          for (int s = 0; s < 5; s ++)
            if (s & 1) b->setServoRaw(s, p);
            else b->setServoRaw(s, 256 - p);

          // print current servo values:
          b->lcdPrintf(7, 3, "%03d", p);

          // make sure we leave some CPU to our listener:
          usleep(50000);
        }
    }

  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
