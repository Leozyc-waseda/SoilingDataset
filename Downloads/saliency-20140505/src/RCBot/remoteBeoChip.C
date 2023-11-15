/*!@file RCBot/remoteBeoChip.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/remoteBeoChip.C $
// $Id: remoteBeoChip.C 7063 2006-08-29 18:26:55Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Devices/BeoChip.H"
#include "Devices/DeviceOpts.H"
#include "Util/MathFunctions.H"
#include "RCBot/BotControl.H"

#include <cstdlib>

#include "Beowulf/Beowulf.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <cstring>
#include <signal.h>
#include <time.h>
#include <unistd.h>

static bool goforever = true;
void terminate(int s) { LERROR("*** INTERRUPT ***"); goforever = false; exit(1);}

// ######################################################################
//! Our own little BeoChipListener
class MyBeoChipListener : public BeoChipListener
{
public:
  MyBeoChipListener(nub::soft_ref<BeoChip> bc) :
    itsBeoChip(bc), minp0(9999), maxp0(0), minp1(9999), maxp1(0),
    counter0(0), counter1(0), kbd(0x1f) { }

  virtual ~MyBeoChipListener() { }

  virtual void event(const BeoChipEventType t, const int valint,
                     const float valfloat)
  {
    LDEBUG("Event: %d val = %d, fval = %f", int(t), valint, valfloat);
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
      case KBD: kbd = valint;  break;
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
  int kbd;
};

// ######################################################################
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
  ModelManager manager("Remote BeoChip");

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeoChip> b(new BeoChip(manager));
  manager.addSubComponent(b);

  nub::soft_ref<Beowulf>
  beo(new Beowulf(manager, "Beowulf Slave", "BeowulfSlave", false));
  manager.addSubComponent(beo);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<serdev>", 1, 1) == false)
    return(1);

  // let's configure our serial device:
  b->setModelParamVal("BeoChipDeviceName", manager.getExtraArg(0));

  // let's register our listener:
  rutz::shared_ptr<MyBeoChipListener> lis(new MyBeoChipListener(b));
  rutz::shared_ptr<BeoChipListener> lis2; lis2.dynCastFrom(lis); // cast down
  b->setListener(lis2);

  // setup signal handling:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);

  TCPmessage rmsg;            // message being received and to process
  TCPmessage smsg;            // message being sent

  // let's get all our ModelComponent instances started:
  manager.start();

  // let's play with the LCD:
  b->lcdClear();   // 01234567890123456789
  b->lcdPrintf(0, 0, "collectFrames: 00000");
  b->lcdPrintf(0, 1, "STEER=XXX  SPEED=XXX");
  b->lcdPrintf(0, 2, "PWM0=0000  0000-0000");
  b->lcdPrintf(0, 3, "PWM1=0000  0000-0000");

  int32 rframe, raction, rnode = -1;  // receive from any node
  while(!beo->receive(rnode, rmsg, rframe, raction, 5));
  LINFO("size: %d", rmsg.getSize());
  rmsg.reset(rframe, raction);

  // wait for keyboard and process it:
  while(goforever) {

    // print keyboard values:
    char kb[6]; kb[5] = '\0';
    for (int i = 0; i < 5; i ++) kb[i] = (lis->kbd>>(4-i))&1 ? '1':'0';

    // quit if both extreme keys pressed simultaneously:
    if (kb[0] == '0' && kb[4] == '0') {
      b->lcdPrintf(15, 0, "QUIT ");
      goforever = false; break;
    }

    if (kb[1] == '0' || kb[3] == '0') {

      smsg.reset(0, 1);
      if (kb[1] == '0')
        {
          // start capturing frames
          smsg.addInt32(1);
          b->lcdPrintf(15, 0, "CAPTR");
          LINFO("start capturing");
        }
      else
        {
          // stop capturing frames
          smsg.addInt32(0);
           b->lcdPrintf(15, 0, "STOP ");
         LINFO("stop capturing");
        }
      beo->send(-1, smsg);
    }
    usleep(50000);

    // receive messages from other nodes
    int sNum, zero, min, max; float steering;
    if(beo->receive(rnode, rmsg, rframe, raction, 5))
      {
        switch(raction)
          {
          case RBEOCHIP_RESET:
            // reset the beochip:
            LINFO("Resetting BeoChip...");
            b->resetChip(); sleep(1);
            rmsg.reset(rframe, raction);

            // keep the gear at the lowest speed/highest torque
            b->setServoRaw(2, 0);

            // turn on the keyboard
            b->debounceKeyboard(true);
            b->captureKeyboard(true);
            break;
          case RBEOCHIP_CAL:
            sNum = int(rmsg.getElementInt32());
            zero = int(rmsg.getElementInt32());
            min = int(rmsg.getElementInt32());
            max = int(rmsg.getElementInt32());

            // calibrate the PWMs:
            LINFO("Calibrating BeoChip Servo: %d",sNum);
            b->calibratePulse(sNum, zero, min, max);
            b->capturePulse(sNum, true);
            break;

          case RBEOCHIP_SETSERVO:
            sNum = int(rmsg.getElementInt32());
            steering = float(rmsg.getElementFloat());
            LINFO("Setting BeoChip Servo: %d to %f",sNum, steering);
            b->setServo(sNum, steering);
            break;
          }
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
