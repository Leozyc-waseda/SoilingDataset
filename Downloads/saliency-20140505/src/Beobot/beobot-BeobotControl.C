/*!@file Beobot/beobot-BeobotControl.C
  controls the Beobot motion input can be from both remote and Board A  */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/beobot-BeobotControl.C $
// $Id: beobot-BeobotControl.C 7063 2006-08-29 18:26:55Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Devices/BeoChip.H"
#include "Devices/DeviceOpts.H"
#include "Beobot/BeobotConfig.H"
#include "Beobot/BeobotControl.H"
#include "Util/MathFunctions.H"

#include <cstdlib>

#include "Beowulf/Beowulf.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <cstring>
#include <signal.h>
#include <time.h>
#include <unistd.h>

static bool goforever = true;
// ######################################################################
void terminate(int s) { LERROR("*** INTERRUPT ***"); goforever = false; exit(1);}

// ######################################################################
//! Our own little BeoChipListener
class MyBeoChipListener : public BeoChipListener
{
public:
  MyBeoChipListener(nub::soft_ref<BeoChip> bc) :
    itsBeoChip(bc), minp0(9999), maxp0(0), minp1(9999), maxp1(0),
    counter0(0), counter1(0), kbd(0x1f)
  {
    performCommand = false;
  }

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

        if(performCommand)
          {
            if (++counter0 >= 10)
              {
                itsBeoChip->lcdPrintf(5, 2, "%04d  %04d-%04d",
                                      valint, minp0, maxp0);
                itsBeoChip->lcdPrintf(6, 1, "%03d",
                                  itsBeoChip->getServoRaw(0));
                counter0 = 0;
              }
          }
        break;
      case PWM1:
        if (valint < minp1) minp1 = valint;
        else if (valint > maxp1) maxp1 = valint;

        if(performCommand)
          {
            itsBeoChip->setServo(1, valfloat);
            if (++counter1 >= 10)
              {
                itsBeoChip->lcdPrintf(5, 3, "%04d  %04d-%04d",
                                      valint, minp1, maxp1);
                itsBeoChip->lcdPrintf(17, 1, "%03d",
                                      itsBeoChip->getServoRaw(1));
                counter1 = 0;
              }
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
  bool performCommand;
};

// ######################################################################
//! Drive the Beobot under remote control and Board A
/*! This simple test program demonstrate how to capture PWM signals
  from the Beochip, which on the Beobot correspond to steering and
  speed inputs from the remote control. The captured signals are then
  directly fed into the Beobot's steering and speed servos controlled
  by BeobotControl class (which has the BeoChip). */
int main(const int argc, const char* argv[])
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("beobot-BeobotControl");

  // Instantiate our various ModelComponents:
  BeobotConfig bbc;
  nub::soft_ref<BeoChip> b(new BeoChip(manager));
  manager.addSubComponent(b);

  nub::soft_ref<BeobotControl> bc(new BeobotControl(b, manager));
  manager.addSubComponent(bc);

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

  // reset the beochip:
  LINFO("Resetting BeoChip...");
  b->resetChip(); sleep(1);

  // keep the gear at the lowest speed/highest torque
  b->setServoRaw(bbc.gearServoNum, bbc.gearMinVal);

  // turn on the keyboard
  b->debounceKeyboard(true);
  b->captureKeyboard(true);

  // calibrate the PWMs:
  b->calibratePulse(0,
                    bbc.pwm0NeutralVal,
                    bbc.pwm0MinVal,
                    bbc.pwm0MaxVal);
  b->calibratePulse(1,
                    bbc.pwm1NeutralVal,
                    bbc.pwm1MinVal,
                    bbc.pwm1MaxVal);
  b->capturePulse(0, true);
  b->capturePulse(1, true);

  // let's play with the LCD:
  b->lcdClear();   // 01234567890123456789
  b->lcdPrintf(0, 0, "BEOBOT-CONTROL: 0000");
  b->lcdPrintf(0, 1, "STEER=XXX  SPEED=XXX");
  b->lcdPrintf(0, 2, "PWM0=0000  0000-0000");
  b->lcdPrintf(0, 3, "PWM1=0000  0000-0000");

  // wait for keyboard and board A process it:
  int32 rframe, raction, rnode = -1;  // receive from any node
  int ct = 0;
  while(goforever) {

    // print keyboard values:
    char kb[6]; kb[5] = '\0';
    for (int i = 0; i < 5; i ++) kb[i] = (lis->kbd>>(4-i))&1 ? '1':'0';

    // quit if both extreme keys pressed simultaneously:
    if (kb[0] == '0' && kb[4] == '0') {
      b->lcdPrintf(15, 0, "QUIT ");
      goforever = false; break;
    }

    // wait up to 5ms to see if we receive a command from Board A
    if( beo->receive( rnode, rmsg, rframe, raction, 5 ) )
    {
      // check what the command is through raction

      // if UPDATE_IMAGE_COLOR: get the PixRGB<byte> image

      // if UPDATE_IMAGE_BW: get the <float> image

      // if SET_CONTROL: 0 remote, 1 Board A

      // if SET_ACTION: steer, speed, gear

      // set the motor to the passed in values
      float steerVal = rmsg.getElementFloat();
      float speedVal = rmsg.getElementFloat();
      float  gearVal = rmsg.getElementFloat();
      LINFO( "Received steer: %f speed: %f gear: %f",
             steerVal, speedVal, gearVal);
      bc->setSteer(steerVal);
      bc->setSpeed(speedVal);
      rmsg.reset(rframe, raction);

      if(++ct >= 10)
        {
          b->lcdPrintf(6,  1, "%03d", b->getServoRaw(0));
          b->lcdPrintf(17, 1, "%03d", b->getServoRaw(1));
          ct = 0;
        }
      // if SET_STEER_PID

      // if SET_SPEED_PID
    }
    usleep(5000);
  }

  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
