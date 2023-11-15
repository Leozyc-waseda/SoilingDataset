/*!@file AppDevices/test-BeoChip.C test suite for Rand Voorhies' BeeSTEM */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-BeeSTEM.C $
// $Id: test-BeeSTEM.C 8623 2007-07-25 17:57:51Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Devices/BeeSTEM.H"
#include "Devices/DeviceOpts.H"
#include "Util/MathFunctions.H"

#include <cstdlib>
#include <iostream>

//! Our own little BeeSTEMListener
class MyBeeSTEMListener : public BeeSTEMListener
{
public:
  MyBeeSTEMListener()
  {  }

  virtual ~MyBeeSTEMListener()
  { }

  virtual void event(const BeeSTEMEventType t, const unsigned char dat1,
                     const unsigned char dat2)
  {
    LDEBUG("Event: %d dat1 = %d, dat2 = %d", int(t), dat1, dat2);

    switch(t) {

    case COMPASS_HEADING_EVENT:   LINFO("Heading: %d", (unsigned int)dat1*2); break;  //Careful here! The heading is sent /2 because of byte limits
    case COMPASS_PITCH_EVENT:     LINFO("Pitch: %d", (signed char)dat1); break;
    case COMPASS_ROLL_EVENT:      LINFO("Roll: %d", (signed char)dat1); break;
    case ACCEL_X_EVENT:           break;
    case ACCEL_Y_EVENT:           break;
    case INT_PRESS_EVENT:         LINFO("Int pressure: %d", (unsigned int)dat1);break;
    case EXT_PRESS_EVENT:         LINFO("External Pressure: %d", (unsigned int)dat1); break;
    case TEMP1_EVENT:             break;
    case TEMP2_EVENT:             break;
    case TEMP3_EVENT:             break;
    case DIG_IN_EVENT:            break;
    case ADC_IN_EVENT:            break;
    case MOTOR_A_CURR_EVENT:      LINFO("Motor A Current: %d", (unsigned int)dat1);break;
    case MOTOR_B_CURR_EVENT:      LINFO("Motor B Current: %d", (unsigned int)dat1);break;
    case MOTOR_C_CURR_EVENT:      LINFO("Motor C Current: %d", (unsigned int)dat1);break;
    case MOTOR_D_CURR_EVENT:      LINFO("Motor D Current: %d", (unsigned int)dat1);break;
    case MOTOR_E_CURR_EVENT:      LINFO("Motor E Current: %d", (unsigned int)dat1);break;
    case ECHO_REPLY_EVENT:        LINFO("BeeSTEM Echo Reply Recieved."); break;
    case RESET_EVENT:             LERROR("BeeSTEM RESET occurred!"); break;
    case SW_OVERFLOW_EVENT:       LERROR("BeeSTEM Software Overflow!"); break;
    case FRAMING_ERR_EVENT:       LERROR("BeeSTEM Framing Error!"); break;
    case OVR_ERR_EVENT:           LERROR("BeeSTEM Hardware Overflow!"); break;
    case HMR3300_LOST_EVENT:      break;
    case ACCEL_LOST_EVENT:        break;
    case TEMP1_LOST_EVENT:        break;
    case TEMP2_LOST_EVENT:        break;
    case HMR_LEVELED_EVENT:       LINFO("HMR3300 Leveling Complete"); break;
    case ESTOP_EVENT:             break;
    case UNRECOGNIZED_EVENT:      break;
    case BAD_OUT_CMD_SEQ_EVENT:   LERROR("BeeSTEM Reports a Bad Command Sequence!"); break;
    case BAD_IN_CMD_SEQ_EVENT:    break;
    case RESET_ACK_EVENT:         LINFO("BeeSTEM Acknowledges Reset Request"); break;
    case HMR3300_CAL_EVENT:       LINFO("HMR3300 is Calibrating!"); break;
    case  NO_EVENT:               break;
    default:                      LERROR("Unknown event %d received!", int(t)); break;

    }
  }
};

//! This program provides basic interfacing to the BeeSTEM
/*! See the BeeSTEM class for details.*/
int main(const int argc, const char* argv[])
{
  char c = 0;
  int num = 0;
  int motor = 0;
  int speed = 0;

  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("BeeSTEM test program");

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeeSTEM> b(new BeeSTEM(manager,"BeeSTEM", "BeeSTEM", "/dev/ttyS1"));
  manager.addSubComponent(b);

  if (manager.parseCommandLine(argc, argv, "No Options Yet", 0,0) == false)
                                   return(1);

  rutz::shared_ptr<MyBeeSTEMListener> lis(new MyBeeSTEMListener);
  rutz::shared_ptr<BeeSTEMListener> lis2; lis2.dynCastFrom(lis); // cast down
  b->setListener(lis2);

  // let's get all our ModelComponent instances started:
  manager.start();

  LINFO("Echo request (should bring an echo reply back)...");
  b->echoRequest(); sleep(2);
  while(c != 'q') {
    std::cin >> c;
    switch(c) {
    case 'h':
      std::cout << "Commands:" << std::endl;
      std::cout << "E: Echo Request" << std::endl;
      std::cout << "R: Reset the chip" << std::endl;
      std::cout << "F: Flood the chip with echo requests" << std::endl;
      std::cout << "C=on c=off: Toggle Compass Reporting" << std::endl;
      std::cout << "M: Set a thruster speed" << std::endl;
      std::cout << "I=on i=off: Toggle Internal Pressure Reporting" << std::endl;
      std::cout << "P=on p=off: Toggle External Pressure Reporting" << std::endl;
      std::cout << "D=on d=off:  Toggle Debugging Mode" << std::endl;
      std::cout << "A=on a=off: Toggle Motor Reporting" << std::endl;
      std::cout << "S: Toggle HMR3300 Calibration Mode" << std::endl;
      std::cout << "L: Level the HMR3300" << std::endl;
      break;
    case 'e':
      LINFO("Echo request (should bring an echo reply back)...");
      b->echoRequest();
      break;
    case 'r':
      LINFO("Reseting the BeeSTEM...");
      b->resetChip();
      break;
    case 'f':
      LINFO("Enter the number of echo requests to flood");
      std::cin >> num;
      LINFO("Flooding the BeeSTEM with %d echo requests...", num);
      for(int i=0; i<num;i++)
        b->echoRequest();
      break;
    case 'C':
      b->setReporting(HMR3300, true);
      break;
    case 'c':
      b->setReporting(HMR3300, false);
      break;
    case 'I':
      b->setReporting(INT_PRESS, true);
      break;
    case 'i':
      b->setReporting(INT_PRESS, false);
      break;
    case 'D':
      b->debugMode(true);
    case 'd':
      b->debugMode(false);
    case 'm':
    case 'M':
      LINFO("Motor:");
      std::cin >> motor;
      LINFO("Speed:");
      std::cin >> speed;
      b->setMotor(motor, speed);
      break;
    case 'P':
      b->setReporting(EXT_PRESS, true);
      break;
    case 'p':
      b->setReporting(EXT_PRESS, false);
      break;
    case 'S':
    case 's':
      b->toggleCalibrateHMR3300();
      break;
    case 'L':
    case 'l':
      b->levelHMR3300();
      break;
    case 'A':
      b->setReporting(MOTOR_CURR, true);
      break;
    case 'a':
      b->setReporting(MOTOR_CURR, false);
      break;
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
