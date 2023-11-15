/*!@file Beobot/beobot-calibrate.C Calibrate the beobot's ESC and turn */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/beobot-calibrate.C $
// $Id: beobot-calibrate.C 7912 2007-02-14 21:12:44Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Devices/BeoChip.H"
#include "Devices/DeviceOpts.H"
#include "Util/MathFunctions.H"

#include <cstdlib>

//! Calibrate the motor hardware of a Beobot
/*! Calibrate the ESC by following the sequence prescribed in the EVX
  manual. */
int main(const int argc, const char* argv[])
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("Beobot Calibrator");

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeoChip> b(new BeoChip(manager));
  manager.addSubComponent(b);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<serdev> <sernum>", 2, 2) == false)
    return(1);

  // let's configure our serial device:
  b->setModelParamVal("BeoChipDeviceName", manager.getExtraArg(0));
  int servo = manager.getExtraArgAs<int>(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  // ESC calibration:
  b->lcdClear();   // 01234567890123456789
  b->lcdPrintf(0, 0, " Beobot Servo Calib");
  const byte neutral = byte(127); // raw value for neutral
  LINFO("Setting all servos to neutral (%d)", neutral);
  b->setServoRaw(servo, neutral);

  LINFO("Press and hold ESC 'SET' button until LED turns solid red.");
  LINFO("Then release button and press [RETURN]");
  getchar();

  LINFO("Setting servo %d to full forward (255)", servo);
  b->setServoRaw(servo, 255);
  LINFO("Press [RETURN] when LED turns solid green.");
  getchar();

  LINFO("Setting servo %d to full reverse (0)", servo);
  b->setServoRaw(servo, 0);
  LINFO("Press [RETURN] when LED blinks green.");
  getchar();

  LINFO("Setting servo %d to neutral (%d)", servo, neutral);
  b->setServoRaw(servo, neutral);
  LINFO("Press [RETURN] when LED gets solid red.");
  getchar();

  LINFO("Calibration complete!");

  while(1) {
    b->setServoRaw(servo, neutral);
    LINFO("Neutral... press [RETURN] to continue");
    getchar();

    b->setServoRaw(servo, 192);
    LINFO("Weak forward... press [RETURN] to continue");
    getchar();

    b->setServoRaw(servo, 255);
    LINFO("Full forward... press [RETURN] to continue");
    getchar();

    b->setServoRaw(servo, neutral);
    LINFO("Neutral... press [RETURN] to continue");
    getchar();

    b->setServoRaw(servo, 64);
    LINFO("Weak reverse... press [RETURN] to continue");
    getchar();

    b->setServoRaw(servo, 0);
    LINFO("Full Reverse... press [RETURN] to continue");
    getchar();
  }

  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
