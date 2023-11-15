/*!@file BeoSub/test-BeoSubBallast.C test submarine ballasts */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoSubBallast.C $
// $Id: test-BeoSubBallast.C 6990 2006-08-11 18:13:51Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Devices/BeoChip.H"
#include "BeoSub/BeoSubBallast.H"

#include <cstdlib>
#include <pthread.h>

//! Our own little BeoChipListener
class MyBeoChipListener : public BeoChipListener
{
public:
  MyBeoChipListener(nub::soft_ref<BeoSubBallast>& ballast,
                    nub::soft_ref<BeoChip>& beochip) :
    itsBallast(ballast), itsBeoChip(beochip)
  { }

  virtual ~MyBeoChipListener()
  { }

  virtual void event(const BeoChipEventType t, const int valint,
                     const float valfloat)
  {
    LINFO("Event: %d val = %d, fval = %f", int(t), valint, valfloat);

    switch(t)
      {
      case NONE: break;
      case KBD:
        // in this test program, we only care about keyboard events:
        itsBallast->input(valint);

        // let the user know what is going on:
        itsBeoChip->lcdPrintf(0, 3, "Current=%04d / %1.3f",
                              itsBallast->getPulses(), itsBallast->get());
        break;
      case RESET: LERROR("BeoChip RESET occurred!"); break;
      case ECHOREP: LINFO("BeoChip Echo reply received."); break;
      case INOVERFLOW: LERROR("BeoChip input overflow!"); break;
      case SERIALERROR: LERROR("BeoChip serial error!"); break;
      case OUTOVERFLOW: LERROR("BeoChip output overflow!"); break;
      default: LERROR("Unexpected event %d received!", int(t)); break;
      }
  }

private:
  nub::soft_ref<BeoSubBallast> itsBallast;
  nub::soft_ref<BeoChip> itsBeoChip;
};

//! This program provides basic testing of a BeoSubBallast
int main(const int argc, const char* argv[])
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("BeoSubBallast test program");

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeoChip> bc(new BeoChip(manager));
  manager.addSubComponent(bc);

  nub::soft_ref<BeoSubBallast> bal(new BeoSubBallast(manager));
  manager.addSubComponent(bal);

  bal->setBeoChip(bc);  // hook BeoChip to Ballast

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<serdev> <front|rear>", 2, 2) == false)
    return(1);

  // let's configure our serial device:
  bc->setModelParamVal("BeoChipDeviceName", manager.getExtraArg(0));
  bc->setModelParamVal("BeoChipUseRTSCTS", false);

  // let's register our listener:
  rutz::shared_ptr<BeoChipListener> lis2(new MyBeoChipListener(bal, bc));
  bc->setListener(lis2);

  // setup for the particular ballast we are testing:
  if (manager.getExtraArg(1).compare("rear") == 0)
    {
      // defaults for ballast that is directly attached to the BeoChip:
      bal->setModelParamVal("BeoSubBallastOutRed", 1);
      bal->setModelParamVal("BeoSubBallastOutWhite", 0);
      bal->setModelParamVal("BeoSubBallastInYellow", 1);
      bal->setModelParamVal("BeoSubBallastInWhite", 0);
      LINFO("Testing REAR ballast (holding BeoChip assembly)");
    }
  else
    {
      // defaults for ballast that is NOT directly attached to the BeoChip:
      bal->setModelParamVal("BeoSubBallastOutRed", 3);
      bal->setModelParamVal("BeoSubBallastOutWhite", 2);
      bal->setModelParamVal("BeoSubBallastInYellow", 3);
      bal->setModelParamVal("BeoSubBallastInWhite", 2);
      LINFO("Testing FRONT ballast (NOT holding BeoChip assembly)");
    }

  // let's get all our ModelComponent instances started:
  manager.start();

  LINFO("Reseting the BeoChip...");
  bc->reset(MC_RECURSE);
  sleep(1);

  LINFO("Waiting for a bit. Turn BeoChip on if not already on."); sleep(2);
  LINFO("Echo request (should bring an echo reply back)...");
  bc->echoRequest(); sleep(1);

  // load a cool custom font:
  bc->lcdLoadFont(1);

  // let's turn everything on:
  bc->capturePulse(0, false);
  bc->capturePulse(1, false);
  bc->captureAnalog(0, false);
  bc->captureAnalog(1, false);
  bc->debounceKeyboard(false);
  bc->captureKeyboard(true);

  // let's play with the LCD:
  LINFO("Start...");
  bc->lcdClear();   // 01234567890123456789
  bc->lcdPrintf(0, 0, "test-BeoSubBallast  ");
  bc->lcdPrintf(0, 1, "                    ");
  bc->lcdPrintf(0, 2, "Desired=0.000       ");
  bc->lcdPrintf(0, 3, "Current=0000 / 0.000");

  // initialize the ballast:
  bc->lcdPrintf(0, 1, "---- Initialize ----");
  bal->mechanicalInitialize();

  // let's play with the ballast:
  while(true)
    {
      bc->lcdPrintf(0, 1, "------ Actuate -----");
      bc->lcdPrintf(0, 2, "Desired=1.000");
      bal->set(1.0F, true);
      bc->lcdPrintf(0, 1, "------- Idle -------");
      sleep(10);

      bc->lcdPrintf(0, 1, "------ Actuate -----");
      bc->lcdPrintf(0, 2, "Desired=0.000");
      bal->set(0.0F, true);
      bc->lcdPrintf(0, 1, "------- Idle -------");
      sleep(10);

      bc->lcdPrintf(0, 1, "------ Actuate -----");
      bc->lcdPrintf(0, 2, "Desired=0.500");
      bal->set(0.5F, true);
      bc->lcdPrintf(0, 1, "------- Idle -------");
      sleep(10);

      bc->lcdPrintf(0, 1, "------ Actuate -----");
      bc->lcdPrintf(0, 2, "Desired=0.600");
      bal->set(0.6F, true);
      bc->lcdPrintf(0, 1, "------- Idle -------");
      sleep(10);

      bc->lcdPrintf(0, 1, "------ Actuate -----");
      bc->lcdPrintf(0, 2, "Desired=0.550");
      bal->set(0.55F, true);
      bc->lcdPrintf(0, 1, "------- Idle -------");
      sleep(10);

      bc->lcdPrintf(0, 1, "------ Actuate -----");
      bc->lcdPrintf(0, 2, "Desired=0.560");
      bal->set(0.56F, true);
      bc->lcdPrintf(0, 1, "------- Idle -------");
      sleep(10);

      bc->lcdPrintf(0, 1, "------ Actuate -----");
      bc->lcdPrintf(0, 2, "Desired=0.555");
      bal->set(0.555F, true);
      bc->lcdPrintf(0, 1, "------- Idle -------");
      sleep(10);

      bc->lcdPrintf(0, 1, "------ Actuate -----");
      bc->lcdPrintf(0, 2, "Desired=0.556");
      bal->set(0.556F, true);
      bc->lcdPrintf(0, 1, "------- Idle -------");
      sleep(10);
    }

  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
