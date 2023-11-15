/*!@file Beobot/test-BeobotControl.C Test the BeobotControl class       */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/test-BeobotControl.C $
// $Id: test-BeobotControl.C 6814 2006-07-08 05:22:35Z beobot $
//

#include "Beobot/BeobotControl.H"
#include "Devices/BeoChip.H"
#include "Component/ModelManager.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include <iostream>
#include <math.h>
#include <signal.h>
#include <unistd.h>

static bool goforever = true;  //!< Will turn false on interrupt signal

// ######################################################################
//! Signal handler (e.g., for control-C)
void terminate(int s){  LERROR("*** INTERRUPT ***"); goforever = false; }

// ######################################################################
//! The main method - run on Beobot
int main(const int argc, const char **argv)
{
  // setup signal handling:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);

  // Instantiate a ModelManager:
  ModelManager manager("Test BeobotControl");

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeoChip> b(new BeoChip(manager));
  manager.addSubComponent(b);
  nub::soft_ref<BeobotControl> bc(new BeobotControl(b, manager));
  manager.addSubComponent(bc);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<serdev>", 1, 1) == false)
    return(1);

  // let's configure our serial device:
  b->setModelParamVal("BeoChipDeviceName", manager.getExtraArg(0));

  // let's get all our ModelComponent instances started:
  manager.start();

  float steer = 0.0F, speed = 0.0F; float sti = 0.1F, ssi = 0.02F;

  // reset the beochip:
  LINFO("Resetting BeoChip...");
  b->resetChip(); sleep(1);

  // let's play with the LCD:
  b->lcdClear();   // 01234567890123456789
  b->lcdPrintf(0, 0, "Test Beobot Control ");
  b->lcdPrintf(0, 1, "STEER=XXX  SPEED=XXX");
  b->lcdPrintf(0, 2, "                    ");
  b->lcdPrintf(0, 3, "                    ");

  // test ramping functions
  bc->setSpeed(0.0F);
  LINFO("Linear ramp to 0.4 speed, expected time 5000ms");
  Timer msecs( 1000 );
  msecs.reset();
  bc->toSpeedLinear( 0.4F, 5000 );
  LINFO("Linear ramp took %llums, ending speed = %f",
        msecs.get(), bc->getSpeed() );
  LINFO("Sigmoid ramp to 0.0 speed, expected time 5000ms");
  msecs.reset();
  bc->toSpeedSigmoid( 0.0F, 5000 );
  LINFO("Sigmoid ramp took %llums, ending speed = %f",
        msecs.get(), bc->getSpeed() );

  bc->rampSpeed( 0.4F, 5000, SPEED_RAMP_SIGMOID );
  LINFO( "Sigmoid ramp to 0.4 speed, expected time 5000ms" );
  msecs.reset();
  while( msecs.get() <= 5000 )
    {
      LINFO( "1. Ramping, current speed = %f", bc->getSpeed() );
      usleep( 100000 );
    }

  LINFO( "Sigmoid ramp complete, ending speed = %f", bc->getSpeed() );
  bc->rampSpeed( -0.25F, 5000, SPEED_RAMP_LINEAR );
  LINFO("Linear ramp to 0.0 speed, expected time 5000ms" );
  msecs.reset();
  while( msecs.get() <= 2500 )
    {
      LINFO( "2. Ramping, current speed = %f", bc->getSpeed() );
      usleep( 100000 );
    }
  bc->rampSpeed( 0.4F, 10000, SPEED_RAMP_LINEAR );
  LINFO( "Adjusting linear ramp before completion" );
  msecs.reset();
  while( msecs.get() <= 5000 )
    {
      LINFO( "3. Ramping, current speed = %f", bc->getSpeed() );
      usleep( 100000 );
    }
  bc->abortRamp();
  bc->setSpeed( 0.0F );
  LINFO( "Prematurely stopping..." );
  LINFO( "Final speed = %f", bc->getSpeed() );

  while(goforever) {
    steer += sti; speed += ssi;
    if (steer < -1.0F) { steer = -1.0F; sti =  fabs(sti); }
    if (steer >  1.0F) { steer =  1.0F; sti = -fabs(sti); }
    if (speed < -0.2F) { speed = -0.2F; ssi =  fabs(ssi); }
    if (speed >  0.2F) { speed =  0.2F; ssi = -fabs(ssi); }

    LINFO("FORCING SPEED TO ZERO"); speed=0;

    LINFO("Steer = %.2f, Speed = %.2f", bc->getSteer(), bc->getSpeed());
    bc->setSteer(steer); bc->setSpeed(speed);
    usleep(300000);
  }

  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
