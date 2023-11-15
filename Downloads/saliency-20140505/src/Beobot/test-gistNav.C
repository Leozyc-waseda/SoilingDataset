/*! @file Beobot/test-gistNav.C -- run on board B (has BeoChip).
  Executes motor commands from board A (master)
  -- Christopher Ackerman  7/30/2003                                    */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/test-gistNav.C $
// $Id: test-gistNav.C 6795 2006-06-29 20:45:32Z rjpeters $

#include "Devices/BeoChip.H"
#include "Beobot/BeobotConfig.H"
#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include <signal.h>
#include <stdio.h>

static bool goforever = true;  //!< Will turn false on interrupt signal

// ######################################################################
//! Signal handler (e.g., for control-C)
void terminate(int s)
{
  LERROR("*** INTERRUPT ***");
  goforever = false;
  exit(1);
}

// ######################################################################
int main(const int argc, const char **argv)
{
  // instantiate a model manager:
  ModelManager manager("GistNavigator - Slave");

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeoChip> b(new BeoChip(manager));
  manager.addSubComponent(b);

  nub::soft_ref<Beowulf> beo(new Beowulf(manager, "Beowulf Slave",
    "BeowulfSlave", false));
  manager.addSubComponent(beo);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<serdev>", 1, 1) == false)
    return(1);

  // let's configure our serial device:
  b->setModelParamVal("BeoChipDeviceName", manager.getExtraArg(0));

  // setup signal handling:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);

  // message being received and to process
  TCPmessage rmsg;

  manager.start();

  // reset the beochip:
  LINFO("Resetting BeoChip...");
  b->resetChip(); sleep(1);
  BeobotConfig bbc;

  // keep the gear at the lowest speed/highest torque
  b->setServoRaw(2, 0);

  while(goforever)
  {
    int32 rframe, raction, rnode = -1;  // receive from any node
    if (beo->receive(rnode, rmsg, rframe, raction, 3)){ // wait up to 3ms

      const double steer = rmsg.getElementDouble();
      const double speed = rmsg.getElementDouble();
      LINFO("Received speed=%f, steer=%f\n",speed,steer);

      bool b1 = b->setServo(bbc.speedServoNum, (float)(steer));
      bool b2 = b->setServo(bbc.steerServoNum, (float)(speed));
      LINFO("setSteering returned %d; setSpeed returned %d\n",b1,b2);
      LINFO("actual speed=%f, steer=%f\n",
            b->getServo(bbc.speedServoNum),
            b->getServo(bbc.steerServoNum));
    }
  }

  manager.stop();
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
