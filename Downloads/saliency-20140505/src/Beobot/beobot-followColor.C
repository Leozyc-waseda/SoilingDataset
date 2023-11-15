/*!@file Beobot/beobot-followColor.C Test color segment following - slave */

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
// Primary maintainer for this file:  T. Nathan Mundhenk <mundhenk@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/beobot-followColor.C $
// $Id: beobot-followColor.C 7063 2006-08-29 18:26:55Z rjpeters $
//

#include "Devices/BeoChip.H"
#include "Beobot/BeobotConfig.H"
#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Util/log.H"
#include "rutz/shared_ptr.h"
#include <signal.h>

static bool goforever = true;
// ######################################################################
void terminate(int s) { LERROR("*** INTERRUPT ***"); goforever = false; exit(1);}

// ######################################################################
//! Receive signals from master node and performs requested actions
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager
  ModelManager manager( "Following Color Segments - Slave" );

  // instantiate our various ModelComponents
  BeobotConfig bbc;
  nub::soft_ref<BeoChip> b(new BeoChip(manager));
  manager.addSubComponent(b);
  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Slave", "BeowulfSlave", false));
  manager.addSubComponent(beo);

  // parse command-line
  if (manager.parseCommandLine(argc, argv, "<serdev>", 1, 1) == false)
    return(1);

  // let's configure our serial device:
  b->setModelParamVal("BeoChipDeviceName", manager.getExtraArg(0));
  // NOTE: may want to add a BeoChip listener

  // setup signal handling:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);

  TCPmessage rmsg;       // received messages

  manager.start();

  // reset the beochip:
  LINFO("Resetting BeoChip...");
  b->resetChip(); sleep(1);

  // keep the gear at the lowest speed/highest torque
  b->setServoRaw(bbc.gearServoNum, bbc.gearMinVal);

  int32 rframe, raction, rnode = -1; // receive from any node
  while( goforever )
  {
    // wait up to 5ms
    if( beo->receive( rnode, rmsg, rframe, raction, 5 ) )
    {
      float steerVal = rmsg.getElementFloat();
      float speedVal = rmsg.getElementFloat();
      float  gearVal = rmsg.getElementFloat();
      LINFO( "Received steer: %f speed: %f gear: %f",
             steerVal, speedVal, gearVal);
      b->setServo(bbc.steerServoNum, steerVal);
      b->setServo(bbc.speedServoNum, speedVal);
      rmsg.reset(rframe, raction);
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
