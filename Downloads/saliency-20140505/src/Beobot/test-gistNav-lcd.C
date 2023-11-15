/*! @file Beobot/test-gistNav-lcd.C -- run on board B.
  It displays status info from  master (board A) sends.
  Christopher Ackerman  7/30/2003                                       */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/test-gistNav-lcd.C $
// $Id: test-gistNav-lcd.C 6795 2006-06-29 20:45:32Z rjpeters $


#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Devices/lcd.H"
#include <signal.h>
#include <stdio.h>

static bool goforever = true;  //!< Will turn false on interrupt signal

// ######################################################################
//! Signal handler (e.g., for control-C)
void terminate(int s) {
  LERROR("*** INTERRUPT ***");
  goforever = false;
  exit(1);
}

// ######################################################################
int main(const int argc, const char **argv)
{
  // message being received and to process
  TCPmessage rmsg;

  // instantiate a model manager:
  ModelManager managerS("GistNavigator - Slave");

  // Instantiate our various ModelComponents:
  nub::soft_ref<Beowulf> beoS(new Beowulf(managerS, "Beowulf Slave",
    "BeowulfSlave", false));
  managerS.addSubComponent(beoS);

  nub::soft_ref<lcd> lcdDisp(new lcd(managerS));
  lcdDisp->setModelParamString("SerialPortDevName",
                               "/dev/ttyS0", MC_RECURSE);
  managerS.addSubComponent(lcdDisp);

  // Parse command-line:
  if (managerS.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // setup signal handling:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);

  managerS.start();

  while(goforever){
    int32 rframe, raction, rnode = -1;  // receive from any node
    if (beoS->receive(rnode, rmsg, rframe, raction, 3)){ // wait up to 3ms
      const int32 sig = rmsg.getElementInt32();
      printf("received sig=%d\n",sig);
      if(sig>0){
        lcdDisp->clear();
        lcdDisp->printf(0, 2, "%d",sig); // counter indicating
          // number of frames processed
      }
      else
        if(sig==0){
          lcdDisp->clear();
          lcdDisp->printf(0, 2, "Press 2 (B) to Start");
        }
        else{
          lcdDisp->clear();
          lcdDisp->printf(0, 2, "paused");
        }

    }
  }
  managerS.stop();
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
