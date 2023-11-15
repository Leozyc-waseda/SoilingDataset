/*!@file Parallel/test-comm.C simple test to send and receive a float */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Parallel/test-comm.C $
// $Id: test-comm.C 6393 2006-03-26 00:57:36Z rjpeters $
//

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <time.h>
#include <unistd.h>

static bool goforever = true;
void terminate(int s) { LERROR("*** INTERRUPT ***"); goforever = false; }

// ######################################################################
/*! This simple executable test to receive a float  */
int main(const int argc, const char **argv)
{
  // instantiate a model manager:
  ModelManager manager("Communication Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Slave", "BeowulfSlave", false));
  manager.addSubComponent(beo);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // setup signal handling:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);

  TCPmessage rmsg;            // message being received and to process
  TCPmessage smsg;            // message being sent

  // let's get all our ModelComponent instances started:
  manager.start();

  // wait for data and process it:
  int i = 0;
  while(goforever) {
    int32 rframe, raction, rnode = -1;  // receive from any node
    if(beo->receive(rnode, rmsg, rframe, raction, 5))
    {
      float val = rmsg.getElementFloat();
      LINFO("rec %f", val);
      rmsg.reset(rframe, raction);

      smsg.reset(rframe, 1);
      smsg.addFloat(3.1212+i);
      LINFO("send: %f",3.1212+i);
      beo->send(rnode, smsg);
      sleep(1);
      i++;
    }
    LINFO("  NOTHING");
  }

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
