/*!@file Beobot/beobot-slave.C A parallel vision slave to use with
 beobot-master.  See the beobotgo script in bin/ for how to launch the
 slaves, and see beobot-master.C for the master program.
*/
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/beobot-slave.C $
// $Id: beobot-slave.C 6793 2006-06-29 20:03:25Z rjpeters $
//

#include "Beobot/BeobotVisualCortex.H"
#include "Beobot/beobot-defs.H"
#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/fancynorm.H"
#include "Util/Types.H"

#include <cstdlib>
#include <iostream>
#include <netdb.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>


#define IMGWIDTH 160
#define IMGHEIGHT 120

static bool goforever = true;  //!< Will turn false on interrupt signal

//! Signal handler (e.g., for control-C)
void terminate(int s)
{
  std::cerr<<"*** INTERRUPT ***"<<std::endl;
  goforever = false;
}

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("Parallel Vision TCP Version 2 - Slave");

  // Instantiate our various ModelComponents:
  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Slave", "BeowulfSlave", false));
  manager.addSubComponent(beo);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // setup signal handling:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  initRandomNumbers();

  // init visual cortex:
  BeobotVisualCortex vcx;
  vcx.init(IMGWIDTH, IMGHEIGHT, LEVEL_MIN, LEVEL_MAX,
           DELTA_MIN, DELTA_MAX, SMLEVEL, NBORIENTS, VCXNORM_MAXNORM,
           JETLEVEL, JETDEPTH, NBNEIGH, beo);

  // let's get all our ModelComponent instances started:
  manager.start();

  // wait for data and process it:
  while(goforever)
    vcx.slaveProcess(); // receive images, process them, and send them back

  // we got broken:
  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
