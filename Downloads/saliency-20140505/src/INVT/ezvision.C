/*!@file INVT/ezvision.C  simplified version of vision.C
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/INVT/ezvision.C $
// $Id: ezvision.C 10711 2009-02-01 04:45:19Z itti $
//

#include "Component/JobServerConfigurator.H"
#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/Point2D.H"
#include "Media/SimFrameSeries.H"
#include "Neuro/NeuroOpts.H"
#include "Neuro/StdBrain.H"
#include "Raster/Raster.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "Util/AllocAux.H"
#include "Util/Pause.H"
#include "Util/Types.H"
#include "Util/csignals.H"
#include "Util/log.H"
#include "Util/sformat.H"
#include "rutz/trace.h"

#include <signal.h>
#include <sys/types.h>

int main(const int argc, const char **argv)
{
GVX_TRACE("ezvision-main");

  // 'volatile' because we will modify this from signal handlers
  volatile int signum = 0;

  // catch signals and redirect them for a clean exit (in particular,
  // this gives us a chance to do useful things like flush and close
  // output files that would otherwise be left in a bogus state, like
  // mpeg output files):
  catchsignals(&signum);

  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Attention Model");

  // Instantiate our various ModelComponents:
  nub::ref<JobServerConfigurator>
    jsc(new JobServerConfigurator(manager));
  manager.addSubComponent(jsc);

  nub::ref<SimEventQueueConfigurator>
    seqc(new SimEventQueueConfigurator(manager));
  manager.addSubComponent(seqc);

  // NOTE: make sure you register your OutputFrameSeries with the
  // manager before you do your InputFrameSeries, to ensure that
  // outputs for the current frame get saved before the next input
  // frame is loaded.
  nub::ref<SimOutputFrameSeries> ofs(new SimOutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::ref<SimInputFrameSeries> ifs(new SimInputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::ref<StdBrain> brain(new StdBrain(manager));
  manager.addSubComponent(brain);

  // Request a bunch of option aliases (shortcuts to lists of options):
  REQUEST_OPTIONALIAS_NEURO(manager);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false)
    return(1);

  nub::ref<SimEventQueue> seq = seqc->getQ();

  // let's get all our ModelComponent instances started:
  manager.start();

  // temporary, for debugging...
  seq->printCallbacks();

  PauseWaiter p;
  int retval = 0;
  SimStatus status = SIM_CONTINUE;

  // main loop:
  while(status == SIM_CONTINUE)
    {
      // Abort if we received a kill or similar signal:
      if (signum != 0) {
        LINFO("quitting because %s was caught", signame(signum));
        retval = -1; break;
      }

      // Are we in pause mode, if so, hold execution:
      if (p.checkPause()) continue;

      // Evolve for one time step and switch to the next one:
      status = seq->evolve();
    }

  // print final memory allocation stats
  LINFO("Simulation terminated.");

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return retval;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
