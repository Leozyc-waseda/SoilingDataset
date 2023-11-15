/*!@file TIGS/tigs-figs.C */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/tigs-figs.C $
// $Id: tigs-figs.C 8295 2007-04-24 21:24:46Z rjpeters $
//

#ifndef TIGS_TIGS_FIGS_C_DEFINED
#define TIGS_TIGS_FIGS_C_DEFINED

#include "Component/GlobalOpts.H"
#include "Component/ModelManager.H"
#include "Media/FrameSeries.H"
#include "TIGS/FourierFeatureExtractor.H"
#include "TIGS/PyramidFeatureExtractor.H"
#include "Util/Pause.H"
#include "Util/csignals.H"
#include "Util/fpe.H"
#include "rutz/prof.h"

int main(int argc, const char** argv)
{
  volatile int signum = 0;
  catchsignals(&signum);

  rutz::prof::print_at_exit(true);

  fpExceptionsUnlock();
  fpExceptionsOff();
  fpExceptionsLock();

  ModelManager mgr("topdown context tester");

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(mgr));
  mgr.addSubComponent(ifs);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(mgr));
  mgr.addSubComponent(ofs);

  nub::ref<PyramidFeatureExtractor> pfx(new PyramidFeatureExtractor(mgr));
  mgr.addSubComponent(pfx);

  nub::ref<FourierFeatureExtractor> ffx(new FourierFeatureExtractor(mgr));
  mgr.addSubComponent(ffx);

  mgr.exportOptions(MC_RECURSE);

  mgr.setOptionValString(&OPT_UseRandom, "false");

  if (mgr.parseCommandLine(argc, argv, "", 0, 0) == false)
    return 1;

  mgr.start();

  PauseWaiter p;

  int nframes = 0;

  while (1)
    {
      if (signum != 0)
        {
          LINFO("caught signal %s; quitting", signame(signum));
          break;
        }

      if (p.checkPause())
        continue;

      const FrameState istat = ifs->updateNext();
      if (istat == FRAME_COMPLETE)
        {
          LINFO("input series complete; quitting");
          break;
        }

      LINFO("trying frame %d", nframes);

      Image<PixRGB<byte> > frame = ifs->readRGB();
      if (!frame.initialized())
        {
          LINFO("input exhausted; quitting");
          break;
        }

      TigsInputFrame tframe(frame, SimTime::ZERO());

      pfx->saveResults(tframe, *ofs);
      ffx->saveResults(tframe, *ofs);

      const FrameState ostat = ofs->updateNext();
      if (ostat == FRAME_COMPLETE)
        {
          LINFO("output series complete; quitting");
          break;
        }

      ++nframes;
    }

  mgr.stop();

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_TIGS_FIGS_C_DEFINED
