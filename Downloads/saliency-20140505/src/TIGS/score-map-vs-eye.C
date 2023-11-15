/*!@file TIGS/score-map-vs-eye.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/score-map-vs-eye.C $
// $Id: score-map-vs-eye.C 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef TIGS_SCORE_MAP_VS_EYE_C_DEFINED
#define TIGS_SCORE_MAP_VS_EYE_C_DEFINED

#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Media/FrameSeries.H"
#include "Psycho/EyeSFile.H"
#include "TIGS/Scorer.H"
#include "TIGS/TigsOpts.H"
#include "Util/Pause.H"
#include "Util/csignals.H"
#include "Util/sformat.H"

#include <fstream>

// Used by: TigsJob
static const ModelOptionDef OPT_MoviePeriod =
  { MODOPT_ARG(SimTime), "MoviePeriod", &MOC_TIGS, OPTEXP_CORE,
    "Inter-frame period (or rate) of input movie",
    "movie-period", '\0', "<float>{s|ms|us|ns|Hz}", "0.0s" };

// Used by: TigsJob
static const ModelOptionDef OPT_TimeShift =
  { MODOPT_ARG(SimTime), "TimeShift", &MOC_TIGS, OPTEXP_CORE,
    "Amount of time shift to use between visual input and eye position",
    "time-shift", '\0', "<float>{s|ms|us|ns|Hz}", "0.0s" };

static void writeStringToFile(const std::string& fname,
                              const std::string& msg)
{
  std::ofstream ofs(fname.c_str());
  if (!ofs.is_open())
    LFATAL("couldn't open %s for writing", fname.c_str());
  ofs << msg << '\n';
  ofs.close();
}

int submain(int argc, const char** argv)
{
  volatile int signum = 0;
  catchsignals(&signum);

  ModelManager mgr("map-eye scorer");
  OModelParam<SimTime> moviePeriod(&OPT_MoviePeriod, &mgr);
  OModelParam<SimTime> timeShift(&OPT_TimeShift, &mgr);

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(mgr));
  mgr.addSubComponent(ifs);

  nub::ref<EyeSFile> eyeS(new EyeSFile(mgr));
  mgr.addSubComponent(eyeS);

  mgr.exportOptions(MC_RECURSE);

  if (mgr.parseCommandLine(argc, argv, "outstem", 1, 1) == false)
    return 1;

  mgr.start();

  const std::string outstem = mgr.getExtraArg(0);

  if (moviePeriod.getVal() == SimTime::ZERO())
    LFATAL("movie period must be non-zero");

  PauseWaiter p;

  int nframes = 0;

  MulticastScorer scorer;

  std::ofstream hist((outstem + ".hist").c_str());
  if (!hist.is_open())
    LFATAL("couldn't open %s.hist for writing", outstem.c_str());

  while (1)
    {
      if (signum != 0)
        {
          LINFO("caught signal %s; quitting", signame(signum));
          break;
        }

      if (p.checkPause())
        continue;

      const SimTime stime =
        moviePeriod.getVal() * (nframes+1) - timeShift.getVal();

      ifs->updateNext();
      Image<float> map = ifs->readFloat();

      if (!map.initialized())
        {
          LINFO("input exhausted; quitting");
          break;
        }

      if (stime > SimTime::ZERO())
        {
          const Point2D<int> eyepos = eyeS->readUpTo(stime);

          LINFO("simtime %.6fs, movie frame %d, eye sample %d, ratio %f, "
                "eyepos (x=%d, y=%d)",
                stime.secs(), nframes+1, eyeS->lineNumber(),
                double(eyeS->lineNumber())/double(nframes+1),
                eyepos.i, eyepos.j);

          const int p = (eyepos.j / 16) * map.getWidth() + (eyepos.i / 16);

          scorer.score("scorer", map, p);

          hist << scorer.itsNssScorer.getCurrentZscore() << ' '
               << scorer.itsPrctileScorer.getCurrentPrctile() << '\n';
        }

      ++nframes;
    }

  writeStringToFile(outstem + ".zscore",
                    sformat("%.5f", scorer.itsNssScorer.getOverallZscore()));

  writeStringToFile(outstem + ".prctile",
                    sformat("%.5f", scorer.itsPrctileScorer.getOverallPrctile()));

  mgr.stop();

  return 0;
}

int main(int argc, const char** argv)
{
  try {
    submain(argc, argv);
  } catch(...) {
    REPORT_CURRENT_EXCEPTION;
    std::terminate();
  }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_SCORE_MAP_VS_EYE_C_DEFINED
