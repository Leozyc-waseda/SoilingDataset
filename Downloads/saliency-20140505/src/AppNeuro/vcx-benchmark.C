/*!@file AppNeuro/vcx-benchmark.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppNeuro/vcx-benchmark.C $
// $Id: vcx-benchmark.C 10982 2009-03-05 05:11:22Z itti $
//

#ifndef APPNEURO_VCX_BENCHMARK_C_DEFINED
#define APPNEURO_VCX_BENCHMARK_C_DEFINED

#include "Channels/ChannelOpts.H"
#include "Channels/IntegerInput.H"
#include "Channels/IntegerMathEngine.H"
#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Component/ModelParam.H"
#include "Channels/IntegerRawVisualCortex.H"
#include "Channels/RawVisualCortex.H"
#include "rutz/rand.h"
#include "rutz/time.h"

#include <stdio.h>
#include <sys/resource.h>
#include <sys/time.h>

static const ModelOptionDef OPT_UseInteger =
  { MODOPT_FLAG, "UseInteger", &MOC_GENERAL, OPTEXP_CORE,
    "Whether to the VisualCortex based on integer arithmetic",
    "use-integer", '\0', "", "true" };

int main(int argc, char** argv)
{
  MYLOGVERB = LOG_CRIT;

  ModelManager manager("VisualCortex Benchmarker");
  OModelParam<bool> useinteger(&OPT_UseInteger, &manager);

  nub::ref<IntegerMathEngine> ieng(new IntegerMathEngine(manager));
  manager.addSubComponent(ieng);

  nub::ref<IntegerRawVisualCortex> ivcx(new IntegerRawVisualCortex(manager, ieng));
  manager.addSubComponent(ivcx);

  nub::ref<RawVisualCortex> fvcx(new RawVisualCortex(manager));
  manager.addSubComponent(fvcx);

  manager.setOptionValString(&OPT_MaxNormType, "Maxnorm");
  manager.setOptionValString(&OPT_DirectionChannelLowThresh, "0");
  manager.setOptionValString(&OPT_IntChannelScaleBits, "16");
  manager.setOptionValString(&OPT_IntMathLowPass5, "lp5optim");
  manager.setOptionValString(&OPT_IntMathLowPass9, "lp9optim");

  if (manager.parseCommandLine(argc, argv, "[numframes=100] [WWWxHHH=512x512]",
                               0, 2) == false)
    return 1;

  const int nframes =
    manager.numExtraArgs() >= 1
    ? manager.getExtraArgAs<int>(0) : 100;

  const Dims dims =
    manager.numExtraArgs() >= 2
    ? manager.getExtraArgAs<Dims>(1) : Dims(512,512);

  manager.start();

  if (nframes > 0)
    {
      // allocate two images with different random
      // (uninitialized) content (so that we excite the
      // dynamic channels with non-static inputs, which may
      // some day make a difference in execute time):
      Image<PixRGB<byte> > in1(dims, NO_INIT);
      Image<PixRGB<byte> > in2(dims, NO_INIT);

      rutz::urand gen(time(NULL)); gen.idraw(1);

      for (Image<PixRGB<byte> >::iterator
             itr = in1.beginw(), stop = in1.endw(); itr != stop; ++itr)
        *itr = PixRGB<byte>(gen.idraw(256),
                            gen.idraw(256),
                            gen.idraw(256));

      for (Image<PixRGB<byte> >::iterator
             itr = in2.beginw(), stop = in2.endw(); itr != stop; ++itr)
        *itr = PixRGB<byte>(gen.idraw(256),
                            gen.idraw(256),
                            gen.idraw(256));

      const Image<byte> clipMask;

      fprintf(stderr, "%s (%s): START: %d frames %dx%d... ",
              argv[0],
              useinteger.getVal() ? "integer" : "floating-point",
              nframes, dims.w(), dims.h());
      fflush(stderr);

      const rutz::time real1 = rutz::time::wall_clock_now();
      const rutz::time user1 = rutz::time::user_rusage();
      const rutz::time sys1 = rutz::time::sys_rusage();

      SimTime t = SimTime::ZERO();

      if (useinteger.getVal())
        for (int c = 0; c < nframes; ++c)
          {
            t += SimTime::HERTZ(30);

            PyramidCache<int> cache;
            ivcx->inputInt(IntegerInput::fromRgb(c & 1 ? in2 : in1,
                                                 ieng->getNbits()),
                           t, &cache, clipMask);

            const Image<int> output = ivcx->getOutputInt();
          }
      else
        for (int c = 0; c < nframes; ++c)
          {
            t += SimTime::HERTZ(30);

            PyramidCache<int> cache;
            fvcx->input(InputFrame::fromRgb(c & 1 ? &in2 : &in1, t));

            const Image<float> output = fvcx->getOutput();
          }

      const rutz::time real2 = rutz::time::wall_clock_now();
      const rutz::time user2 = rutz::time::user_rusage();
      const rutz::time sys2 = rutz::time::sys_rusage();

      const double real_secs = (real2 - real1).sec();
      const double user_secs = (real2 - real1).sec();
      const double sys_secs = (real2 - real1).sec();

      const double frame_rate = nframes / real_secs;
      const double msec_per_frame = (1000.0*real_secs) / nframes;

      fprintf(stderr, "DONE.\n");
      fprintf(stderr, "%s (%s): real %.3fs; user %.3fs; "
              "sys %.3fs\n", argv[0],
              useinteger.getVal() ? "integer" : "floating-point",
              real_secs, user_secs, sys_secs);
      fprintf(stderr, "%s (%s): %.3ffps; %.3fmsec/frame\n",
              argv[0],
              useinteger.getVal() ? "integer" : "floating-point",
              frame_rate, msec_per_frame);
    }

  manager.stop();
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPNEURO_VCX_BENCHMARK_C_DEFINED
