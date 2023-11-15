/*!@file INVT/mgzvisualize.C  Display combined .mgz streams
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/INVT/mgzvisualize.C $
// $Id: mgzvisualize.C 10794 2009-02-08 06:21:09Z itti $
//

#include "Component/ModelManager.H"
#include "Image/DrawOps.H"
#include "Image/Layout.H"
#include "Image/MathOps.H"
#include "Image/ShapeOps.H"
#include "Media/FrameSeries.H"
#include "Media/MgzDecoder.H"
#include "Raster/GenericFrame.H"
#include "Transport/FrameInfo.H"
#include "Util/csignals.H"
#include "Util/StringConversions.H"
#include "Util/StringUtil.H"
#include "Util/sformat.H"
#include "rutz/trace.h"

//! Visualize mgz streams
/*! Typically you would use that with VCX output files computed and
    saved by running ezvision. */
int main(const int argc, const char **argv)
{
GVX_TRACE("mgzvisualize");

  // 'volatile' because we will modify this from signal handlers
  volatile int signum = 0;

  // catch signals and redirect them for a clean exit (in particular,
  // this gives us a chance to do useful things like flush and close
  // output files that would otherwise be left in a bogus state, like
  // mpeg output files):
  catchsignals(&signum);

  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("MGZ Visualizer");

  // Instantiate our various ModelComponents:
  nub::ref<InputFrameSeries> mpgifs(new InputFrameSeries(manager));
  manager.addSubComponent(mpgifs);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  // Parse command-line:
  if (manager.
      parseCommandLine(argc, argv,
                       "<in1.mgz:weight:av:st> ... <inN.mgz:weight:av:st>",
                       1, -1) == false)
    return(1);

  std::vector<rutz::shared_ptr<MgzDecoder> > ifs;
  std::vector<float> weight, avg, std;

  for (uint i = 0; i < manager.numExtraArgs(); i ++)
    {
      std::vector<std::string> tokens;
      split(manager.getExtraArg(i), ":", std::back_inserter(tokens));

      rutz::shared_ptr<MgzDecoder> in(new MgzDecoder(tokens[0]));
      ifs.push_back(in);

      float w = 1.0F; if (tokens.size() > 1) convertFromString(tokens[1], w);
      weight.push_back(w);

      float av = 0.0F;  if (tokens.size() > 2) convertFromString(tokens[2], av);
      avg.push_back(av);

      float st = 1.0F;  if (tokens.size() > 3) convertFromString(tokens[3], st);
      if (st == 0.0F) LFATAL("Cannot handle zero stdev!");
      std.push_back(st);
    }

  // let's get all our ModelComponent instances started:
  manager.start();

  int retval = 0;

  // main loop:
  while(true)
    {
      if (signum != 0)
        { LINFO("Quit: %s was caught", signame(signum)); retval = -1; break; }

      if (ofs->becameVoid())
        { LINFO("Quit: output stream was closed or became void"); break; }

      // Get the next input frames:
      mpgifs->updateNext();
      GenericFrame input = mpgifs->readFrame();
      if (input.initialized() == false)
        { LINFO("Quit: input stream exhausted"); break; }

      Image<float> sm;
      for (uint i = 0; i < ifs.size(); i ++)
        {
          GenericFrame frame = ifs[i]->readFrame();
          if (frame.initialized() == false)
            { LINFO("Quit: input MGZ stream %d exhausted", i); break; }
          Image<float> cm = frame.asFloat();

          if (avg[i] != 0.0F) cm -= avg[i];

          const float w = weight[i] / std[i];
          if (w != 1.0F) cm *= w;

          if (sm.initialized()) sm += cm; else sm = cm;
        }

      // normalize the map:
      float mi, ma; getMinMax(sm, mi, ma);
      const float factor = 0.0F;
      if (factor == 0.0F) inplaceNormalize(sm, 0.0F, 255.0F);
      else if (factor != 1.0F) sm *= factor;

      // zoom the map:
      Image<byte> smb = sm; // clamped conversion
      smb = rescaleOpt(smb, input.getDims(), false);
      Image<PixRGB<byte> > smc = smb;

      writeText(smc, Point2D<int>(1,1), sformat("[%f ... %f]", mi, ma).c_str());

      Layout<PixRGB<byte> > out = hcat(input.asRgb(), smc);
      ofs->writeRgbLayout(out, "mgzvisualize",
                          FrameInfo("mgzvisualize", SRC_POS));
    }

  // print final memory allocation stats
  LINFO("Done.");

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
