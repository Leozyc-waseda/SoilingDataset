/*!@file AppNeuro/app-test-spatiotemporal.C */

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
// Primary maintainer for this file: David J. Berg <dberg@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppNeuro/app-test-spatiotemporal.C $


#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"

#include "Image/Image.H"
#include "Image/Range.H"
#include "Image/MathOps.H"
#include "Image/SteerableFilters.H"
#include "Media/FrameSeries.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Transport/FrameInfo.H"
#include "Util/Pause.H"
#include "Util/csignals.H"
#include "Util/Timer.H"

int submain(const int argc, const char **argv)
{
  volatile int signum = 0;
  catchsignals(&signum);

  ModelManager manager("Spatio-temporal energy filters");

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  SpatioTemporalEnergy ste;
  std::vector<Gauss2ndDerivParams> params = 
    {
      Gauss2ndDerivParams(0.0F,  90.0F), //static vertical
      Gauss2ndDerivParams(90.0F, 90.0F), //static horizontal
      Gauss2ndDerivParams(0.0F, 0.0F),   //flicker
      Gauss2ndDerivParams(0.0F, 75.0F),  //rightward
      Gauss2ndDerivParams(0.0F, -75.0F), //leftward
      Gauss2ndDerivParams(90.0F, 75.0F), //upward
      Gauss2ndDerivParams(90.0F, -75.0F) //downward
    };

  std::vector<std::string> names = 
    {
      "static vertical",
      "static horizontal",
      "flicker",
      "rightward",
      "leftward",
      "upward",
      "downward"
    };

  ste.setupFilters(params);
  Range<float> const outrange(0.0F, 255.0F);

  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false)
    return(1);

  manager.start();

  ifs->startStream();

  int c = 0;
  PauseWaiter p;

  while (true)
    {
      if (signum != 0)
        {
          LINFO("quitting because %s was caught", signame(signum));
          return -1;
        }

      if (ofs->becameVoid())
        {
          LINFO("quitting because output stream was closed or became void");
          return 0;
        }

      if (p.checkPause())
        continue;

      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE)
        break;

      GenericFrame input = ifs->readFrame();
      if (!input.initialized())
        break;

      //get input as gray
      const Image<float> img = input.asGray();
      ImageSet<float> resp = ste.getNextOutput(img);
      
      const FrameState os = ofs->updateNext();

      Range<float> range;
      for (uint ii = 0; ii < resp.size(); ++ii)
        range.merge(rangeOf(resp[ii]));

      //write frames
      ofs->writeFrame(GenericFrame(img, FLOAT_NORM_0_255), "input");
      for (uint ii = 0; ii < resp.size(); ++ii)
        ofs->writeFrame(GenericFrame(remapRange(resp[ii], range, outrange), FLOAT_NORM_PRESERVE), names[ii]);
      
      if (os == FRAME_FINAL)
        break;
      
      LDEBUG("frame %d", c++);
      
      if (ifs->shouldWait() || ofs->shouldWait())
        Raster::waitForKey();
    }

  return 0;
}

int main(const int argc, const char **argv)
{
  try
    {
      return submain(argc, argv);
    }
  catch (...)
    {
      REPORT_CURRENT_EXCEPTION;
    }

  return 1;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */
