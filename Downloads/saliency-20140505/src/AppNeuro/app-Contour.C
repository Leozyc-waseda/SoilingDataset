/*!@file AppNeuro/app-Contour.C
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
// pending; filed July 23, 2001, following provisional applications     //
// No. 60/274,674 filed March 8, 2001 and 60/288,724 filed May 4, 2001).//
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
// Primary maintainer for this file: Rob Peters <rjpeters@klab.caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppNeuro/app-Contour.C $
// $Id: app-Contour.C 15310 2012-06-01 02:29:24Z itti $
//

#include "Channels/ChannelBase.H"
#include "Channels/ChannelOpts.H"
#include "Channels/ContourChannel.H"
#include "Component/ModelManager.H"
#include "Image/CutPaste.H"   // for crop() etc.
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Media/FrameSeries.H"
#include "Raster/Raster.H"
#include "Util/Assert.H"
#include "Util/CpuTimer.H"
#include "Util/SimTime.H"
#include "Util/log.H"
#include "rutz/compat_snprintf.h"
#include "rutz/shared_ptr.h"

#include <exception>
#include <iostream>

#ifdef HAVE_FENV_H
#include <fenv.h>
#endif

namespace
{
  // Parameters
  double OVERLAP_FRACTION = 0.2;

  // If SPLIT_SIZE is negative, then just do a "dry run" with
  // abs(SPLIT_SIZE) but without actually running the contour channel
  int SPLIT_SIZE = 1400000;

  int NEST_DEPTH = 0;

  // Prototypes
  Image<float> processWholeImage(const Image<byte>& img,
                                 ModelManager& manager,
                                 const std::string& saveprefix,
                                 int& pixused,
                                 nub::ref<FrameOstream> ofs);

  Image<float> processSplitImage(const Image<byte>& img,
                                 ModelManager& manager,
                                 const std::string& saveprefix,
                                 int& pixused,
                                 nub::ref<FrameOstream> ofs);

  // Implementations
  Image<float> processImage(const Image<byte>& img,
                            ModelManager& manager,
                            const std::string& saveprefix,
                            int& pixused,
                            nub::ref<FrameOstream> ofs)
  {
    Image<float> output;

    if (img.getSize() > abs(SPLIT_SIZE))
      {
        LINFO("%*sin PARTS (%s, w x h = %d x %d = %d)",
              NEST_DEPTH*4, "",
              saveprefix.c_str(), img.getWidth(), img.getHeight(),
              img.getSize());

        ++NEST_DEPTH;

        output = processSplitImage(img, manager, saveprefix,
                                   pixused, ofs);

        LINFO("%*seffective tiling ratio: %d/%d = %.4f",
              NEST_DEPTH*4, "",
              pixused, img.getSize(),
              double(pixused) / double(img.getSize()));

        --NEST_DEPTH;
      }
    else
      {
        LINFO("%*sin whole (%s, w x h = %d x %d = %d)",
              NEST_DEPTH*4, "",
              saveprefix.c_str(), img.getWidth(), img.getHeight(),
              img.getSize());
        output = processWholeImage(img, manager, saveprefix,
                                   pixused, ofs);
      }

    ASSERT(output.initialized());

    return output;
  }


  Image<float> processWholeImage(const Image<byte>& img,
                                 ModelManager& manager,
                                 const std::string& saveprefix,
                                 int& pixused,
                                 nub::ref<FrameOstream> ofs)
  {
    pixused = img.getSize();

    if (SPLIT_SIZE > 0)
      {
        nub::ref<ChannelBase> contourChan =
          makeContourChannel(manager, saveprefix);

        manager.addSubComponent(contourChan);
        contourChan->exportOptions(MC_RECURSE);

        contourChan->start();

        const Image<float> gray(img);
        contourChan->input(InputFrame::fromGrayFloat(&gray, SimTime::ZERO()));

        const Image<float> result = contourChan->getOutput();

        contourChan->saveResults(ofs);

        contourChan->stop();
        manager.removeSubComponent(*contourChan);

        return result;
      }
    else
      {
        return Image<float>(img);
      }
  }

  Image<float> processSplitImage(const Image<byte>& input,
                                 ModelManager& manager,
                                 const std::string& saveprefix,
                                 int& pixused,
                                 nub::ref<FrameOstream> ofs)
  {
    // split image into 2x2 temporary image files
    const int w = input.getWidth();
    const int h = input.getHeight();

    /*

          0    w1 wm  w2    w        0    w1 wm  w2    w
        0 +-----+--+--+-----+      0 +-----+--+--+-----+
          |        ://|     |        |     |//:        |
          |        ://|     |        |     |//:        |
          |        ://|     |        |     |//:        |
       h1 +     1  ://|     +     h1 +     |//:  2     +
          |        ://|     |        |     |//:        |
       hm +........://|     |     hm +     |//:........+
          |///////////|     |        |     |///////////|
       h2 +-----------+     +     h2 +     +-----------+
          |                 |        |                 |
          |                 |        |                 |
          |                 |        |                 |
        h +-----+--+--+-----+      h +-----+-----+-----+


          0    w1 wm  w2    w        0    w1 wm  w2    w
        0 +-----+--+--+-----+      0 +-----+--+--+-----+
          |                 |        |                 |
          |                 |        |                 |
          |                 |        |                 |
       h1 +-----------+     +     h1 +     +-----------+
          |///////////|     |        |     |///////////|
       hm +........://|     |     hm +     |//:........+
          |        ://|     |        |     |//:        |
       h2 +     3  ://|     +     h2 +     |//:  4     +
          |        ://|     |        |     |//:        |
          |        ://|     |        |     |//:        |
          |        ://|     |        |     |//:        |
        h +-----+--+--+-----+      h +-----+--+--+-----+


        Regions hatched with //////// are regions of overlap between
        adjacent tiles; these regions are discarded when the final output
        image is re-joined.

     */

    const int wm = w/2;
    const int hm = h/2;

    ASSERT(OVERLAP_FRACTION >= 0.0);
    ASSERT(OVERLAP_FRACTION <= 1.0);

    const int wtile = int(w*(0.5+0.5*OVERLAP_FRACTION));
    const int htile = int(h*(0.5+0.5*OVERLAP_FRACTION));

    LINFO("%*stile area is %dx%d",
          NEST_DEPTH*4, "", wtile, htile);

    const int w1 = w - wtile;
    const int w2 = 0 + wtile - 1;

    const int h1 = h - htile;
    const int h2 = 0 + htile - 1;

    const Rectangle input_rect[4] = {
      Rectangle::tlbrI (0,  0,  h2,   w2),
      Rectangle::tlbrI (0,  w1, h2,   w-1 ),
      Rectangle::tlbrI (h1, 0,  h-1,  w2),
      Rectangle::tlbrI (h1, w1, h-1,  w-1 )
    };

    for (size_t i = 0; i < 4; ++i)
      {
        LDEBUG("input_rect[%" ZU "].dims() = %dx%d",
               i, input_rect[i].dims().w(), input_rect[i].dims().h());
        ASSERT(input.rectangleOk(input_rect[i]));
        ASSERT(input_rect[i].dims() == Dims(wtile, htile));
      }

    const Rectangle subcrop_rect[4] = {
      Rectangle (Point2D<int>(0,     0),     Dims(wm,   hm)   ),
      Rectangle (Point2D<int>(wm-w1, 0),     Dims(w-wm, hm)   ),
      Rectangle (Point2D<int>(0,     hm-h1), Dims(wm,   h-hm) ),
      Rectangle (Point2D<int>(wm-w1, hm-h1), Dims(w-wm, h-hm) ),
    };

    const Point2D<int> subcrop_paste_pt[4] = {
      Point2D<int>(0,  0  ),
      Point2D<int>(wm, 0  ),
      Point2D<int>(0,  hm),
      Point2D<int>(wm, hm)
    };

    Image<float> output(input.getDims(), NO_INIT);

    // Use this to reconstruct the input image using the same
    // crop/paste functions as we are using to reconstruct the output,
    // so that we can check the reconstructed input against the
    // original input to make sure we haven't screwed anything up
    Image<byte> reconstruct_input(input.getDims(), ZEROS);

    pixused = 0;

    // process those four images
    for (int i = 0; i < 4; ++i)
      {
        const Image<byte> subinput =
          crop(input,
               input_rect[i].topLeft(),
               input_rect[i].dims());

        const std::string subprefix =
          sformat("%s-sub%d", saveprefix.c_str(), i);

        int subpix = 0;

        const Image<float> suboutput =
          processImage(subinput, manager, subprefix, subpix, ofs);

        if (suboutput.getDims() != subinput.getDims())
          LFATAL("Oops! In order to handle a subdivided image, the "
                 "contour channel must give output the same size as "
                 "its input (%dx%d), but the actual output was %dx%d. "
                 "Make sure that the --levelspec option includes a "
                 "map level of 0 (e.g., --levelspec=2,4,3,4,0); its "
                 "current value is --levelspec=%s.",
                 subinput.getWidth(), subinput.getHeight(),
                 suboutput.getWidth(), suboutput.getHeight(),
                 manager.getOptionValString(&OPT_LevelSpec).c_str());

        pixused += subpix;

        const Image<float> subsuboutput =
          crop(suboutput,
               subcrop_rect[i].topLeft(),
               subcrop_rect[i].dims());

        const Image<float> subsubinput =
          crop(subinput,
               subcrop_rect[i].topLeft(),
               subcrop_rect[i].dims());

        inplacePaste(output, subsuboutput, subcrop_paste_pt[i]);
        inplacePaste(reconstruct_input, Image<byte>(subsubinput),
                     subcrop_paste_pt[i]);
      }

    if (!(reconstruct_input == input))
      {
        LFATAL("crop/paste error: reconstructed input differs from original input");
      }

    return output;
  }
}

int main(int argc, char* argv[])
{
  try
    {
      ModelManager manager("Contour");

      nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
      manager.addSubComponent(ofs);

      // make a dummy contour channel to get the command-line options
      // exported
      nub::ref<ChannelBase> dummyContourChannel =
        makeContourChannel(manager);
      manager.addSubComponent(dummyContourChannel);

      // force the map level to 0 so that we get full-sized output
      // from the ContourChannel
      manager.setOptionValString(&OPT_LevelSpec, "2,4,3,4,0");
      manager.setOptionValString(&OPT_MaxNormType, "Ignore");

      MYLOGVERB = LOG_DEBUG;

      if (manager.parseCommandLine(argc, argv,
                                   "input.pnm [save_prefix] "
                                   "[split_size] [overlap_fraction]",
                                   1, 4) == false)
        return 1;

#ifdef HAVE_FEENABLEEXCEPT
      fedisableexcept(FE_ALL_EXCEPT);
      feenableexcept(FE_DIVBYZERO|FE_INVALID);
#endif

      const Image<byte> input = Raster::ReadGray(manager.getExtraArg(0));

      const std::string saveprefix =
        (manager.numExtraArgs() >= 2 && manager.getExtraArg(1).length() > 0)
        ? manager.getExtraArg(1) : "contourout";

      if (manager.numExtraArgs() >= 3 && manager.getExtraArg(2).length() > 0)
        SPLIT_SIZE = manager.getExtraArgAs<int>(2);

      if (manager.numExtraArgs() >= 4 && manager.getExtraArg(3).length() > 0)
        OVERLAP_FRACTION = manager.getExtraArgAs<double>(3);

      manager.start();

      CpuTimer t;

      int pixused = 0;
      const Image<float> output =
        processImage(input, manager, saveprefix, pixused, ofs);

      t.mark();
      t.report("full contour operation");

      if (SPLIT_SIZE < 0)
        {
          ofs->writeGray(Image<byte>(output),
                         sformat("%s.output", saveprefix.c_str()));
        }
      else
        {
          ofs->writeFloat(output,
                          FLOAT_NORM_0_255,
                          sformat("%s.output", saveprefix.c_str()));

          ofs->writeFloat(output,
                          FLOAT_NORM_0_255|FLOAT_NORM_WITH_SCALE,
                          sformat("%s.output-s", saveprefix.c_str()));
        }

      manager.stop();

      return 0;
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
/* indent-tabs-mode: nil */
/* End: */
