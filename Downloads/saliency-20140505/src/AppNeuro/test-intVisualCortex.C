/*!@file AppNeuro/test-intVisualCortex.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppNeuro/test-intVisualCortex.C $
// $Id: test-intVisualCortex.C 10987 2009-03-05 19:47:46Z itti $
//

#ifndef APPNEURO_TEST_INTVISUALCORTEX_C_DEFINED
#define APPNEURO_TEST_INTVISUALCORTEX_C_DEFINED

#include "Channels/ChannelOpts.H"
#include "Channels/IntegerInput.H"
#include "Channels/IntegerMathEngine.H"
#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Component/ModelParam.H"
#include "Image/Image.H"
#include "Image/ColorOps.H"
#include "Image/DrawOps.H"
#include "Image/Layout.H"
#include "Image/IntegerMathOps.H"
#include "Image/MathOps.H"
#include "Image/PyrBuilder.H"
#include "Image/PyramidCache.H"
#include "Image/PyramidOps.H"
#include "Image/ShapeOps.H"
#include "Media/FrameSeries.H"
#include "Neuro/NeuroOpts.H"
#include "Channels/IntegerRawVisualCortex.H"
#include "Channels/RawVisualCortex.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Transport/FrameInfo.H"
#include "Util/AllocAux.H"
#include "Util/Pause.H"
#include "Util/csignals.H"
#include "rutz/time.h"

static const ModelOptionDef OPT_SaveInputCopy =
  { MODOPT_FLAG, "SaveInputCopy", &MOC_DISPLAY, OPTEXP_SAVE,
    "Save a copy of the input frame",
    "save-input-copy", '\0', "", "false" };

static const ModelOptionDef OPT_CompareToVC =
  { MODOPT_FLAG, "CompareToVC", &MOC_DISPLAY, OPTEXP_SAVE,
    "Compare int and float visual cortex outputs",
    "compare-to-vc", '\0', "", "true" };

static const ModelOptionDef OPT_ALIASsaveall =
  { MODOPT_ALIAS, "ALIASsaveall", &MOC_DISPLAY, OPTEXP_SAVE,
    "Default save option for test-intVisualCortex",
    "saveall", '\0', "",
    "--save-channel-outputs --save-vcx-output --save-input-copy" };

// ######################################################################
int submain(int argc, const char** argv)
{
  volatile int signum = 0;
  catchsignals(&signum);

  ModelManager manager("test-intChannel");

  OModelParam<bool> saveInput(&OPT_SaveInputCopy, &manager);
  OModelParam<bool> compareToVC(&OPT_CompareToVC, &manager);
  OModelParam<IntegerDecodeType> decodeType(&OPT_IntInputDecode, &manager);

  // Instantiate our various ModelComponents:
  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::ref<IntegerMathEngine> ieng(new IntegerMathEngine(manager));
  manager.addSubComponent(ieng);

  nub::ref<IntegerRawVisualCortex> ivc(new IntegerRawVisualCortex(manager, ieng));
  manager.addSubComponent(ivc);

  nub::ref<RawVisualCortex> vc(new RawVisualCortex(manager));
  manager.addSubComponent(vc);

  REQUEST_OPTIONALIAS_CHANNEL(manager);
  manager.requestOptionAlias(&OPT_ALIASsaveall);

  manager.setOptionValString(&OPT_MaxNormType, "Maxnorm");
  manager.setOptionValString(&OPT_DirectionChannelLowThresh, "0");
  manager.setOptionValString(&OPT_IntChannelScaleBits, "16");
  manager.setOptionValString(&OPT_IntMathLowPass5, "lp5optim");
  manager.setOptionValString(&OPT_IntMathLowPass9, "lp9optim");
  manager.setOptionValString(&OPT_RawVisualCortexOutputFactor, "1.0"); // look at raw values

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  manager.start();

  LINFO("using %u bits for integer arithmetic", ieng->getNbits());
  LINFO("using '%s' decoding algorithm", convertToString(decodeType.getVal()).c_str());

  int c = 0; PauseWaiter p; SimTime t; PyramidCache<int> cache; size_t npixels = 0;

  rutz::time r1 = rutz::time::wall_clock_now();
  rutz::time u1 = rutz::time::user_rusage();
  rutz::time s1 = rutz::time::sys_rusage();

  while (true)
    {
      if (signum != 0) { LINFO("quitting because %s was caught", signame(signum)); return -1; }

      t += SimTime::HERTZ(30);

      if (p.checkPause()) continue;

      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE) break;

      GenericFrame input = ifs->readFrame();
      if (!input.initialized()) break;

      npixels = size_t(input.getDims().sz());
      Image< PixRGB<byte> > inputimg = input.asRgb();
      const IntegerInput ii = IntegerInput::decode(input, decodeType.getVal(), ieng->getNbits());
      ivc->inputInt(ii, t, &cache);
      if (compareToVC.getVal()) vc->input(InputFrame::fromRgb(&inputimg, t));

      Image<int> ivcout = ivc->getOutputInt();
      Image<float> vcout; if (compareToVC.getVal()) vcout = vc->getOutput();

      const FrameState os = ofs->updateNext();
      if (saveInput.getVal()) ofs->writeFrame(input, "input", FrameInfo("copy of input frame", SRC_POS));
      ivc->saveResults(ofs);
      if (compareToVC.getVal())
        {
          // do the int map:
          int mini, maxi, a, b; getMinMax(ivcout, mini, maxi);
          const std::string irng = sformat("[%d .. %d]", mini, maxi);
          intgInplaceNormalize(ivcout, 0, 255, &a, &b); Image<byte> bivcout = ivcout;

          Image<PixRGB<byte> > map1 = rescaleOpt(makeRGB(bivcout,bivcout,bivcout), input.getDims(), false);
          drawRect(map1, Rectangle(Point2D<int>(0, 0), map1.getDims()), PixRGB<byte>(128, 128, 255));
          writeText(map1, Point2D<int>(1,1), " Integer Salmap ", PixRGB<byte>(255,255,64),
                    PixRGB<byte>(0), SimpleFont::FIXED(10), true);
          writeText(map1, Point2D<int>(3, map1.getHeight() - 3), irng.c_str(), PixRGB<byte>(255,255,64),
                    PixRGB<byte>(0), SimpleFont::FIXED(6), true, ANCHOR_BOTTOM_LEFT);

          // do the float map:
          float fmini, fmaxi; getMinMax(vcout, fmini, fmaxi);
          inplaceNormalize(vcout, 0.0F, 255.0F); Image<byte> bvcout = vcout;
          const std::string rng = sformat("[%f .. %f]", fmini, fmaxi);

          Image<PixRGB<byte> > map2 = rescaleOpt(makeRGB(bvcout,bvcout,bvcout), input.getDims(), false);
          drawRect(map2, Rectangle(Point2D<int>(0, 0), map2.getDims()), PixRGB<byte>(128, 128, 255));
          writeText(map2, Point2D<int>(1,1), " Float Salmap ", PixRGB<byte>(255,255,64),
                    PixRGB<byte>(0), SimpleFont::FIXED(10), true);
          writeText(map2, Point2D<int>(3, map2.getHeight() - 3), rng.c_str(), PixRGB<byte>(255,255,64),
                    PixRGB<byte>(0), SimpleFont::FIXED(6), true, ANCHOR_BOTTOM_LEFT);

          // do the difference map (note some promoting to int will occur and prevent overflows):
          Image<byte> diff = ((bvcout - bivcout)) / 2 + 127;
          Image<PixRGB<byte> > map3 = rescaleOpt(makeRGB(diff,diff,diff), input.getDims(), false);
          drawRect(map3, Rectangle(Point2D<int>(0, 0), map3.getDims()), PixRGB<byte>(128, 128, 255));
          writeText(map3, Point2D<int>(1,1), " Float - Int ", PixRGB<byte>(255,255,64),
                    PixRGB<byte>(0), SimpleFont::FIXED(10), true);
          std::string msg = sformat("[float = %g * int]", fmaxi / maxi);
          writeText(map3, Point2D<int>(3, map2.getHeight() - 3), msg.c_str(), PixRGB<byte>(255,255,64),
                    PixRGB<byte>(0), SimpleFont::FIXED(6), true, ANCHOR_BOTTOM_LEFT);

          Layout< PixRGB<byte> > disp = vcat(hcat(input.asRgb(), map3), hcat(map1, map2));
          ofs->writeRGB(disp.render(), "combo", FrameInfo("combo display output", SRC_POS));
        }

      if (os == FRAME_FINAL) break;
      LINFO("frame %d", c); ++c;

      if (ifs->shouldWait() || ofs->shouldWait())  Raster::waitForKey();
    }

  rutz::time r2 = rutz::time::wall_clock_now();
  rutz::time u2 = rutz::time::user_rusage();
  rutz::time s2 = rutz::time::sys_rusage();

  const rutz::time irreal = (r2 - r1);
  const rutz::time iruser = (u2 - u1);
  const rutz::time irsys = (s2 - s1);
  invt_allocation_show_stats(1, "final", npixels);
  LINFO("integer code: %.2es real; %.2es user + sys", irreal.sec(), (iruser + irsys).sec());

  manager.stop();
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

#endif // APPNEURO_TEST_INTVISUALCORTEX_C_DEFINED
