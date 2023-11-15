/*!@file AppNeuro/test-intChannel.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppNeuro/test-intChannel.C $
// $Id: test-intChannel.C 11208 2009-05-20 02:03:21Z itti $
//

#ifndef APPNEURO_TEST_INTCHANNEL_C_DEFINED
#define APPNEURO_TEST_INTCHANNEL_C_DEFINED

#include "Channels/BlueYellowChannel.H"
#include "Channels/ChannelOpts.H"
#include "Channels/ColorChannel.H"
#include "Channels/DirectionChannel.H"
#include "Channels/FlickerChannel.H"
#include "Channels/GaborChannel.H"
#include "Channels/IntegerInput.H"
#include "Channels/IntegerMathEngine.H"
#include "Channels/IntegerOrientationChannel.H"
#include "Channels/IntegerSimpleChannel.H"
#include "Channels/IntensityChannel.H"
#include "Channels/MotionChannel.H"
#include "Channels/OrientationChannel.H"
#include "Channels/RedGreenChannel.H"
#include "Channels/SingleChannel.H"
#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Image/ColorOps.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/IntegerMathOps.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/PyrBuilder.H"
#include "Image/PyramidCache.H"
#include "Image/PyramidOps.H"
#include "Image/ShapeOps.H"
#include "Media/FrameSeries.H"
#include "Neuro/NeuroOpts.H"
#include "Channels/RawVisualCortex.H"
#include "Channels/IntegerRawVisualCortex.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Transport/FrameInfo.H"
#include "Util/Pause.H"
#include "Util/csignals.H"
#include "rutz/time.h"

#include <map>

static const ModelOptionDef OPT_ALIASsaveall =
  { MODOPT_ALIAS, "ALIASsaveall", &MOC_DISPLAY, OPTEXP_SAVE,
    "Default save option for test-intChannel",
    "saveall", '\0', "",
    "--save-channel-outputs --save-vcx-output" };

namespace
{
  struct ChannelCorrcoefData
  {
    ChannelCorrcoefData() : sum(0.0), n(0) {}

    void add(double d) { sum += d; ++n; }

    double sum;
    int n;
  };

  std::map<IntegerChannel*, ChannelCorrcoefData> ccdata;

  void compareChannels(int c,
                       IntegerChannel& ic,
                       ChannelBase& fc)
  {
    IntegerComplexChannel* icc =
      dynamic_cast<IntegerComplexChannel*>(&ic);

    if (icc != 0)
      {
        ComplexChannel* fcc =
          dynamic_cast<ComplexChannel*>(&fc);

        ASSERT(icc->numChans() == fcc->numChans());

        for (uint i = 0; i < icc->numChans(); ++i)
          compareChannels(c, *(icc->subChan(i)), *(fcc->subChan(i)));
      }

    float fmi, fma; getMinMax(fc.getOutput(), fmi, fma);
    int imi, ima; getMinMax(ic.getOutputInt(), imi, ima);

    const double cc = corrcoef(fc.getOutput(),
                               Image<float>(ic.getOutputInt()));
    LINFO("%06d corrcoef(%-20s,%-20s) = %.4f, "
          "frange = %.2e .. %.2e, irange = %.2e .. %.2e, imax/fmax = %.2e",
          c, fc.tagName().c_str(), ic.tagName().c_str(), cc,
          fmi, fma, float(imi), float(ima), float(ima) / fma);

    ccdata[&ic].add(cc);
  }
}

// ######################################################################
int submain(int argc, const char** argv)
{
  volatile int signum = 0;
  catchsignals(&signum);

  IntgTrigTable<256, 8> trig;

  ModelManager manager("test-intChannel");

  OModelParam<IntegerDecodeType> decodeType
    (&OPT_IntInputDecode, &manager);

  // Instantiate our various ModelComponents:
  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::ref<RawVisualCortex> fvc(new RawVisualCortex(manager));
  manager.addSubComponent(fvc);

  nub::ref<IntegerMathEngine> ieng(new IntegerMathEngine(manager));
  manager.addSubComponent(ieng);

  nub::ref<IntegerRawVisualCortex> ivc(new IntegerRawVisualCortex(manager, ieng));
  manager.addSubComponent(ivc);

  REQUEST_OPTIONALIAS_CHANNEL(manager);
  manager.requestOptionAlias(&OPT_ALIASsaveall);

  manager.setOptionValString(&OPT_MaxNormType, "Maxnorm");
  manager.setOptionValString(&OPT_DirectionChannelLowThresh, "0");
  manager.setOptionValString(&OPT_IntChannelScaleBits, "16");
  manager.setOptionValString(&OPT_IntMathLowPass5, "lp5optim");
  manager.setOptionValString(&OPT_IntMathLowPass9, "lp9optim");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "?prof-only?", 0, 1) == false)
    return(1);

  const bool profOnly =
    manager.numExtraArgs() > 0 ? manager.getExtraArgAs<bool>(0) : false;

  manager.start();

  LINFO("using %u bits for integer arithmetic", ieng->getNbits());

  LINFO("using '%s' decoding algorithm",
        convertToString(decodeType.getVal()).c_str());

  int c = 0;

  PauseWaiter p;

  SimTime t;

  rutz::time iruser, irsys, fruser, frsys;

  PyramidCache<int> cache;

  while (true)
    {
      if (signum != 0)
        {
          LINFO("quitting because %s was caught", signame(signum));
          break;
        }

      t += SimTime::HERTZ(30);

      if (p.checkPause())
        continue;

      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE)
        break;

      GenericFrame input = ifs->readFrame();
      if (!input.initialized())
        break;

      const FrameState os = ofs->updateNext();

      rutz::time u1 = rutz::time::user_rusage();
      rutz::time s1 = rutz::time::sys_rusage();

      const Image<PixRGB<byte> > rgb = input.asRgb();

      InputFrame finput = InputFrame::fromRgb(&rgb, t);

      fvc->input(finput);

      (void) fvc->getOutput();

      rutz::time u2 = rutz::time::user_rusage();
      rutz::time s2 = rutz::time::sys_rusage();

      const IntegerInput ii =
        IntegerInput::decode(input, decodeType.getVal(),
                             ieng->getNbits());

      ivc->inputInt(ii, t, &cache);

      (void) ivc->getOutputInt();

      rutz::time u3 = rutz::time::user_rusage();
      rutz::time s3 = rutz::time::sys_rusage();

      fruser += (u2 - u1);
      frsys += (s2 - s1);
      iruser += (u3 - u2);
      irsys += (s3 - s2);

      if (!profOnly)
        {
          Image<float> frg, fby;
          const float lumthresh = 25.5f;
          getRGBY(finput.colorFloat(), frg, fby, lumthresh);

          ofs->writeFrame(input, "input",
                          FrameInfo("copy of input frame", SRC_POS));

          ofs->writeFloat(finput.grayFloat(), FLOAT_NORM_0_255, "fBW",
                          FrameInfo("floating-point luminance", SRC_POS));

          ofs->writeFloat(frg, FLOAT_NORM_0_255, "fRG",
                          FrameInfo("floating-point Red-Green", SRC_POS));

          ofs->writeFloat(fby, FLOAT_NORM_0_255, "fBY",
                          FrameInfo("floating-point Blue-Yellow", SRC_POS));

          ofs->writeFloat(ii.grayInt(), FLOAT_NORM_0_255, "iBW",
                          FrameInfo("fixed-point luminance", SRC_POS));

          ofs->writeFloat(ii.rgInt(), FLOAT_NORM_0_255, "iRG",
                          FrameInfo("fixed-point Red-Green", SRC_POS));

          ofs->writeFloat(ii.byInt(), FLOAT_NORM_0_255, "iBY",
                          FrameInfo("fixed-point Blue-Yellow", SRC_POS));

          fvc->saveResults(ofs);
          ivc->saveResults(ofs);

          {
            const double cc = corrcoef(finput.grayFloat(), Image<float>(ii.grayInt()));
            LINFO("%06d corrcoef(fbw,ibw) = %.4f", c, cc);
          }

          {
            const double cc = corrcoef(frg, Image<float>(ii.rgInt()));
            LINFO("%06d corrcoef(frg,irg) = %.4f", c, cc);
          }

          {
            const double cc = corrcoef(fby, Image<float>(ii.byInt()));
            LINFO("%06d corrcoef(fby,iby) = %.4f", c, cc);
          }

          if (fvc->hasSubChan("orientation")
              && ivc->hasSubChan("int-orientation"))
            {
              const ImageSet<float> fpyr =
                dyn_cast<OrientationChannel>(fvc->subChan("orientation"))->gabor(2).pyramid(0);
              const ImageSet<int> ipyr =
                dyn_cast<IntegerOrientationChannel>(ivc->subChan("int-orientation"))->gabor(2).intgPyramid();

              for (uint i = 0; i < fpyr.size(); ++i)
                if (fpyr[i].initialized())
                  ofs->writeFloat(fpyr[i], FLOAT_NORM_0_255,
                                  sformat("fpyr[%u]", i),
                                  FrameInfo(sformat("floating-point pyramid "
                                                    "level %u of %u", i,
                                                    fpyr.size()), SRC_POS));

              for (uint i = 0; i < ipyr.size(); ++i)
                if (ipyr[i].initialized())
                  ofs->writeFloat(Image<float>(ipyr[i]), FLOAT_NORM_0_255,
                                  sformat("ipyr[%u]", i),
                                  FrameInfo(sformat("integer pyramid "
                                                    "level %u of %u", i,
                                                    fpyr.size()), SRC_POS));

              for (uint i = 0; i < fpyr.size(); ++i)
                if (fpyr[i].initialized())
                  {
                    const double cc = corrcoef(fpyr[i], Image<float>(ipyr[i]));
                    LINFO("%06d pyr[%u] corrcoef = %.4f", c, i, cc);
                  }
            }

          compareChannels(c, *ivc, *fvc);
        }

      if (os == FRAME_FINAL)
        break;

      LDEBUG("frame %d", c);
      ++c;

      if (ifs->shouldWait() || ofs->shouldWait())
        Raster::waitForKey();
    }

  manager.stop();

  for (std::map<IntegerChannel*, ChannelCorrcoefData>::const_iterator
         itr = ccdata.begin(), stop = ccdata.end();
       itr != stop; ++itr)
    {
      LINFO("AVG corrcoef(%-20s) = %.4f",
            (*itr).first->tagName().c_str(),
            (*itr).second.sum / (*itr).second.n);
    }

  LINFO("floating-point code: %.3es user + %.3es sys (%.3fms per frame)",
        fruser.sec(), frsys.sec(),
        c <= 0 ? 0.0 : (fruser.msec() + frsys.msec()) / c);
  LINFO("integer code:        %.3es user + %.3es sys (%.3fms per frame)",
        iruser.sec(), irsys.sec(),
        c <= 0 ? 0.0 : (iruser.msec() + irsys.msec()) / c);

  if (iruser.sec() + irsys.sec() > 0.0)
    LINFO("floating-point/integer runtime ratio: %.2f",
          (fruser.sec() + frsys.sec()) / (iruser.sec() + irsys.sec()));

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

#endif // APPNEURO_TEST_INTCHANNEL_C_DEFINED
