/*!@file TIGS/FourierFeatureExtractor.C Extract topdown features using fourier decomposition */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/FourierFeatureExtractor.C $
// $Id: FourierFeatureExtractor.C 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef TIGS_FOURIERFEATUREEXTRACTOR_C_DEFINED
#define TIGS_FOURIERFEATUREEXTRACTOR_C_DEFINED

#include "TIGS/FourierFeatureExtractor.H"

#include "Image/CutPaste.H" // for crop()
#include "Image/DrawOps.H"
#include "Image/FourierEngine.H"
#include "Image/MathOps.H"
#include "Image/Normalize.H"
#include "Image/Range.H"
#include "Image/ShapeOps.H"
#include "TIGS/Drawing.H"
#include "TIGS/TigsOpts.H"
#include "Transport/FrameOstream.H"
#include "Util/log.H"
#include "rutz/trace.h"

FourierFeatureExtractor::FourierFeatureExtractor(OptionManager& mgr) :
  FeatureExtractor(mgr, "ffx"),
  itsEngine(0),
  itsSaveIllustrations(&OPT_FxSaveIllustrations, this),
  itsSaveRawMaps(&OPT_FxSaveRawMaps, this)
{}

FourierFeatureExtractor::~FourierFeatureExtractor() { itsEngine = 0; }

Image<PixRGB<byte> > FourierFeatureExtractor::
illustrate(const TigsInputFrame& fin) const
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  if (fin.isGhost())
    LFATAL("FourierFeatureExtractor needs non-ghost frames");

  using tigs::labelImage;
  using tigs::boxify;

  const PixRGB<byte> bg(255, 255, 255);

  if (itsEngine == 0)
    itsEngine = new FourierEngine<double>(fin.lum().getDims());

  const PixRGB<byte> red(255, 64, 64);
  const PixRGB<byte> green(96, 192, 96);
  const PixRGB<byte> blue(128, 128, 255);
  const PixRGB<byte> yellow(160, 160, 0);

  const Image<float> lum = fin.lum();
  const Image<complexd> fft = itsEngine->fft(fin.lum());

  Image<float> logmag = logmagnitude(fft);

  logmag *= 15.0f;

  {
    const Range<float> r = rangeOf(logmag);
    LINFO("log(mag(fft)) range: [%f .. %f]", r.min(), r.max());
  }

  const int flags = 0;

  Image<PixRGB<byte> > cart =
    normalizeFloat(zoomXY(cartesian(logmag, Dims(128, 128)), 2, 4),
                   flags);

  drawRect(cart, Rectangle::tlbrI(0, 96, 511, 193), red, 1);

  Image<float> cropped =
    zoomXY(crop(cartesian(logmag, Dims(64, 16)),
                Point2D<int>(24, 0), Dims(24, 16)),
           8, 32);

  cropped -= 80.0f;
  cropped *= 5.0f;

  {
    const Range<float> r = rangeOf(cropped);
    LINFO("cropped range: [%f .. %f]", r.min(), r.max());
  }

  Image<PixRGB<byte> > result =
    labelImage(boxify(lum, 4, green), "luminance", green, bg);

  result =
    concatLooseX(result,
                 labelImage(boxify(normalizeFloat(logmag, flags),
                                   4, blue),
                            "log(|fft|)", blue, bg));

  result =
    concatLooseX(result,
                 labelImage(boxify(cart, 4, yellow),
                            "cartesian(log(|fft|))", yellow, bg));

  result =
    concatLooseX(result,
                 labelImage(boxify(normalizeFloat(cropped, flags),
                                   4, red),
                            "cropped(cartesian)", red, bg));

  return result;
}

void FourierFeatureExtractor::
saveRawIllustrationParts(const TigsInputFrame& fin,
                         FrameOstream& ofs) const
{
  if (itsEngine == 0)
    itsEngine = new FourierEngine<double>(fin.lum().getDims());

  const Image<float> lum = fin.lum();
  const Image<complexd> fft = itsEngine->fft(fin.lum());

  const Image<float> logmag = logmagnitude(fft);

  const Image<float> cart = cartesian(logmag, Dims(128, 128));

  const Image<float> cropped = crop(cartesian(logmag, Dims(64, 16)),
                                    Point2D<int>(24, 0), Dims(24, 16));

  ofs.writeFloat(lum, FLOAT_NORM_PRESERVE, "ffx-luminance");
  ofs.writeFloat(logmag, FLOAT_NORM_PRESERVE, "ffx-logmag");
  ofs.writeFloat(cart, FLOAT_NORM_PRESERVE, "ffx-cartesian");
  ofs.writeFloat(cropped, FLOAT_NORM_PRESERVE, "ffx-cropped");
}

void FourierFeatureExtractor::saveResults(const TigsInputFrame& fin,
                                          FrameOstream& ofs) const
{
  if (itsSaveIllustrations.getVal())
    ofs.writeRGB(this->illustrate(fin), "ffx");

  if (itsSaveRawMaps.getVal())
    this->saveRawIllustrationParts(fin, ofs);
}

Image<float> FourierFeatureExtractor::doExtract(const TigsInputFrame& fin)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  if (fin.isGhost())
    LFATAL("VisualCortexFeatureExtractor needs non-ghost frames");

  if (itsEngine == 0)
    itsEngine = new FourierEngine<double>(fin.lum().getDims());

  const Image<complexd> res = itsEngine->fft(fin.lum());

  const Image<float> cart =
    cartesian(logmagnitude(res), Dims(64, 16));

  Image<float> result =
    crop(cart, Point2D<int>(24, 0), Dims(24, 16));

  result -= 5.5f;
  result *= 80.0f;

  const Range<float> r = rangeOf(result);

  LINFO("log(mag(fft)) range: [%f .. %f]", r.min(), r.max());

  return result;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_FOURIERFEATUREEXTRACTOR_C_DEFINED
