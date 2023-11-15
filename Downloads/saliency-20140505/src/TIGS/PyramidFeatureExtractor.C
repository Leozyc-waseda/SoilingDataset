/*!@file TIGS/PyramidFeatureExtractor.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/PyramidFeatureExtractor.C $
// $Id: PyramidFeatureExtractor.C 12546 2010-01-12 15:46:00Z sychic $
//

#ifndef TIGS_PYRAMIDFEATUREEXTRACTOR_C_DEFINED
#define TIGS_PYRAMIDFEATUREEXTRACTOR_C_DEFINED

#include "TIGS/PyramidFeatureExtractor.H"

#include "Image/CutPaste.H"
#include "Image/ImageSet.H"
#include "Image/PyramidOps.H"
#include "Image/ShapeOps.H" // for downSize()
#include "TIGS/Drawing.H"
#include "TIGS/TigsOpts.H"
#include "Transport/FrameOstream.H"
#include "Util/sformat.H"
#include "rutz/trace.h"

namespace
{
  float* insertLocalAvg(const Image<float>& img,
                        float* p, float* const stop,
                        const double factor)
  {
    Image<float> ds = downSize(img, 4, 4, 3);

    for (int i = 0; i < ds.getSize() && p < stop; ++i, ++p)
      *p = factor * ds[i];

    return p;
  }

  float* insertLocalMax(const Image<float>& img,
                        float* p, float* const stop,
                        const double factor)
  {

    if (p+16 > stop)
      return p;

    for (int y = 0; y < img.getHeight(); ++y)
      for (int x = 0; x < img.getWidth(); ++x)
        {
          const float val = factor * img[Point2D<int>(x,y)];
          const int pos =
            (y*4 / img.getHeight()) * 4 +
            (x*4 / img.getWidth());

          p[pos] = std::max(p[pos], val);
        }

    return p+16;
  }

  float* insertLocalVar(const Image<float>& img,
                        float* p, float* const stop,
                        const double factor)
  {

    if (p+16 > stop)
      return p;

    float ss[16] = { 0.0f }, ssq[16] = { 0.0f };
    int N[16] = { 0 };

    for (int y = 0; y < img.getHeight(); ++y)
      for (int x = 0; x < img.getWidth(); ++x)
        {
          const double val = img[Point2D<int>(x,y)];
          const int pos =
            (y*4 / img.getHeight()) * 4 +
            (x*4 / img.getWidth());

          ssq[pos] += val*val;
          ss[pos] += val;
          ++N[pos];
        }

    for (int i = 0; i < 16; ++i)
      {
        if (N[i] > 1 && (ss[i]/N[i]) > 0.0f)
          {
            const float numer1 = (ssq[i] - (ss[i]*ss[i]/N[i]));
            const float denom1 = (N[i]-1);

            if (numer1 > 0.0f && denom1 > 0.0f)
              // coefficient of variation = 100*stdev/mean
              p[i] = factor*100.0f*sqrt(numer1/denom1)/(ss[i]/N[i]);
          }
        else
          {
            p[i] = 0.0f;
          }
      }

    return p+16;
  }

  float* insertPyrFeatures(const ImageSet<float>& pyr,
                           float* p, float* const stop,
                           const double factor)
  {
    p = insertLocalAvg(pyr[2], p, stop, factor);
    p = insertLocalVar(pyr[2], p, stop, 1.5);
    p = insertLocalAvg(pyr[5], p, stop, factor);
    p = insertLocalVar(pyr[5], p, stop, 1.5);

    return p;
  }

  Image<PixRGB<byte> > illustrate1(const ImageSet<float>& pyr,
                                   const double factor,
                                   const char* name,
                                   const PixRGB<byte>& bg)
  {
    using tigs::labelImage;
    using tigs::boxify;

    const PixRGB<byte> red(255, 64, 64);
    const PixRGB<byte> green(96, 192, 96);
    const PixRGB<byte> blue(128, 128, 255);
    const PixRGB<byte> yellow(160, 160, 0);

    const Image<PixRGB<byte> > top = // 256x256
      labelImage(boxify(pyr[1] * float(factor), 8, green),
                 name, green, bg);

    const Image<PixRGB<byte> > mid = // 256x128
      concatX(labelImage(boxify(pyr[2] * float(factor), 4, yellow), "fine", yellow, bg),
              labelImage(boxify(zoomXY(pyr[5] * float(factor), 8, 8), 4, yellow), "coarse", yellow, bg));

    Image<float> avg2(4,4,NO_INIT);
    Image<float> var2(4,4,NO_INIT);
    Image<float> avg5(4,4,NO_INIT);
    Image<float> var5(4,4,NO_INIT);

    insertLocalAvg(pyr[2], &avg2[0], &avg2[0]+16, factor);
    insertLocalVar(pyr[2], &var2[0], &var2[0]+16, 1.5);
    insertLocalAvg(pyr[5], &avg5[0], &avg5[0]+16, factor);
    insertLocalVar(pyr[5], &var5[0], &var5[0]+16, 1.5);

    Image<PixRGB<byte> > cavg2 = avg2;
    Image<PixRGB<byte> > cvar2 = var2;
    Image<PixRGB<byte> > cavg5 = avg5;
    Image<PixRGB<byte> > cvar5 = var5;

    const Image<PixRGB<byte> > low2 =
      concatX(labelImage(boxify(zoomXY(cavg2, 16, 16), 2, red), "mean", red, bg),
              labelImage(boxify(zoomXY(cvar2, 16, 16), 2, blue), "var", blue, bg));

    const Image<PixRGB<byte> > low5 =
      concatX(labelImage(boxify(zoomXY(cavg5, 16, 16), 2, red), "mean", red, bg),
              labelImage(boxify(zoomXY(cvar5, 16, 16), 2, blue), "var", blue, bg));

    const Image<PixRGB<byte> > low = concatX(low2, low5);

    return concatY(concatY(top, mid), low);
  }

  void saveRaw1(const ImageSet<float>& pyr,
                const double factor,
                const char* name,
                FrameOstream& ofs)
  {
    // ofs.writeFloat(pyr[0], FLOAT_NORM_PRESERVE,
    //             (sformat("%s-base", name)));

    //  ofs.writeFloat(pyr[2], FLOAT_NORM_PRESERVE,
    //               (sformat("%s-fine", name)));

  // ofs.writeFloat(pyr[5], FLOAT_NORM_PRESERVE,
    //             (sformat("%s-coarse", name)));

    Image<float> avg2(4,4,NO_INIT);
    Image<float> var2(4,4,NO_INIT);
    Image<float> avg5(4,4,NO_INIT);
    Image<float> var5(4,4,NO_INIT);

    insertLocalAvg(pyr[2], &avg2[0], &avg2[0]+16, factor);
    insertLocalVar(pyr[2], &var2[0], &var2[0]+16, 1.5);
    insertLocalAvg(pyr[5], &avg5[0], &avg5[0]+16, factor);
    insertLocalVar(pyr[5], &var5[0], &var5[0]+16, 1.5);

    ofs.writeFloat(avg2, FLOAT_NORM_PRESERVE,
                   (sformat("%s-fine-avg", name)));

    ofs.writeFloat(var2, FLOAT_NORM_PRESERVE,
                   (sformat("%s-fine-var", name)));

    ofs.writeFloat(avg5, FLOAT_NORM_PRESERVE,
                   (sformat("%s-coarse-avg", name)));

    ofs.writeFloat(var5, FLOAT_NORM_PRESERVE,
                   (sformat("%s-coarse-var", name)));
  }
}

PyramidFeatureExtractor::PyramidFeatureExtractor(OptionManager& mgr) :
  FeatureExtractor(mgr, "pfx"),
  itsSaveIllustrations(&OPT_FxSaveIllustrations, this),
  itsSaveRawMaps(&OPT_FxSaveRawMaps, this)
{}

PyramidFeatureExtractor::~PyramidFeatureExtractor() {}

Image<PixRGB<byte> > PyramidFeatureExtractor::
illustrate(const TigsInputFrame& fin) const
{
  if (fin.isGhost())
    LFATAL("PyramidFeatureExtractor needs non-ghost frames");

  const PixRGB<byte> bg(255, 255, 255);

  Image<PixRGB<byte> > top =
    illustrate1(buildPyrGaussian(fin.lum(), 0, 10, 9), 1.0, "luminance", bg);

  top = concatX(top,
                illustrate1(buildPyrGaussian(fin.rg(), 0, 10, 9), 1.0, "red/green", bg));

  top = concatX(top,
                illustrate1(buildPyrGaussian(fin.by(), 0, 10, 9), 1.0, "blue/yellow", bg));

  const double angles[] = { 0.0, 45.0, 90.0, 135.0 };

  Image<PixRGB<byte> > bottom;

  for (int i = 0; i < 4; i += 2)
    {
      Image<PixRGB<byte> > x =
        illustrate1(buildPyrOriented(fin.lum(), 0, 10, 9,
                                     angles[i], 15.0),
                    5.0,
                    sformat("%d degrees", int(angles[i])).c_str(), bg);

      if (bottom.initialized())
        bottom = concatX(bottom, x);
      else
        bottom = x;
    }

  Image<PixRGB<byte> > result = concatX(top, bottom);
  return rescale(result,
                 int(0.75*result.getWidth()),
                 int(0.75*result.getHeight()));
}

void PyramidFeatureExtractor::
saveRawIllustrationParts(const TigsInputFrame& fin,
                         FrameOstream& ofs) const
{
  saveRaw1(buildPyrGaussian(fin.lum(), 0, 10, 9), 1.0, "luminance", ofs);

  saveRaw1(buildPyrGaussian(fin.rg(), 0, 10, 9), 1.0, "red-green", ofs);

  saveRaw1(buildPyrGaussian(fin.by(), 0, 10, 9), 1.0, "blue-yellow", ofs);

  const double angles[] = { 0.0, 45.0, 90.0, 135.0 };

  for (int i = 0; i < 4; ++i)
    {
      saveRaw1(buildPyrOriented(fin.lum(), 0, 10, 9,
                                angles[i], 15.0),
               5.0,
               sformat("ori%d", int(angles[i])).c_str(),
               ofs);
    }
}

void PyramidFeatureExtractor::saveResults(const TigsInputFrame& fin,
                                          FrameOstream& ofs) const
{
  if (itsSaveIllustrations.getVal())
    ofs.writeRGB(this->illustrate(fin), "pfx");

  if (itsSaveRawMaps.getVal())
    this->saveRawIllustrationParts(fin, ofs);
}

Image<float> PyramidFeatureExtractor::doExtract(const TigsInputFrame& fin)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  if (fin.isGhost())
    LFATAL("PyramidFeatureExtractor needs non-ghost frames");

  Image<float> result(4, 112, ZEROS);

  float*       p    = result.getArrayPtr();
  float* const stop = p + result.getSize();

  {
    GVX_TRACE("PyramidFeatureExtractor::extract-laplacian");

    p = insertPyrFeatures(buildPyrGaussian(fin.lum(), 0, 10, 9),
                          p, stop, 1.0);

    p = insertPyrFeatures(buildPyrGaussian(fin.rg(), 0, 10, 9),
                          p, stop, 1.0);

    p = insertPyrFeatures(buildPyrGaussian(fin.by(), 0, 10, 9),
                          p, stop, 1.0);
  }

  const double angles[] = { 0.0, 45.0, 90.0, 135.0 };

  for (int i = 0; i < 4; ++i)
    {
      GVX_TRACE("PyramidFeatureExtractor::extract-oriented");

      p = insertPyrFeatures(buildPyrOriented(fin.lum(), 0, 10, 9,
                                             angles[i], 15.0),
                            p, stop, 15.0);
    }

  return result;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_PYRAMIDFEATUREEXTRACTOR_C_DEFINED
