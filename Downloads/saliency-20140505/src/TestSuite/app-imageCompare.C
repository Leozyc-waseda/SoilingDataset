/*!@file TestSuite/app-imageCompare.C Helper program for the test suite to compare two or more images */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/app-imageCompare.C $
// $Id: app-imageCompare.C 15310 2012-06-01 02:29:24Z itti $
//

#if defined INVT_HAVE_QT3 || defined INVT_HAVE_QT4
#  define IMAGECOMPARE_USE_QT 1
#endif

#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Component/ModelParam.H"
#ifdef IMAGECOMPARE_USE_QT
# ifdef INVT_HAVE_QT4
#  include "GUI/QtDisplayStream4.H"  // use Qt4 if available
# else
#  include "GUI/QtDisplayStream.H"
# endif
#else
#  include "GUI/ImageDisplayStream.H"
#endif
#include "Image/ColorOps.H"
#include "Image/MathOps.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Transport/FrameInfo.H"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

static Image<float> rectify(const Image<float>& x)
{
  Image<float> y(x);
  inplaceRectify(y);
  return y;
}

static Image<PixRGB<float> > rectifyRgb(const Image<PixRGB<float> >& x)
{
  Image<float> r, g, b;
  getComponents(x, r, g, b);
  return makeRGB(rectify(r), rectify(g), rectify(b));
}

template <class T>
static Image<byte> findNonZero(const Image<T>& x)
{
  Image<byte> result(x.getDims(), NO_INIT);

  typename Image<T>::const_iterator sptr = x.begin();
  Image<byte>::iterator dptr = result.beginw();
  Image<byte>::iterator stop = result.endw();

  const T zero = T();

  while (dptr != stop) { *dptr++ = (*sptr++ == zero) ? 0 : 255; }

  return result;
}

static const ModelOptionDef OPT_Compare =
  { MODOPT_FLAG, "Compare", &MOC_GENERAL, OPTEXP_CORE,
    "Whether to show comparisons between images (otherwise, just the images themselves will be shown)",
    "compare", '\0', "", "true" };

int main(int argc, char** argv)
{
  MYLOGVERB = LOG_CRIT;

  ModelManager mgr("imageCompare");
  OModelParam<bool> docompare(&OPT_Compare, &mgr);

#ifdef IMAGECOMPARE_USE_QT
  nub::ref<FrameOstream> ofs(new QtDisplayStream(mgr));
#else
  nub::ref<FrameOstream> ofs(new ImageDisplayStream(mgr));
#endif
  mgr.addSubComponent(ofs);

  if (mgr.parseCommandLine(argc, argv, "image1 [image2 [image3 ...]]", 1, -1) == false)
    return(1);

  mgr.start();

  std::vector<GenericFrame> im;
  std::vector<std::string> imname;

  for (uint i = 0; i < mgr.numExtraArgs(); ++i)
    {
      imname.push_back(mgr.getExtraArg(i));

      im.push_back(Raster::ReadFrame(imname.back()));

      im.back().setFloatFlags(FLOAT_NORM_0_255);

      ofs->writeFrame(im.back(), sformat("image%u",i+1), FrameInfo(imname.back(), SRC_POS));
    }

  if (docompare.getVal())
    for (size_t i = 0; i < im.size(); ++i)
      {
        const GenericFrame im1 = im[i];

        for (size_t j = i+1; j < im.size(); ++j)
          {
            const GenericFrame im2 = im[j];

            if (im1.getDims() != im2.getDims())
              continue;

            GenericFrame im1_greater_than_im2, im2_greater_than_im1, im1_neq_im2;

            switch (im1.nativeType())
              {
              case GenericFrame::NONE:
              case GenericFrame::RGB_U8:
              case GenericFrame::RGBD:
              case GenericFrame::RGB_F32:
              case GenericFrame::VIDEO:
                im1_greater_than_im2 = GenericFrame(rectifyRgb(im1.asRgbF32() - im2.asRgbF32()), FLOAT_NORM_0_255);
                im2_greater_than_im1 = GenericFrame(rectifyRgb(im2.asRgbF32() - im1.asRgbF32()), FLOAT_NORM_0_255);
                im1_neq_im2 = GenericFrame(findNonZero(im1.asRgbF32() - im2.asRgbF32()));
                break;

              case GenericFrame::GRAY_U8:
              case GenericFrame::GRAY_F32:
                im1_greater_than_im2 = GenericFrame(rectify(im1.asGrayF32() - im2.asGrayF32()), FLOAT_NORM_0_255);
                im2_greater_than_im1 = GenericFrame(rectify(im2.asGrayF32() - im1.asGrayF32()), FLOAT_NORM_0_255);
                im1_neq_im2 = GenericFrame(findNonZero(im1.asGrayF32() - im2.asGrayF32()));
                break;

              case GenericFrame::RGB_U16: break;
              case GenericFrame::GRAY_U16: break;
              }

            ofs->writeFrame(im1_greater_than_im2, sformat("image%" ZU ">image%" ZU "", i+1, j+1),
                            FrameInfo(imname[i] + " > " + imname[j], SRC_POS));
            ofs->writeFrame(im2_greater_than_im1, sformat("image%" ZU ">image%" ZU "", j+1, i+1),
                            FrameInfo(imname[j] + " > " + imname[i], SRC_POS));
            ofs->writeFrame(im1_neq_im2, sformat("image%" ZU "!=image%" ZU "", i+1, j+1),
                            FrameInfo(imname[i] + " != " + imname[j], SRC_POS));
          }
      }

  while (!ofs->isVoid()) sleep(1);

  mgr.stop();
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */
