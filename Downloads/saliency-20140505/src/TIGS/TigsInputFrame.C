/*!@file TIGS/TigsInputFrame.C Class that lets us do lazy computation of luminance/rg/by from an input frame */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/TigsInputFrame.C $
// $Id: TigsInputFrame.C 8297 2007-04-25 00:26:17Z rjpeters $
//

#ifndef TIGS_TIGSINPUTFRAME_C_DEFINED
#define TIGS_TIGSINPUTFRAME_C_DEFINED

#include "TIGS/TigsInputFrame.H"

#include "Image/ColorOps.H"
#include "Util/StringConversions.H"
#include "rutz/trace.h"

#include <sstream>

// specialization of rescale() for PixRGB<byte>; here we unpack the
// PixRGB operations so that they can be better optimized by the
// compiler for at least a 2x speedup -- unfortunately (with gcc
// anyway) the overloaded PixRGB operators (i.e. PixRGB+PixRGB,
// PixRGB*float, etc.) make for slow code, so here we unpack things
// and do the interpolation one element at a time using builtin,
// scalar arithmetic only
Image<PixRGB<float> > rescaleAndPromote(const Image<PixRGB<byte> >& src,
                                        const int new_w, const int new_h)
{
GVX_TRACE(__PRETTY_FUNCTION__);

  ASSERT(src.initialized()); ASSERT(new_w > 0 && new_h > 0);

  const int orig_w = src.getWidth();
  const int orig_h = src.getHeight();

  // check if same size already
  if (new_w == orig_w && new_h == orig_h) return src;

  const float sw = float(orig_w) / float(new_w);
  const float sh = float(orig_h) / float(new_h);

  Image<PixRGB<float> > result(new_w, new_h, NO_INIT);
  Image<PixRGB<float> >::iterator dptr = result.beginw();
  Image<PixRGB<byte> >::const_iterator const sptr = src.begin();

  for (int j = 0; j < new_h; ++j)
    {
      const float y = std::max(0.0f, (j+0.5f) * sh - 0.5f);

      const int y0 = int(y);
      const int y1 = std::min(y0 + 1, orig_h - 1);

      const float fy = y - float(y0);

      const int wy0 = orig_w * y0;
      const int wy1 = orig_w * y1;

      for (int i = 0; i < new_w; ++i)
        {
          const float x = std::max(0.0f, (i+0.5f) * sw - 0.5f);

          const int x0 = int(x);
          const int x1 = std::min(x0 + 1, orig_w - 1);

          const float fx = x - float(x0);

#define RGB_BILINEAR_INTERP(EL)                                 \
  do {                                                          \
    const float                                                 \
      d00( sptr[x0 + wy0].p[EL] ), d10( sptr[x1 + wy0].p[EL] ), \
      d01( sptr[x0 + wy1].p[EL] ), d11( sptr[x1 + wy1].p[EL] ); \
                                                                \
    const float                                                 \
      dx0( d00 + (d10 - d00) * fx ),                            \
      dx1( d01 + (d11 - d01) * fx );                            \
                                                                \
    dptr->p[EL] = float( int( dx0 + (dx1 - dx0) * fy ) );       \
  } while(0)

          // yes, I'm doing that funny float(byte()) conversion on
          // purpose, in order to maintain backward compatibility with
          // when I used to do
          // Image<PixRGB<float>>(rescale(byteimage))

          RGB_BILINEAR_INTERP(0);
          RGB_BILINEAR_INTERP(1);
          RGB_BILINEAR_INTERP(2);

#undef RGB_BILINEAR_INTERP

          ++dptr;
        }
    }
  return result;
}

rutz::shared_ptr<TigsInputFrame>
TigsInputFrame::fromGhostString(const std::string& s)
{
  std::istringstream iss(s);
  int64 nanosecs;
  std::string dimsstr;
  std::string hashstr;
  iss >> nanosecs >> dimsstr >> hashstr;

  const Dims dims = fromStr<Dims>(dimsstr);
  const Digest<16> hash = Digest<16>::fromString(hashstr);

  return rutz::shared_ptr<TigsInputFrame>
    (new TigsInputFrame(SimTime::NSECS(nanosecs), dims, hash));
}

std::string TigsInputFrame::toGhostString() const
{
  std::ostringstream oss;
  oss << itsTime.nsecs()
      << ' ' << convertToString(itsOrigbounds.dims())
      << ' ' << this->getHash().asString();
  return oss.str();
}

void TigsInputFrame::initialize() const
{
  if (itsLum.initialized()
      && itsRG.initialized()
      && itsBY.initialized()
      && itsRGB.initialized())
    return;

  GVX_TRACE(__PRETTY_FUNCTION__);

  const Image<PixRGB<float> > fframe =
    rescaleAndPromote(itsOrigframe, 512, 512);

  itsRGB = Image<PixRGB<byte> >(fframe);

  itsLum = luminance(fframe);

  getRGBYsimple(fframe, itsRG, itsBY, float(5.0f));

  {GVX_TRACE("re-range");
  itsRG += 1.0f;
  itsBY += 1.0f;
  itsRG *= 127.5f;
  itsBY *= 127.5f;
  }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_TIGSINPUTFRAME_C_DEFINED
