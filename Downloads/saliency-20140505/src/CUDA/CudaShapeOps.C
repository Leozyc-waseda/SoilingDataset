/*!@file CUDA/CudaMathOps.C C++ wrapper for CUDA Math operations */

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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaShapeOps.C $
// $Id: CudaShapeOps.C 12962 2010-03-06 02:13:53Z irock $
//

#include "CUDA/CudaImage.H"
#include "Util/Assert.H"
#include "CUDA/cudadefs.h"
#include "CudaShapeOps.H"
#include "CUDA/CudaLowPass.H"
#include "CudaDevices.H"
#include "wrap_c_cuda.h"

#include <cmath>

// ######################################################################
CudaImage<float> cudaQuickLocalAvg(const CudaImage<float>& array, const int scale)
{
  const MemoryPolicy mp = array.getMemoryPolicy();
  const int dev = array.getMemoryDevice();
  ASSERT(array.initialized());
  ASSERT(mp != HOST_MEMORY);
  int lw = array.getWidth(), lh = array.getHeight();
  int sw = std::max(1, lw / scale), sh = std::max(1, lh / scale);

  Dims tile = CudaDevices::getDeviceTileSize(dev);

  CudaImage<float> result(sw, sh, NO_INIT, mp, dev);

  float fac = 1.0f / float(scale * scale);

  cuda_c_quickLocalAvg(array.getCudaArrayPtr(),result.getCudaArrayPtr(),fac,lw,lh,sw,sh,tile.w(),tile.h());

  return result;

}

// ######################################################################
CudaImage<float> cudaQuickLocalAvg2x2(const CudaImage<float>& array)
{
  const MemoryPolicy mp = array.getMemoryPolicy();
  const int dev = array.getMemoryDevice();
  ASSERT(array.initialized());
  ASSERT(mp != HOST_MEMORY);

  int lw = array.getWidth(), lh = array.getHeight();
  int sw = lw / 2, sh = lh / 2;

  // Just do default averaging if this is smaller than 2 along a side
  if(lw < 2 || lh < 2)
    return cudaQuickLocalAvg(array,2);

  Dims tile = CudaDevices::getDeviceTileSize(dev);

  CudaImage<float> result(sw, sh, NO_INIT, mp, dev);

  cuda_c_quickLocalAvg2x2(array.getCudaArrayPtr(),result.getCudaArrayPtr(),lw,lh,sw,sh,tile.w(),tile.h());
  return result;
}

CudaImage<float> cudaQuickLocalMax(const CudaImage<float>& array, const int scale)
{

  ASSERT(array.initialized());
  int lw = array.getWidth(), lh = array.getHeight();
  int sw = std::max(1, lw / scale), sh = std::max(1, lh / scale);

  const MemoryPolicy mp = array.getMemoryPolicy();
  const int dev = array.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize(dev);
  CudaImage<float> result(sw, sh, NO_INIT,mp,dev);

  cuda_c_quickLocalMax(array.getCudaArrayPtr(),result.getCudaArrayPtr(),lw,lh,sw,sh,tile.w(),tile.h());
  return result;
}


CudaImage<float> cudaDecXY(const CudaImage<float>& src, const int xfactor, const int yfactor_raw)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int yfactor = yfactor_raw >= 0 ? yfactor_raw : xfactor;

  const int dev = src.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  const int w = src.getWidth();
  const int h = src.getHeight();
  // Set up output image memory
  CudaImage<float> res = CudaImage<float>(Dims(w/xfactor,h/yfactor), NO_INIT, src.getMemoryPolicy(), dev);

  // Call CUDA implementation
  cuda_c_dec_xy(src.getCudaArrayPtr(), res.getCudaArrayPtr(), xfactor, yfactor, w, h, tile.sz());
  return res;
}

CudaImage<float> cudaDecX(const CudaImage<float>& src, const int xfactor)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  const int w = src.getWidth();
  const int h = src.getHeight();
  // Set up output image memory
  CudaImage<float> res = CudaImage<float>(Dims(w/xfactor,h), NO_INIT, src.getMemoryPolicy(), dev);

  // Call CUDA implementation
  cuda_c_dec_x(src.getCudaArrayPtr(), res.getCudaArrayPtr(), xfactor, w, h, tile.sz());
  return res;
}

CudaImage<float> cudaDecY(const CudaImage<float>& src, const int yfactor)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  const int w = src.getWidth();
  const int h = src.getHeight();
  // Set up output image memory
  CudaImage<float> res = CudaImage<float>(Dims(w,h/yfactor), NO_INIT, src.getMemoryPolicy(), dev);

  // Call CUDA implementation
  cuda_c_dec_y(src.getCudaArrayPtr(), res.getCudaArrayPtr(), yfactor, w, h, tile.sz());
  return res;
}

// ######################################################################
CudaImage<float> cudaDownSize(const CudaImage<float>& src, const Dims& dims,
                  const int filterWidth)
{
  return cudaDownSize(src, dims.w(), dims.h(), filterWidth);
}

// ######################################################################
CudaImage<float> cudaDownSize(const CudaImage<float>& src, const int new_w, const int new_h,
                  const int filterWidth)
{

  if (src.getWidth() == new_w && src.getHeight() == new_h) return src;

  ASSERT(src.getWidth() / new_w > 1 && src.getHeight() / new_h > 1);

  const int wdepth = int(0.5+log(double(src.getWidth() / new_w)) / M_LN2);
  const int hdepth = int(0.5+log(double(src.getHeight() / new_h)) / M_LN2);

  if (wdepth != hdepth)
    LFATAL("arrays must have same proportions");

  CudaImage<float> result = src;
  for (int i = 0; i < wdepth; ++i)
    {
      switch(filterWidth)
        {
        case 5:
          result = cudaLowPass5Dec(result,true,true);
          break;
        case 9:
          result = cudaLowPass9Dec(result,true,true);
          break;
        default:
          result = cudaDecX(cudaLowPassX(filterWidth, result));
          result = cudaDecY(cudaLowPassY(filterWidth, result));
          break;
        }
    }
  return result;
}

// ######################################################################
CudaImage<float> cudaDownSizeClean(const CudaImage<float>& src, const Dims& new_dims,
                           const int filterWidth)
{

  if (src.getDims() == new_dims) return src;

  ASSERT(new_dims.isNonEmpty());
  ASSERT(filterWidth >= 1);

  CudaImage<float> result = src;

  while (result.getWidth() > new_dims.w() * 2 &&
         result.getHeight() > new_dims.h() * 2)
    {
      if (filterWidth == 1)
        {
          result = cudaDecX(result);
          result = cudaDecY(result);
        }
      else if (filterWidth == 2)
        {
          result = cudaQuickLocalAvg2x2(result);
        }
      else
        {
          result = cudaDecX(cudaLowPassX(filterWidth, result));
          result = cudaDecY(cudaLowPassY(filterWidth, result));
        }
    }

  return cudaRescaleBilinear(result, new_dims);
}


// ######################################################################
template <class T> CudaImage<T> cudaRescaleBilinear(const CudaImage<T>& src, const Dims& dims)
{
  return cudaRescaleBilinear(src, dims.w(), dims.h());
}

// ######################################################################
CudaImage<float> cudaRescaleBilinear(const CudaImage<float>& src, const int new_w, const int new_h)
{

  const int dev = src.getMemoryDevice();
  const MemoryPolicy mp = src.getMemoryPolicy();
  ASSERT(src.initialized()); ASSERT(new_w > 0 && new_h > 0);
  ASSERT(mp != HOST_MEMORY);
  const int orig_w = src.getWidth();
  const int orig_h = src.getHeight();

  // check if same size already
  if (new_w == orig_w && new_h == orig_h) return src;

  const float sw = float(orig_w) / float(new_w);
  const float sh = float(orig_h) / float(new_h);

  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> result(new_w, new_h, NO_INIT, mp, dev);
  cuda_c_rescaleBilinear(src.getCudaArrayPtr(),result.getCudaArrayPtr(),sw,sh,orig_w,orig_h,new_w,new_h,tile.w(),tile.h());
  return result;
}

// ######################################################################
CudaImage<PixRGB<float> > cudaRescaleBilinear(const CudaImage<PixRGB<float> >& src, const int new_w, const int new_h)
{

  const int dev = src.getMemoryDevice();
  const MemoryPolicy mp = src.getMemoryPolicy();
  ASSERT(src.initialized()); ASSERT(new_w > 0 && new_h > 0);
  ASSERT(mp != HOST_MEMORY);
  const int orig_w = src.getWidth();
  const int orig_h = src.getHeight();

  // check if same size already
  if (new_w == orig_w && new_h == orig_h) return src;

  const float sw = float(orig_w) / float(new_w);
  const float sh = float(orig_h) / float(new_h);

  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<PixRGB<float> > result(new_w, new_h, NO_INIT, mp, dev);
  cuda_c_rescaleBilinearRGB((float3_t *)src.getCudaArrayPtr(),(float3_t *)result.getCudaArrayPtr(),sw,sh,orig_w,orig_h,new_w,new_h,tile.w(),tile.h());
  return result;
}


// ######################################################################
template <class T> CudaImage<T> cudaRescale(const CudaImage<T>& src, const Dims& newdims,
                 RescaleType ftype)
{
  switch (ftype)
    {
    case RESCALE_SIMPLE_BILINEAR: return cudaRescaleBilinear(src, newdims);
    default: LFATAL("unhandled ftype '%c'", ftype);
    }
  ASSERT(0);
  /* never reached */ return CudaImage<T>();
}

// ######################################################################
template <class T> CudaImage<T> cudaRescale(const CudaImage<T>& src, const int width, const int height,
                 RescaleType ftype)
{
  return cudaRescale(src, Dims(width, height), ftype);
}

// Explicit template instantiations
template CudaImage<float> cudaRescale(const CudaImage<float>& src, const Dims& newdims,
                             RescaleType ftype = RESCALE_SIMPLE_BILINEAR);
template CudaImage<float> cudaRescale(const CudaImage<float>& src, const int width, const int height,
                             RescaleType ftype = RESCALE_SIMPLE_BILINEAR);
template CudaImage<float> cudaRescaleBilinear(const CudaImage<float>& src, const Dims& dims);
template CudaImage<PixRGB<float> > cudaRescale(const CudaImage<PixRGB<float> >& src, const Dims& newdims,
                             RescaleType ftype = RESCALE_SIMPLE_BILINEAR);
template CudaImage<PixRGB<float> > cudaRescale(const CudaImage<PixRGB<float> >& src, const int width, const int height,
                             RescaleType ftype = RESCALE_SIMPLE_BILINEAR);
template CudaImage<PixRGB<float> > cudaRescaleBilinear(const CudaImage<PixRGB<float> >& src, const Dims& dims);
