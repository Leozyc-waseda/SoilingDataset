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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaMathOps.C $
// $Id: CudaMathOps.C 12962 2010-03-06 02:13:53Z irock $
//

#include "CUDA/CudaImage.H"
#include "Util/Assert.H"
#include "CUDA/cudadefs.h"
#include "CudaMathOps.H"
#include "CudaDevices.H"
#include "wrap_c_cuda.h"


void cudaGetMin(const CudaImage<float>& src, CudaImage<float>& minim, CudaImage<float> *buf)
{
  MemoryPolicy mp = src.getMemoryPolicy();
  const int dev = src.getMemoryDevice();

  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(mp != HOST_MEMORY);

  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> tmp;
  if(minim.size() != 1 || minim.getMemoryDevice() != dev || minim.getMemoryPolicy() != mp)
    minim = CudaImage<float>(1,1,NO_INIT,mp, dev);

  if(buf == 0)
  {
    // Set up output image memory
    tmp = CudaImage<float>(Dims(iDivUp(src.size(),tile.sz()),1), NO_INIT, src.getMemoryPolicy(), dev);
    buf = &tmp;
  }
  // Call CUDA implementation
  cuda_c_getMin(src.getCudaArrayPtr(), minim.getCudaArrayPtr(), buf->getCudaArrayPtr(), tile.sz(), src.size());

}

void cudaGetMax(const CudaImage<float>& src, CudaImage<float>& maxim, CudaImage<float> *buf)
{
  MemoryPolicy mp = src.getMemoryPolicy();
  const int dev = src.getMemoryDevice();

  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(mp != HOST_MEMORY);

  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> tmp;
  if(maxim.size() != 1 || maxim.getMemoryDevice() != dev || maxim.getMemoryPolicy() != mp)
    maxim = CudaImage<float>(1,1,NO_INIT,mp, dev);

  if(buf == 0)
  {
    // Set up output image memory
    tmp = CudaImage<float>(Dims(iDivUp(src.size(),tile.sz()),1), NO_INIT, src.getMemoryPolicy(), dev);
    buf = &tmp;
  }

  // Call CUDA implementation
  cuda_c_getMax(src.getCudaArrayPtr(), maxim.getCudaArrayPtr(),buf->getCudaArrayPtr(), tile.sz(), src.size());

}

void cudaGetAvg(const CudaImage<float>& src, CudaImage<float>& avgim, CudaImage<float> *buf)
{
  MemoryPolicy mp = src.getMemoryPolicy();
  const int dev = src.getMemoryDevice();

  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(mp != HOST_MEMORY);

  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> tmp;
  if(avgim.size() != 1 || avgim.getMemoryDevice() != dev || avgim.getMemoryPolicy() != mp)
    avgim = CudaImage<float>(1,1,NO_INIT,mp, dev);

  if(buf == 0)
  {
    // Set up output image memory
    tmp = CudaImage<float>(Dims(iDivUp(src.size(),tile.sz()),1), NO_INIT, src.getMemoryPolicy(), dev);
    buf = &tmp;
  }

  // Call CUDA implementation
  cuda_c_getAvg(src.getCudaArrayPtr(), avgim.getCudaArrayPtr(), buf->getCudaArrayPtr(), tile.sz(), src.size());

}

CudaImage<float> cudaGetAvg(const CudaImage<float>& src)
{
  MemoryPolicy mp = src.getMemoryPolicy();
  const int dev = src.getMemoryDevice();

  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(mp != HOST_MEMORY);

  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> avgim = CudaImage<float>(1,1,NO_INIT,mp, dev);

  // Set up output image memory
  CudaImage<float> buf = CudaImage<float>(Dims(iDivUp(src.size(),tile.sz()),1), NO_INIT, src.getMemoryPolicy(), dev);

  // Call CUDA implementation
  cuda_c_getAvg(src.getCudaArrayPtr(), avgim.getCudaArrayPtr(), buf.getCudaArrayPtr(), tile.sz(), src.size());
  return avgim;
}


CudaImage<float> cudaGetSum(const CudaImage<float>& src)
{
  MemoryPolicy mp = src.getMemoryPolicy();
  const int dev = src.getMemoryDevice();

  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(mp != HOST_MEMORY);

  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> sumim = CudaImage<float>(1,1,NO_INIT,mp, dev);

  // Set up output image memory
  CudaImage<float> buf = CudaImage<float>(Dims(iDivUp(src.size(),tile.sz()),1), NO_INIT, src.getMemoryPolicy(), dev);

  // Call CUDA implementation
  cuda_c_getSum(src.getCudaArrayPtr(), sumim.getCudaArrayPtr(), buf.getCudaArrayPtr(), tile.sz(), src.size());
  return sumim;
}

CudaImage<float> cudaSquared(const CudaImage<float>& src)
{
  MemoryPolicy mp = src.getMemoryPolicy();
  const int dev = src.getMemoryDevice();

  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(mp != HOST_MEMORY);

  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> res = CudaImage<float>(src.getDims(),NO_INIT,mp,dev);

  // Call CUDA implementation
  cuda_c_squared(src.getCudaArrayPtr(), res.getCudaArrayPtr(), tile.sz(), src.size());
  return res;
}

CudaImage<float> cudaSqrt(const CudaImage<float>& src)
{
  MemoryPolicy mp = src.getMemoryPolicy();
  const int dev = src.getMemoryDevice();

  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(mp != HOST_MEMORY);

  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> res = CudaImage<float>(src.getDims(),NO_INIT,mp,dev);

  // Call CUDA implementation
  cuda_c_sqrt(src.getCudaArrayPtr(), res.getCudaArrayPtr(), tile.sz(), src.size());
  return res;
}

void cudaGetMinMax(const CudaImage<float>& src, CudaImage<float>& minim, CudaImage<float>& maxim, CudaImage<float> *buf)
{
  const MemoryPolicy mp = src.getMemoryPolicy();
  const int dev = src.getMemoryDevice();
  // Ensure that the data is valid
  ASSERT(src.initialized());
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> tmp;
  if(buf == 0)
  {
    // Save time by only allocating mem once
    tmp = CudaImage<float>(Dims(iDivUp(src.size(),tile.sz()),1), NO_INIT, mp, dev);
    buf = &tmp;
  }
  cudaGetMin(src,minim,buf);
  cudaGetMax(src,maxim,buf);
}


void cudaGetMinMaxAvg(const CudaImage<float>& src, CudaImage<float>& minim, CudaImage<float>& maxim, CudaImage<float>& avgim, CudaImage<float> *buf)
{
  const int dev = src.getMemoryDevice();
  const MemoryPolicy mp = src.getMemoryPolicy();
  // Ensure that the data is valid
  ASSERT(src.initialized());
  ASSERT(mp != HOST_MEMORY);

  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> tmp;
  if(buf == 0)
  {
    // Save time by only allocating mem once
    tmp = CudaImage<float>(Dims(iDivUp(src.size(),tile.sz()),1), NO_INIT, mp, dev);
    buf = &tmp;
  }
  cudaGetMin(src,minim,buf);
  cudaGetMax(src,maxim,buf);
  cudaGetAvg(src,avgim,buf);
}

// Extract a single value from a 1x1 CudaImage
template <class T> T cudaGetScalar(const CudaImage<T>& src)
{
  ASSERT(src.size() == 1);
  Image<T> im = src.exportToImage();
  return im.getVal(0,0);
}

template float cudaGetScalar(const CudaImage<float>& src);
template PixRGB<float> cudaGetScalar(const CudaImage<PixRGB<float> >& src);
template int cudaGetScalar(const CudaImage<int>& src);

void cudaFindMin(const CudaImage<float>& src, Point2D<int>& p, float& val)
{
  const int dev = src.getMemoryDevice();
  const MemoryPolicy mp = src.getMemoryPolicy();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  // Ensure that the data is valid
  ASSERT(src.initialized());
  ASSERT(mp != HOST_MEMORY);
  CudaImage<int> tmp = CudaImage<int>(Dims(iDivUp(src.size(),tile.sz()),1), NO_INIT, mp, dev);
  CudaImage<float> buf = CudaImage<float>(Dims(iDivUp(src.size(),tile.sz()),1), NO_INIT, src.getMemoryPolicy(), dev);
  cuda_c_findMin(src.getCudaArrayPtr(),buf.getCudaArrayPtr(),tmp.getCudaArrayPtr(),tile.sz(),src.size());
  Image<int> resLoc = tmp.exportToImage();
  Image<float> res = buf.exportToImage();
  int idx = resLoc.getVal(0,0);
  val = res.getVal(0,0);
  int x,y;
  y = idx / src.getWidth();
  x = idx % src.getWidth();
  p = Point2D<int>(x,y);
}

void cudaFindMax(const CudaImage<float>& src, Point2D<int>& p, float& val)
{
  const int dev = src.getMemoryDevice();
  const MemoryPolicy mp = src.getMemoryPolicy();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  // Ensure that the data is valid
  ASSERT(src.initialized());
  ASSERT(mp != HOST_MEMORY);
  CudaImage<int> tmp = CudaImage<int>(Dims(iDivUp(src.size(),tile.sz()),1), NO_INIT, mp, dev);
  CudaImage<float> buf = CudaImage<float>(Dims(iDivUp(src.size(),tile.sz()),1), NO_INIT, src.getMemoryPolicy(), dev);
  cuda_c_findMax(src.getCudaArrayPtr(),buf.getCudaArrayPtr(),tmp.getCudaArrayPtr(),tile.sz(),src.size());
  Image<int> resLoc = tmp.exportToImage();
  Image<float> res = buf.exportToImage();
  int idx = resLoc.getVal(0,0);
  val = res.getVal(0,0);
  int x,y;
  y = idx / src.getWidth();
  x = idx % src.getWidth();
  p = Point2D<int>(x,y);
}


// ######################################################################
void cudaInplaceNormalize(CudaImage<float>& dst, const float nmin, const float nmax)
{
  ASSERT(dst.initialized());
  const int dev = dst.getMemoryDevice();
  const MemoryPolicy mp = dst.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  if (!dst.initialized()) return;
  CudaImage<float> oldmin,oldmax;
  cudaGetMin(dst, oldmin);
  cudaGetMax(dst, oldmax);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_inplaceNormalize(dst.getCudaArrayPtr(), oldmin.getCudaArrayPtr(), oldmax.getCudaArrayPtr(), nmin, nmax,tile.sz(),dst.size());
}

void cudaInplaceRectify(CudaImage<float>& dst)
{
  ASSERT(dst.initialized());
  if (!dst.initialized()) return;
  const int dev = dst.getMemoryDevice();
  const MemoryPolicy mp = dst.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_inplaceRectify(dst.getCudaArrayPtr(), tile.sz(),dst.size());
}

void cudaInplaceClamp(CudaImage<float>& dst, const float cmin, const float cmax)
{
  ASSERT(dst.initialized());
  if (!dst.initialized()) return;
  const int dev = dst.getMemoryDevice();
  const MemoryPolicy mp = dst.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_inplaceClamp(dst.getCudaArrayPtr(),cmin,cmax,tile.sz(),dst.size());
}

void cudaClear(CudaImage<float>& dst, const float val)
{
  const int dev = dst.getMemoryDevice();
  const MemoryPolicy mp = dst.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_clear(dst.getCudaArrayPtr(),val,tile.sz(),dst.size());
}

void cudaAbs(CudaImage<float>& src)
{
  const int dev = src.getMemoryDevice();
  const MemoryPolicy mp = src.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_abs(src.getCudaArrayPtr(),tile.sz(),src.size());
}

void cudaInplaceAddScalar(CudaImage<float>& dst, const CudaImage<float>& offset)
{
  ASSERT(dst.initialized());
  const MemoryPolicy mp = dst.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  const int dev = dst.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_inplaceAddScalar(dst.getCudaArrayPtr(), offset.getCudaArrayPtr(), tile.sz(), dst.size());
}

void cudaInplaceSubtractScalar(CudaImage<float>& dst, const CudaImage<float>& offset)
{
  ASSERT(dst.initialized());
  const int dev = dst.getMemoryDevice();
  const MemoryPolicy mp = dst.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_inplaceSubtractScalar(dst.getCudaArrayPtr(), offset.getCudaArrayPtr(), tile.sz(), dst.size());
}

void cudaInplaceMultiplyScalar(CudaImage<float>& dst, const CudaImage<float>& offset)
{
  ASSERT(dst.initialized());
  const int dev = dst.getMemoryDevice();
  const MemoryPolicy mp = dst.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_inplaceMultiplyScalar(dst.getCudaArrayPtr(), offset.getCudaArrayPtr(), tile.sz(), dst.size());
}

void cudaInplaceDivideScalar(CudaImage<float>& dst, const CudaImage<float>& offset)
{
  ASSERT(dst.initialized());
  const int dev = dst.getMemoryDevice();
  const MemoryPolicy mp = dst.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_inplaceDivideScalar(dst.getCudaArrayPtr(), offset.getCudaArrayPtr(), tile.sz(), dst.size());
}

void cudaInplaceAddImages(CudaImage<float>& im1, const CudaImage<float>& im2)
{
  ASSERT(im1.initialized() && im2.initialized());
  ASSERT(im1.getMemoryDevice() == im2.getMemoryDevice());
  ASSERT(im1.size() == im2.size());
  const int dev = im1.getMemoryDevice();
  const MemoryPolicy mp = im1.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_inplaceAddImages(im1.getCudaArrayPtr(), im2.getCudaArrayPtr(), tile.sz(), im1.size());
}

void cudaInplaceSubtractImages(CudaImage<float>& im1, const CudaImage<float>& im2)
{
  ASSERT(im1.initialized() && im2.initialized());
  ASSERT(im1.getMemoryDevice() == im2.getMemoryDevice());
  ASSERT(im1.size() == im2.size());
  const int dev = im1.getMemoryDevice();
  const MemoryPolicy mp = im1.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_inplaceSubtractImages(im1.getCudaArrayPtr(), im2.getCudaArrayPtr(), tile.sz(), im1.size());
}

void cudaInplaceMultiplyImages(CudaImage<float>& im1, const CudaImage<float>& im2)
{
  ASSERT(im1.initialized() && im2.initialized());
  ASSERT(im1.getMemoryDevice() == im2.getMemoryDevice());
  ASSERT(im1.size() == im2.size());
  const int dev = im1.getMemoryDevice();
  const MemoryPolicy mp = im1.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_inplaceMultiplyImages(im1.getCudaArrayPtr(), im2.getCudaArrayPtr(), tile.sz(), im1.size());
}

void cudaInplaceDivideImages(CudaImage<float>& im1, const CudaImage<float>& im2)
{
  ASSERT(im1.initialized() && im2.initialized());
  ASSERT(im1.getMemoryDevice() == im2.getMemoryDevice());
  ASSERT(im1.size() == im2.size());
  const int dev = im1.getMemoryDevice();
  const MemoryPolicy mp = im1.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_inplaceDivideImages(im1.getCudaArrayPtr(), im2.getCudaArrayPtr(), tile.sz(), im1.size());
}

CudaImage<float> cudaAddImages(const CudaImage<float>& im1, const CudaImage<float>& im2)
{
  ASSERT(im1.initialized() && im2.initialized());
  ASSERT(im1.getMemoryDevice() == im2.getMemoryDevice());
  ASSERT(im1.size() == im2.size());
  MemoryPolicy mp = im1.getMemoryPolicy();
  const int dev = im1.getMemoryDevice();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> res = CudaImage<float>(im1.getDims(),NO_INIT,mp,dev);
  cuda_c_addImages(im1.getCudaArrayPtr(), im2.getCudaArrayPtr(), res.getCudaArrayPtr(), tile.sz(), im1.size());
  return res;
}

CudaImage<float> cudaSubtractImages(const CudaImage<float>& im1, const CudaImage<float>& im2)
{
  ASSERT(im1.initialized() && im2.initialized());
  ASSERT(im1.getMemoryDevice() == im2.getMemoryDevice());
  ASSERT(im1.size() == im2.size());
  MemoryPolicy mp = im1.getMemoryPolicy();
  const int dev = im1.getMemoryDevice();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> res = CudaImage<float>(im1.getDims(),NO_INIT,mp,dev);
  cuda_c_subtractImages(im1.getCudaArrayPtr(), im2.getCudaArrayPtr(), res.getCudaArrayPtr(), tile.sz(), im1.size());
  return res;
}

CudaImage<float> cudaMultiplyImages(const CudaImage<float>& im1, const CudaImage<float>& im2)
{
  ASSERT(im1.initialized() && im2.initialized());
  ASSERT(im1.getMemoryDevice() == im2.getMemoryDevice());
  ASSERT(im1.size() == im2.size());
  MemoryPolicy mp = im1.getMemoryPolicy();
  const int dev = im1.getMemoryDevice();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> res = CudaImage<float>(im1.getDims(),NO_INIT,mp,dev);
  cuda_c_multiplyImages(im1.getCudaArrayPtr(), im2.getCudaArrayPtr(), res.getCudaArrayPtr(), tile.sz(), im1.size());
  return res;
}

CudaImage<float> cudaDivideImages(const CudaImage<float>& im1, const CudaImage<float>& im2)
{
  ASSERT(im1.initialized() && im2.initialized());
  ASSERT(im1.getMemoryDevice() == im2.getMemoryDevice());
  ASSERT(im1.size() == im2.size());
  MemoryPolicy mp = im1.getMemoryPolicy();
  const int dev = im1.getMemoryDevice();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> res = CudaImage<float>(im1.getDims(),NO_INIT,mp,dev);
  cuda_c_divideImages(im1.getCudaArrayPtr(), im2.getCudaArrayPtr(), res.getCudaArrayPtr(), tile.sz(), im1.size());
  return res;
}

CudaImage<float> cudaTakeMax(const CudaImage<float>& im1, const CudaImage<float>& im2)
{
  ASSERT(im1.initialized() && im2.initialized());
  ASSERT(im1.getMemoryDevice() == im2.getMemoryDevice());
  ASSERT(im1.size() == im2.size());
  MemoryPolicy mp = im1.getMemoryPolicy();
  const int dev = im1.getMemoryDevice();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> res = CudaImage<float>(im1.getDims(),NO_INIT,mp,dev);
  cuda_c_takeMax(im1.getCudaArrayPtr(), im2.getCudaArrayPtr(), res.getCudaArrayPtr(), tile.sz(), im1.size());
  return res;
}


// ######################################################################
CudaImage<float> cudaQuadEnergy(const CudaImage<float>& real, const CudaImage<float>& imag)
{
  ASSERT(real.initialized() && imag.initialized());
  ASSERT(real.getMemoryDevice() == imag.getMemoryDevice());
  ASSERT(real.isSameSize(imag));
  MemoryPolicy mp = real.getMemoryPolicy();
  const int dev = real.getMemoryDevice();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> out(real.getDims(), NO_INIT,mp,dev);
  cuda_c_quadEnergy(real.getCudaArrayPtr(), imag.getCudaArrayPtr(), out.getCudaArrayPtr(), tile.sz(), real.size());

  return out;
}


void cudaInplaceAttenuateBorders(CudaImage<float>& a, int size)
{
  ASSERT(a.initialized());

  Dims dims = a.getDims();

  if (size * 2 > dims.w()) size = dims.w() / 2;
  if (size * 2 > dims.h()) size = dims.h() / 2;
  if (size < 1) return;  // forget it
  const int dev = a.getMemoryDevice();
  MemoryPolicy mp = a.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  cuda_c_inplaceAttenuateBorders(a.getCudaArrayPtr(), size, tile.sz(), a.getWidth(), a.getHeight());
}


