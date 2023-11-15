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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaFilterOps.C $
// $Id: CudaFilterOps.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Image/Rectangle.H"
#include "CUDA/CudaImage.H"
#include "Util/Assert.H"
#include "CUDA/cudadefs.h"
#include "CUDA/CudaMathOps.H"
#include "CUDA/CudaLowPass.H"
#include "CUDA/CudaShapeOps.H"
#include "CUDA/CudaCutPaste.H"
#include "CUDA/CudaKernels.H"
#include "CUDA/CudaConvolutions.H"
#include "CUDA/CudaNorm.H"
#include "CudaFilterOps.H"
#include "CudaDevices.H"
#include "Util/Timer.H"
#include "wrap_c_cuda.h"

// ######################################################################
CudaImage<float> cudaOrientedFilter(const CudaImage<float>& src, const float k,
                                    const float theta, const float intensity)
{
  double kx = double(k) * cos((theta + 90.0) * M_PI / 180.0);
  double ky = double(k) * sin((theta + 90.0) * M_PI / 180.0);
  MemoryPolicy mp = src.getMemoryPolicy();
  int dev = src.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> re(src.getDims(), NO_INIT, mp, dev);
  CudaImage<float> im(src.getDims(), NO_INIT, mp, dev);

  cuda_c_orientedFilter(src.getCudaArrayPtr(),re.getCudaArrayPtr(),im.getCudaArrayPtr(),(float)kx,(float)ky,intensity,src.getWidth(),src.getHeight(),tile.sz());

  re = cudaLowPass9(re);
  im = cudaLowPass9(im);

  return cudaQuadEnergy(re, im);
}


// ######################################################################
CudaImage<float> cudaCenterSurround(const CudaImage<float>& center, const CudaImage<float>& surround,
                                    const bool absol)
{
  ASSERT(center.initialized() && surround.initialized());
  ASSERT(center.getMemoryDevice() == surround.getMemoryDevice());

  const int lw = center.getWidth(), lh = center.getHeight();
  const int sw = surround.getWidth(), sh = surround.getHeight();

  if (sw > lw || sh > lh) LFATAL("center must be larger than surround");

  MemoryPolicy mp = center.getMemoryPolicy();
  int dev = center.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  // result has the size of the larger image:
  CudaImage<float> result(center.getDims(), NO_INIT, mp, dev);
  // scan large image and subtract corresponding pixel from small image:

  if(absol)
    cuda_c_centerSurroundAbs(center.getCudaArrayPtr(), surround.getCudaArrayPtr(), result.getCudaArrayPtr(), lw, lh, sw, sh, tile.sz());
  else
    cuda_c_centerSurroundClamped(center.getCudaArrayPtr(), surround.getCudaArrayPtr(), result.getCudaArrayPtr(), lw, lh, sw, sh, tile.sz());

  // attenuate borders:
  cudaInplaceAttenuateBorders(result, result.getDims().max() / 20);


  return result;

}


void cudaCenterSurround(const CudaImage<float>& center, const CudaImage<float>& surround,
                    CudaImage<float>& pos, CudaImage<float>& neg)
{

  const int lw = center.getWidth(), lh = center.getHeight();
  const int sw = surround.getWidth(), sh = surround.getHeight();

  if (sw > lw || sh > lh) LFATAL("center must be larger than surround");

  MemoryPolicy mp = center.getMemoryPolicy();
  int dev = center.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  // result has the size of the larger image:
  pos = CudaImage<float>(center.getDims(), NO_INIT,mp,dev);
  neg = CudaImage<float>(center.getDims(), NO_INIT,mp,dev);

  cuda_c_centerSurroundDirectional(center.getCudaArrayPtr(), surround.getCudaArrayPtr(), pos.getCudaArrayPtr(), neg.getCudaArrayPtr(),
                                   lw,lh,sw,sh,tile.sz());
  // attenuate borders:
  cudaInplaceAttenuateBorders(pos, pos.getDims().max() / 20);
  cudaInplaceAttenuateBorders(neg, neg.getDims().max() / 20);
}


// ######################################################################
CudaImage<float> cudaDoubleOpp(const CudaImage<float>& cplus, const CudaImage<float>& cminus,
                               const CudaImage<float>& splus, const CudaImage<float>& sminus)
{
  ASSERT(cplus.isSameSize(cminus)); ASSERT(splus.isSameSize(sminus));


  // compute difference between both center arrays
  CudaImage<float> cdiff = cplus;
  cdiff -= cminus;

  // compute difference between both surround arrays
  CudaImage<float> sdiff = splus;
  sdiff -= sminus;

  // compute center-surround cdiff - sdiff = (cp - cm) [-] (sp - sm)
  return cudaCenterSurround(cdiff, sdiff, true);  // take abs
}


// ######################################################################
CudaImage<float> cudaCenterSurroundAbsDownScaleNormalize(const CudaImage<float>& center, const CudaImage<float>& surround,
                                                         const bool absol, int newWidth, int newHeight, float mi, float ma, int nIter, float weakness)
{


  ASSERT(center.initialized() && surround.initialized());
  ASSERT(center.getMemoryDevice() == surround.getMemoryDevice());

  const int lw = center.getWidth(), lh = center.getHeight();
  const int sw = surround.getWidth(), sh = surround.getHeight();

  if (sw > lw || sh > lh) LFATAL("center must be larger than surround");

  MemoryPolicy mp = center.getMemoryPolicy();
  int dev = center.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize(dev);


  // result has the size of the larger image:
  CudaImage<float> result(center.getDims(), NO_INIT, mp, dev);

  // Attenuate at the edge of the image of this border size
  const int attBorder = std::max(lw, lh)/20;
  //T zero = T();
  cuda_c_centerSurroundAbsAttenuate(center.getCudaArrayPtr(), surround.getCudaArrayPtr(), result.getCudaArrayPtr(), lw, lh, sw, sh, attBorder, tile.w(), tile.h());

  result = cudaDownSize(result, newWidth, newHeight);
  cudaInplaceRectify(result);

  // then, normalize between mi and ma if not zero
  if (mi != 0.0F || ma != 0.0F) cudaInplaceNormalize(result, mi, ma);

  // Normalize using fancy normalization: multiple iterations of
  // filtering by a DoG

  const int w = result.getWidth();
  const int h = result.getHeight();
  int siz = std::max(w, h);
  int maxhw = std::max(0, std::min(w, h) / 2 - 1);

  // Allocate max buffer so that it can be reused
  CudaImage<float> maxim = CudaImage<float>(1,1,NO_INIT,mp,dev);
  // first clamp negative values to zero


  // build separable Gaussians for DoG computation:
  float esig = (float(siz) * FANCYESIG) * 0.01F;
  float isig = (float(siz) * FANCYISIG) * 0.01F;
  CudaImage<float> gExc = cudaGaussian(mp,dev,FANCYCOEX/(esig*sqrt(2.0*M_PI)) * weakness, esig, maxhw);
  CudaImage<float> gInh = cudaGaussian(mp,dev,FANCYCOIN/(isig*sqrt(2.0*M_PI)) * weakness, isig, maxhw);

  for (int i = 0; i < nIter; ++i)
    {

      CudaImage<float> excit = cudaSepFilter(result, gExc, gExc, CONV_BOUNDARY_CLEAN); // excitatory part
      CudaImage<float> inhib = cudaSepFilter(result, gInh, gInh, CONV_BOUNDARY_CLEAN); // inhibitory part

      // result = result + excit - inhib + maxim*(-0.01F*FANCYINHI)
      excit -= inhib;
      cudaGetMax(result, maxim);

      result += excit;        // we get broad inhibition from everywhere

      result += cudaGetScalar(maxim)*(-0.01F * FANCYINHI) ;    // we get fixed global inhibition

      cudaInplaceRectify(result);

      //      sigmoid(FANCYG, FANCYH, FANCYS);
    }


  return result;

}


// ######################################################################
CudaImage<float> cudaSpatialPoolMax(const CudaImage<float>& src, const int si, const int sj,
               const int sti, const int stj)
{
  const int w = src.getWidth(), h = src.getHeight();
  MemoryPolicy mp = src.getMemoryPolicy();
  int dev = src.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize(dev);

  // This is confusing... we need to oversize the result, so that the intermediate results can be stored
  // in between cuda kernels
  CudaImage<float> buf1(src.getSize(),1, NO_INIT,mp,dev);
  CudaImage<float> buf2(src.getSize(),1, NO_INIT,mp,dev);
  int res_w = (int)ceil((float)w / (float)si);
  int res_h =  (int)ceil((float)h / (float)sj);
  CudaImage<float> result(res_w,res_h,NO_INIT,mp,dev);
  cuda_c_spatialPoolMax(src.getCudaArrayPtr(),result.getCudaArrayPtr(),buf1.getCudaArrayPtr(),buf2.getCudaArrayPtr(),w,h,si,sj,sti,stj,tile.w(),tile.h());
  return result;
}


