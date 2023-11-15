/*!@file CUDA/CudaLowPass.C C++ wrapper for CUDA Low pass operations */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaLowPass.C $
// $Id: CudaLowPass.C 12962 2010-03-06 02:13:53Z irock $
//

#include "CUDA/CudaImage.H"
#include "CUDA/CudaDevices.H"
#include "CUDA/CudaLowPass.H"
#include "wrap_c_cuda.h"


CudaImage<float> cudaLowPass5xDec(const CudaImage<float>&src)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  // Set up output image memory

  CudaImage<float> result = CudaImage<float>(src.getWidth()/2, src.getHeight(), NO_INIT, src.getMemoryPolicy(), dev);

  const Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  cuda_c_lowpass_5_x_dec_x(src.getCudaArrayPtr(),src.getWidth(),src.getHeight(),result.getCudaArrayPtr(),tile.w());

  return result;
}


CudaImage<float> cudaLowPass5yDec(const CudaImage<float>&src)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  // Set up output image memory

  CudaImage<float> result = CudaImage<float>(src.getWidth(), src.getHeight()/2, NO_INIT, src.getMemoryPolicy(), dev);

  const Dims tile = CudaDevices::getDeviceTileSize(dev);

  cuda_c_lowpass_5_y_dec_y(src.getCudaArrayPtr(),src.getWidth(),src.getHeight(),result.getCudaArrayPtr(),tile.w(),tile.h());
  return result;
}

CudaImage<float> cudaLowPass5Dec(const CudaImage<float>& src, const bool go_x, const bool go_y)
{
  CudaImage<float> result = src;
  if(go_x) result = cudaLowPass5xDec(result);
  if(go_y) result = cudaLowPass5yDec(result);
  return result;
}

CudaImage<float> cudaLowPass9xDec(const CudaImage<float>&src)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  // Set up output image memory

  const int rw = src.getWidth()/2;
  const int rh = src.getHeight();

  CudaImage<float> result = CudaImage<float>(rw,rh, NO_INIT, src.getMemoryPolicy(), dev);

  const Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  cuda_c_lowpass_9_x_dec_x(src.getCudaArrayPtr(),src.getWidth(),src.getHeight(),result.getCudaArrayPtr(),rw,rh,tile.w());

  return result;
}


CudaImage<float> cudaLowPass9yDec(const CudaImage<float>&src)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  // Set up output image memory

  const int rw = src.getWidth();
  const int rh = src.getHeight()/2;

  CudaImage<float> result = CudaImage<float>(rw,rh, NO_INIT, src.getMemoryPolicy(), dev);

  const Dims tile = CudaDevices::getDeviceTileSize(dev);

  cuda_c_lowpass_9_y_dec_y(src.getCudaArrayPtr(),src.getWidth(),src.getHeight(),result.getCudaArrayPtr(),rw,rh,tile.w(),tile.h());
  return result;
}

CudaImage<float> cudaLowPass9Dec(const CudaImage<float>& src, const bool go_x, const bool go_y)
{
  CudaImage<float> result = src;
  if(go_x) result = cudaLowPass9xDec(result);
  if(go_y) result = cudaLowPass9yDec(result);
  return result;
}



CudaImage<float> cudaLowPass9x(const CudaImage<float>&src)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  // Set up output image memory

  CudaImage<float> result = CudaImage<float>(src.getWidth(), src.getHeight(), NO_INIT, src.getMemoryPolicy(), dev);

  const Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  cuda_c_lowpass_9_x(src.getCudaArrayPtr(),src.getWidth(),src.getHeight(),result.getCudaArrayPtr(),tile.w());

  return result;
}


CudaImage<float> cudaLowPass9xyDec(const CudaImage<float>&src)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  // Set up output image memory
  const int rw = src.getWidth()/2;
  const int rh = src.getHeight()/2;
  CudaImage<float> tmp = CudaImage<float>(rw, src.getHeight(), ZEROS, src.getMemoryPolicy(), dev);

  const Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_lowpass_texture_9_x_dec_x(src.getCudaArrayPtr(),src.getWidth(),src.getHeight(),tmp.getCudaArrayPtr(),rw,src.getHeight(),tile.sz(),1);

  CudaImage<float> res = CudaImage<float>(rw, rh, ZEROS, src.getMemoryPolicy(), dev);
  cuda_c_lowpass_texture_9_y_dec_y(tmp.getCudaArrayPtr(),tmp.getWidth(),tmp.getHeight(),res.getCudaArrayPtr(),rw,rh,1,tile.sz());
  return res;
}



CudaImage<float> cudaLowPass9y(const CudaImage<float>&src)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  // Set up output image memory

  CudaImage<float> result = CudaImage<float>(src.getWidth(), src.getHeight(), NO_INIT, src.getMemoryPolicy(), dev);

  const Dims tile = CudaDevices::getDeviceTileSize(dev);

  cuda_c_lowpass_9_y(src.getCudaArrayPtr(),src.getWidth(),src.getHeight(),result.getCudaArrayPtr(),tile.w(),tile.h());
  return result;
}

CudaImage<float> cudaLowPass5x(const CudaImage<float>&src)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  // Set up output image memory

  CudaImage<float> result = CudaImage<float>(src.getWidth(), src.getHeight(), NO_INIT, src.getMemoryPolicy(), dev);

  const Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  cuda_c_lowpass_5_x(src.getCudaArrayPtr(),src.getWidth(),src.getHeight(),result.getCudaArrayPtr(),tile.w());

  return result;
}


CudaImage<float> cudaLowPass5y(const CudaImage<float>&src)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  // Set up output image memory

  CudaImage<float> result = CudaImage<float>(src.getWidth(), src.getHeight(), NO_INIT, src.getMemoryPolicy(), dev);

  const Dims tile = CudaDevices::getDeviceTileSize(dev);

  cuda_c_lowpass_5_y(src.getCudaArrayPtr(),src.getWidth(),src.getHeight(),result.getCudaArrayPtr(),tile.w(),tile.h());
  return result;
}

CudaImage<float> cudaLowPass3x(const CudaImage<float>&src)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  // Set up output image memory

  CudaImage<float> result = CudaImage<float>(src.getWidth(), src.getHeight(), NO_INIT, src.getMemoryPolicy(), dev);

  const Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  cuda_c_lowpass_3_x(src.getCudaArrayPtr(),src.getWidth(),src.getHeight(),result.getCudaArrayPtr(),tile.w());

  return result;
}


CudaImage<float> cudaLowPass3y(const CudaImage<float>&src)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  // Set up output image memory

  CudaImage<float> result = CudaImage<float>(src.getWidth(), src.getHeight(), NO_INIT, src.getMemoryPolicy(), dev);

  const Dims tile = CudaDevices::getDeviceTileSize(dev);

  cuda_c_lowpass_3_y(src.getCudaArrayPtr(),src.getWidth(),src.getHeight(),result.getCudaArrayPtr(),tile.w(),tile.h());
  return result;
}

CudaImage<float> cudaLowPass9(const CudaImage<float>& src, const bool go_x, const bool go_y)
{
  CudaImage<float> result = src;
  if(go_x) result = cudaLowPass9x(result);
  if(go_y) result = cudaLowPass9y(result);
  return result;
}

// ######################################################################
CudaImage<float> cudaLowPass(const int N, const CudaImage<float>& src, const bool go_x, const bool go_y)
{
  CudaImage<float> result = src;
  if (go_x) result = cudaLowPassX(N,result);
  if (go_y) result = cudaLowPassY(N,result);
  return result;
}

// ######################################################################
CudaImage<float> cudaLowPassX(const int N, const CudaImage<float>& src)
{
  switch (N)
    {
    case 3: return cudaLowPass3x(src);
    case 5: return cudaLowPass5x(src);
    case 9: return cudaLowPass9x(src);
    default:
      LERROR("Only 3,5, and 9 tap kernels implemented");
      return CudaImage<float>();
      break;
    }
//   const Image<float> kern = binomialKernel(N);
//   ASSERT(kern.getWidth() == N);
//   ASSERT(kern.getHeight() == 1);
//   return sepFilter(src, kern.getArrayPtr(), NULL, N, 0,
//                    CONV_BOUNDARY_CLEAN);
}

// ######################################################################
CudaImage<float> cudaLowPassY(const int N, const CudaImage<float>& src)
{
  switch (N)
    {
    case 3: return cudaLowPass3y(src);
    case 5: return cudaLowPass5y(src);
    case 9: return cudaLowPass9y(src);
    default:
      LERROR("Only 3,5, and 9 tap kernels implemented");
      return CudaImage<float>();
      break;
    }

//   const Image<float> kern = binomialKernel(N);
//   ASSERT(kern.getWidth() == N);
//   ASSERT(kern.getHeight() == 1);
//   return sepFilter(src, NULL, kern.getArrayPtr(), 0, N,
//                    CONV_BOUNDARY_CLEAN);
}
