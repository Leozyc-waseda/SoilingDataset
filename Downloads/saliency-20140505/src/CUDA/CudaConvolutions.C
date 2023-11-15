/*!@file CUDA/CudaConvolutions.C C++ wrapper for CUDA convolution methods */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaConvolutions.C $
// $Id: CudaConvolutions.C 12962 2010-03-06 02:13:53Z irock $
//

#include "CUDA/CudaImage.H"
#include "Util/Assert.H"
#include "CUDA/cudadefs.h"
#include "Image/Convolutions.H"
#include "CudaConvolutions.H"
#include "CudaDevices.H"
#include "wrap_c_cuda.h"
#include <algorithm>

// ######################################################################

CudaImage<float> cudaOptConvolve(const CudaImage<float>& src, const CudaImage<float>& f)
{
  MemoryPolicy mp = src.getMemoryPolicy();
  int dev = src.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize(dev);

  ASSERT(src.initialized());
  ASSERT(mp != HOST_MEMORY);
  const int src_w = src.getWidth();
  const int src_h = src.getHeight();

  const int fil_w = f.getWidth();
  const int fil_h = f.getHeight();

  ASSERT((fil_w & 1) && (fil_h & 1));
  CudaImage<float> result = CudaImage<float>(src_w, src_h, NO_INIT,mp,dev);

  cuda_c_optConvolve(result.getCudaArrayPtr(),src.getCudaArrayPtr(),src_w,src_h,f.getCudaArrayPtr(),fil_w,fil_h,tile.w(),tile.h());
  return result;
}


// ######################################################################

CudaImage<float> cudaConvolveZeroHelper(const CudaImage<float>& src, const CudaImage<float>& filter,
                                        const int Nx, const int Ny, bool runOptimized)
{
  MemoryPolicy mp = src.getMemoryPolicy();
  int dev = src.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize(dev);

  ASSERT(src.initialized()); //ASSERT((Nx & 1)  && (Ny & 1));
  ASSERT(mp != HOST_MEMORY);
  const int src_w = src.getWidth(), src_h = src.getHeight();
  CudaImage<float> result = CudaImage<float>(src_w, src_h, NO_INIT,mp,dev);

  int mem_size = CudaDevices::getDeviceSharedMemorySize(dev)/int(sizeof(float));
  // Decide whether we can run optimized versions based on size of filter
  if(runOptimized && mem_size < Nx*Ny+(tile.w()+Nx)*(tile.h()+Ny))
  {
    printf("Unable to run convolveZeroHelper optimized\n");
    runOptimized = false;
  }

  if(runOptimized)
    cuda_c_convolveZeroHelperOptimized(result.getCudaArrayPtr(),src.getCudaArrayPtr(),src_w,src_h,filter.getCudaArrayPtr(),Nx,Ny,tile.w(),tile.h());
  else
    cuda_c_convolveZeroHelper(result.getCudaArrayPtr(),src.getCudaArrayPtr(),src_w,src_h,filter.getCudaArrayPtr(),Nx,Ny,tile.w(),tile.h());

  return result;
}

// ######################################################################

CudaImage<float> cudaConvolveCleanHelper(const CudaImage<float>& src, const CudaImage<float>& filter,
                                         const int Nx, const int Ny, bool runOptimized)
{
  MemoryPolicy mp = src.getMemoryPolicy();
  int dev = src.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize(dev);

  ASSERT(src.initialized()); //ASSERT((Nx & 1)  && (Ny & 1));
  ASSERT(mp != HOST_MEMORY);
  const int src_w = src.getWidth(), src_h = src.getHeight();
  CudaImage<float> result = CudaImage<float>(src_w, src_h, NO_INIT,mp,dev);

  cuda_c_convolveCleanHelper(result.getCudaArrayPtr(),src.getCudaArrayPtr(),src_w,src_h,filter.getCudaArrayPtr(),Nx,Ny,tile.w(),tile.h());

  return result;
}

// ######################################################################
CudaImage<float> cudaConvolveHmax(const CudaImage<float>& src, const CudaImage<float>& filter, bool runOptimized)
{
  MemoryPolicy mp = src.getMemoryPolicy();
  int dev = src.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize(dev);
  const int Nx = filter.getWidth(), Ny = filter.getHeight();
  ASSERT(src.initialized());
  ASSERT((Nx & 1)  && (Ny & 1));
  ASSERT(mp != HOST_MEMORY);

  const int src_w = src.getWidth(), src_h = src.getHeight();
  CudaImage<float> result = CudaImage<float>(src_w, src_h, NO_INIT,mp,dev);

  int mem_size = CudaDevices::getDeviceSharedMemorySize(dev)/int(sizeof(float));
  // Decide whether we can run optimized versions based on size of filter
  if(runOptimized && mem_size < Nx*Ny+(tile.w()+Nx)*(tile.h()+Ny))
  {
    //printf("Unable to run convolveHmaxHelper optimized Nx %d Ny %d tw %d th %d\n",Nx,Ny,tile.w(),tile.h());
    runOptimized = false;
  }

  if(runOptimized)
    cuda_c_convolveHmaxHelperOptimized(result.getCudaArrayPtr(),src.getCudaArrayPtr(),src_w,src_h,filter.getCudaArrayPtr(),Nx,Ny,tile.w(),tile.h());
  else
    cuda_c_convolveHmaxHelper(result.getCudaArrayPtr(),src.getCudaArrayPtr(),src_w,src_h,filter.getCudaArrayPtr(),Nx,Ny,tile.w(),tile.h());

  return result;

}


// ######################################################################
CudaImage<float> cudaConvolve(const CudaImage<float>& src, const CudaImage<float>& filter,
         const int Nx, const int Ny,
                              ConvolutionBoundaryStrategy boundary, bool runOptimized)
{
  switch (boundary)
    {
    case CONV_BOUNDARY_ZERO:
      return cudaConvolveZeroHelper(src, filter, Nx, Ny, runOptimized);
      break;
    case CONV_BOUNDARY_CLEAN:
      return cudaConvolveCleanHelper(src, filter, Nx, Ny, runOptimized);
      break;
    case CONV_BOUNDARY_REPLICATE:
      // not implemented yet -- pass through to error
    default:
      LFATAL("convolution boundary strategy %d not supported",
             (int) boundary);
    }
  /* can't happen */ return CudaImage<float>();
}

// ######################################################################
CudaImage<float> cudaXFilter(const CudaImage<float>& src, const CudaImage<float>& hFilt, const int hfs,
                             ConvolutionBoundaryStrategy boundary, bool runOptimized)
{
  MemoryPolicy mp = src.getMemoryPolicy();
  int dev = src.getMemoryDevice();
  ASSERT(src.initialized());
  ASSERT(mp != HOST_MEMORY);
  if (hfs == 0)
    {
      return src;  // no filter
    }

  Dims tile = CudaDevices::getDeviceTileSize(dev);

  // Needed for non-optimized functions
  int mem_size = CudaDevices::getDeviceSharedMemorySize(dev)/int(sizeof(float));
  int share_len = std::min(mem_size,hfs);
  // Decide whether we can run optimized versions based on size of filter
  if(runOptimized && mem_size < hfs*2+tile.sz())
  {
    Dims tileOpt = Dims(mem_size-hfs*2,1);
    if(tileOpt.sz() < 16)
      runOptimized = false;
    else
      tile = tileOpt;
  }
  const int w = src.getWidth(), h = src.getHeight();
  CudaImage<float> result = CudaImage<float>(w, h, NO_INIT,mp,dev);

  // *** horizontal pass ***
  if(runOptimized)
    {
      switch(boundary)
        {
        case CONV_BOUNDARY_ZERO:
          cuda_c_optXFilterZero(result.getCudaArrayPtr(),src.getCudaArrayPtr(),w,h,hFilt.getCudaArrayPtr(),hfs,tile.sz());
          break;
        case CONV_BOUNDARY_CLEAN:
          cuda_c_optXFilterClean(result.getCudaArrayPtr(),src.getCudaArrayPtr(),w,h,hFilt.getCudaArrayPtr(),hfs,tile.sz());
          break;
        case CONV_BOUNDARY_REPLICATE:
          cuda_c_optXFilterReplicate(result.getCudaArrayPtr(),src.getCudaArrayPtr(),w,h,hFilt.getCudaArrayPtr(),hfs,tile.sz());
          break;
        default:
          LFATAL("convolution boundary strategy %d not supported",
                 (int) boundary);
          break;
        }
    }
  else
    {
      switch(boundary)
        {
        case CONV_BOUNDARY_ZERO:
          cuda_c_xFilterZero(result.getCudaArrayPtr(),src.getCudaArrayPtr(),w,h,hFilt.getCudaArrayPtr(),hfs,share_len,tile.sz());
          break;
        case CONV_BOUNDARY_CLEAN:
          cuda_c_xFilterClean(result.getCudaArrayPtr(),src.getCudaArrayPtr(),w,h,hFilt.getCudaArrayPtr(),hfs,share_len,tile.sz());
          break;
        case CONV_BOUNDARY_REPLICATE:
          cuda_c_xFilterReplicate(result.getCudaArrayPtr(),src.getCudaArrayPtr(),w,h,hFilt.getCudaArrayPtr(),hfs,share_len,tile.sz());
          break;
        default:
          LFATAL("convolution boundary strategy %d not supported",
                 (int) boundary);
          break;
        }
    }
  return result;
}

// ######################################################################
CudaImage<float> cudaYFilter(const CudaImage<float>& src, const CudaImage<float>& hFilt, const int hfs,
                             ConvolutionBoundaryStrategy boundary, bool runOptimized)
{
  MemoryPolicy mp = src.getMemoryPolicy();
  int dev = src.getMemoryDevice();
  ASSERT(src.initialized());
  ASSERT(mp != HOST_MEMORY);
  if (hfs == 0)
    {
      return src;  // no filter
    }

  Dims tile = CudaDevices::getDeviceTileSize(dev);
  // Needed for non-optimized functions
  int mem_size = CudaDevices::getDeviceSharedMemorySize(dev)/int(sizeof(float));
  int share_len = std::min(mem_size,hfs);

  // Decide whether we can run optimized versions based on size of filter
  if(runOptimized && mem_size < hfs*2+tile.sz())
  {
    // Modifying tile size
    Dims tileOpt = Dims(mem_size-hfs*2,1);
    if(tileOpt.sz() < 16)
      runOptimized = false;
    else
      tile = tileOpt;
  }
  const int w = src.getWidth(), h = src.getHeight();
  CudaImage<float> result = CudaImage<float>(w, h, NO_INIT,mp,dev);

  // *** horizontal pass ***
  if(runOptimized)
    {
      switch(boundary)
        {
        case CONV_BOUNDARY_ZERO:
          cuda_c_optYFilterZero(result.getCudaArrayPtr(),src.getCudaArrayPtr(),w,h,hFilt.getCudaArrayPtr(),hfs,tile.sz());
          break;
        case CONV_BOUNDARY_CLEAN:
          cuda_c_optYFilterClean(result.getCudaArrayPtr(),src.getCudaArrayPtr(),w,h,hFilt.getCudaArrayPtr(),hfs,tile.sz());
          break;
        case CONV_BOUNDARY_REPLICATE:
          cuda_c_optYFilterReplicate(result.getCudaArrayPtr(),src.getCudaArrayPtr(),w,h,hFilt.getCudaArrayPtr(),hfs,tile.sz());
          break;
        default:
          LFATAL("convolution boundary strategy %d not supported",
                 (int) boundary);
          break;
        }
    }
  else
    {
      switch(boundary)
        {
        case CONV_BOUNDARY_ZERO:
          cuda_c_yFilterZero(result.getCudaArrayPtr(),src.getCudaArrayPtr(),w,h,hFilt.getCudaArrayPtr(),hfs,share_len,tile.sz());
          break;
        case CONV_BOUNDARY_CLEAN:
          cuda_c_yFilterClean(result.getCudaArrayPtr(),src.getCudaArrayPtr(),w,h,hFilt.getCudaArrayPtr(),hfs,share_len,tile.sz());
          break;
        case CONV_BOUNDARY_REPLICATE:
          cuda_c_yFilterReplicate(result.getCudaArrayPtr(),src.getCudaArrayPtr(),w,h,hFilt.getCudaArrayPtr(),hfs,share_len,tile.sz());
          break;
        default:
          LFATAL("convolution boundary strategy %d not supported",
                 (int) boundary);
          break;
        }
    }
  return result;
}


// ######################################################################
CudaImage<float> cudaSepFilter(const CudaImage<float>& src, const CudaImage<float>& hFilter,
          const CudaImage<float>& vFilter,
                               ConvolutionBoundaryStrategy boundary, bool runOptimized)
{
  ASSERT(hFilter.is1D() || hFilter.getSize() == 0);
  ASSERT(vFilter.is1D() || vFilter.getSize() == 0);
  return cudaSepFilter(src, hFilter, vFilter,
                       hFilter.getSize(), vFilter.getSize(), boundary, runOptimized);
}

// ######################################################################
CudaImage<float> cudaSepFilter(const CudaImage<float>& src, const CudaImage<float>& hFilt, const CudaImage<float>& vFilt,
          const int hfs, const int vfs,
                               ConvolutionBoundaryStrategy boundary, bool runOptimized)
{

  CudaImage<float> res=src;
  if (hfs > 0) res = cudaXFilter(src, hFilt, hfs, boundary, runOptimized);
  if (vfs > 0) res = cudaYFilter(res, vFilt, vfs, boundary, runOptimized);
  return res;
}
