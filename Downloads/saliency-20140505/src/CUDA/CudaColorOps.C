/*!@file CUDA/CudaColorOps.C C++ wrapper for CUDA Color operations */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaColorOps.C $
// $Id: CudaColorOps.C 12962 2010-03-06 02:13:53Z irock $
//

#include "CUDA/CudaImage.H"
#include "Util/Assert.H"
#include "CUDA/cudadefs.h"
#include "CudaColorOps.H"
#include "CudaDevices.H"
#include "wrap_c_cuda.h"


void cudaGetRGBY(const CudaImage<PixRGB<float> >& src, CudaImage<float>& rg, CudaImage<float>& by,
                 const float thresh, const float min_range, const float max_range)
{
  ASSERT(src.initialized());
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);
  const MemoryPolicy mp = src.getMemoryPolicy();
  const int dev = src.getMemoryDevice();
  rg = CudaImage<float>(src.getDims(), NO_INIT, mp, dev);
  by = CudaImage<float>(src.getDims(), NO_INIT, mp, dev);
  const Dims tile = CudaDevices::getDeviceTileSize(dev);
  cuda_c_getRGBY((float3_t *)src.getCudaArrayPtr(),rg.getCudaArrayPtr(),by.getCudaArrayPtr(),thresh,
                 min_range,max_range,src.getWidth(),src.getHeight(),tile.w(),tile.h());
}

CudaImage<PixRGB<float> >  cudaToRGB(const CudaImage<float>& src)
{

  ASSERT(src.initialized());
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);
  const MemoryPolicy mp = src.getMemoryPolicy();
  const int dev = src.getMemoryDevice();
  CudaImage<PixRGB<float> > dst = CudaImage<PixRGB<float> >(src.getDims(), NO_INIT, mp, dev);

  const Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_toRGB((float3_t *)dst.getCudaArrayPtr(),src.getCudaArrayPtr(),src.size(),tile.sz());
  return dst;
}


void cudaGetComponents(const CudaImage<PixRGB<float> >& src, CudaImage<float>& red, CudaImage<float>& green, CudaImage<float>& blue)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  // Set up output image memory
  red = CudaImage<float>(src.getDims(), NO_INIT, src.getMemoryPolicy(), dev);
  green = CudaImage<float>(src.getDims(), NO_INIT, src.getMemoryPolicy(), dev);
  blue = CudaImage<float>(src.getDims(), NO_INIT, src.getMemoryPolicy(), dev);

  const Dims tile = CudaDevices::getDeviceTileSize(dev);
  // Call CUDA implementation
  cuda_c_getComponents((float3_t *)src.getCudaArrayPtr(), red.getCudaArrayPtr(), green.getCudaArrayPtr(), blue.getCudaArrayPtr(),
                       src.getWidth(), src.getHeight(),tile.w(),tile.h());
}

// Our CUDA library only supports float implementation, no use pretending to support others with template style
CudaImage<float> cudaLuminance(const CudaImage<PixRGB<float> >& src)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  // Output is the same size as the input for this filter
  CudaImage<float> result(src.getDims(), NO_INIT, src.getMemoryPolicy(), dev);

  const Dims tile = CudaDevices::getDeviceTileSize(dev);
  // Now call the CUDA implementation
  cuda_c_luminance((float3_t *)src.getCudaArrayPtr(),result.getCudaArrayPtr(),result.getWidth(),result.getHeight(),
                       tile.w(),tile.h());
  return result;
}


// Our CUDA library only supports float implementation, no use pretending to support others with template style
CudaImage<float> cudaLuminanceNTSC(const CudaImage<PixRGB<float> >& src)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());
  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  // Output is the same size as the input for this filter
  CudaImage<float> result(src.getDims(), NO_INIT, src.getMemoryPolicy(), dev);

  const Dims tile = CudaDevices::getDeviceTileSize(dev);
  // Now call the CUDA implementation
  cuda_c_luminanceNTSC((float3_t *)src.getCudaArrayPtr(),result.getCudaArrayPtr(),result.getWidth(),result.getHeight(),
                       tile.w(),tile.h());
  return result;
}
