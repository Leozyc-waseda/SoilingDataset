/*!@file CUDA/CudaKernels.C C++ wrapper for CUDA Kernel generation */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaKernels.C $
// $Id: CudaKernels.C 12962 2010-03-06 02:13:53Z irock $
//


#include "CUDA/CudaImage.H"
#include "CUDA/CudaMathOps.H"
#include "Util/Assert.H"
#include "CUDA/cudadefs.h"
#include "CudaKernels.H"
#include "CudaDevices.H"
#include "wrap_c_cuda.h"


// ######################################################################
CudaImage<float> cudaDogFilterHmax(MemoryPolicy mp, int dev, const float theta, const float gamma, const int size, const float div)
{
  // Ensure that we are on a CUDA device
  ASSERT(mp != HOST_MEMORY);

  // resize the data buffer
  CudaImage<float> dest(size, size, NO_INIT,mp,dev);
  Dims tile = CudaDevices::getDeviceTileSize(dev);
  cuda_c_dogFilterHmax(dest.getCudaArrayPtr(), theta, gamma, size, div, tile.w(),tile.h());
  return dest;
}

CudaImage<float> cudaDogFilter(MemoryPolicy mp, int dev, const float stddev, const float theta, const int halfsize_in)
{
  // Ensure that we are on a CUDA device
  ASSERT(mp != HOST_MEMORY);

  // resize the data buffer
  int halfsize = halfsize_in;
  if (halfsize <= 0) halfsize = int(ceil(stddev * sqrt(7.4F)));
  int size = 2*halfsize+1;
  CudaImage<float> dest(size,size, NO_INIT,mp,dev);
  Dims tile = CudaDevices::getDeviceTileSize(dev);
  cuda_c_dogFilter(dest.getCudaArrayPtr(),stddev,theta,halfsize,size,tile.w(),tile.h());
  return dest;
}


// ######################################################################
// On CUDA device produces a Gabor kernel with optionally unequal major+minor axis lengths.
CudaImage<float> cudaGaborFilter3(MemoryPolicy mp, int dev, const float major_stddev, const float minor_stddev,
                          const float period, const float phase,
                                  const float theta, int size)
{
  // Ensure that we are on a CUDA device
  ASSERT(mp != HOST_MEMORY);
  const float max_stddev =
    major_stddev > minor_stddev ? major_stddev : minor_stddev;

  // figure the proper size for the result
  if (size == -1) size = int(ceil(max_stddev * sqrt(-2.0F * log(exp(-5.0F)))));
  else size = size/2;

  CudaImage<float> result = CudaImage<float>(2 * size + 1, 2 * size + 1,NO_INIT,mp,dev);
  CudaImage<float> avg;
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cuda_c_gaborFilter3(result.getCudaArrayPtr(),major_stddev,minor_stddev,period,phase,theta,size,tile.sz(),result.size());
  cudaGetAvg(result,avg);
  result -= avg;
  return result;
}

// ######################################################################
CudaImage<float> cudaGaussian(MemoryPolicy mp, int dev, const float coeff, const float sigma,
                              const int maxhw, const float threshperc)
{
  // Ensure that we are on a CUDA device
  ASSERT(mp != HOST_MEMORY);

  // determine size: keep only values larger that threshperc*max (here max=1)
  int hw = (int)(sigma * sqrt(-2.0F * log(threshperc / 100.0F)));

  // if kernel turns out to be too large, cut it off:
  if (maxhw > 0 && hw > maxhw) hw = maxhw;

  // allocate image for result:
  CudaImage<float> result = CudaImage<float>(2 * hw + 1, 1,NO_INIT,mp,dev);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  // if coeff is given as 0, compute it from sigma:
  float c = coeff;
  if (coeff == 0.0F) c = 1.0F / (sigma * sqrtf(2.0f * float(M_PI)));
  const float sig22 = - 0.5F / (sigma * sigma);
  cuda_c_gaussian(result.getCudaArrayPtr(),c,sig22,hw,tile.sz(),result.size());
  return result;
}


