/*!@file CUDA/CudaRandom.C C++ wrapper for CUDA random number generation */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaRandom.C $
// $Id: CudaRandom.C 12962 2010-03-06 02:13:53Z irock $
//

#include "CUDA/CudaImage.H"
#include "Util/Assert.H"
#include "CUDA/cudadefs.h"
#include "CudaRandom.H"
#include "CUDA/cuda_mersennetwister.h"
#include "CudaDevices.H"
#include "wrap_c_cuda.h"
#include <sys/time.h>

void cudaSetSeed(int dev)
{
  // Set the random seed for the device
  struct timeval tv; gettimeofday(&tv, NULL);
  unsigned int seed = tv.tv_usec;
  cudaSetSeed(seed,dev);
}

void cudaSetSeed(unsigned int seed, int dev)
{
  CudaDevices::setCurrentDevice(dev);
  cuda_c_seedMT(seed);
}


void cudaRandomMT(CudaImage<float>& out)
{
  ASSERT(out.size() > 0);
  const int dev = out.getMemoryDevice();
  const MemoryPolicy mp = out.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  cudaSizeRandomBuffer(out,mp,dev,out.size());
  cuda_c_randomMT(out.getCudaArrayPtr(),out.size(),tile.sz());
}

// Ensure the size of the random number buffer is valid
void cudaSizeRandomBuffer(CudaImage<float>&buf, MemoryPolicy mp, int dev, int minSize)
{
  ASSERT(mp != HOST_MEMORY);
  int bufDev = buf.getMemoryDevice();
  MemoryPolicy bufMP = buf.getMemoryPolicy();
  int bufSize = buf.size();
  if(minSize == 0 || bufSize < minSize || bufSize % MT_RNG_COUNT || bufMP != mp || bufDev != dev)
    {
      bufSize = MT_RNG_COUNT;
      while(bufSize < minSize)
        {
          bufSize += MT_RNG_COUNT;
        }
      buf = CudaImage<float>(bufSize,1,ZEROS,mp,dev);
    }
}
