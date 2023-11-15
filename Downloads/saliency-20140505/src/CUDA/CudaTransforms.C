/*!@file CUDA/CudaTransforms.C C++ wrapper for CUDA transformation ops */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaTransforms.C $
// $Id: CudaTransforms.C 12962 2010-03-06 02:13:53Z irock $
//

#include "CUDA/CudaImage.H"
#include "Util/Assert.H"
#include "CUDA/cudadefs.h"
#include "CUDA/CudaRandom.H"
#include "CudaTransforms.H"
#include "CudaDevices.H"
#include "wrap_c_cuda.h"

// ######################################################################
void cudaInplaceAddBGnoise(CudaImage<float>& src, const float range, CudaImage<float>& randBuf)
{
// background noise level: as coeff of map full dynamic range:
#define BGNOISELEVEL 0.00001

  cudaInplaceAddBGnoise2(src, range * BGNOISELEVEL, randBuf);
}

// ######################################################################
void cudaInplaceAddBGnoise2(CudaImage<float>& src, const float range, CudaImage<float>& randBuf)
{
  MemoryPolicy mp = src.getMemoryPolicy();
  const int dev = src.getMemoryDevice();

  // Ensure that the data is valid
  ASSERT(src.initialized());
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  const int w = src.getWidth(), h = src.getHeight();
  cudaSizeRandomBuffer(randBuf,mp,dev,src.size());
  // do not put noise very close to image borders:
  int siz = std::min(w, h) / 10;
  cuda_c_inplaceAddBGnoise2(src.getCudaArrayPtr(),randBuf.getCudaArrayPtr(),siz,range,w,h,tile.sz());
}



