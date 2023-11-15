/*!@file CUDA/cuda-drawops.h CUDA/GPU optimized drawing operations code */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/cuda_drawops.h $
// $Id: cuda_drawops.h 12962 2010-03-06 02:13:53Z irock $
//

#ifndef CUDA_DRAWOPS_H_DEFINED
#define CUDA_DRAWOPS_H_DEFINED

#include <cuda.h>
#include "CUDA/cutil.h"
#include "cudadefs.h"

// Fill in rectangle with intensity value
__global__ void cuda_global_drawFilledRect(float *dst, int top, int left, int bottom, int right, const float intensity, const int w, const int h, const int tile_width, const int tile_height)
{
  const int x_pos = left + IMUL(blockIdx.x,tile_width) + threadIdx.x;
  const int y_pos = top + IMUL(blockIdx.y,tile_height) + threadIdx.y;
  const int idx = IMUL(y_pos,w) + x_pos;
  int x_max = MIN(w,right+1);
  int y_max = MIN(h,bottom+1);
  if(x_pos < x_max && y_pos < y_max)
  {
    dst[idx] = intensity;
  }
}

// Fill in rectangle with rgb color value
__global__ void cuda_global_drawFilledRectRGB(float3_t *dst, int top, int left, int bottom, int right, const float c1, const float c2, const float c3, const int w, const int h, const int tile_width, const int tile_height)
{
  const int x_pos = left + IMUL(blockIdx.x,tile_width) + threadIdx.x;
  const int y_pos = top + IMUL(blockIdx.y,tile_height) + threadIdx.y;
  const int idx = IMUL(y_pos,w) + x_pos;
  int x_max = MIN(w,right+1);
  int y_max = MIN(h,bottom+1);
  if(x_pos < x_max && y_pos < y_max)
  {
    dst[idx].p[0] = c1;
    dst[idx].p[1] = c2;
    dst[idx].p[2] = c3;
  }
}

#endif
