/*!@file CUDA/cuda_saliencyops.h CUDA/GPU optimized saliency calculations */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/cuda_saliencyops.h $
// $Id: cuda_saliencyops.h 13227 2010-04-15 01:38:09Z dparks $
//

#ifndef CUDA_SALIENCYOPS_H_DEFINED
#define CUDA_SALIENCYOPS_H_DEFINED

#include <cuda.h>
#include "CUDA/cutil.h"
#include "cudadefs.h"

__global__ void cuda_global_inertiaMap(float *dst, float s, float r_inv, int px, int py, int tile_width, int tile_height, int w, int h)
{
  // Destination column
  const int dest_col = blockIdx.x*tile_width + threadIdx.x;
  // Destination row index
  const int dest_row = blockIdx.y*tile_height + threadIdx.y;
  // Destination index
  const int dest_idx = dest_row*w + dest_col;

  if(dest_col < w && dest_row < h)
    {
      const int dsq = (px - dest_col)*(px - dest_col) + (py - dest_row)*(py - dest_row);
      dst[dest_idx] = s * exp(-dsq * r_inv);
    }
}


__global__ void cuda_global_inhibitionMap(float *dst, float factorOld, float factorNew, float radius, int px, int py, int tile_width, int tile_height, int w, int h)
{
  // Destination column
  const int dest_col = blockIdx.x*tile_width + threadIdx.x;
  // Destination row index
  const int dest_row = blockIdx.y*tile_height + threadIdx.y;
  // Destination index
  const int dest_idx = dest_row*w + dest_col;

  if(dest_col < w && dest_row < h)
    {
      const int dsq = (px - dest_col)*(px - dest_col) + (py - dest_row)*(py - dest_row);
      const float newval =
              dst[dest_idx] * factorOld
                  + (factorNew * exp(- dsq / radius));
      dst[dest_idx] =  newval < 0.0F
        ? 0.0F
        : newval > 255.0F
        ? 255.0F
        : newval;
    }
}



#endif
