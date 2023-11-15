/*!@file CUDA/cuda-colorops.h CUDA/GPU optimized color operations code */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/cuda_colorops.h $
// $Id: cuda_colorops.h 12962 2010-03-06 02:13:53Z irock $
//

#ifndef CUDA_COLOROPS_H_DEFINED
#define CUDA_COLOROPS_H_DEFINED

#include <cuda.h>
#include "CUDA/cutil.h"
#include "cudadefs.h"

// General note, all device functions are inlined automatically

// Get double color opponency maps
__global__ void cuda_global_getRGBY(const float3_t *src, float *rgptr, float *byptr, const float thresh, const float min_range, const float max_range, const int w, const int h, const int tile_width, const int tile_height)
{
  const int x_pos = IMUL(blockIdx.x,tile_width) + threadIdx.x;
  const int y_pos = IMUL(blockIdx.y,tile_height) + threadIdx.y;
  const int idx = IMUL(y_pos,w) + x_pos;
  float thresh3 = 3.0F * thresh;

  if(x_pos < w && y_pos < h)
  {
    float r = src[idx].p[0], g = src[idx].p[1], b = src[idx].p[2];

    // first do the luminanceNormalization:
    float lum = r + g + b;
    if (lum < thresh3)  // too dark... no response from anybody
      {
        rgptr[idx] = min_range;
        byptr[idx] = min_range;
      }
    else
      {
        // normalize chroma by luminance:
        float fac = (max_range-min_range) / lum;
        r *= fac; g *= fac; b *= fac;

        // red = [r - (g+b)/2]        [.] = clamp between 0 and 255
        // green = [g - (r+b)/2]
        // blue = [b - (r+g)/2]
        // yellow = [2*((r+g)/2 - |r-g| - b)]

        // now compute color opponencies:
        // yellow gets a factor 2 to compensate for its previous attenuation
        // by luminanceNormalize():
        float red = r - 0.5f * (g + b), green = g - 0.5f * (r + b);
        float blue = b - 0.5f * (r + g), yellow = -2.0f * (blue + fabs(r-g));

        if (red < min_range) red = min_range;
        else if (red > max_range) red = max_range;
        if (green < min_range) green = min_range;
        else if (green > max_range) green = max_range;
        if (blue < min_range) blue = min_range;
        else if (blue > max_range) blue = max_range;
        if (yellow < min_range) yellow=min_range;
        else if (yellow > max_range) yellow=max_range;

        rgptr[idx] = red - green;
        byptr[idx] = blue - yellow;
      }
  }
}


// Get double color opponency maps
__global__ void cuda_global_toRGB(float3_t *dst, const float *src, const int sz, const int tile_len)
{
  const int idx = IMUL(blockIdx.x,tile_len) + threadIdx.x;
  if(idx < sz)
  {
    float val = src[idx];
    dst[idx].p[0] = val;
    dst[idx].p[1] = val;
    dst[idx].p[2] = val;
  }
}


// Actual CUDA Implementation, set up as a __device__ function to allow it to be called
//  from other CUDA functions
__device__ void cuda_device_getComponents(const float3_t *aptr, float *rptr, float *gptr, float *bptr,
                                          const int w, const int h, const int idx)
{
  if(idx < w*h)
  {
    rptr[idx] = aptr[idx].p[0];
    gptr[idx] = aptr[idx].p[1];
    bptr[idx] = aptr[idx].p[2];
  }
}




// Wrap the device function in a global wrapper so it is also callable from the host
__global__ void cuda_global_getComponents(const float3_t *aptr, float *rptr, float *gptr, float *bptr, int w, int h, int tile_width, int tile_height)
{
  // Optimization, as this will be frequently calculated across many functions, why don't we just pass it along?
  const int idx = blockIdx.y*tile_height*w + threadIdx.y*w + blockIdx.x*tile_width + threadIdx.x;
  cuda_device_getComponents(aptr,rptr,gptr,bptr,w,h,idx);
}


// Actual CUDA Implementation, set up as a __device__ function to allow it to be called
//  from other CUDA functions
__device__ void cuda_device_luminance(const float3_t *aptr, float *dptr, const int w, const int h, const int idx, const int secidx)
{
  if(idx < w*h)
    dptr[idx] = (aptr[idx].p[0] + aptr[idx].p[1] + aptr[idx].p[2])/3.0F;
  // Second index is used for a border pixel if we are layering filters together
  // For calls not using this layer, it is a 1 boolean comparision charge
  // For calls using this, however, it allows simple/complex filters to be stacked (as long as the array size doesn't change)
  if(secidx >= 0 && secidx < w*h)
    dptr[secidx] = (aptr[secidx].p[0] + aptr[secidx].p[1] + aptr[secidx].p[2])/3.0F;
}


// Wrap the device function in a global wrapper so it is also callable from the host
__global__ void cuda_global_luminance(const float3_t *aptr, float *dptr, const int w, const int h, const int tile_width, const int tile_height)
{
  // Optimization, as this will be frequently calculated across many functions, why don't we just pass it along?
  const int idx = blockIdx.y*tile_height*w + threadIdx.y*w + blockIdx.x*tile_width + threadIdx.x;
  // Secidx is set to -1 because we are not doing a border with this direct call
  cuda_device_luminance(aptr,dptr,w,h,idx,-1);
}


// Actual CUDA Implementation, set up as a __device__ function to allow it to be called
//  from other CUDA functions
__device__ void cuda_device_luminanceNTSC(const float3_t *aptr, float *dptr, const int w, const int h, const int idx, const int secidx)
{
  //Taken from Matlab's rgb2gray() function
  // T = inv([1.0 0.956 0.621; 1.0 -0.272 -0.647; 1.0 -1.106 1.703]);
  // coef = T(1,:)';
  const float coef1 =  0.298936F,coef2 = 0.587043F, coef3 = 0.114021F;;
  if(idx < w*h)
    dptr[idx] = roundf(aptr[idx].p[0]*coef1 + aptr[idx].p[1]*coef2 + aptr[idx].p[2]*coef3);
  // Second index is used for a border pixel if we are layering filters together
  // For calls not using this layer, it is a 1 boolean comparision charge
  // For calls using this, however, it allows simple/complex filters to be stacked (as long as the array size doesn't change)
  if(secidx >= 0 && secidx < w*h)
    dptr[secidx] = roundf(aptr[secidx].p[0]*coef1 + aptr[secidx].p[1]*coef2 + aptr[secidx].p[2]*coef3);
}


// Wrap the device function in a global wrapper so it is also callable from the host
__global__ void cuda_global_luminanceNTSC(const float3_t *aptr, float *dptr, const int w, const int h, const int tile_width, const int tile_height)
{
  // Optimization, as this will be frequently calculated across many functions, why don't we just pass it along?
  const int idx = blockIdx.y*tile_height*w + threadIdx.y*w + blockIdx.x*tile_width + threadIdx.x;
  // Secidx is set to -1 because we are not doing a border with this direct call
  cuda_device_luminanceNTSC(aptr,dptr,w,h,idx,-1);
}

#endif
