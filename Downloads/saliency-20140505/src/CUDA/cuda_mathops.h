/*!@file CUDA/cuda-mathops.h CUDA/GPU optimized math operations code */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/cuda_mathops.h $
// $Id: cuda_mathops.h 12962 2010-03-06 02:13:53Z irock $
//


#ifndef CUDA_MATHOPS_H_DEFINED
#define CUDA_MATHOPS_H_DEFINED

#include <cuda.h>
#include "CUDA/cutil.h"
#include "cudadefs.h"


// Actual CUDA Implementation, set up as a __device__ function to allow it to be called
//  from other CUDA functions
__device__ void cuda_device_inplaceRectify(float *ptr, const int idx)
{
  if(ptr[idx] < 0.0F)
  {
    ptr[idx] = 0.0F;
  }
}

__global__ void cuda_global_inplaceRectify(float *ptr, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    cuda_device_inplaceRectify(ptr,src_idx);
}

__device__ void cuda_device_inplaceClamp(float *ptr, const float cmin, const float cmax, const int idx)
{
  if(ptr[idx] < cmin)
  {
    ptr[idx] = cmin;
  }
  else if(ptr[idx] > cmax)
  {
    ptr[idx] = cmax;
  }
}

__global__ void cuda_global_inplaceClamp(float *ptr, const float cmin, const float cmax, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    cuda_device_inplaceClamp(ptr,cmin,cmax,src_idx);
}

__device__ void cuda_device_inplaceNormalize(float *ptr, const int idx, const float omin, const float nmin, const float nscale)
{
  ptr[idx] = nmin + (ptr[idx] - omin) * nscale;
}

__global__ void cuda_global_inplaceNormalize(float *ptr, const float* omin, const float* omax, const float nmin, const float nmax, const int tile_len, const int sz)
{
  const float scale = omax[0] - omin[0];
  const float nscale = (nmax - nmin) / scale;

  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    cuda_device_inplaceNormalize(ptr,src_idx,omin[0],nmin,nscale);
}

__device__ void cuda_device_abs(float *ptr, const int idx)
{
  if(ptr[idx] < 0)
    ptr[idx] *= -1;
}

__global__ void cuda_global_abs(float *src, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    cuda_device_abs(src,src_idx);
}


__global__ void cuda_global_clear(float *src, const float val, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    src[src_idx] = val;
}



__global__ void cuda_global_inplaceAddScalar(float *ptr, const float *offset, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    ptr[src_idx] += offset[0];
}

__global__ void cuda_global_inplaceSubtractScalar(float *ptr, const float *offset, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    ptr[src_idx] -= offset[0];
}

__global__ void cuda_global_inplaceMultiplyScalar(float *ptr, const float *offset, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    ptr[src_idx] *= offset[0];
}

__global__ void cuda_global_inplaceDivideScalar(float *ptr, const float *offset, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    ptr[src_idx] /= offset[0];
}


__global__ void cuda_global_inplaceAddHostScalar(float *ptr, const float val, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    ptr[src_idx] += val;
}

__global__ void cuda_global_inplaceSubtractHostScalar(float *ptr, const float val, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    ptr[src_idx] -= val;
}

__global__ void cuda_global_inplaceMultiplyHostScalar(float *ptr, const float val, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    ptr[src_idx] *= val;
}

__global__ void cuda_global_inplaceDivideHostScalar(float *ptr, const float val, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    ptr[src_idx] /= val;
}



__global__ void cuda_global_inplaceAddImages(float *im1, const float *im2, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    im1[src_idx] += im2[src_idx];
}

__global__ void cuda_global_inplaceSubtractImages(float *im1, const float *im2, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    im1[src_idx] -= im2[src_idx];
}

__global__ void cuda_global_inplaceMultiplyImages(float *im1, const float *im2, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    im1[src_idx] *= im2[src_idx];
}

__global__ void cuda_global_inplaceDivideImages(float *im1, const float *im2, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    im1[src_idx] /= im2[src_idx];
}


__global__ void cuda_global_addScalar(const float *im1, const float *offset, float *res, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    res[src_idx] = im1[src_idx] + offset[0];
}

__global__ void cuda_global_subtractScalar(const float *im1, const float *offset, float *res, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    res[src_idx] = im1[src_idx] - offset[0];
}

__global__ void cuda_global_multiplyScalar(const float *im1, const float *offset, float *res, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    res[src_idx] = im1[src_idx] * offset[0];
}

__global__ void cuda_global_divideScalar(const float *im1, const float *offset, float *res, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    res[src_idx] = im1[src_idx] / offset[0];
}


__global__ void cuda_global_addHostScalar(const float *im1, const float val, float *res, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    res[src_idx] = im1[src_idx] + val;
}

__global__ void cuda_global_subtractHostScalar(const float *im1, const float val, float *res, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    res[src_idx] = im1[src_idx] - val;
}

__global__ void cuda_global_multiplyHostScalar(const float *im1, const float val, float *res, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    res[src_idx] = im1[src_idx] * val;
}

__global__ void cuda_global_divideHostScalar(const float *im1, const float val, float *res, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    res[src_idx] = im1[src_idx] / val;
}


__global__ void cuda_global_addImages(const float *im1, const float *im2, float *res, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    res[src_idx] = im1[src_idx] + im2[src_idx];
}

__global__ void cuda_global_subtractImages(const float *im1, const float *im2, float *res, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    res[src_idx] = im1[src_idx] - im2[src_idx];
}

__global__ void cuda_global_multiplyImages(const float *im1, const float *im2, float *res, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    res[src_idx] = im1[src_idx] * im2[src_idx];
}

__global__ void cuda_global_divideImages(const float *im1, const float *im2, float *res, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    res[src_idx] = im1[src_idx] / im2[src_idx];
}


__global__ void cuda_global_takeMax(const float *im1, const float *im2, float *res, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
  {
    float f1 = im1[src_idx];
    float f2 = im2[src_idx];
    res[src_idx] = (f1 > f2) ? f1 : f2;
  }
}


__device__ void cuda_device_getMin(const float *src, float *dest, float *buf, float *shr, const int tile_len, const int sz)
{
  //ASSERT(blockDim.y == 1 && gridDim.y == 1);

  const int shr_idx = threadIdx.x;
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  const bool threadActive = src_idx < sz && shr_idx < tile_len;


  if(threadActive)
  {
    shr[shr_idx] = src[src_idx];
  }
  __syncthreads();

  // Come up with a per block min
  int incr = 1;
  int mod = 2;
  while(incr < sz)
  {
    if(shr_idx % mod == 0 && shr_idx+incr < tile_len && src_idx+incr < sz)
    {
      // Check neighbor
      if(shr[shr_idx] > shr[shr_idx+incr])
        shr[shr_idx] = shr[shr_idx+incr];
    }
    __syncthreads();

    incr *= 2;
    mod *= 2;
  }
  // Now load the global output
  if(threadIdx.x == 0 && threadActive)
  {
    int dst_idx = blockIdx.x;
    buf[dst_idx] = shr[0];
    // Have the first block put the value in the final destination, this will eventually be the answer
    if(dst_idx == 0)
      dest[0] = shr[0];
  }
  __syncthreads();
}

__global__ void cuda_global_getMin(const float *src, float *dest, float *buf, const int tile_len, const int sz)
{
  cuda_device_getMin(src,dest,buf,shared_data,tile_len,sz);
}

__device__ void cuda_device_getMax(const float *src, float *dest, float *buf, float *shr, const int tile_len, const int sz)
{
  //ASSERT(blockDim.y == 1 && gridDim.y == 1);

  const int shr_idx = threadIdx.x;
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  const bool threadActive = src_idx < sz && shr_idx < tile_len;


  if(threadActive)
  {
    shr[shr_idx] = src[src_idx];
  }
  __syncthreads();

  // Come up with a per block max
  int incr = 1;
  int mod = 2;
  while(incr < sz)
  {
    if(shr_idx % mod == 0 && shr_idx+incr < tile_len && src_idx+incr < sz)
    {
      // Check neighbor
      if(shr[shr_idx] < shr[shr_idx+incr])
        shr[shr_idx] = shr[shr_idx+incr];
    }
    __syncthreads();

    incr *= 2;
    mod *= 2;
  }
  // Now load the global output
  if(threadIdx.x == 0 && threadActive)
  {
    int dst_idx = blockIdx.x;
    buf[dst_idx] = shr[0];
    // Have the first block put the value in the final destination, this will eventually be the answer
    if(dst_idx == 0)
      dest[0] = shr[0];

  }
  __syncthreads();
}


__global__ void cuda_global_getMax(const float *src, float *dest, float *buf, const int tile_len, const int sz)
{
  cuda_device_getMax(src,dest,buf,shared_data,tile_len,sz);
}

__device__ void cuda_device_getSum(const float *src, float *dest, float *buf, float *shr, const int tile_len, const int sz)
{
  //ASSERT(blockDim.y == 1 && gridDim.y == 1);

  const int shr_idx = threadIdx.x;
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  const bool threadActive = src_idx < sz && shr_idx < tile_len;

  if(threadActive)
  {
    shr[shr_idx] = src[src_idx];
  }
  __syncthreads();

  // Come up with a per block sum
  int incr = 1;
  int mod = 2;
  while(incr < sz)
  {
    if(shr_idx % mod == 0 && shr_idx+incr < tile_len && src_idx+incr < sz)
    {
      // Sum with neighbor
      shr[shr_idx] += shr[shr_idx+incr];
    }
    __syncthreads();

    incr *= 2;
    mod *= 2;
  }
  // Now load the global output
  if(threadIdx.x == 0 && threadActive)
  {
    int dst_idx = blockIdx.x;
    buf[dst_idx] = shr[0];
    // Have the first block put the value in the final destination, this will eventually be the answer
    if(dst_idx == 0)
      dest[0] = shr[0];

  }
  __syncthreads();
}

__global__ void cuda_global_getSum(const float *src, float *dest, float *buf, const int tile_len, const int sz, const int orig_sz)
{
  cuda_device_getSum(src,dest,buf,shared_data,tile_len,sz);
}

__global__ void cuda_global_getAvg(const float *src, float *dest, float *buf, const int tile_len, const int sz, const int orig_sz)
{
  cuda_device_getSum(src,dest,buf,shared_data,tile_len,sz);
  // If the size of the image is smaller than the tile, than we have a complete sum, which we can then get the average from
  if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && sz <= tile_len)
    dest[0] = dest[0]/orig_sz;

}

__global__ void cuda_global_squared(const float *im, float *res, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    {
      float in = im[src_idx];
      res[src_idx] = in*in;
    }
}

__global__ void cuda_global_sqrt(const float *im, float *res, const int tile_len, const int sz)
{
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  if(src_idx < sz)
    {
      res[src_idx] = sqrt(im[src_idx]);
    }
}



__global__ void cuda_global_quadEnergy(const float *real, const float *imag, float *out, int tile_len, int sz)
{
  const int idx = blockIdx.x*tile_len + threadIdx.x;
  if(idx < sz)
  {
    float re = real[idx];
    float im = imag[idx];
    out[idx] = sqrtf(re*re + im*im);
  }

}

__global__ void cuda_global_inplaceAttenuateBorders_x(float *im, int borderSize, int tile_width, int w, int h)
{

  const float increment = 1.0 / (float)(borderSize + 1);

  const int x_pos = IMUL(blockIdx.x,tile_width) + threadIdx.x;
  if(x_pos < w)
  {
    // In the top lines of the border
    if(blockIdx.y < borderSize)
      {
        const int idx = x_pos + IMUL(blockIdx.y,w);
        float coeff = increment*(blockIdx.y+1);
        im[idx] *= coeff;
      }
    // In the bottom lines of the border
    else if(blockIdx.y < IMUL(borderSize,2))
      {
        const int idx = x_pos + IMUL((h-borderSize+blockIdx.y-borderSize),w);
        float coeff = increment*(IMUL(borderSize,2) - blockIdx.y);
        im[idx] *= coeff;
      }
  }
}

__global__ void cuda_global_inplaceAttenuateBorders_y(float *im, int borderSize, int tile_height, int w, int h)
{
  const float increment = 1.0 / (float)(borderSize + 1);

  const int y_pos = IMUL(blockIdx.y,tile_height) + threadIdx.y;
  if(y_pos < h)
  {
    // In the left lines of the border
    if(blockIdx.x < borderSize)
      {
        const int idx = IMUL(y_pos,w) + blockIdx.x;
        float coeff = increment*(blockIdx.x+1);
        im[idx] *= coeff;
      }
    // In the right lines of the border
    else if(blockIdx.x < IMUL(borderSize,2))
      {
        const int idx = IMUL(y_pos,w)+ (blockIdx.x-borderSize) + (w-borderSize);
        float coeff = increment*(IMUL(borderSize,2) - blockIdx.x);
        im[idx] *= coeff;
      }
  }
}


__device__ void cuda_device_findMax(const float *src, const int *srcloc, float *buf, int *loc, const int tile_len, const int sz)
{
  //ASSERT(blockDim.y == 1 && gridDim.y == 1);

  const int shr_idx = threadIdx.x;
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  const bool threadActive = src_idx < sz && shr_idx < tile_len;
  float *shrVal = &shared_data[0]; // size of tile_len
  float *shrLoc = (float *) &(shared_data[tile_len]); // size of tile_len

  if(threadActive)
  {
    shrVal[shr_idx] = src[src_idx];
    if(srcloc == NULL)
      shrLoc[shr_idx] = src_idx;
    else
      shrLoc[shr_idx] = srcloc[src_idx];
  }
  __syncthreads();

  // Come up with a per block max
  int incr = 1;
  int mod = 2;
  while(incr < sz)
  {
    if(shr_idx % mod == 0 && shr_idx+incr < tile_len && src_idx+incr < sz)
    {
      // Check neighbor
      if(shrVal[shr_idx+incr] > shrVal[shr_idx])
        {
          shrVal[shr_idx] = shrVal[shr_idx+incr];
          shrLoc[shr_idx] = shrLoc[shr_idx+incr];
        }
    }
    __syncthreads();

    incr *= 2;
    mod *= 2;
  }
  // Now load the global output
  if(threadIdx.x == 0 && threadActive)
  {
    int dst_idx = blockIdx.x;
    buf[dst_idx] = shrVal[0];
    loc[dst_idx] = shrLoc[0];
  }
  __syncthreads();
}


__global__ void cuda_global_findMax(const float *src, const int *srcloc, float *buf, int *loc, const int tile_len, const int sz)
{
  cuda_device_findMax(src,srcloc,buf,loc,tile_len,sz);
}


__device__ void cuda_device_findMin(const float *src, const int *srcloc, float *buf, int *loc, const int tile_len, const int sz)
{
  //ASSERT(blockDim.y == 1 && gridDim.y == 1);

  const int shr_idx = threadIdx.x;
  const int src_idx = blockIdx.x*tile_len + threadIdx.x;
  const bool threadActive = src_idx < sz && shr_idx < tile_len;
  float *shrVal = &shared_data[0]; // size of tile_len
  float *shrLoc = (float *) &(shared_data[tile_len]); // size of tile_len

  if(threadActive)
  {
    shrVal[shr_idx] = src[src_idx];
    if(srcloc == NULL)
      shrLoc[shr_idx] = src_idx;
    else
      shrLoc[shr_idx] = srcloc[src_idx];
  }
  __syncthreads();

  // Come up with a per block min
  int incr = 1;
  int mod = 2;
  while(incr < sz)
  {
    if(shr_idx % mod == 0 && shr_idx+incr < tile_len && src_idx+incr < sz)
    {
      // Check neighbor
      if(shrVal[shr_idx+incr] < shrVal[shr_idx])
        {
          shrVal[shr_idx] = shrVal[shr_idx+incr];
          shrLoc[shr_idx] = shrLoc[shr_idx+incr];
        }
    }
    __syncthreads();

    incr *= 2;
    mod *= 2;
  }
  // Now load the global output
  if(threadIdx.x == 0 && threadActive)
  {
    int dst_idx = blockIdx.x;
    buf[dst_idx] = shrVal[0];
    loc[dst_idx] = shrLoc[0];
  }
  __syncthreads();
}


__global__ void cuda_global_findMin(const float *src, const int *srcloc, float *buf, int *loc, const int tile_len, const int sz)
{
  cuda_device_findMin(src,srcloc,buf,loc,tile_len,sz);
}


#endif
