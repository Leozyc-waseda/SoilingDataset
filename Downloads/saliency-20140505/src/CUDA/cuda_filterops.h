/*!@file CUDA/cuda-filterops.h CUDA/GPU optimized filter operations code */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/cuda_filterops.h $
// $Id: cuda_filterops.h 12962 2010-03-06 02:13:53Z irock $
//


#ifndef CUDA_FILTEROPS_H_DEFINED
#define CUDA_FILTEROPS_H_DEFINED

#include <cuda.h>
#include "CUDA/cutil.h"
#include "cudadefs.h"
#include <float.h>


__global__ void cuda_global_orientedFilter(const float *src, float *re, float *im, const float kx, const float ky, const float intensity, const int w, const int h, const int tile_width)
{
  // Determine index
  const int x_idx = IMUL(blockIdx.x,tile_width) + threadIdx.x;
  const int y_idx = blockIdx.y;
  const int idx = IMUL(y_idx,w) + x_idx;

  const float val = src[idx] * intensity;
  // (x,y) = (0,0) at center of image:
  const int w2l = w >> 1, w2r = w - w2l;
  const int h2l = h >> 1, h2r = h - h2l;

  // Position in coordinate space if the center of the image is (0,0)
  //for (int j = -h2l; j < h2r; ++j)
  //  for (int i = -w2l; i < w2r; ++i)
  const int x_pos = x_idx-w2l;
  const int y_pos = y_idx-h2l;
  if(x_pos < w2r && y_pos < h2r)
  {
    const float arg = kx*x_pos + ky*y_pos;
    //const float arg = kx * i + ky * j;
    // NOTE: sin and cos are single cycle calls on GPU, no need for trig tables
    // Since we're using the same arg, sincosf() is what we need
    // If we want to be less accurate, we can use __sincosf() directly
    float sinarg, cosarg;
    sincosf(arg, &sinarg, &cosarg);


    re[idx] = val * cosarg;
    im[idx] = val * sinarg;
  }

}


__global__ void cuda_global_centerSurroundAbs(const float *center, const float *surround, float *res, int lw, int lh, int sw, int sh, int scalex, int scaley, int remx, int remy, int tile_width )
{
  // compute abs(hires - lowres):
  const int x_pos = IMUL(blockIdx.x,tile_width) + threadIdx.x;
  const int y_pos = blockIdx.y;
  const int cidx = IMUL(y_pos,lw) + x_pos;
  // sidx = y_pos/scaley*sw + x_pos/scalex
  int sidx = IMUL(y_pos/scaley,sw) + x_pos/scalex;
  // This seems to lock the indices to one less than the last at the edges of the image...
  // don't understand justification, but it was in the cpu version of centerSurround()
  if(x_pos > remx) sidx--;
  if(y_pos > remy) sidx-=sw;

  if(x_pos < lw && y_pos < lh)
    {
      float cval = center[cidx];
      float sval = surround[sidx];
      if(cval > sval)
        {
          res[cidx] = cval - sval;
        }
      else
        {
          res[cidx] = sval - cval;
        }
    }
}

__global__ void cuda_global_centerSurroundClamped(const float *center, const float *surround, float *res, int lw, int lh, int sw, int sh, int scalex, int scaley, int remx, int remy, int tile_width )
{
  // compute hires - lowres, clamped to 0:
  const int x_pos = IMUL(blockIdx.x,tile_width) + threadIdx.x;
  const int y_pos = blockIdx.y;
  const int cidx = IMUL(y_pos,lw) + x_pos;
  // sidx = y_pos/scaley*sw + x_pos/scalex
  int sidx = IMUL(y_pos/scaley,sw) + x_pos/scalex;
  // This seems to lock the indices to one less than the last at the edges of the image...
  // don't understand justification, but it was in the cpu version of centerSurround()
  if(x_pos > remx) sidx--;
  if(y_pos > remy) sidx-=sw;

  if(x_pos < lw && y_pos < lh)
    {
      float cval = center[cidx];
      float sval = surround[sidx];
      if(cval > sval)
        {
          res[cidx] = cval - sval;
        }
      else
        {
          res[cidx] = 0;
        }
    }
}

__global__ void cuda_global_centerSurroundDirectional(const float *center, const float *surround, float *pos, float *neg, int lw, int lh, int sw, int sh, int scalex, int scaley, int remx, int remy, int tile_width )
{
  // compute abs(hires - lowres):
  const int x_pos = IMUL(blockIdx.x,tile_width) + threadIdx.x;
  const int y_pos = blockIdx.y;
  const int cidx = IMUL(y_pos,lw) + x_pos;
  // sidx = y_pos/scaley*sw + x_pos/scalex
  int sidx = IMUL(y_pos/scaley,sw) + x_pos/scalex;
  // This seems to lock the indices to one less than the last at the edges of the image...
  // don't understand justification, but it was in the cpu version of centerSurround()
  if(x_pos > remx) sidx--;
  if(y_pos > remy) sidx-=sw;

  if(x_pos < lw && y_pos < lh)
    {
      float cval = center[cidx];
      float sval = surround[sidx];
      if(cval > sval)
        {
          pos[cidx] = cval - sval;
          neg[cidx] = 0.0F;
        }
      else
        {
          pos[cidx] = 0.0F;
          neg[cidx] = sval - cval;
        }
    }
}



__global__ void cuda_global_centerSurroundAbsAttenuate(const float *center, const float *surround, float *res, int lw, int lh, int sw, int sh, int borderSize, int scalex, int scaley, int remx, int remy, int tile_width, int tile_height)
{
  // compute abs(hires - lowres):
  const float increment = 1.0 / (float)(borderSize + 1);
  const int x_pos = IMUL(blockIdx.x,tile_width) + threadIdx.x;
  const int y_pos = blockIdx.y;
  const int cidx = IMUL(y_pos,lw) + x_pos;
  float result;
  // sidx = y_pos/scaley*sw + x_pos/scalex
  int sidx = IMUL(y_pos/scaley,sw) + x_pos/scalex;
  // This seems to lock the indices to one less than the last at the edges of the image...
  // don't understand justification, but it was in the cpu version of centerSurround()
  if(x_pos > remx) sidx--;
  if(y_pos > remy) sidx-=sw;

  if(x_pos < lw && y_pos < lh)
    {
      // Perform the center/surround absolute calculation
      float cval = center[cidx];
      float sval = surround[sidx];
      if(cval > sval)
        {
          result = cval - sval;
        }
      else
        {
          result = sval - cval;
        }
      // Perform the attenuate border calculation
      // Y Border: In the top lines of the border
      if(y_pos < borderSize)
        {
          float coeff = increment*(y_pos+1);
          result *= coeff;
        }
      // Y Border: In the bottom lines of the border
      else if(y_pos > lh-borderSize-1)
        {
          float coeff = increment*(borderSize-lh);
          result *= coeff;
        }
      // X Border: In the left lines of the border
      if(x_pos < borderSize)
        {
          float coeff = increment*(x_pos+1);
          result *= coeff;
        }
      // Y Border: In the right lines of the border
      else if(x_pos < lw-borderSize-1)
        {
          float coeff = increment*(borderSize-lw);
          result *= coeff;
        }
    }

}


__global__ void cuda_global_spatialPoolMax(const float *src, float *res, const int src_w, const int src_h, int skip_w, int skip_h, int reg_w, int reg_h, int tile_width, int tile_height)
{
  // Determine which region you are working on, and which tile within that region
  const int tilesperregion_w = IDIVUP(reg_w,tile_width);
  const int tilesperregion_h = IDIVUP(reg_h,tile_height);
  // Find out which region we are in
  const int reg_x = blockIdx.x / tilesperregion_w;
  const int reg_y = blockIdx.y / tilesperregion_h;
  // Within the region, which tile are we
  const int tile_x = blockIdx.x % tilesperregion_w;
  const int tile_y = blockIdx.y % tilesperregion_h;
  // What are the src x and y positions for this thread
  const int reg_x_pos = IMUL(tile_x,tile_width) + threadIdx.x;
  const int reg_y_pos = IMUL(tile_y,tile_height) + threadIdx.y;
  const int x_pos = IMUL(reg_x, skip_w) + reg_x_pos;
  const int y_pos = IMUL(reg_y, skip_h) + reg_y_pos;
  const int ld_idx = IMUL(threadIdx.y,tile_width) + threadIdx.x;
  const int src_idx = IMUL(y_pos,src_w)+x_pos;
  const int src_sz = IMUL(src_w,src_h);

  // Load the spatial pool into shared memory
  float *data = (float *) shared_data; // size of tile_width*tile_height
  const int tile_sz = IMUL(tile_height,tile_width);

  // Make sure we are in bounds of the region and in bounds of the image
  if(y_pos < src_h && x_pos < src_w && reg_x_pos < reg_w && reg_y_pos < reg_h)
    data[ld_idx] = src[src_idx];
  else
    data[ld_idx] = -FLT_MAX;

  __syncthreads();

  if(y_pos < src_h && x_pos < src_w)
  {
    // Come up with a per block max
    int incr = 1;
    int mod = 2;
    while(incr < tile_sz)
    {
      if(ld_idx % mod == 0 && ld_idx+incr < tile_sz)
      {
        // Max between neighbors
        if(data[ld_idx] < data[ld_idx+incr])
          data[ld_idx] = data[ld_idx+incr];
      }
      __syncthreads();

      incr *= 2;
      mod *= 2;
    }
  }
  // Have the first thread load the output, should be done regardless of whether we are in the actual image or not
  if(ld_idx == 0)
    {
      const int res_idx = IMUL(blockIdx.y,gridDim.x)+blockIdx.x;
      res[res_idx] = data[ld_idx];
    }

}



#endif
