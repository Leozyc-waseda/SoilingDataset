/*!@file CUDA/cuda-lowpass.h CUDA/GPU optimized lowpass filter code */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/cuda_shapeops.h $
// $Id: cuda_shapeops.h 12962 2010-03-06 02:13:53Z irock $
//

#ifndef CUDA_SHAPEOPS_H_DEFINED
#define CUDA_SHAPEOPS_H_DEFINED

#include <cuda.h>
#include "CUDA/cutil.h"
#include "cudadefs.h"

__global__ void cuda_global_dec_xy(const float *src,  float* dst, const int x_factor, const int y_factor, const unsigned int w, const unsigned int h, int tile_width)
{

  // Destination width, height
  const int new_width = w/x_factor;
  const int new_height = h/y_factor;
  const int new_size = new_width*new_height;

  const int dx = threadIdx.x;                    // dest pixel within dest tile
  const int dts = IMUL(blockIdx.x, tile_width);  // tile start for source, relative to row start
  const int drs = IMUL(blockIdx.y, new_width);   // Row start index in dest data

  const int srx = IMUL((dx+dts),x_factor);             // src pixel in src row
  const int srs = IMUL(IMUL(blockIdx.y, w),y_factor);  // Row start index in source data

  const int writeIdx = drs + dts + dx; // write index
  const int loadIdx = srs + srx;  // load index

  // only process every so many pixels
  if(writeIdx < new_size  && loadIdx < w*h && dts+dx < new_width) {
    dst[writeIdx] = src[loadIdx];
  }

}

__global__ void cuda_global_dec_x(const float *src,  float* dst, const int x_factor, const unsigned int w, const unsigned int h, int tile_width)
{

  // Destination width, height
  const int new_width = w/x_factor;
  const int new_size = new_width*h;

  const int dx = threadIdx.x;                    // dest pixel within dest tile
  const int dts = IMUL(blockIdx.x, tile_width);  // tile start for source, relative to row start
  const int drs = IMUL(blockIdx.y, new_width);   // Row start index in dest data

  const int srx = IMUL((dx+dts),x_factor);       // src pixel in src row
  const int srs = IMUL(blockIdx.y, w);           // Row start index in source data

  const int writeIdx = drs + dts + dx; // write index
  const int loadIdx = srs + srx;  // load index

  // only process every so many pixels
  if(writeIdx < new_size  && loadIdx < w*h && dts+dx < new_width) {
    dst[writeIdx] = src[loadIdx];
  }

}


__global__ void cuda_global_dec_y(const float *src,  float* dst, const int y_factor, const unsigned int w, const unsigned int h, int tile_width)
{

  // Destination width, height
  const int new_height = h/y_factor;
  const int new_size = w*new_height;

  const int dx = threadIdx.x;                    // dest pixel within dest tile
  const int dts = IMUL(blockIdx.x, tile_width);  // tile start for source, relative to row start
  const int drs = IMUL(blockIdx.y, w);           // Row start index in dest data

  const int srx = dx+dts;                              // src pixel in src row
  const int srs = IMUL(IMUL(blockIdx.y, w),y_factor);  // Row start index in source data

  const int writeIdx = drs + dts + dx; // write index
  const int loadIdx = srs + srx;  // load index

  // only process every so many pixels
  if(writeIdx < new_size  && loadIdx < w*h && dts+dx < w) {
    dst[writeIdx] = src[loadIdx];
  }

}

__global__ void cuda_global_quickLocalAvg(const float *in, float *res, float fac, int scalex, int scaley, int remx, int remy, int lw, int lh, int sw, int sh, int tile_width, int tile_height)
{
  const int srs = IMUL(blockIdx.y, tile_height) + threadIdx.y; // row start for scaled avg
  const int scs = IMUL(blockIdx.x, tile_width) + threadIdx.x; // column index for scaled avg
  int sidx = IMUL(srs,sw) + scs;

  if(scs < sw && srs < sh)
  {
    res[sidx] = 0;
    int offx = 0; int offy=0;
    // Remaining input pixels will be taken up by the last averaging pixel in each dimension
    if(scs == sw-1) offx+=remx;
    if(srs == sh-1) offy+=remy;

    for(int j=0;j<scaley+offy;j++)
    {
      for(int i=0;i<scalex+offx;i++)
        {
          const int x_pos = IMUL(scs,scalex)+i;
          const int y_pos = IMUL(srs,scaley)+j;
          int lidx = IMUL(y_pos,lw) + x_pos;
          if(x_pos < lw && y_pos < lh)
            res[sidx] += in[lidx];
        }
    }
    // Normalize by the area of the average
    res[sidx] *= fac;
  }

}


__global__ void cuda_global_quickLocalAvg2x2(const float *in, float *res, int lw, int lh, int sw, int sh, int tile_width, int tile_height)
{
  const int srs = IMUL(blockIdx.y, tile_height) + threadIdx.y; // row start for scaled avg
  const int scs = IMUL(blockIdx.x, tile_width) + threadIdx.x; // column index for scaled avg
  const int sidx = IMUL(srs,sw) + scs;

  if(scs < sw && srs < sh)
  {
    res[sidx] = 0;
    const int x_pos = IMUL(scs,2);
    const int y_pos = IMUL(srs,2);
    int lidx = IMUL(y_pos,lw) + x_pos;

    if(x_pos+1 < lw && y_pos+1 < lh)
    {
      res[sidx] = (in[lidx] + in[lidx+1] + in[lidx+lw] + in[lidx+lw+1])*0.25F;
    }
  }

}

__global__ void cuda_global_quickLocalMax(const float *in, float *res, int scalex, int scaley, int remx, int remy, int lw, int lh, int sw, int sh, int tile_width, int tile_height)
{
  const int srs = IMUL(blockIdx.y, tile_height) + threadIdx.y; // row start for scaled avg
  const int scs = IMUL(blockIdx.x, tile_width) + threadIdx.x; // column index for scaled avg
  int sidx = IMUL(srs,sw) + scs;
  float curRes = -10000.0F;
  if(scs < sw && srs < sh)
  {
    int offx = 0; int offy=0;
    // Remaining input pixels will be taken up by the last averaging pixel in each dimension
    if(scs == sw-1) offx+=remx;
    if(srs == sh-1) offy+=remy;

    for(int j=0;j<scaley+offy;j++)
    {
      for(int i=0;i<scalex+offx;i++)
        {
          const int x_pos = IMUL(scs,scalex)+i;
          const int y_pos = IMUL(srs,scaley)+j;
          int lidx = IMUL(y_pos,lw) + x_pos;
          if(x_pos < lw && y_pos < lh)
            curRes= fmaxf(in[lidx],curRes);
        }
    }
    // Normalize by the area of the average
    res[sidx] = curRes;
  }

}



__global__ void cuda_global_rescaleBilinear(const float *src, float *res, float sw, float sh, int orig_w, int orig_h, int new_w, int new_h, int tile_width, int tile_height)
{
  // code inspired from one of the Graphics Gems book:
  /*
    (1) (x,y) are the original coords corresponding to scaled coords (i,j)
    (2) (x0,y0) are the greatest lower bound integral coords from (x,y)
    (3) (x1,y1) are the least upper bound integral coords from (x,y)
    (4) d00, d10, d01, d11 are the values of the original image at the corners
        of the rect (x0,y0),(x1,y1)
    (5) the value in the scaled image is computed from bilinear interpolation
        among d00,d10,d01,d11
  */

  // Destination column
  const int dest_col = blockIdx.x*tile_width + threadIdx.x;
  // Destination row index
  const int dest_row = blockIdx.y*tile_height + threadIdx.y;
  // Destination index
  const int dest_idx = dest_row*new_w + dest_col;

  if(dest_col < new_w && dest_row < new_h)
  {
    // Src column
    const float y = fmaxf(0.0f,(dest_row+0.5f)*sh - 0.5f);
    const int src_row0 = int(y);
    const int src_row1 = (int) fminf(src_row0+1,orig_h-1);
    const float x = fmaxf(0.0f,(dest_col+0.5f)*sw - 0.5f);
    const int src_col0 = int(x);
    const int src_col1 = (int) fminf(src_col0+1,orig_w-1);
    const float fy = y - float(src_row0);
    const float fx = x - float(src_col0);
    const int yw0 = IMUL(src_row0,orig_w);
    const int yw1 = IMUL(src_row1,orig_w);

    const float d00 = src[yw0+src_col0];
    const float d10 = src[yw0+src_col1];
    const float d01 = src[yw1+src_col0];
    const float d11 = src[yw1+src_col1];
    float dx0 = d00 + (d10 - d00) * fx;
    float dx1 = d01 + (d11 - d01) * fx;
    res[dest_idx] = dx0 + (dx1 - dx0)*fy;
  }
}


__global__ void cuda_global_rescaleBilinearRGB(const float3_t *src, float3_t *res, float sw, float sh, int orig_w, int orig_h, int new_w, int new_h, int tile_width, int tile_height)
{
  // code inspired from one of the Graphics Gems book:
  /*
    (1) (x,y) are the original coords corresponding to scaled coords (i,j)
    (2) (x0,y0) are the greatest lower bound integral coords from (x,y)
    (3) (x1,y1) are the least upper bound integral coords from (x,y)
    (4) d00, d10, d01, d11 are the values of the original image at the corners
        of the rect (x0,y0),(x1,y1)
    (5) the value in the scaled image is computed from bilinear interpolation
        among d00,d10,d01,d11
  */

  // Destination column
  const int dest_col = blockIdx.x*tile_width + threadIdx.x;
  // Destination row index
  const int dest_row = blockIdx.y*tile_height + threadIdx.y;
  // Destination index
  const int dest_idx = dest_row*new_w + dest_col;

  if(dest_col < new_w && dest_row < new_h)
  {
    // Src column
    const float y = fmaxf(0.0f,(dest_row+0.5f)*sh - 0.5f);
    const int src_row0 = int(y);
    const int src_row1 = (int) fminf(src_row0+1,orig_h-1);
    const float x = fmaxf(0.0f,(dest_col+0.5f)*sw - 0.5f);
    const int src_col0 = int(x);
    const int src_col1 = (int) fminf(src_col0+1,orig_w-1);
    const float fy = y - float(src_row0);
    const float fx = x - float(src_col0);
    const int yw0 = IMUL(src_row0,orig_w);
    const int yw1 = IMUL(src_row1,orig_w);

    float d00,d10,d01,d11;
    float dx0,dx1;
    for(int i=0;i<3;i++)
      {
        d00 = src[yw0+src_col0].p[i];
        d10 = src[yw0+src_col1].p[i];
        d01 = src[yw1+src_col0].p[i];
        d11 = src[yw1+src_col1].p[i];
        dx0 = d00 + (d10 - d00) * fx;
        dx1 = d01 + (d11 - d01) * fx;
        res[dest_idx].p[i] = dx0 + (dx1 - dx0)*fy;
      }
  }
}


#endif
