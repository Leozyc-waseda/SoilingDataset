/*!@file CUDA/cuda-kernels.h CUDA/GPU convolution kernel generation code  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/cuda_cutpaste.h $
// $Id: cuda_cutpaste.h 13228 2010-04-15 01:49:10Z itti $
//


#ifndef CUDA_CUTPASTE_H_DEFINED
#define CUDA_CUTPASTE_H_DEFINED

#include <cuda.h>
#include "CUDA/cutil.h"
#include "cudadefs.h"

__global__ void cuda_global_crop(const float *src, float *res, int srcw, int srch, int startx, int starty, int endx, int endy, int maxx,int maxy, int tile_width, int tile_height)
{
  int resw = endx-startx;
  int resh = endy-starty;
  int dy = IMUL(blockIdx.y,tile_height)+threadIdx.y;
  int dx = IMUL(blockIdx.x,tile_width)+threadIdx.x;
  int res_idx = IMUL(dy,resw) + dx;
  int src_idx = IMUL(starty+dy,srcw) + startx+dx;

  if(dx < resw && dy < resh && startx+dx < endx && starty+dy < endy)
    {
      if(startx+dx < maxx && starty+dy < maxy)
        {
          res[res_idx] = src[src_idx];
        }
      else
        {
          res[res_idx] = 0.0F;
        }
    }
}

__global__ void cuda_global_shiftImage(const float *src, float *dst, int w, int h, float deltax, float deltay, int tile_width, int tile_height)
{
  // Save bottom row and right column for the border
  float *data = (float *) shared_data; //tile_width * tile_height size
  float *borderY = (float *) &data[tile_width*tile_height]; // size of (tile_height)
  float *borderX = (float *) &data[tile_width*tile_height+tile_height]; // size of (tile_width+1)
  const int sy = threadIdx.y; // source pixel column within source tile
  const int sx = threadIdx.x; // source pixel row within source tile
  const int sts = IMUL(blockIdx.y, tile_height); // tile start for source, in rows
  //const int ste = sts + tile_height; // tile end for source, in rows


  // Current column index
  const int scs = IMUL(blockIdx.x, tile_width) + sx;

  int smemPos = IMUL(sy, tile_width) + sx;
  int gmemPos = IMUL(sts + sy, w) + scs;
  const int ypos=sts+sy, xpos=scs;
  float val=0.0F;

  // prepare a couple of variable for the x direction
  int xt = (int)floor(deltax);
  float xfrac = deltax - xt;
  int startx = MAX(0,xt);
  int endx = MIN(0,xt) + w;
  if (fabs(xfrac) < 1.0e-10F) xfrac = 0.0F;
  else endx--;

  // prepare a couple of variable for the y direction
  int yt = (int)floor(deltay);
  float yfrac = deltay - yt;
  int starty = MAX(0,yt);
  int endy = MIN(0,yt) + h;
  if (fabs(yfrac) < 1.0e-10F) yfrac = 0.0F;
  else endy--;

  // dispatch to faster shiftClean() if displacements are roughly integer:
  //if (fabs(xfrac) < 1.0e-10 && fabs(yfrac) < 1.0e-10)
  //  return shiftClean(srcImg, xt, yt);

  if (xfrac > 0.0)
    {
      xfrac = 1.0 - xfrac;
      xt++;
    }

  if (yfrac > 0.0)
    {
      yfrac = 1.0 - yfrac;
      yt++;
    }

  // prepare the coefficients
  float tl = (1.0F - xfrac) * (1.0F - yfrac);
  float tr = xfrac * (1.0F - yfrac);
  float bl = (1.0F - xfrac) * yfrac;
  float br = xfrac * yfrac;

  // only process columns that are actually within image bounds:
  if (xpos < w && ypos < h) {
    // Shared and global (source) memory indices for current column

    // Load data
    data[smemPos] = src[gmemPos];
    // Bottom of a tile, not of the image
    bool bot = (sy == tile_height-1 && ypos+1 < h);
    // Bottom of the image
    bool tbot = (ypos+1 >= h);
    // Right of a tile, not of the image
    bool rig = (sx == tile_width-1 && xpos+1 < w);
    // Right of the image
    bool trig = (xpos+1 >= w);
    // Load Y border
    if(rig)
      borderY[threadIdx.y] = src[gmemPos + 1];

    // Load X border
    if(bot)
      borderX[threadIdx.x] = src[gmemPos+w];

    // Load corner
    if(bot && rig)
      borderX[tile_width] = src[gmemPos+w+1];

    // Ensure the completness of loading stage because results emitted
    // by each thread depend on the data loaded by other threads:
    __syncthreads();

    int dx = xpos+xt;
    int dy = ypos+yt;

    val=0;
    if(dy >= starty && dy < endy && dx >= startx && dx < endx)
      {
        val += data[smemPos]*tl;
        if(bot)
          {
            if (rig)
              {
                val += borderY[threadIdx.y]*tr;
                val += borderX[threadIdx.x]*bl;
                val += borderX[tile_width]*br;
              }
            else if(trig)
              {
                val += data[smemPos]*tr; // Duplicate value
                val += borderX[threadIdx.x]*bl;
                val += borderX[threadIdx.x]*br; // Duplicate value
              }
            else // Not at the right
              {
                val += data[smemPos+1]*tr;
                val += borderX[threadIdx.x]*bl;
                val += borderX[threadIdx.x+1]*br;
              }
          }
        else if(tbot)
          {
            if (rig)
              {
                val += borderY[threadIdx.y]*tr;
                val += data[smemPos]*bl; // Duplicate value
                val += borderY[threadIdx.y]*br; // Duplicate value
              }
            else if(trig)
              {
                // Nothing to add
                val += data[smemPos]*tr; // Duplicate value
                val += data[smemPos]*bl; // Duplicate value
                val += data[smemPos]*br; // Duplicate value
              }
            else // Not at the right
              {
                val += data[smemPos+1]*tr;
                val += data[smemPos]*bl; // Duplicate value
                val += data[smemPos+1]*br; // Duplicate value
              }
          }
        else // Not at the bottom
          {
            if (rig)
              {
                val += borderY[threadIdx.y]*tr;
                val += data[smemPos+tile_width]*bl;
                val += borderY[threadIdx.y+1]*br;
              }
            else if(trig)
              {
                val += data[smemPos]*tr; // Duplicate value
                val += data[smemPos+tile_width]*bl;
                val += data[smemPos+tile_width]*br; // Duplicate value
              }
            else // Not at the right
              {
                val += data[smemPos+1]*tr;
                val += data[smemPos+tile_width]*bl;
                val += data[smemPos+tile_width+1]*br;
              }
          }
      }


    // Determine new memory location
    int dmemPos = IMUL(dy,w) + dx;
    if(dx >= 0 && dx < w && dy >= 0 && dy < h)
      {
        dst[dmemPos] = val;
      }
  }
}


__global__ void cuda_global_inplacePaste(float *dst, const float *img, int w, int h, int iw, int ih, int dx, int dy, int tile_width, int tile_height)
{
  int sx = threadIdx.x;
  int stsx = blockIdx.x*tile_width;
  int xpos = stsx + sx;
  int sy = threadIdx.y;
  int stsy = blockIdx.y*tile_height;
  int ypos = stsy + sy;
  int didx = (ypos+dy)*w + xpos+dx;
  int iidx = ypos*iw + xpos;
  if(xpos < iw && ypos < ih)
    {
      dst[didx] = img[iidx];
    }
}


__global__ void cuda_global_inplacePasteRGB(float3_t *dst, const float3_t *img, int w, int h, int iw, int ih, int dx, int dy, int tile_width, int tile_height)
{
  int sx = threadIdx.x;
  int stsx = blockIdx.x*tile_width;
  int xpos = stsx + sx;
  int sy = threadIdx.y;
  int stsy = blockIdx.y*tile_height;
  int ypos = stsy + sy;
  int didx = (ypos+dy)*w + xpos+dx;
  int iidx = ypos*iw + xpos;
  if(xpos < iw && ypos < ih)
    {
      dst[didx].p[0] = img[iidx].p[0];
      dst[didx].p[1] = img[iidx].p[1];
      dst[didx].p[2] = img[iidx].p[2];
    }
}

__global__ void cuda_global_inplaceOverlay(float *dst, const float *img, int w, int h, int iw, int ih, int dx, int dy, int tile_width, int tile_height)
{
  int sx = threadIdx.x;
  int stsx = blockIdx.x*tile_width;
  int xpos = stsx + sx;
  int sy = threadIdx.y;
  int stsy = blockIdx.y*tile_height;
  int ypos = stsy + sy;
  int didx = (ypos+dy)*w + xpos+dx;
  int iidx = ypos*iw + xpos;
  if(xpos < iw && ypos < ih)
    {
      float val = img[iidx];
      if(val > 0.0F)
        dst[didx] = val;
    }
}


__global__ void cuda_global_inplaceOverlayRGB(float3_t *dst, const float3_t *img, int w, int h, int iw, int ih, int dx, int dy, int tile_width, int tile_height)
{
  int sx = threadIdx.x;
  int stsx = blockIdx.x*tile_width;
  int xpos = stsx + sx;
  int sy = threadIdx.y;
  int stsy = blockIdx.y*tile_height;
  int ypos = stsy + sy;
  int didx = (ypos+dy)*w + xpos+dx;
  int iidx = ypos*iw + xpos;
  if(xpos < iw && ypos < ih)
    {
      float p0,p1,p2;
      p0 = img[iidx].p[0];
      p1 = img[iidx].p[1];
      p2 = img[iidx].p[2];
      if(p0+p1+p2 > 0.0F)
        {
          dst[didx].p[0] = p0;
          dst[didx].p[1] = p1;
          dst[didx].p[2] = p2;
        }
    }
}


#endif
