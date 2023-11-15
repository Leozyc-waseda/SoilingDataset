/*!@file CUDA/cuda-lowpass.cu CUDA/GPU optimized lowpass code */

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
// $HeadURL$
// $Id$
//

#include "CUDA/cuda-lowpass.h"
#include <cuda.h>
#include "CUDA/cutil.h"
// define int as 32 bits on CUDA architecture to satisfy env_config.h
#define INT_IS_32_BITS
#include "Envision/env_types.h"

// 24-bit multiplication is faster on G80, but we must be sure to
// multiply integers only within [-8M, 8M - 1] range
#define IMUL(a, b) __mul24(a, b)

#define ROW_TILE_W 128
#define COLUMN_TILE_W 16
#define COLUMN_TILE_H 16 //48



////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void cudalowpass5xdecx(const int *src,  const unsigned int w, const unsigned int h, int* dst)
{
  __shared__ int data[ROW_TILE_W]; // Data cache in shared memory

  const int sx = threadIdx.x;                    // source pixel within source tile
  const int dx = (sx >> 1);                      // dest pixel within dest tile (decimated 2x)
  const int sts = IMUL(blockIdx.x, ROW_TILE_W);  // tile start for source, relative to row start
  const int dts = (sts >>1);                     // tile start for dest, relative to row start
  const int srs = IMUL(blockIdx.y, w);           // Row start index in source data
  const int drs = IMUL(blockIdx.y, (w >> 1));    // Row start index in dest data

  // Load global memory values into our data cache:
  const int loadIdx = sts + sx;  // index of one pixel value to load
  if (loadIdx < w) data[sx] = src[srs + loadIdx];

  int border; if (sx == 0 && sts > 0) border = src[srs + sts - 1]; else border = 0;
  const int ww = (w & 0xfffe); // evened-down source size

  // Ensure the completness of loading stage because results emitted
  // by each thread depend on the data loaded by other threads:
  __syncthreads();

  // only process every other pixel
  if ( (sx & 1) == 0 && loadIdx < ww) {
    const int writeIdx = dts + dx; // write index relative to row start
    const int *dptr = data + sx;

    if (loadIdx == 0) dst[drs + writeIdx] = (dptr[1] + ((*dptr) << 1)) / 3;            // first pixel of image
    else if (sx == 0) dst[drs + writeIdx] = (border + dptr[1] + ((*dptr) << 1) ) >> 2; // first of tile
    else dst[drs + writeIdx] = (dptr[-1] + dptr[1] + ((*dptr) << 1)) >> 2;             // all other pixels
  }
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void cudalowpass5ydecy(const int *src,  const unsigned int w, const unsigned int h,
                                  int* dst, int sms, int gms)
{
  // Data cache
  __shared__ int data[COLUMN_TILE_W * COLUMN_TILE_H];
  __shared__ int border[COLUMN_TILE_W];

  const int sy = threadIdx.y; // source pixel row within source tile
  const int dy = (sy >> 1);   // dest pixel row within dest tile (decimated 2x)

  const int sts = IMUL(blockIdx.y, COLUMN_TILE_H); // tile start for source, in rows
  const int ste = sts + COLUMN_TILE_H; // tile end for source, in rows

  const int dts = (sts >> 1);
  const int dte = (ste >> 1);

  // Clamp tile and apron limits by image borders
  const int stec = min(ste, h);
  const int dtec = min(dte, (h >> 1));

  // Current column index
  const int scs = IMUL(blockIdx.x, COLUMN_TILE_W) + threadIdx.x;
  const int dcs = scs;

  // only process columns that are actually within image bounds:
  if (scs < w) {
    // Shared and global memory indices for current column
    int smemPos = IMUL(sy, COLUMN_TILE_W) + threadIdx.x;
    int gmemPos = IMUL(sts + sy, w) + scs;

    // Cycle through the entire data cache
    // Load global memory values, if indices are within the image borders:
    for (int y = sts + sy; y < stec; y += blockDim.y) {
      data[smemPos] = src[gmemPos];
      smemPos += sms; gmemPos += gms;
    }

    if (sy == 0 && sts > 0) border[threadIdx.x] = src[IMUL(sts - 1, w) + scs];

    // Ensure the completness of loading stage because results emitted
    // by each thread depend on the data loaded by other threads:
    __syncthreads();

    // only process every other row
    if ((sy & 1) == 0) {
      // Shared and global memory indices for current column
      smemPos = IMUL(sy, COLUMN_TILE_W) + threadIdx.x;
      gmemPos = IMUL(dts + dy, w) + dcs;
  
      // Cycle through the tile body, clamped by image borders
      // Calculate and output the results
      int *dptr = data + smemPos;
      int dgms = (gms >> 1);  // memory stride for dest

      if (sts + sy == 0) { // top row of image
        dst[gmemPos] = (dptr[COLUMN_TILE_W] + ((*dptr) << 1)) / 3;
        dptr += sms; gmemPos += dgms;
        for (int y = sts + sy + blockDim.y; y < stec; y += blockDim.y) {
          dst[gmemPos] = (dptr[-COLUMN_TILE_W] + dptr[COLUMN_TILE_W] + ((*dptr) << 1)) >> 2;
          dptr += sms; gmemPos += dgms;
        }
      } else if (sy == 0) { // top row of a tile
        dst[gmemPos] = (border[threadIdx.x] + dptr[COLUMN_TILE_W] + ((*dptr) << 1)) >> 2;
        dptr += sms; gmemPos += dgms;
        for (int y = sts + sy + blockDim.y; y < stec; y += blockDim.y) {
          dst[gmemPos] = (dptr[-COLUMN_TILE_W] + dptr[COLUMN_TILE_W] + ((*dptr) << 1)) >> 2;
          dptr += sms; gmemPos += dgms;
        }
      } else { // all other rows
        for (int y = sts + sy; y < stec; y += blockDim.y) {
          dst[gmemPos] = (dptr[-COLUMN_TILE_W] + dptr[COLUMN_TILE_W] + ((*dptr) << 1)) >> 2;
          dptr += sms; gmemPos += dgms;
        }
      }
    }
  }
}

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// ######################################################################
void cuda_lowpass_5_x_dec_x_fewbits_optim(const int* src, const unsigned int w, const unsigned int h, int* dst)
{
  dim3 blockGridRows(iDivUp(w, ROW_TILE_W), h);
  dim3 threadBlockRows(ROW_TILE_W);

  cudalowpass5xdecx<<<blockGridRows, threadBlockRows>>>(src, w, h, dst);
}

// ######################################################################
void cuda_lowpass_5_y_dec_y_fewbits_optim(const int* src, const unsigned int w, const unsigned int h, int* dst)
{
  dim3 blockGridColumns(iDivUp(w, COLUMN_TILE_W), iDivUp(h, COLUMN_TILE_H));
  dim3 threadBlockColumns(COLUMN_TILE_W, 8);

  cudalowpass5ydecy<<<blockGridColumns, threadBlockColumns>>>(src, w, h, dst, 
                                                              COLUMN_TILE_W * threadBlockColumns.y,
                                                              w * threadBlockColumns.y);
}


__global__ void cudalowpass9x(const int* src, const unsigned int w, const unsigned int h, int* dst)
{
  __shared__ int data[ROW_TILE_W]; // Data cache in shared memory
  __shared__ int border[6];  // Bordering data flanking this tile
  const int sx = threadIdx.x;                   // source pixel within source tile
  const int sts = IMUL(blockIdx.x, ROW_TILE_W); // tile start for source, relative to row start
  const int srs = IMUL(blockIdx.y,w);           // Row start index in source data

  const int loadIdx = sts + sx; // index of one pixel value to load
  const int off = sx - 3; // Offset for the data storing

  // Load border pixels
  if (sx < 3 && sts > 0) border[sx] = src[srs + sts - (3-sx)];
  if (sx >= ROW_TILE_W-3 && sts+ROW_TILE_W < w-3) border[3+sx-(ROW_TILE_W-3)] = src[srs + sts + sx + 3];
 
 // Load the row into shared memory among the thread block
  if (loadIdx < w) 
    data[sx] = src[srs + loadIdx];
  else
    return; // Threads that are over the edge of the image on the right most tile...

  // Ensure the completness of loading stage because results emitted
  // by each thread depend on the data loaded by other threads:
  __syncthreads();

  // First part of source row, just reduce sample
  if(sts+sx < 3)
  {
    switch(sx)
    {
    case 0:
      dst[srs + loadIdx] =                 // [ (72^) 56 28 8 ]
                           (data[0]* 72 +
                            data[1] * 56 +
                            data[2] * 28 +
                            data[3] *  8
                            ) / 164;
      break;
    case 1:
      dst[srs + loadIdx] =                  // [ 56^ (72) 56 28 8 ]
                           ((data[0] + data[2]) * 56 +
                            data[1] * 72 +
                            data[3] * 28 +
                            data[4] *  8
                            ) / 220;
      break;
    case 2:
      dst[srs + loadIdx] =                  // [ 28^ 56 (72) 56 28 8 ]
                           ((data[0] + data[4]) * 28 +
                            (data[1] + data[3]) * 56 +
                            data[2] * 72 +
                            data[5] *  8
                            ) / 248;
    default:
      //LERROR();
      break;
    }
  }
  // If not the first part of the soure row, but is the first bit of this tile, use border
  else if(sx < 3 && sts+sx < w-3)
  {
    switch(sx)
    {
    case 0:
      dst[srs + loadIdx] =              // [ 8^ 28 56 (72) 56 28 8 ]
        ((border[0] + data[off+6]) *  8 +
         (border[1] + data[off+5]) * 28 +
         (border[2] + data[off+4]) * 56 +
         data[off+3] * 72
         ) >> 8;
      break;
    case 1:
      dst[srs + loadIdx] =              // [ 8^ 28 56 (72) 56 28 8 ]
        ((border[1] + data[off+6]) *  8 +
         (border[2] + data[off+5]) * 28 +
         (data[off+2] + data[off+4]) * 56 +
         data[off+3] * 72
         ) >> 8;
      break;    
    case 2:
      dst[srs + loadIdx] =              // [ 8^ 28 56 (72) 56 28 8 ]
        ((border[2] + data[off+6]) *  8 +
         (data[off+1] + data[off+5]) * 28 +
         (data[off+2] + data[off+4]) * 56 +
         data[off+3] * 72
         ) >> 8;
      break;
    }
  }
  // If we are not near the edge of this tile, do standard way
  else if(sx < ROW_TILE_W-3 && sts +sx < w-3)
  {
    dst[srs + loadIdx] =              // [ 8^ 28 56 (72) 56 28 8 ]
      ((data[off+0] + data[off+6]) *  8 +
       (data[off+1] + data[off+5]) * 28 +
       (data[off+2] + data[off+4]) * 56 +
       data[off+3] * 72
       ) >> 8;
  }
  // If not the last part of the source row, but in the last bit of this tile, use border
  else if(sts+sx < w-3)
  {
    switch(sx)
      {
      case ROW_TILE_W-3:
        dst[srs + loadIdx] =              // [ 8^ 28 56 (72) 56 28 8 ]
          ((data[off+0] + border[3]) *  8 +
           (data[off+1] + data[off+5]) * 28 +
           (data[off+2] + data[off+4]) * 56 +
           data[off+3] * 72
           ) >> 8;
        break;
      case ROW_TILE_W-2:
        dst[srs + loadIdx] =              // [ 8^ 28 56 (72) 56 28 8 ]
          ((data[off+0] + border[4]) *  8 +
           (data[off+1] + border[3]) * 28 +
           (data[off+2] + data[off+4]) * 56 +
           data[off+3] * 72
           ) >> 8;
        break;    
      case ROW_TILE_W-1:
        dst[srs + loadIdx] =              // [ 8^ 28 56 (72) 56 28 8 ]
          ((data[off+0] + border[5]) *  8 +
           (data[off+1] + border[4]) * 28 +
           (data[off+2] + border[3]) * 56 +
           data[off+3] * 72
           ) >> 8;
        break;
  }
  }
  // If in the last bit of the source row, reduce sample
  else if(sts + sx < w)
  {
//      dst[srs  + loadIdx] =                  // [ 8^ 28 56 (72) ]
//         (data[off+0] *  8 +
//          data[off+1] * 28 +
//          data[off+2] * 56 +
//          data[off+3] * 72
//          ) / 164;
//   }
//   else if(sx < 0)
//   {
    switch(w-(sts+sx))
    {
    case 3:
      dst[srs + loadIdx] =                  // [ 8^ 28 56 (72) 56 28 ]
        (data[off+0] *  8 +
         (data[off+1] + data[off+5]) * 28 +
         (data[off+2] + data[off+4]) * 56 +
         data[off+3] * 72
         ) / 248;
      break;
    case 2:
      dst[srs + loadIdx] =                  // [ 8^ 28 56 (72) 56 ]
        (data[off+0] *  8 +
         data[off+1] * 28 +
         (data[off+2] + data[off+4]) * 56 +
         data[off+3] * 72
         ) / 220;
      break;
    case 1:
      dst[srs  + loadIdx] =                  // [ 8^ 28 56 (72) ]
        (data[off+0] *  8 +
         data[off+1] * 28 +
         data[off+2] * 56 +
         data[off+3] * 72
         ) / 164;
      break;
    default:
      dst[srs + loadIdx] = sx;
    }
  }
}

// ######################################################################
__global__ void cudalowpass9y(const int* src,
                                     const unsigned int w,
                                     const unsigned int h,
                                     int* dst, int sms, int gms)
{

  // Data cache
  __shared__ int data[COLUMN_TILE_W * COLUMN_TILE_H];
  __shared__ int border[COLUMN_TILE_W * 6];

  const int sy = threadIdx.y; // source pixel row within source tile

  const int sts = IMUL(blockIdx.y, COLUMN_TILE_H); // tile start for source, in rows
  const int ste = sts + COLUMN_TILE_H; // tile end for source, in rows


  // Clamp tile and apron limits by image borders
  const int stec = min(ste, h);

  // Current column index
  const int scs = IMUL(blockIdx.x, COLUMN_TILE_W) + threadIdx.x;

  // only process columns that are actually within image bounds:
  if (scs < w && sts+sy < stec) 
  {
    // Shared and global memory indices for current column
    int smemPos = IMUL(sy, COLUMN_TILE_W) + threadIdx.x;
    int gmemPos = IMUL(sts + sy, w) + scs;

    // Cycle through the entire data cache
    // Load global memory values, if indices are within the image borders:
//     for (int y = sts + sy; y < stec; y += blockDim.y) {
//       data[smemPos] = src[gmemPos];
//       smemPos += sms; gmemPos += gms;
//     }
    data[smemPos] = src[gmemPos];

    if (sy < 3 && gmemPos > IMUL(3,w)) 
      border[smemPos] = src[gmemPos-IMUL(3,w)];
      //border[threadIdx.x+IMUL(sy,COLUMN_TILE_W)] = src[IMUL(sts-(3-sy), w) + scs];

    int bordOff = 6+sy-COLUMN_TILE_H;

    if (sy >= COLUMN_TILE_H-3 && ste+3 < h) //blockIdx.y < blockDim.y-1) 
      border[threadIdx.x+IMUL(bordOff,COLUMN_TILE_W)] = src[gmemPos+IMUL(3,w)];
      //border[threadIdx.x+IMUL(bordOff,COLUMN_TILE_W)] = src[IMUL(ste-1+COLUMN_TILE_H-sy, w) + scs];

    // Ensure the completness of loading stage because results emitted
    // by each thread depend on the data loaded by other threads:
    __syncthreads();

    // Shared and global memory indices for current column
    smemPos = IMUL(sy, COLUMN_TILE_W) + threadIdx.x;
    gmemPos = IMUL(sts + sy, w) + scs;
  

    // Setup the offsets to get to the correct smem points in the arrays for both the data and the border
    int *dptr = data + smemPos;
    const int sw = COLUMN_TILE_W, sw2 = sw + sw, sw3 = sw2 + sw;
    const int nsw = -sw, nsw2 = nsw - sw, nsw3 = nsw2 - sw;
    const int bn3 = threadIdx.x, bn2 = bn3 + COLUMN_TILE_W, bn1 = bn2 + COLUMN_TILE_W;
    const int bp1 = bn1+COLUMN_TILE_W, bp2 = bp1 + COLUMN_TILE_W, bp3 = bp2 + COLUMN_TILE_W;

    // Are we in the top 3 rows of the whole image
    if(sts + sy < 3)
    {
      switch(sts+sy)
      {
      case 0:
        dst[gmemPos] =
          (dptr[0] * 72 +
           dptr[sw] * 56 +
           dptr[sw2] * 28 +
           dptr[sw3] *  8
           ) / 164;
        break;
      case 1:
        dst[gmemPos] =
          (dptr[0] * 72 +
           (dptr[nsw] + dptr[sw]) * 56 +
           dptr[sw2] * 28 +
           dptr[sw3] *  8
           ) / 220;
        break;
      case 2:
        dst[gmemPos] =
          (dptr[0] * 72 +
           (dptr[nsw] + dptr[sw]) * 56 +
           (dptr[nsw2] + dptr[sw2]) * 28 +
           dptr[sw3] *  8
           ) / 248;
        break;
      }
    }
    else if(sy < 3 && sts+sy<h-3) // If not top 3 in the whole image, are we in the top 3 rows of this tile
    {
      switch(sy)
      {
      case 0:
        dst[gmemPos] =
          (dptr[0] * 72 +
           (border[bn1] + dptr[sw]) * 56 +
           (border[bn2] + dptr[sw2]) * 28 +
           (border[bn3] + dptr[sw3]) *  8
           ) >> 8;
        break;
      case 1:
        dst[gmemPos] =
          (dptr[0] * 72 +
           (dptr[nsw] + dptr[sw]) * 56 +
           (border[bn1] + dptr[sw2]) * 28 +
           (border[bn2] + dptr[sw3]) *  8
           ) >> 8;
        break;
      case 2:
        dst[gmemPos] =
          (dptr[0] * 72 +
           (dptr[nsw] + dptr[sw]) * 56 +
           (dptr[nsw2] + dptr[sw2]) * 28 +
           (border[bn1] + dptr[sw3]) *  8
           ) >> 8;
        break;
      }      
    }
    else if(sy <COLUMN_TILE_H-3 && sts+sy<h-3)//(sy < COLUMN_TILE_H-4 && sts+sy<h-3) // Are we in the middle of the tile
    {
        dst[gmemPos] =
          ((dptr[nsw3] + dptr[sw3]) *  8 +
           (dptr[nsw2] + dptr[sw2]) * 28 +
           (dptr[nsw] + dptr[sw]) * 56 +
           dptr[0] * 72
           ) >> 8;
    }
    else if(sts + sy < h-3) // Are we not at the bottom of the image, but bottom 3 of the tile
    {
      switch(sy)
      {
      case COLUMN_TILE_H-3:
        dst[gmemPos] =
          (dptr[0] * 72 +
           (dptr[nsw] + dptr[sw]) * 56 +
           (dptr[nsw2] + dptr[sw2]) * 28 +
           (dptr[nsw3] + border[bp1]) *  8
           ) >> 8;
        break;
      case COLUMN_TILE_H-2:
        dst[gmemPos] =
          (dptr[0] * 72 +
           (dptr[nsw] + dptr[sw]) * 56 +
           (dptr[nsw2] + border[bp1]) * 28 +
           (dptr[nsw3] + border[bp2]) *  8
           ) >> 8;
        break;
      case COLUMN_TILE_H-1:
        dst[gmemPos] =
          (dptr[0] * 72 +
           (dptr[nsw] + border[bp1]) * 56 +
           (dptr[nsw2] + border[bp2]) * 28 +
           (dptr[nsw3] + border[bp3]) *  8
           ) >> 8;
        break;
      }
    }
    else // We must be at the bottom 3 of the image
    {
      switch(h-(sts+sy))
      {
      case 3:
        dst[gmemPos] =
          (dptr[0] * 72 +
           (dptr[nsw] + dptr[sw]) * 56 +
           (dptr[nsw2] + dptr[sw2]) * 28 +
           dptr[nsw3] *  8 
           ) / 248;
        break;
      case 2:
        dst[gmemPos] =
          (dptr[0] * 72 +
           (dptr[nsw] + dptr[sw]) * 56 +
           dptr[nsw2] * 28 +
           dptr[nsw3] *  8
           ) / 220;
        break;
      case 1:
        dst[gmemPos] =
          (dptr[0] * 72 +
           dptr[nsw] * 56 +
           dptr[nsw2] * 28 +
           dptr[nsw3] *  8
           ) / 164;
        break;
      }
    }   
  }
}


// #####################################################################
void cuda_lowpass_9_x_fewbits_optim(const int* src,
                                     const unsigned int w,
                                     const unsigned int h,
                                     int* dst)
{
  //ENV_ASSERT(w >= 9);
  dim3 blockGridRows(iDivUp(w, ROW_TILE_W), h);
  dim3 threadBlockRows(ROW_TILE_W);
  cudalowpass9x<<<blockGridRows, threadBlockRows>>>(src, w, h, dst);
}

void cuda_lowpass_9_y_fewbits_optim(const int* src,
                                     const unsigned int w,
                                     const unsigned int h,
                                     int* dst)
{
  //ENV_ASSERT(h >= 9);
  dim3 blockGridColumns(iDivUp(w, COLUMN_TILE_W), iDivUp(h, COLUMN_TILE_H));
  dim3 threadBlockColumns(COLUMN_TILE_W, COLUMN_TILE_H);

  cudalowpass9y<<<blockGridColumns, threadBlockColumns>>>(src, w, h, dst, COLUMN_TILE_W*threadBlockColumns.y,
                                                          w*threadBlockColumns.y);
}





// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */
