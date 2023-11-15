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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/cuda_lowpass.h $
// $Id: cuda_lowpass.h 12962 2010-03-06 02:13:53Z irock $
//

#ifndef CUDA_LOWPASS_H_DEFINED
#define CUDA_LOWPASS_H_DEFINED

#include <cuda.h>
#include "CUDA/cutil.h"
#include "cudadefs.h"

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_global_lowpass_5_x_dec_x(const float *src,  const unsigned int w, const unsigned int h, float* dst, int tile_width)
{
  // Data cache in shared memory
  //__shared__ float data[CUDA_1D_TILE_W];
  //__shared__ float border[4];
  //tile_width=CUDA_1D_TILE_W;

  // Save 4 pixels for the border
  float *data = (float *) shared_data; // size of tile_width
  float *border = (float *) &data[tile_width]; // size of 4

  const int sx = threadIdx.x;                    // source pixel within source tile
  const int dx = (sx >> 1);                      // dest pixel within dest tile (decimated 2x)
  const int sts = IMUL(blockIdx.x, tile_width);  // tile start for source, relative to row start
  const int dts = (sts >>1);                     // tile start for dest, relative to row start
  const int srs = IMUL(blockIdx.y, w);           // Row start index in source data
  const int drs = IMUL(blockIdx.y, (w >> 1));    // Row start index in dest data

  // Load global memory values into our data cache:
  const int loadIdx = sts + sx;  // index of one pixel value to load
  if (loadIdx < w) data[sx] = src[srs + loadIdx];

  // Load beginning border
  if (sx < 2 && sts > 0)
    border[sx] = src[srs + loadIdx - 2];
  // Load ending border
  else if(sx >= tile_width-2 && sts+tile_width < w-2)
    border[4+sx-tile_width] = src[srs + sts + sx + 2];

  const int ww = (w & 0xfffe); // evened-down source size

  // Ensure the completness of loading stage because results emitted
  // by each thread depend on the data loaded by other threads:
  __syncthreads();

  // only process every other pixel
  if ( (sx & 1) == 0 && loadIdx < ww) {
    const int writeIdx = dts + dx; // write index relative to row start
    const float *dptr = data + sx;
    // [ 1 4 (6) 4 1 ] / 16.0

    // If we are smaller than the Gaussian filter we are using, special case this
    // this is not very efficient, just making sure it can handle small images
    if(w < 5)
    {
      int numAhead = max(0,min(w-1-(sts+sx),2));
      int numBehind = max(0,min(sts+sx,2));
      int situation = numBehind*10+numAhead;
      switch(situation)
      {
      case 0: // 00
        dst[drs + writeIdx] = dptr[0];
        break;
      case 1: // 01
        dst[drs + writeIdx] = (dptr[0] * (6.0F / 10.0F) +
                               dptr[1] * (4.0F / 10.0F));
        break;
      case 2: // 02
        dst[drs + writeIdx] = (dptr[0] * (6.0F / 11.0F) +
                               dptr[1] * (4.0F / 11.0F) +
                               dptr[2] * (1.0F / 11.0F));
        break;
      case 10:
        dst[drs + writeIdx] = (dptr[0] * (6.0F / 10.0F) +
                               dptr[-1] * (4.0F / 10.0F));
        break;
      case 11:
        dst[drs + writeIdx] = (dptr[0]              * (6.0F / 14.0F) +
                               (dptr[-1] + dptr[1]) * (4.0F / 14.0F));
        break;
      case 12:
        dst[drs + writeIdx] = (dptr[0]              * (6.0F / 15.0F) +
                               (dptr[-1] + dptr[1]) * (4.0F / 15.0F) +
                               dptr[2]              * (1.0F / 15.0F));
        break;
      case 20:
        dst[drs + writeIdx] = (dptr[0] * (6.0F / 11.0F) +
                               dptr[-1] * (4.0F / 11.0F) +
                               dptr[-2] * (1.0F / 11.0F));
        break;
      case 21:
        dst[drs + writeIdx] = (dptr[0]              * (6.0F / 15.0F) +
                               (dptr[-1] + dptr[1]) * (4.0F / 15.0F) +
                               dptr[-2]              * (1.0F / 15.0F));
        break;
      }
    }
    // First set of pixels in the row
    else if(sts+sx < 2)
    {
      switch(sx)
      {
      case 0:
        dst[drs + writeIdx] = (dptr[0] * (6.0F / 11.0F) +
                               dptr[1] * (4.0F / 11.0F) +
                               dptr[2] * (1.0F / 11.0F));
        break;
      case 1:
        dst[drs + writeIdx] = (dptr[0]              * (6.0F / 15.0F) +
                               (dptr[-1] + dptr[1]) * (4.0F / 15.0F) +
                               dptr[2]              * (1.0F / 15.0F));
        break;
      }
    }
    // First two pixels in tile
    else if(sx < 2 && sts+sx < w-2)
    {
      switch(sx)
      {
      case 0:
        dst[drs + writeIdx] = (dptr[0]               * (6.0F / 16.0F) +
                               (border[1] + dptr[1]) * (4.0F / 16.0F) +
                               (border[0] + dptr[2]) * (1.0F / 16.0F));
        break;
      case 1:
        dst[drs + writeIdx] = (dptr[0]               * (6.0F / 16.0F) +
                               (dptr[-1] + dptr[1])  * (4.0F / 16.0F) +
                               (border[1] + dptr[2]) * (1.0F / 16.0F));
        break;
      }
    }
    // In the middle of the tile
    else if(sx < tile_width-2 && sts+sx < w-2)
    {
      dst[drs + writeIdx] = (dptr[0]              * (6.0F / 16.0F) +
                             (dptr[-1] + dptr[1]) * (4.0F / 16.0F) +
                             (dptr[-2] + dptr[2]) * (1.0F / 16.0F));
    }
    // Last two pixels of the tile
    else if(sts+sx < w-2)
    {
      switch(tile_width-sx)
      {
      case 2:
        dst[drs + writeIdx] = (dptr[0]                * (6.0F / 16.0F) +
                               (dptr[-1] + dptr[1])   * (4.0F / 16.0F) +
                               (dptr[-2] + border[2]) * (1.0F / 16.0F));
        break;
      case 1:
        dst[drs + writeIdx] = (dptr[0]                * (6.0F / 16.0F) +
                               (dptr[-1] + border[2]) * (4.0F / 16.0F) +
                               (dptr[-2] + border[3]) * (1.0F / 16.0F));
        break;
      }
    }
    // Last two pixels of the row
    else
    {
      switch(w-(sts+sx))
      {
      case 2:
        dst[drs + writeIdx] = (dptr[0]              * (6.0F / 15.0F) +
                               (dptr[-1] + dptr[1]) * (4.0F / 15.0F) +
                               (dptr[-2])           * (1.0F / 15.0F));
        break;
      case 1:
        dst[drs + writeIdx] = (dptr[0]    * (6.0F / 11.0F) +
                               (dptr[-1]) * (4.0F / 11.0F) +
                               (dptr[-2]) * (1.0F / 11.0F));
        break;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void cuda_global_lowpass_5_y_dec_y(const float *src,  const unsigned int w, const unsigned int h, float* dst, int tile_width, int tile_height)
{
  // Data cache
  //__shared__ float data[CUDA_TILE_W*CUDA_TILE_H];
  //__shared__ float border[CUDA_TILE_W*4];
  //const int tile_width=CUDA_TILE_W;
  //const int tile_height= CUDA_TILE_H;

  // Save 4 rows for the border
  float *data = (float *) shared_data; //tile_width * tile_height size
  float *border = (float *) &data[tile_width*tile_height]; // size of tile_width*4

  const int sy = threadIdx.y; // source pixel row within source tile
  const int dy = (sy >> 1);   // dest pixel row within dest tile (decimated 2x)

  const int sts = IMUL(blockIdx.y, tile_height); // tile start for source, in rows
  const int ste = sts + tile_height; // tile end for source, in rows

  const int dts = (sts >> 1);
  const int dte = (ste >> 1);

  // Clamp tile and apron limits by image borders
  const int stec = min(ste, h);
  const int dtec = min(dte, (h >> 1));

  // Current column index
  const int scs = IMUL(blockIdx.x, tile_width) + threadIdx.x;
  const int dcs = scs;

  // only process columns that are actually within image bounds:
  if (scs < w) {
    // Shared and global (source) memory indices for current column
    int smemPos = IMUL(sy, tile_width) + threadIdx.x;
    int gmemPos = IMUL(sts + sy, w) + scs;

    // Load data
    data[smemPos] = src[gmemPos];

    // Load border
    if (sy < 2 && gmemPos > IMUL(2,w))
      border[smemPos] = src[gmemPos-IMUL(2,w)];

    int bordOff = 4+sy-tile_height;

    if (sy >= tile_height-2 && ste+2 < h)
      border[threadIdx.x+IMUL(bordOff,tile_width)] = src[gmemPos+IMUL(2,w)];

    // Ensure the completness of loading stage because results emitted
    // by each thread depend on the data loaded by other threads:
    __syncthreads();

    // only process every other row
    if ((sy & 1) == 0) {
      // Shared and global (destination) memory indices for current column
      smemPos = IMUL(sy, tile_width) + threadIdx.x;
      gmemPos = IMUL(dts + dy, w) + dcs;

      // Cycle through the tile body, clamped by image borders
      // Calculate and output the results
      float *dptr = data + smemPos;
      const int sw = tile_width, sw2 = sw + sw;
      const int nsw = -sw, nsw2 = nsw - sw;
      const int bn2 = threadIdx.x, bn1 = bn2 + tile_width;
      const int bp1 = bn1+tile_width, bp2 = bp1 + tile_width;

      //  [ 1 4 (6) 4 1 ] / 16

    // If we are smaller than the Gaussian filter we are using, special case this
    // this is not very efficient, just making sure it can handle small images
      if(h < 5)
      {
        int numAhead = max(0,min(h-1-(sts+sy),2));
        int numBehind = max(0,min(sts+sy,2));
        int situation = numBehind*10+numAhead;
        switch(situation)
        {
        case 0: // 00
          dst[gmemPos] = dptr[0];
          break;
        case 1: // 01
          dst[gmemPos] = (dptr[0] * (6.0F / 10.0F) +
                          dptr[sw] * (4.0F / 10.0F));
          break;
        case 2: // 02
          dst[gmemPos] = (dptr[0] * (6.0F / 11.0F) +
                          dptr[sw] * (4.0F / 11.0F) +
                          dptr[sw2] * (1.0F / 11.0F));
          break;
        case 10:
          dst[gmemPos] = (dptr[0] * (6.0F / 10.0F) +
                          dptr[nsw] * (4.0F / 10.0F));
          break;
        case 11:
          dst[gmemPos] = (dptr[0]              * (6.0F / 14.0F) +
                          (dptr[nsw] + dptr[sw]) * (4.0F / 14.0F));
          break;
        case 12:
          dst[gmemPos] = (dptr[0]              * (6.0F / 15.0F) +
                          (dptr[nsw] + dptr[sw]) * (4.0F / 15.0F) +
                          dptr[sw2]              * (1.0F / 15.0F));
          break;
        case 20:
          dst[gmemPos] = (dptr[0] * (6.0F / 11.0F) +
                          dptr[nsw] * (4.0F / 11.0F) +
                          dptr[nsw2] * (1.0F / 11.0F));
          break;
        case 21:
          dst[gmemPos] = (dptr[0]              * (6.0F / 15.0F) +
                          (dptr[nsw] + dptr[sw]) * (4.0F / 15.0F) +
                          dptr[nsw2]              * (1.0F / 15.0F));
          break;
        }
      }
      // Are we in the top 2 rows of the whole image
      else if(sts + sy < 2)
      {
        switch(sts+sy)
        {
        case 0:
          dst[gmemPos] =
            (dptr[0]   * (6.0F / 11.0F) +
             dptr[sw]  * (4.0F / 11.0F) +
             dptr[sw2] * (1.0F / 11.0F)
             );
          break;
        case 1:
          dst[gmemPos] =
            (dptr[0]                * (6.0F / 15.0F) +
             (dptr[nsw] + dptr[sw]) * (4.0F / 15.0F) +
             dptr[sw2]              * (1.0F / 15.0F)
             );
          break;
        }
      }
      else if(sy < 2 && sts+sy<h-2) // If not top 2 in the whole image, are we in the top 2 rows of this tile
      {
        switch(sy)
        {
        case 0:
          dst[gmemPos] =
            (dptr[0]                   * (6.0F / 16.0F) +
             (border[bn1] + dptr[sw])  * (4.0F / 16.0F) +
             (border[bn2] + dptr[sw2]) * (1.0F / 16.0F)
             );
          break;
        case 1:
          dst[gmemPos] =
            (dptr[0]                   * (6.0F / 16.0F) +
             (dptr[nsw] + dptr[sw])    * (4.0F / 16.0F) +
             (border[bn1] + dptr[sw2]) * (1.0F / 16.0F)
             );
          break;
        }
      }
      else if(sy <tile_height-2 && sts+sy<h-2) // Are we in the middle of the tile
      {
        dst[gmemPos] =
          ((dptr[nsw2] + dptr[sw2]) * (1.0F / 16.0F) +
           (dptr[nsw] + dptr[sw])   * (4.0F / 16.0F) +
           dptr[0]                  * (6.0F / 16.0F)
           );
      }
      else if(sts + sy < h-2) // Are we not at the bottom of the image, but bottom 4 of the tile
      {
        switch(tile_height-sy)
        {
        case 2:
          dst[gmemPos] =
            (dptr[0]                    * (6.0F / 16.0F) +
             (dptr[nsw] + dptr[sw])     * (4.0F / 16.0F) +
             (dptr[nsw2] + border[bp1]) * (1.0F / 16.0F)
             );
          break;
        case 1:
          dst[gmemPos] =
            (dptr[0]                    * (6.0F / 16.0F) +
             (dptr[nsw] +  border[bp1]) * (4.0F / 16.0F) +
             (dptr[nsw2] + border[bp2]) * (1.0F / 16.0F)
             );
          break;
        }
      }
      else // We must be at the bottom 4 of the image
      {
        switch(h-(sts+sy))
          {
          case 2:
            dst[gmemPos] =
              (dptr[0]                * (6.0F / 15.0F) +
               (dptr[nsw] + dptr[sw]) * (4.0F / 15.0F) +
               dptr[nsw2]             * (1.0F / 15.0F)
               );
            break;
          case 1:
            dst[gmemPos] =
              (dptr[0]    * (6.0F / 11.0F) +
               dptr[nsw]  * (4.0F / 11.0F) +
               dptr[nsw2] * (1.0F / 11.0F)
               );
            break;
          }
      }

    }
  }
}


__global__ void cuda_global_lowpass_9_x_dec_x(const float* src, const unsigned int w, const unsigned int h, float* dst, const int dw, const int dh, int tile_width)
{
  // w and h are from the original source image
  const int src_tile_width = tile_width<<1; // Size of the tile used from the source data
  // Data cache in shared memory
  float *data = (float *) shared_data; // size of src_tile_width
  // Bordering data flanking this tile
  float *border = (float *) &data[src_tile_width]; // size of 8
  const int dx = threadIdx.x;                   // dest pixel within dest tile (decimated 2x)
  const int dts = IMUL(blockIdx.x, tile_width); // tile start for dest, relative to row start
  const int drs = IMUL(blockIdx.y,dw);           // Row start index in dest data
  const int sx = dx<<1;                         // source pixel within source tile
  const int sts = (dts <<1);                    // tile start for source, relative to row start
  const int srs = IMUL(blockIdx.y, w);   // Row start index in source data

  const int loadIdx = sts + sx; // index of one pixel value to load
  const int writeIdx = dts + dx; // index of one pixel value to load
  const int bn4 = 0, bn3 = 1, bn2 = 2, bn1 = 3;
  const int bp1 = 4, bp2 = 5, bp3 = 6, bp4 = 7;
  float *dptr = &data[sx];

  // Load border pixels
  if (dx < 4 && sts > 0) border[dx] = src[srs + sts - (4-dx)];
  if (dx >= tile_width-4 && sts+src_tile_width < w-4) border[4+dx-(tile_width-4)] = src[srs + sts + tile_width + dx + 4];

  // Load the row into shared memory among the thread block
  if (loadIdx < w)
    {
      data[sx] = src[srs + loadIdx];
      if(sx+1 < src_tile_width && loadIdx+1 < w)
          data[sx+1] = src[srs + loadIdx + 1];
    }

  // Ensure the completness of loading stage because results emitted
  // by each thread depend on the data loaded by other threads:
  __syncthreads();

  // [ 1 8 28 56 (70) 56 28 8 1 ]




  if(writeIdx < dw && blockIdx.y < dh)
    {
      /* if(true) */
      /*         { */
      /*           dst[drs+writeIdx] = dptr[0]; */
      /*         } */
      if(w < 9)
        {
          // This is not at all efficient, just here to ensure filter behaves properly in small image cases
          const int numAhead = max(0,min(w-1-(sts+sx),4));
          const int numBehind = max(0,min(sts+sx,4));
          const int situation = numBehind*10+numAhead;

          switch(situation)
            {
            case 0: // 00
              dst[drs+writeIdx] = dptr[0];
              break;
            case 2: // 02
              dst[drs+writeIdx] = (dptr[0] * (70.0F / 154.0F) +
                                  dptr[1]     * (56.0F / 154.0F) +
                                  dptr[2]     * (28.0F / 154.0F));
              break;
            case 4:
              dst[drs+writeIdx] = (dptr[0] * (70.0F / 163.0F) +
                                  dptr[1] * (56.0F / 163.0F) +
                                  dptr[2] * (28.0F / 163.0F) +
                                  dptr[3] * ( 8.0F / 163.0F) +
                                  dptr[4] * ( 1.0F / 163.0F));
              break;
            case 10:
              dst[drs+writeIdx] = (dptr[0]  * (70.0F / 126.0F) +
                                  dptr[-1] * (56.0F / 126.0F));
              break;
            case 12:
              dst[drs+writeIdx] = (dptr[0]              * (70.0F / 210.0F) +
                                  (dptr[-1] + dptr[1]) * (56.0F / 210.0F) +
                                  dptr[2]              * (28.0F / 210.0F));
              break;
            case 14:
              dst[drs+writeIdx] = (dptr[0]               * (70.0F / 219.0F) +
                                  (dptr[-1] + dptr[1])  * (56.0F / 219.0F) +
                                  dptr[2]               * (28.0F / 219.0F) +
                                  dptr[3]               * ( 8.0F / 219.0F) +
                                  dptr[4]               * ( 1.0F / 219.0F));
              break;
            case 20:
              dst[drs+writeIdx] = (dptr[0]     * (70.0F / 154.0F) +
                                  dptr[-1]    * (56.0F / 154.0F) +
                                  dptr[-2]    * (28.0F / 154.0F));
              break;
            case 22:
              dst[drs+writeIdx] = (dptr[0]              * (70.0F / 238.0F) +
                                  (dptr[-1] + dptr[1]) * (56.0F / 238.0F) +
                                  (dptr[-2] + dptr[2]) * (28.0F / 238.0F));
              break;
            case 24:
              dst[drs+writeIdx] = (dptr[0]              * (70.0F / 247.0F) +
                                  (dptr[-1] + dptr[1]) * (56.0F / 247.0F) +
                                  (dptr[-2] + dptr[2]) * (28.0F / 247.0F) +
                                  dptr[3]              * ( 8.0F / 247.0F) +
                                  dptr[4]              * ( 1.0F / 247.0F));
              break;
            case 30:
              dst[drs+writeIdx] = (dptr[0]  * (70.0F / 162.0F) +
                                  dptr[-1] * (56.0F / 162.0F) +
                                  dptr[-2] * (28.0F / 162.0F) +
                                  dptr[-3] * ( 8.0F / 162.0F));
              break;
            case 32:
              dst[drs+writeIdx] = (dptr[0]              * (70.0F / 246.0F) +
                                  (dptr[-1] + dptr[1]) * (56.0F / 246.0F) +
                                  (dptr[-2] + dptr[2]) * (28.0F / 246.0F) +
                                  dptr[-3]             * ( 8.0F / 246.0F));
              break;
            case 34:
              dst[drs+writeIdx] = (dptr[0]              * (70.0F / 255.0F) +
                                  (dptr[-1] + dptr[1]) * (56.0F / 255.0F) +
                                  (dptr[-2] + dptr[2]) * (28.0F / 255.0F) +
                                  (dptr[-3] + dptr[3]) * ( 8.0F / 255.0F) +
                                  dptr[4]              * ( 1.0F / 255.0F));
              break;
            case 40:
              dst[drs+writeIdx] = (dptr[0]  * (70.0F / 163.0F) +
                                  dptr[-1] * (56.0F / 163.0F) +
                                  dptr[-2] * (28.0F / 163.0F) +
                                  dptr[-3] * ( 8.0F / 163.0F) +
                                  dptr[-4] * ( 1.0F / 163.0F));
              break;
            case 42:
              dst[drs+writeIdx] = (dptr[0]              * (70.0F / 247.0F) +
                                  (dptr[-1] + dptr[1]) * (56.0F / 247.0F) +
                                  (dptr[-2] + dptr[2]) * (28.0F / 247.0F) +
                                  dptr[-3]             * ( 8.0F / 247.0F) +
                                  dptr[-4]             * ( 1.0F / 247.0F));
              break;
            }
        }
      // First part of source row, just reduce sample
      else if(sts+sx < 4)
        {
          switch(sx)
            {
            case 0:
              dst[drs + writeIdx] =
                (dptr[0] * (70.0F / 163.0F) +
                 dptr[1] * (56.0F / 163.0F) +
                 dptr[2] * (28.0F / 163.0F) +
                 dptr[3] * ( 8.0F / 163.0F) +
                 dptr[4] * ( 1.0F / 163.0F)
                 );
              break;
            case 1:
              dst[drs + writeIdx] =
                (dptr[0]              * (70.0F / 219.0F) +
                 (dptr[-1] + dptr[1]) * (56.0F / 219.0F) +
                 dptr[2]              * (28.0F / 219.0F) +
                 dptr[3]              * ( 8.0F / 219.0F) +
                 dptr[4]              * ( 1.0F / 219.0F)
                 );
              break;
            case 2:
              dst[drs + writeIdx] =
                (dptr[0]              * (70.0F / 247.0F) +
                 (dptr[-1] + dptr[1]) * (56.0F / 247.0F) +
                 (dptr[-2] + dptr[2]) * (28.0F / 247.0F) +
                 dptr[3]              * ( 8.0F / 247.0F) +
                 dptr[4]              * ( 1.0F / 247.0F)
                 );
              break;
            case 3:
              dst[drs + writeIdx] =
                (dptr[0]              * (70.0F / 255.0F) +
                 (dptr[-1] + dptr[1]) * (56.0F / 255.0F) +
                 (dptr[-2] + dptr[2]) * (28.0F / 255.0F) +
                 (dptr[-3] + dptr[3]) * ( 8.0F / 255.0F) +
                 dptr[4]              * ( 1.0F / 255.0F)
                 );
              break;
            default:
              //LERROR();
              break;
            }
        }
      // If not the first part of the source row, but is the first bit of this tile, use border
      else if(sx < 4 && sts+sx < w-4)
        {
          switch(sx)
            {
            case 0:
              dst[drs + writeIdx] =
                ((border[bn4] + dptr[4]) * ( 1.0F / 256.0F) +
                 (border[bn3] + dptr[3]) * ( 8.0F / 256.0F) +
                 (border[bn2] + dptr[2]) * (28.0F / 256.0F) +
                 (border[bn1] + dptr[1]) * (56.0F / 256.0F) +
                 dptr[0]                 * (70.0F / 256.0F)
                 );
              break;
            case 1:
              dst[drs + writeIdx] =
                ((border[bn3] + dptr[4])   * ( 1.0F / 256.0F) +
                 (border[bn2] + dptr[3])   * ( 8.0F / 256.0F) +
                 (border[bn1] + dptr[2])   * (28.0F / 256.0F) +
                 (dptr[-1] + dptr[1])      * (56.0F / 256.0F) +
                 dptr[0]                   * (70.0F / 256.0F)
                 );
              break;
            case 2:
              dst[drs + writeIdx] =
                ((border[bn2] + dptr[4]) * ( 1.0F / 256.0F) +
                 (border[bn1] + dptr[3]) * ( 8.0F / 256.0F) +
                 (dptr[-2] + dptr[2])    * (28.0F / 256.0F) +
                 (dptr[-1] + dptr[1])    * (56.0F / 256.0F) +
                 dptr[0]                 * (70.0F / 256.0F)
                 );
              break;
            case 3:
              dst[drs + writeIdx] =
                ((border[bn1] + dptr[4]) * ( 1.0F / 256.0F) +
                 (dptr[-3] + dptr[3])    * ( 8.0F / 256.0F) +
                 (dptr[-2] + dptr[2])    * (28.0F / 256.0F) +
                 (dptr[-1] + dptr[1])    * (56.0F / 256.0F) +
                 dptr[0]                 * (70.0F / 256.0F)
                 );
              break;
            }
        }
      // If we are not near the edge of this tile, do standard way
      else if(sx>=4 && sx<src_tile_width-4 && sts+sx < w-4)
        {
          dst[drs + writeIdx] =
            ((dptr[-4] + dptr[4]) * ( 1.0F / 256.0F) +
             (dptr[-3] + dptr[3]) * ( 8.0F / 256.0F) +
             (dptr[-2] + dptr[2]) * (28.0F / 256.0F) +
             (dptr[-1] + dptr[1]) * (56.0F / 256.0F) +
             dptr[0]              * (70.0F / 256.0F)
             );
        }
      // If not the last part of the source row, but in the last bit of this tile, use border
      else if(sx>=4 && sts+sx < w-4)
        {
          switch(src_tile_width-sx)
            {
            case 4:
              dst[drs + writeIdx] =
                ((dptr[-4] + border[bp1]) * ( 1.0F / 256.0F) +
                 (dptr[-3] + dptr[3])     * ( 8.0F / 256.0F) +
                 (dptr[-2] + dptr[2])     * (28.0F / 256.0F) +
                 (dptr[-1] + dptr[1])     * (56.0F / 256.0F) +
                 dptr[0]                  * (70.0F / 256.0F)
                 );
              break;
            case 3:
              dst[drs + writeIdx] =
                ((dptr[-4] + border[bp2]) * ( 1.0F / 256.0F) +
                 (dptr[-3] + border[bp1]) * ( 8.0F / 256.0F) +
                 (dptr[-2] + dptr[2])     * (28.0F / 256.0F) +
                 (dptr[-1] + dptr[1])     * (56.0F / 256.0F) +
                 dptr[0]                  * (70.0F / 256.0F)
                 );
              break;
            case 2:
              dst[drs + writeIdx] =
                ((dptr[-4] + border[bp3]) * ( 1.0F / 256.0F) +
                 (dptr[-3] + border[bp2]) * ( 8.0F / 256.0F) +
                 (dptr[-2] + border[bp1]) * (28.0F / 256.0F) +
                 (dptr[-1] + dptr[1])     * (56.0F / 256.0F) +
                 dptr[0]                  * (70.0F / 256.0F)
                 );
              break;
            case 1:
              dst[drs + writeIdx] =
                ((dptr[-4] + border[bp4]) * ( 1.0F / 256.0F) +
                 (dptr[-3] + border[bp3]) * ( 8.0F / 256.0F) +
                 (dptr[-2] + border[bp2]) * (28.0F / 256.0F) +
                 (dptr[-1] + border[bp1]) * (56.0F / 256.0F) +
                 dptr[0]                  * (70.0F / 256.0F)
                 );
              break;
            }
        }
      else if(sx < 4 && sts + sx >= w-4) // We are in the beginning of a tile close to the end of the image
        {
          switch(sx)
            {
            case 0:
              switch(w-(sts+sx))
                {
                case 4:
                  dst[drs + writeIdx] =
                    (dptr[0]                 * (70.0F / 255.0F) +
                     (border[bn1] + dptr[1]) * (56.0F / 255.0F) +
                     (border[bn2] + dptr[2]) * (28.0F / 255.0F) +
                     (border[bn3] + dptr[3]) * ( 8.0F / 255.0F) +
                     border[bn4]             * ( 1.0F / 255.0F)
                     );
                  break;
                case 3:
                  dst[drs + writeIdx] =
                    (dptr[0]                 * (70.0F / 247.0F) +
                     (border[bn1] + dptr[1]) * (56.0F / 247.0F) +
                     (border[bn2] + dptr[2]) * (28.0F / 247.0F) +
                     border[bn3]             * ( 8.0F / 247.0F) +
                     border[bn4]             * ( 1.0F / 247.0F)
                     );
                  break;
                case 2:
                  dst[drs + writeIdx] =
                    (dptr[0]                 * (70.0F / 219.0F) +
                     (border[bn1] + dptr[1]) * (56.0F / 219.0F) +
                     border[bn2]             * (28.0F / 219.0F) +
                     border[bn3]             * ( 8.0F / 219.0F) +
                     border[bn4]             * ( 1.0F / 219.0F)
                     );
                  break;
                case 1:
                  dst[drs + writeIdx] =
                    (dptr[0]     * (70.0F / 163.0F) +
                     border[bn1] * (56.0F / 163.0F) +
                     border[bn2] * (28.0F / 163.0F) +
                     border[bn3] * ( 8.0F / 163.0F) +
                     border[bn4] * ( 1.0F / 163.0F)
                     );
                  break;
                }
              break;
            case 1:
              switch(w-(sts+sx))
                {
                case 4:
                  dst[drs + writeIdx] =
                    (dptr[0]                 * (70.0F / 255.0F) +
                     (dptr[1] + dptr[1])     * (56.0F / 255.0F) +
                     (border[bn1] + dptr[2]) * (28.0F / 255.0F) +
                     (border[bn2] + dptr[3]) * ( 8.0F / 255.0F) +
                     border[bn3]             * ( 1.0F / 255.0F)
                     );
                  break;
                case 3:
                  dst[drs + writeIdx] =
                    (dptr[0]                 * (70.0F / 247.0F) +
                     (dptr[1] + dptr[1])     * (56.0F / 247.0F) +
                     (border[bn1] + dptr[2]) * (28.0F / 247.0F) +
                     border[bn2]             * ( 8.0F / 247.0F) +
                     border[bn3]             * ( 1.0F / 247.0F)
                     );
                  break;
                case 2:
                  dst[drs + writeIdx] =
                    (dptr[0]                * (70.0F / 219.0F) +
                     (dptr[1] + dptr[1])    * (56.0F / 219.0F) +
                     border[bn1]            * (28.0F / 219.0F) +
                     border[bn2]            * ( 8.0F / 219.0F) +
                     border[bn3]            * ( 1.0F / 219.0F)
                     );
                  break;
                case 1:
                  dst[drs + writeIdx] =
                    (dptr[0]     * (70.0F / 163.0F) +
                     dptr[1]     * (56.0F / 163.0F) +
                     border[bn1] * (28.0F / 163.0F) +
                     border[bn2] * ( 8.0F / 163.0F) +
                     border[bn3] * ( 1.0F / 163.0F)
                     );
                  break;
                }
              break;
            case 2:
              switch(w-(sts+sx))
                {
                case 4:
                  dst[drs + writeIdx] =
                    (dptr[0]                  * (70.0F / 255.0F) +
                     (dptr[1] + dptr[1])      * (56.0F / 255.0F) +
                     (dptr[2] + dptr[2])      * (28.0F / 255.0F) +
                     (border[bn1] + dptr[3])  * ( 8.0F / 255.0F) +
                     border[bn2]              * ( 1.0F / 255.0F)
                     );
                  break;
                case 3:
                  dst[drs + writeIdx] =
                    (dptr[0]                 * (70.0F / 247.0F) +
                     (dptr[1] + dptr[1])     * (56.0F / 247.0F) +
                     (dptr[2] + dptr[2])     * (28.0F / 247.0F) +
                     border[bn1]             * ( 8.0F / 247.0F) +
                     border[bn2]             * ( 1.0F / 247.0F)
                     );
                  break;
                case 2:
                  dst[drs + writeIdx] =
                    (dptr[0]                * (70.0F / 219.0F) +
                     (dptr[1] + dptr[1])    * (56.0F / 219.0F) +
                     dptr[2]                * (28.0F / 219.0F) +
                     border[bn1]            * ( 8.0F / 219.0F) +
                     border[bn2]            * ( 1.0F / 219.0F)
                     );
                  break;
                case 1:
                  dst[drs + writeIdx] =
                    (dptr[0]          * (70.0F / 163.0F) +
                     dptr[1]          * (56.0F / 163.0F) +
                     dptr[2]          * (28.0F / 163.0F) +
                     border[bn1]      * ( 8.0F / 163.0F) +
                     border[bn2]      * ( 1.0F / 163.0F)
                     );
                  break;
                }
              break;
            case 3:
              switch(w-(sts+sx))
                {
                case 4:
                  dst[drs + writeIdx] =
                    (dptr[0]               * (70.0F / 255.0F) +
                     (dptr[1] + dptr[1])   * (56.0F / 255.0F) +
                     (dptr[2] + dptr[2])   * (28.0F / 255.0F) +
                     (dptr[3] + dptr[3])   * ( 8.0F / 255.0F) +
                     border[bn1]           * ( 1.0F / 255.0F)
                     );
                  break;
                case 3:
                  dst[drs + writeIdx] =
                    (dptr[0]               * (70.0F / 247.0F) +
                     (dptr[1] + dptr[1])   * (56.0F / 247.0F) +
                     (dptr[2] + dptr[2])   * (28.0F / 247.0F) +
                     dptr[3]               * ( 8.0F / 247.0F) +
                     border[bn1]           * ( 1.0F / 247.0F)
                     );
                  break;
                case 2:
                  dst[drs + writeIdx] =
                    (dptr[0]             * (70.0F / 219.0F) +
                     (dptr[1] + dptr[1]) * (56.0F / 219.0F) +
                     dptr[2]             * (28.0F / 219.0F) +
                     dptr[3]             * ( 8.0F / 219.0F) +
                     border[bn1]         * ( 1.0F / 219.0F)
                     );
                  break;
                case 1:
                  dst[drs + writeIdx] =
                    (dptr[0]     * (70.0F / 163.0F) +
                     dptr[1]     * (56.0F / 163.0F) +
                     dptr[2]     * (28.0F / 163.0F) +
                     dptr[3]     * ( 8.0F / 163.0F) +
                     border[bn1] * ( 1.0F / 163.0F)
                     );
                  break;
                }
              break;
            }
        }

      // If in the last bit of the source row, reduce sample
      else if(sts + sx < w)
        {
          switch(w-(sts+sx))
            {
            case 4:
              dst[drs + writeIdx] =
                (dptr[-4]             * ( 1.0F / 255.0F) +
                 (dptr[-3] + dptr[3]) * ( 8.0F / 255.0F) +
                 (dptr[-2] + dptr[2]) * (28.0F / 255.0F) +
                 (dptr[-1] + dptr[1]) * (56.0F / 255.0F) +
                 dptr[0]              * (70.0F / 255.0F)
                 );
              break;
            case 3:
              dst[drs + writeIdx] =
                (dptr[-4]             * ( 1.0F / 247.0F) +
                 dptr[-3]             * ( 8.0F / 247.0F) +
                 (dptr[-2] + dptr[2]) * (28.0F / 247.0F) +
                 (dptr[-1] + dptr[1]) * (56.0F / 247.0F) +
                 dptr[0]              * (70.0F / 247.0F)
                 );
              break;
            case 2:
              dst[drs + writeIdx] =
                (dptr[-4]             * ( 1.0F / 219.0F) +
                 dptr[-3]             * ( 8.0F / 219.0F) +
                 dptr[-2]             * (28.0F / 219.0F) +
                 (dptr[-1] + dptr[1]) * (56.0F / 219.0F) +
                 dptr[0]              * (70.0F / 219.0F)
                 );
              break;
            case 1:
              dst[drs + writeIdx] =
                (dptr[-4]             * ( 1.0F / 163.0F) +
                 dptr[-3]             * ( 8.0F / 163.0F) +
                 dptr[-2]             * (28.0F / 163.0F) +
                 dptr[-1]             * (56.0F / 163.0F) +
                 dptr[ 0]             * (70.0F / 163.0F)
                 );
              break;
            }
        }
    }
}

// ######################################################################
__global__ void cuda_global_lowpass_9_y_dec_y(const float* src, const unsigned int w, const unsigned int h, float* dst, const int dw, const int dh, int tile_width, int tile_height)
{
  // Data cache
  const int src_tile_height = tile_height<<1;
  float *data = (float *) shared_data; // size of tile_width * src_tile_height
  float *border = (float *) &data[tile_width*src_tile_height]; // size of tile_width * 8

  const int dy = threadIdx.y;                   // dest pixel within dest tile (decimated 2x)
  const int dts = IMUL(blockIdx.y, tile_height); // tile start for dest, relative to row start
  const int dcs = IMUL(blockIdx.x, tile_width) + threadIdx.x;  // Current column index in dest data
  const int sy = dy<<1;                         // source pixel within source tile
  const int sts = dts<<1;                       // tile start for source, in rows
  const int ste = sts + src_tile_height;        // tile end for source, in rows
  const int scs = dcs;                           // Current column index

  // Clamp tile and apron limits by image borders
  const int stec = min(ste, h);


  // Load the top border
  if (dy < 4 && sts+sy > 4)
    {
      border[IMUL(dy,tile_width)+threadIdx.x] = src[IMUL(sts+dy,w)+scs-IMUL(4,w)];
    }
  // Load the bottom border
  else if (dy >= tile_height-4)
    {
      int bordOff = 8+dy-tile_height;
      if(ste+4 <= h)
        //border[threadIdx.x+IMUL(bordOff,tile_width)] = src[gmemPos+IMUL(4,w)];
        border[threadIdx.x+IMUL(bordOff,tile_width)] = src[IMUL(sts+dy+tile_height+4,w)+scs];
      else
        // This is explicitly handling the case where the second to last tile is so close to the edge of the image, that the post border
        // can't fit in the remaining space, maybe we should be handling this as yet another special case, but there are a lot of special
        // cases already in this function (handling smaller than 9 imgs, handling all cases of small last tile)
        border[threadIdx.x+IMUL(bordOff,tile_width)] = 0;
    }


  // only process columns that are actually within image bounds:
  if (scs < w && sts+sy < stec)
  {
    // Shared and global memory indices for current column
    int smemPos = IMUL(sy, tile_width) + threadIdx.x;
    int gmemPos = IMUL(sts + sy, w) + scs;

    // Each thread loads up to two pixels of data
    data[smemPos] = src[gmemPos];
    if(sts+sy+1 < stec)
      data[smemPos+tile_width] = src[gmemPos+w];


    // Ensure the completness of loading stage because results emitted
    // by each thread depend on the data loaded by other threads:
    __syncthreads();

    // Shared and global memory indices for current column
    smemPos = IMUL(sy, tile_width) + threadIdx.x;
    gmemPos = IMUL(dts + dy, dw) + dcs;


    // Setup the offsets to get to the correct smem points in the arrays for both the data and the border
    float *dptr = data + smemPos;
    const int sw = tile_width, sw2 = sw + sw, sw3 = sw2 + sw, sw4 = sw3 + sw;
    const int nsw = -sw, nsw2 = nsw - sw, nsw3 = nsw2 - sw, nsw4 = nsw3 - sw;
    const int bn4 = threadIdx.x, bn3 = bn4 + tile_width, bn2 = bn3 + tile_width, bn1 = bn2 + tile_width;
    const int bp1 = bn1+tile_width, bp2 = bp1 + tile_width, bp3 = bp2 + tile_width, bp4 = bp3 + tile_width;

    if(h < 9)
    {
      // This is not at all efficient, just here to ensure filter behaves properly in small image cases
      const int numAhead = max(0,min(h-1-(sts+sy),4));
      const int numBehind = max(0,min(sts+sy,4));
      const int situation = numBehind*10+numAhead;
      switch(situation)
      {
      case 0: // 00
        dst[gmemPos] = dptr[0];
        break;
      case 1: // 01
        dst[gmemPos] = (dptr[0] * (70.0F / 126.0F) +
                        dptr[sw] * (56.0F / 126.0F));
        break;
      case 2: // 02
        dst[gmemPos] = (dptr[0]   * (70.0F / 154.0F) +
                        dptr[sw]  * (56.0F / 154.0F) +
                        dptr[sw2] * (28.0F / 154.0F));
        break;
      case 3:
        dst[gmemPos] = (dptr[0]   * (70.0F / 162.0F) +
                        dptr[sw]  * (56.0F / 162.0F) +
                        dptr[sw2] * (28.0F / 162.0F) +
                        dptr[sw3] * ( 8.0F / 162.0F));
        break;
      case 4:
        dst[gmemPos] = (dptr[0]   * (70.0F / 163.0F) +
                        dptr[sw]  * (56.0F / 163.0F) +
                        dptr[sw2] * (28.0F / 163.0F) +
                        dptr[sw3] * ( 8.0F / 163.0F) +
                        dptr[sw4] * ( 1.0F / 163.0F));
        break;
      case 20:
        dst[gmemPos] = (dptr[0]    * (70.0F / 154.0F) +
                        dptr[nsw]  * (56.0F / 154.0F) +
                        dptr[nsw2] * (28.0F / 154.0F));
        break;
      case 21:
        dst[gmemPos] = (dptr[0]                 * (70.0F / 210.0F) +
                        (dptr[nsw] + dptr[sw])  * (56.0F / 210.0F) +
                        dptr[nsw2]              * (28.0F / 210.0F));
        break;
      case 22:
        dst[gmemPos] = (dptr[0]                  * (70.0F / 238.0F) +
                        (dptr[nsw] + dptr[sw])   * (56.0F / 238.0F) +
                        (dptr[nsw2] + dptr[sw2]) * (28.0F / 238.0F));
        break;
      case 23:
        dst[gmemPos] = (dptr[0]                  * (70.0F / 246.0F) +
                        (dptr[nsw] + dptr[sw])   * (56.0F / 246.0F) +
                        (dptr[nsw2] + dptr[sw2]) * (28.0F / 246.0F) +
                        dptr[sw3]                * ( 8.0F / 246.0F));
        break;
      case 24:
        dst[gmemPos] = (dptr[0]                  * (70.0F / 247.0F) +
                        (dptr[nsw] + dptr[sw])   * (56.0F / 247.0F) +
                        (dptr[nsw2] + dptr[sw2]) * (28.0F / 247.0F) +
                        dptr[sw3]                * ( 8.0F / 247.0F) +
                        dptr[sw4]                * ( 1.0F / 247.0F));
        break;
      case 40:
        dst[gmemPos] = (dptr[0]    * (70.0F / 163.0F) +
                        dptr[nsw]  * (56.0F / 163.0F) +
                        dptr[nsw2] * (28.0F / 163.0F) +
                        dptr[nsw3] * ( 8.0F / 163.0F) +
                        dptr[nsw4] * ( 1.0F / 163.0F));
        break;
      case 41:
        dst[gmemPos] = (dptr[0]                 * (70.0F / 219.0F) +
                        (dptr[nsw] + dptr[sw])  * (56.0F / 219.0F) +
                        dptr[nsw2]              * (28.0F / 219.0F) +
                        dptr[nsw3]              * ( 8.0F / 219.0F) +
                        dptr[nsw4]              * ( 1.0F / 219.0F));
        break;
      case 42:
        dst[gmemPos] = (dptr[0]                  * (70.0F / 247.0F) +
                        (dptr[nsw] + dptr[sw])   * (56.0F / 247.0F) +
                        (dptr[nsw2] + dptr[sw2]) * (28.0F / 247.0F) +
                        dptr[nsw3]               * ( 8.0F / 247.0F) +
                        dptr[nsw4]               * ( 1.0F / 247.0F));
        break;
      case 43:
        dst[gmemPos] = (dptr[0]                  * (70.0F / 255.0F) +
                        (dptr[nsw] + dptr[sw])   * (56.0F / 255.0F) +
                        (dptr[nsw2] + dptr[sw2]) * (28.0F / 255.0F) +
                        (dptr[nsw3] + dptr[sw3]) * ( 8.0F / 255.0F) +
                        dptr[nsw4]               * ( 1.0F / 255.0F));
        break;
      }
    }
    // Are we in the top 4 rows of the whole image
    else if(sts + sy < 4)
    {
      switch(sts+sy)
      {
      case 0:
        dst[gmemPos] =
          (dptr[0]   * (70.0F / 163.0F) +
           dptr[sw]  * (56.0F / 163.0F) +
           dptr[sw2] * (28.0F / 163.0F) +
           dptr[sw3] * ( 8.0F / 163.0F) +
           dptr[sw4] * ( 1.0F / 163.0F)
           );
        break;
      case 1:
        dst[gmemPos] =
          (dptr[0]                * (70.0F / 219.0F) +
           (dptr[nsw] + dptr[sw]) * (56.0F / 219.0F) +
           dptr[sw2]              * (28.0F / 219.0F) +
           dptr[sw3]              * ( 8.0F / 219.0F) +
           dptr[sw4]              * ( 1.0F / 219.0F)
           );
        break;
      case 2:
        dst[gmemPos] =
          (dptr[0]                  * (70.0F / 247.0F) +
           (dptr[nsw] + dptr[sw])   * (56.0F / 247.0F) +
           (dptr[nsw2] + dptr[sw2]) * (28.0F / 247.0F) +
           dptr[sw3]                * ( 8.0F / 247.0F) +
           dptr[sw4]                * ( 1.0F / 247.0F)
           );
        break;
      case 3:
        dst[gmemPos] =
          (dptr[0]                  * (70.0F / 255.0F) +
           (dptr[nsw] + dptr[sw])   * (56.0F / 255.0F) +
           (dptr[nsw2] + dptr[sw2]) * (28.0F / 255.0F) +
           (dptr[nsw3] + dptr[sw3]) * ( 8.0F / 255.0F) +
           dptr[sw4]                * ( 1.0F / 255.0F)
           );
        break;
      }
    }
    else if(sy < 4 && sts+sy<h-4) // If not top 4 in the whole image, are we in the top 4 rows of this tile
    {
      switch(sy)
      {
      case 0:
        dst[gmemPos] =
          (dptr[0]                   * (70.0F / 256.0F) +
           (border[bn1] + dptr[sw])  * (56.0F / 256.0F) +
           (border[bn2] + dptr[sw2]) * (28.0F / 256.0F) +
           (border[bn3] + dptr[sw3]) * ( 8.0F / 256.0F) +
           (border[bn4] + dptr[sw4]) * ( 1.0F / 256.0F)
           );
        break;
      case 1:
        dst[gmemPos] =
          (dptr[0]                   * (70.0F / 256.0F) +
           (dptr[nsw] + dptr[sw])    * (56.0F / 256.0F) +
           (border[bn1] + dptr[sw2]) * (28.0F / 256.0F) +
           (border[bn2] + dptr[sw3]) * ( 8.0F / 256.0F) +
           (border[bn3] + dptr[sw4]) * ( 1.0F / 256.0F)
           );
        break;
      case 2:
        dst[gmemPos] =
          (dptr[0]                   * (70.0F / 256.0F) +
           (dptr[nsw] + dptr[sw])    * (56.0F / 256.0F) +
           (dptr[nsw2] + dptr[sw2])  * (28.0F / 256.0F) +
           (border[bn1] + dptr[sw3]) * ( 8.0F / 256.0F) +
           (border[bn2] + dptr[sw4]) * ( 1.0F / 256.0F)
           );
        break;
      case 3:
        dst[gmemPos] =
          (dptr[0]                   * (70.0F / 256.0F) +
           (dptr[nsw] + dptr[sw])    * (56.0F / 256.0F) +
           (dptr[nsw2] + dptr[sw2])  * (28.0F / 256.0F) +
           (dptr[nsw3] + dptr[sw3])  * ( 8.0F / 256.0F) +
           (border[bn1] + dptr[sw4]) * ( 1.0F / 256.0F)
           );
        break;
      }
    }
    else if(sy >= 4 && sy <src_tile_height-4 && sts+sy<h-4) // Are we in the middle of the tile
    {
        dst[gmemPos] =
          ((dptr[nsw4] + dptr[sw4]) * ( 1.0F / 256.0F) +
           (dptr[nsw3] + dptr[sw3]) * ( 8.0F / 256.0F) +
           (dptr[nsw2] + dptr[sw2]) * (28.0F / 256.0F) +
           (dptr[nsw] + dptr[sw])   * (56.0F / 256.0F) +
           dptr[0]                  * (70.0F / 256.0F)
           );
    }
    else if(sy >= 4 && sts + sy < h-4) // Are we not at the bottom of the image, but bottom 4 of the tile
    {
      switch(src_tile_height-sy)
      {
      case 4:
        dst[gmemPos] =
          (dptr[0]                    * (70.0F / 256.0F) +
           (dptr[nsw] + dptr[sw])     * (56.0F / 256.0F) +
           (dptr[nsw2] + dptr[sw2])   * (28.0F / 256.0F) +
           (dptr[nsw3] + dptr[sw3])   * ( 8.0F / 256.0F) +
           (dptr[nsw4] + border[bp1]) * ( 1.0F / 256.0F)
           );
        break;
      case 3:
        dst[gmemPos] =
          (dptr[0]                    * (70.0F / 256.0F) +
           (dptr[nsw] + dptr[sw])     * (56.0F / 256.0F) +
           (dptr[nsw2] + dptr[sw2])   * (28.0F / 256.0F) +
           (dptr[nsw3] + border[bp1]) * ( 8.0F / 256.0F) +
           (dptr[nsw4] + border[bp2]) * ( 1.0F / 256.0F)
           );
        break;
      case 2:
        dst[gmemPos] =
          (dptr[0]                    * (70.0F / 256.0F) +
           (dptr[nsw] + dptr[sw])     * (56.0F / 256.0F) +
           (dptr[nsw2] + border[bp1]) * (28.0F / 256.0F) +
           (dptr[nsw3] + border[bp2]) * ( 8.0F / 256.0F) +
           (dptr[nsw4] + border[bp3]) * ( 1.0F / 256.0F)
           );
        break;
      case 1:
        dst[gmemPos] =
          (dptr[0]                    * (70.0F / 256.0F) +
           (dptr[nsw] +  border[bp1]) * (56.0F / 256.0F) +
           (dptr[nsw2] + border[bp2]) * (28.0F / 256.0F) +
           (dptr[nsw3] + border[bp3]) * ( 8.0F / 256.0F) +
           (dptr[nsw4] + border[bp4]) * ( 1.0F / 256.0F)
           );
        break;
      }
    }
    else if(sy < 4 && sts + sy >= h-4) // We are in the top of a tile close to the bottom of the image
    {
      switch(sy)
      {
      case 0:
        switch(h-(sts+sy))
        {
          case 4:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 255.0F) +
               (border[bn1] + dptr[sw])   * (56.0F / 255.0F) +
               (border[bn2] + dptr[sw2]) * (28.0F / 255.0F) +
               (border[bn3] + dptr[sw3]) * ( 8.0F / 255.0F) +
               border[bn4]               * ( 1.0F / 255.0F)
               );
            break;
          case 3:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 247.0F) +
               (border[bn1] + dptr[sw])   * (56.0F / 247.0F) +
               (border[bn2] + dptr[sw2]) * (28.0F / 247.0F) +
               border[bn3]               * ( 8.0F / 247.0F) +
               border[bn4]               * ( 1.0F / 247.0F)
               );
            break;
          case 2:
            dst[gmemPos] =
              (dptr[0]                * (70.0F / 219.0F) +
               (border[bn1] + dptr[sw]) * (56.0F / 219.0F) +
               border[bn2]             * (28.0F / 219.0F) +
               border[bn3]             * ( 8.0F / 219.0F) +
               border[bn4]             * ( 1.0F / 219.0F)
               );
            break;
          case 1:
            dst[gmemPos] =
              (dptr[0]    * (70.0F / 163.0F) +
               border[bn1]  * (56.0F / 163.0F) +
               border[bn2] * (28.0F / 163.0F) +
               border[bn3] * ( 8.0F / 163.0F) +
               border[bn4] * ( 1.0F / 163.0F)
               );
            break;
        }
        break;
      case 1:
        switch(h-(sts+sy))
        {
          case 4:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 255.0F) +
               (dptr[nsw] + dptr[sw])   * (56.0F / 255.0F) +
               (border[bn1] + dptr[sw2]) * (28.0F / 255.0F) +
               (border[bn2] + dptr[sw3]) * ( 8.0F / 255.0F) +
               border[bn3]               * ( 1.0F / 255.0F)
               );
            break;
          case 3:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 247.0F) +
               (dptr[nsw] + dptr[sw])   * (56.0F / 247.0F) +
               (border[bn1] + dptr[sw2]) * (28.0F / 247.0F) +
               border[bn2]               * ( 8.0F / 247.0F) +
               border[bn3]               * ( 1.0F / 247.0F)
               );
            break;
          case 2:
            dst[gmemPos] =
              (dptr[0]                * (70.0F / 219.0F) +
               (dptr[nsw] + dptr[sw]) * (56.0F / 219.0F) +
               border[bn1]             * (28.0F / 219.0F) +
               border[bn2]             * ( 8.0F / 219.0F) +
               border[bn3]             * ( 1.0F / 219.0F)
               );
            break;
          case 1:
            dst[gmemPos] =
              (dptr[0]    * (70.0F / 163.0F) +
               dptr[nsw]  * (56.0F / 163.0F) +
               border[bn1] * (28.0F / 163.0F) +
               border[bn2] * ( 8.0F / 163.0F) +
               border[bn3] * ( 1.0F / 163.0F)
               );
            break;
        }
        break;
      case 2:
        switch(h-(sts+sy))
        {
          case 4:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 255.0F) +
               (dptr[nsw] + dptr[sw])   * (56.0F / 255.0F) +
               (dptr[nsw2] + dptr[sw2]) * (28.0F / 255.0F) +
               (border[bn1] + dptr[sw3]) * ( 8.0F / 255.0F) +
               border[bn2]               * ( 1.0F / 255.0F)
               );
            break;
          case 3:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 247.0F) +
               (dptr[nsw] + dptr[sw])   * (56.0F / 247.0F) +
               (dptr[nsw2] + dptr[sw2]) * (28.0F / 247.0F) +
               border[bn1]               * ( 8.0F / 247.0F) +
               border[bn2]               * ( 1.0F / 247.0F)
               );
            break;
          case 2:
            dst[gmemPos] =
              (dptr[0]                * (70.0F / 219.0F) +
               (dptr[nsw] + dptr[sw]) * (56.0F / 219.0F) +
               dptr[nsw2]             * (28.0F / 219.0F) +
               border[bn1]             * ( 8.0F / 219.0F) +
               border[bn2]             * ( 1.0F / 219.0F)
               );
            break;
          case 1:
            dst[gmemPos] =
              (dptr[0]    * (70.0F / 163.0F) +
               dptr[nsw]  * (56.0F / 163.0F) +
               dptr[nsw2] * (28.0F / 163.0F) +
               border[bn1] * ( 8.0F / 163.0F) +
               border[bn2] * ( 1.0F / 163.0F)
               );
            break;
        }
        break;
      case 3:
        switch(h-(sts+sy))
        {
          case 4:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 255.0F) +
               (dptr[nsw] + dptr[sw])   * (56.0F / 255.0F) +
               (dptr[nsw2] + dptr[sw2]) * (28.0F / 255.0F) +
               (dptr[nsw3] + dptr[sw3]) * ( 8.0F / 255.0F) +
               border[bn1]               * ( 1.0F / 255.0F)
               );
            break;
          case 3:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 247.0F) +
               (dptr[nsw] + dptr[sw])   * (56.0F / 247.0F) +
               (dptr[nsw2] + dptr[sw2]) * (28.0F / 247.0F) +
               dptr[nsw3]               * ( 8.0F / 247.0F) +
               border[bn1]               * ( 1.0F / 247.0F)
               );
            break;
          case 2:
            dst[gmemPos] =
              (dptr[0]                * (70.0F / 219.0F) +
               (dptr[nsw] + dptr[sw]) * (56.0F / 219.0F) +
               dptr[nsw2]             * (28.0F / 219.0F) +
               dptr[nsw3]             * ( 8.0F / 219.0F) +
               border[bn1]             * ( 1.0F / 219.0F)
               );
            break;
          case 1:
            dst[gmemPos] =
              (dptr[0]    * (70.0F / 163.0F) +
               dptr[nsw]  * (56.0F / 163.0F) +
               dptr[nsw2] * (28.0F / 163.0F) +
               dptr[nsw3] * ( 8.0F / 163.0F) +
               border[bn1] * ( 1.0F / 163.0F)
               );
            break;
        }
        break;
      }
    }
    else  // We must be at the bottom 4 of the image
    {
      switch(h-(sts+sy))
      {
      case 4:
        dst[gmemPos] =
          (dptr[0]                  * (70.0F / 255.0F) +
           (dptr[nsw] + dptr[sw])   * (56.0F / 255.0F) +
           (dptr[nsw2] + dptr[sw2]) * (28.0F / 255.0F) +
           (dptr[nsw3] + dptr[sw3]) * ( 8.0F / 255.0F) +
           dptr[nsw4]               * ( 1.0F / 255.0F)
           );
        break;
      case 3:
        dst[gmemPos] =
          (dptr[0]                  * (70.0F / 247.0F) +
           (dptr[nsw] + dptr[sw])   * (56.0F / 247.0F) +
           (dptr[nsw2] + dptr[sw2]) * (28.0F / 247.0F) +
           dptr[nsw3]               * ( 8.0F / 247.0F) +
           dptr[nsw4]               * ( 1.0F / 247.0F)
           );
        break;
      case 2:
        dst[gmemPos] =
          (dptr[0]                * (70.0F / 219.0F) +
           (dptr[nsw] + dptr[sw]) * (56.0F / 219.0F) +
           dptr[nsw2]             * (28.0F / 219.0F) +
           dptr[nsw3]             * ( 8.0F / 219.0F) +
           dptr[nsw4]             * ( 1.0F / 219.0F)
           );
        break;
      case 1:
        dst[gmemPos] =
          (dptr[0]    * (70.0F / 163.0F) +
           dptr[nsw]  * (56.0F / 163.0F) +
           dptr[nsw2] * (28.0F / 163.0F) +
           dptr[nsw3] * ( 8.0F / 163.0F) +
           dptr[nsw4] * ( 1.0F / 163.0F)
           );
        break;
      }
    }
  }
}



__global__ void cuda_global_lowpass_9_x(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width)
{
  // Data cache in shared memory
  float *data = (float *) shared_data; // size of tile_width
  // Bordering data flanking this tile
  float *border = (float *) &data[tile_width]; // size of 8
  const int sx = threadIdx.x;                   // source pixel within source tile
  const int sts = IMUL(blockIdx.x, tile_width); // tile start for source, relative to row start
  const int srs = IMUL(blockIdx.y,w);           // Row start index in source data

  const int loadIdx = sts + sx; // index of one pixel value to load
  const int bn4 = 0, bn3 = 1, bn2 = 2, bn1 = 3;
  const int bp1 = 4, bp2 = 5, bp3 = 6, bp4 = 7;
  float *dptr = &data[sx];
  // Load border pixels
  if (sx < 4 && sts > 0) border[sx] = src[srs + sts - (4-sx)];
  if (sx >= tile_width-4 && sts+tile_width < w-4) border[4+sx-(tile_width-4)] = src[srs + sts + sx + 4];

 // Load the row into shared memory among the thread block
  if (loadIdx < w)
    data[sx] = src[srs + loadIdx];
  else
    return; // Threads that are over the edge of the image on the right most tile...

  // Ensure the completness of loading stage because results emitted
  // by each thread depend on the data loaded by other threads:
  __syncthreads();

  // [ 1 8 28 56 (70) 56 28 8 1 ]

  if(w < 9)
  {
    // This is not at all efficient, just here to ensure filter behaves properly in small image cases
    const int numAhead = max(0,min(w-1-(sts+sx),4));
    const int numBehind = max(0,min(sts+sx,4));
    const int situation = numBehind*10+numAhead;

    switch(situation)
    {
    case 0: // 00
      dst[srs+loadIdx] = dptr[0];
      break;
    case 1: // 01
      dst[srs+loadIdx] = (dptr[0] * (70.0F / 126.0F) +
                      dptr[1]     * (56.0F / 126.0F));
      break;
    case 2: // 02
      dst[srs+loadIdx] = (dptr[0] * (70.0F / 154.0F) +
                      dptr[1]     * (56.0F / 154.0F) +
                      dptr[2]     * (28.0F / 154.0F));
      break;
    case 3:
      dst[srs+loadIdx] = (dptr[0] * (70.0F / 162.0F) +
                          dptr[1] * (56.0F / 162.0F) +
                          dptr[2] * (28.0F / 162.0F) +
                          dptr[3] * ( 8.0F / 162.0F));
      break;
    case 4:
      dst[srs+loadIdx] = (dptr[0] * (70.0F / 163.0F) +
                          dptr[1] * (56.0F / 163.0F) +
                          dptr[2] * (28.0F / 163.0F) +
                          dptr[3] * ( 8.0F / 163.0F) +
                          dptr[4] * ( 1.0F / 163.0F));
      break;
    case 10:
      dst[srs+loadIdx] = (dptr[0]  * (70.0F / 126.0F) +
                          dptr[-1] * (56.0F / 126.0F));
      break;
    case 11:
      dst[srs+loadIdx] = (dptr[0]              * (70.0F / 182.0F) +
                          (dptr[-1] + dptr[1]) * (56.0F / 182.0F));
      break;
    case 12:
      dst[srs+loadIdx] = (dptr[0]              * (70.0F / 210.0F) +
                          (dptr[-1] + dptr[1]) * (56.0F / 210.0F) +
                          dptr[2]              * (28.0F / 210.0F));
      break;
    case 13:
      dst[srs+loadIdx] = (dptr[0]               * (70.0F / 218.0F) +
                          (dptr[-1] + dptr[1])  * (56.0F / 218.0F) +
                          dptr[2]               * (28.0F / 218.0F) +
                          dptr[3]               * ( 8.0F / 218.0F));
      break;
    case 14:
      dst[srs+loadIdx] = (dptr[0]               * (70.0F / 219.0F) +
                          (dptr[-1] + dptr[1])  * (56.0F / 219.0F) +
                          dptr[2]               * (28.0F / 219.0F) +
                          dptr[3]               * ( 8.0F / 219.0F) +
                          dptr[4]               * ( 1.0F / 219.0F));
      break;
    case 20:
      dst[srs+loadIdx] = (dptr[0]     * (70.0F / 154.0F) +
                          dptr[-1]    * (56.0F / 154.0F) +
                          dptr[-2]    * (28.0F / 154.0F));
      break;
    case 21:
      dst[srs+loadIdx] = (dptr[0]               * (70.0F / 210.0F) +
                          (dptr[-1] + dptr[1])  * (56.0F / 210.0F) +
                          dptr[-2]              * (28.0F / 210.0F));
      break;
    case 22:
      dst[srs+loadIdx] = (dptr[0]              * (70.0F / 238.0F) +
                          (dptr[-1] + dptr[1]) * (56.0F / 238.0F) +
                          (dptr[-2] + dptr[2]) * (28.0F / 238.0F));
      break;
    case 23:
      dst[srs+loadIdx] = (dptr[0]              * (70.0F / 246.0F) +
                          (dptr[-1] + dptr[1]) * (56.0F / 246.0F) +
                          (dptr[-2] + dptr[2]) * (28.0F / 246.0F) +
                          dptr[3]              * ( 8.0F / 246.0F));
      break;
    case 24:
      dst[srs+loadIdx] = (dptr[0]              * (70.0F / 247.0F) +
                          (dptr[-1] + dptr[1]) * (56.0F / 247.0F) +
                          (dptr[-2] + dptr[2]) * (28.0F / 247.0F) +
                          dptr[3]              * ( 8.0F / 247.0F) +
                          dptr[4]              * ( 1.0F / 247.0F));
      break;
    case 30:
      dst[srs+loadIdx] = (dptr[0]  * (70.0F / 162.0F) +
                          dptr[-1] * (56.0F / 162.0F) +
                          dptr[-2] * (28.0F / 162.0F) +
                          dptr[-3] * ( 8.0F / 162.0F));
      break;
    case 31:
      dst[srs+loadIdx] = (dptr[0]               * (70.0F / 218.0F) +
                          (dptr[-1] + dptr[1])  * (56.0F / 218.0F) +
                          dptr[-2]              * (28.0F / 218.0F) +
                          dptr[-3]              * ( 8.0F / 218.0F));
      break;
    case 32:
      dst[srs+loadIdx] = (dptr[0]              * (70.0F / 246.0F) +
                          (dptr[-1] + dptr[1]) * (56.0F / 246.0F) +
                          (dptr[-2] + dptr[2]) * (28.0F / 246.0F) +
                          dptr[-3]             * ( 8.0F / 246.0F));
      break;
    case 33:
      dst[srs+loadIdx] = (dptr[0]              * (70.0F / 254.0F) +
                          (dptr[-1] + dptr[1]) * (56.0F / 254.0F) +
                          (dptr[-2] + dptr[2]) * (28.0F / 254.0F) +
                          (dptr[-3] + dptr[3]) * ( 8.0F / 254.0F));
      break;
    case 34:
      dst[srs+loadIdx] = (dptr[0]              * (70.0F / 255.0F) +
                          (dptr[-1] + dptr[1]) * (56.0F / 255.0F) +
                          (dptr[-2] + dptr[2]) * (28.0F / 255.0F) +
                          (dptr[-3] + dptr[3]) * ( 8.0F / 255.0F) +
                          dptr[4]              * ( 1.0F / 255.0F));
      break;
    case 40:
      dst[srs+loadIdx] = (dptr[0]  * (70.0F / 163.0F) +
                          dptr[-1] * (56.0F / 163.0F) +
                          dptr[-2] * (28.0F / 163.0F) +
                          dptr[-3] * ( 8.0F / 163.0F) +
                          dptr[-4] * ( 1.0F / 163.0F));
      break;
    case 41:
      dst[srs+loadIdx] = (dptr[0]               * (70.0F / 219.0F) +
                          (dptr[-1] + dptr[1])  * (56.0F / 219.0F) +
                          dptr[-2]              * (28.0F / 219.0F) +
                          dptr[-3]              * ( 8.0F / 219.0F) +
                          dptr[-4]              * ( 1.0F / 219.0F));
      break;
    case 42:
      dst[srs+loadIdx] = (dptr[0]              * (70.0F / 247.0F) +
                          (dptr[-1] + dptr[1]) * (56.0F / 247.0F) +
                          (dptr[-2] + dptr[2]) * (28.0F / 247.0F) +
                          dptr[-3]             * ( 8.0F / 247.0F) +
                          dptr[-4]             * ( 1.0F / 247.0F));
      break;
    case 43:
      dst[srs+loadIdx] = (dptr[0]              * (70.0F / 255.0F) +
                          (dptr[-1] + dptr[1]) * (56.0F / 255.0F) +
                          (dptr[-2] + dptr[2]) * (28.0F / 255.0F) +
                          (dptr[-3] + dptr[3]) * ( 8.0F / 255.0F) +
                          dptr[-4]             * ( 1.0F / 255.0F));
      break;
    }
  }
  // First part of source row, just reduce sample
  else if(sts+sx < 4)
  {
    switch(sx)
    {
    case 0:
      dst[srs + loadIdx] =
        (dptr[0] * (70.0F / 163.0F) +
         dptr[1] * (56.0F / 163.0F) +
         dptr[2] * (28.0F / 163.0F) +
         dptr[3] * ( 8.0F / 163.0F) +
         dptr[4] * ( 1.0F / 163.0F)
         );
      break;
    case 1:
      dst[srs + loadIdx] =
        (dptr[0]              * (70.0F / 219.0F) +
         (dptr[-1] + dptr[1]) * (56.0F / 219.0F) +
         dptr[2]              * (28.0F / 219.0F) +
         dptr[3]              * ( 8.0F / 219.0F) +
         dptr[4]              * ( 1.0F / 219.0F)
         );
      break;
    case 2:
      dst[srs + loadIdx] =
        (dptr[0]              * (70.0F / 247.0F) +
         (dptr[-1] + dptr[1]) * (56.0F / 247.0F) +
         (dptr[-2] + dptr[2]) * (28.0F / 247.0F) +
         dptr[3]              * ( 8.0F / 247.0F) +
         dptr[4]              * ( 1.0F / 247.0F)
         );
      break;
    case 3:
      dst[srs + loadIdx] =
        (dptr[0]              * (70.0F / 255.0F) +
         (dptr[-1] + dptr[1]) * (56.0F / 255.0F) +
         (dptr[-2] + dptr[2]) * (28.0F / 255.0F) +
         (dptr[-3] + dptr[3]) * ( 8.0F / 255.0F) +
         dptr[4]              * ( 1.0F / 255.0F)
         );
      break;
    default:
      //LERROR();
      break;
    }
  }
  // If not the first part of the source row, but is the first bit of this tile, use border
  else if(sx < 4 && sts+sx < w-4)
  {
    switch(sx)
    {
    case 0:
      dst[srs + loadIdx] =
        ((border[bn4] + dptr[4]) * ( 1.0F / 256.0F) +
         (border[bn3] + dptr[3]) * ( 8.0F / 256.0F) +
         (border[bn2] + dptr[2]) * (28.0F / 256.0F) +
         (border[bn1] + dptr[1]) * (56.0F / 256.0F) +
         dptr[0]                 * (70.0F / 256.0F)
         );
      break;
    case 1:
      dst[srs + loadIdx] =
        ((border[bn3] + dptr[4])   * ( 1.0F / 256.0F) +
         (border[bn2] + dptr[3])   * ( 8.0F / 256.0F) +
         (border[bn1] + dptr[2])   * (28.0F / 256.0F) +
         (dptr[-1] + dptr[1])      * (56.0F / 256.0F) +
         dptr[0]                   * (70.0F / 256.0F)
         );
      break;
    case 2:
      dst[srs + loadIdx] =
        ((border[bn2] + dptr[4]) * ( 1.0F / 256.0F) +
         (border[bn1] + dptr[3]) * ( 8.0F / 256.0F) +
         (dptr[-2] + dptr[2])    * (28.0F / 256.0F) +
         (dptr[-1] + dptr[1])    * (56.0F / 256.0F) +
         dptr[0]                 * (70.0F / 256.0F)
         );
      break;
    case 3:
      dst[srs + loadIdx] =
        ((border[bn1] + dptr[4]) * ( 1.0F / 256.0F) +
         (dptr[-3] + dptr[3])    * ( 8.0F / 256.0F) +
         (dptr[-2] + dptr[2])    * (28.0F / 256.0F) +
         (dptr[-1] + dptr[1])    * (56.0F / 256.0F) +
         dptr[0]                 * (70.0F / 256.0F)
         );
      break;
    }
  }
  // If we are not near the edge of this tile, do standard way
  else if(sx>=4 && sx<tile_width-4 && sts+sx < w-4)
  {
    dst[srs + loadIdx] =
      ((dptr[-4] + dptr[4]) * ( 1.0F / 256.0F) +
       (dptr[-3] + dptr[3]) * ( 8.0F / 256.0F) +
       (dptr[-2] + dptr[2]) * (28.0F / 256.0F) +
       (dptr[-1] + dptr[1]) * (56.0F / 256.0F) +
       dptr[0]              * (70.0F / 256.0F)
       );
  }
  // If not the last part of the source row, but in the last bit of this tile, use border
  else if(sx>=4 && sts+sx < w-4)
  {
    switch(tile_width-sx)
      {
      case 4:
        dst[srs + loadIdx] =
          ((dptr[-4] + border[bp1]) * ( 1.0F / 256.0F) +
           (dptr[-3] + dptr[3])     * ( 8.0F / 256.0F) +
           (dptr[-2] + dptr[2])     * (28.0F / 256.0F) +
           (dptr[-1] + dptr[1])     * (56.0F / 256.0F) +
           dptr[0]                  * (70.0F / 256.0F)
           );
        break;
      case 3:
        dst[srs + loadIdx] =
          ((dptr[-4] + border[bp2]) * ( 1.0F / 256.0F) +
           (dptr[-3] + border[bp1]) * ( 8.0F / 256.0F) +
           (dptr[-2] + dptr[2])     * (28.0F / 256.0F) +
           (dptr[-1] + dptr[1])     * (56.0F / 256.0F) +
           dptr[0]                  * (70.0F / 256.0F)
           );
        break;
      case 2:
        dst[srs + loadIdx] =
          ((dptr[-4] + border[bp3]) * ( 1.0F / 256.0F) +
           (dptr[-3] + border[bp2]) * ( 8.0F / 256.0F) +
           (dptr[-2] + border[bp1]) * (28.0F / 256.0F) +
           (dptr[-1] + dptr[1])     * (56.0F / 256.0F) +
           dptr[0]                  * (70.0F / 256.0F)
           );
        break;
      case 1:
        dst[srs + loadIdx] =
          ((dptr[-4] + border[bp4]) * ( 1.0F / 256.0F) +
           (dptr[-3] + border[bp3]) * ( 8.0F / 256.0F) +
           (dptr[-2] + border[bp2]) * (28.0F / 256.0F) +
           (dptr[-1] + border[bp1]) * (56.0F / 256.0F) +
           dptr[0]                  * (70.0F / 256.0F)
           );
        break;
      }
  }
  else if(sx < 4 && sts + sx >= w-4) // We are in the beginning of a tile close to the end of the image
  {
    switch(sx)
    {
    case 0:
      switch(w-(sts+sx))
      {
      case 4:
        dst[srs + loadIdx] =
          (dptr[0]                 * (70.0F / 255.0F) +
           (border[bn1] + dptr[1]) * (56.0F / 255.0F) +
           (border[bn2] + dptr[2]) * (28.0F / 255.0F) +
           (border[bn3] + dptr[3]) * ( 8.0F / 255.0F) +
           border[bn4]             * ( 1.0F / 255.0F)
           );
        break;
      case 3:
        dst[srs + loadIdx] =
          (dptr[0]                 * (70.0F / 247.0F) +
           (border[bn1] + dptr[1]) * (56.0F / 247.0F) +
           (border[bn2] + dptr[2]) * (28.0F / 247.0F) +
           border[bn3]             * ( 8.0F / 247.0F) +
           border[bn4]             * ( 1.0F / 247.0F)
           );
        break;
      case 2:
        dst[srs + loadIdx] =
          (dptr[0]                 * (70.0F / 219.0F) +
           (border[bn1] + dptr[1]) * (56.0F / 219.0F) +
           border[bn2]             * (28.0F / 219.0F) +
           border[bn3]             * ( 8.0F / 219.0F) +
           border[bn4]             * ( 1.0F / 219.0F)
           );
        break;
      case 1:
        dst[srs + loadIdx] =
          (dptr[0]     * (70.0F / 163.0F) +
           border[bn1] * (56.0F / 163.0F) +
           border[bn2] * (28.0F / 163.0F) +
           border[bn3] * ( 8.0F / 163.0F) +
           border[bn4] * ( 1.0F / 163.0F)
           );
        break;
      }
      break;
    case 1:
      switch(w-(sts+sx))
        {
          case 4:
            dst[srs + loadIdx] =
              (dptr[0]                 * (70.0F / 255.0F) +
               (dptr[1] + dptr[1])     * (56.0F / 255.0F) +
               (border[bn1] + dptr[2]) * (28.0F / 255.0F) +
               (border[bn2] + dptr[3]) * ( 8.0F / 255.0F) +
               border[bn3]             * ( 1.0F / 255.0F)
               );
            break;
          case 3:
            dst[srs + loadIdx] =
              (dptr[0]                 * (70.0F / 247.0F) +
               (dptr[1] + dptr[1])     * (56.0F / 247.0F) +
               (border[bn1] + dptr[2]) * (28.0F / 247.0F) +
               border[bn2]             * ( 8.0F / 247.0F) +
               border[bn3]             * ( 1.0F / 247.0F)
               );
            break;
          case 2:
            dst[srs + loadIdx] =
              (dptr[0]                * (70.0F / 219.0F) +
               (dptr[1] + dptr[1])    * (56.0F / 219.0F) +
               border[bn1]            * (28.0F / 219.0F) +
               border[bn2]            * ( 8.0F / 219.0F) +
               border[bn3]            * ( 1.0F / 219.0F)
               );
            break;
          case 1:
            dst[srs + loadIdx] =
              (dptr[0]     * (70.0F / 163.0F) +
               dptr[1]     * (56.0F / 163.0F) +
               border[bn1] * (28.0F / 163.0F) +
               border[bn2] * ( 8.0F / 163.0F) +
               border[bn3] * ( 1.0F / 163.0F)
               );
            break;
        }
        break;
      case 2:
        switch(w-(sts+sx))
        {
          case 4:
            dst[srs + loadIdx] =
              (dptr[0]                  * (70.0F / 255.0F) +
               (dptr[1] + dptr[1])      * (56.0F / 255.0F) +
               (dptr[2] + dptr[2])      * (28.0F / 255.0F) +
               (border[bn1] + dptr[3])  * ( 8.0F / 255.0F) +
               border[bn2]              * ( 1.0F / 255.0F)
               );
            break;
          case 3:
            dst[srs + loadIdx] =
              (dptr[0]                 * (70.0F / 247.0F) +
               (dptr[1] + dptr[1])     * (56.0F / 247.0F) +
               (dptr[2] + dptr[2])     * (28.0F / 247.0F) +
               border[bn1]             * ( 8.0F / 247.0F) +
               border[bn2]             * ( 1.0F / 247.0F)
               );
            break;
          case 2:
            dst[srs + loadIdx] =
              (dptr[0]                * (70.0F / 219.0F) +
               (dptr[1] + dptr[1])    * (56.0F / 219.0F) +
               dptr[2]                * (28.0F / 219.0F) +
               border[bn1]            * ( 8.0F / 219.0F) +
               border[bn2]            * ( 1.0F / 219.0F)
               );
            break;
          case 1:
            dst[srs + loadIdx] =
              (dptr[0]          * (70.0F / 163.0F) +
               dptr[1]          * (56.0F / 163.0F) +
               dptr[2]          * (28.0F / 163.0F) +
               border[bn1]      * ( 8.0F / 163.0F) +
               border[bn2]      * ( 1.0F / 163.0F)
               );
            break;
        }
        break;
      case 3:
        switch(w-(sts+sx))
        {
          case 4:
            dst[srs + loadIdx] =
              (dptr[0]               * (70.0F / 255.0F) +
               (dptr[1] + dptr[1])   * (56.0F / 255.0F) +
               (dptr[2] + dptr[2])   * (28.0F / 255.0F) +
               (dptr[3] + dptr[3])   * ( 8.0F / 255.0F) +
               border[bn1]           * ( 1.0F / 255.0F)
               );
            break;
          case 3:
            dst[srs + loadIdx] =
              (dptr[0]               * (70.0F / 247.0F) +
               (dptr[1] + dptr[1])   * (56.0F / 247.0F) +
               (dptr[2] + dptr[2])   * (28.0F / 247.0F) +
               dptr[3]               * ( 8.0F / 247.0F) +
               border[bn1]           * ( 1.0F / 247.0F)
               );
            break;
          case 2:
            dst[srs + loadIdx] =
              (dptr[0]             * (70.0F / 219.0F) +
               (dptr[1] + dptr[1]) * (56.0F / 219.0F) +
               dptr[2]             * (28.0F / 219.0F) +
               dptr[3]             * ( 8.0F / 219.0F) +
               border[bn1]         * ( 1.0F / 219.0F)
               );
            break;
          case 1:
            dst[srs + loadIdx] =
              (dptr[0]     * (70.0F / 163.0F) +
               dptr[1]     * (56.0F / 163.0F) +
               dptr[2]     * (28.0F / 163.0F) +
               dptr[3]     * ( 8.0F / 163.0F) +
               border[bn1] * ( 1.0F / 163.0F)
               );
            break;
        }
        break;
      }
    }

  // If in the last bit of the source row, reduce sample
  else if(sts + sx < w)
  {
    switch(w-(sts+sx))
    {
    case 4:
      dst[srs + loadIdx] =
        (dptr[-4]             * ( 1.0F / 255.0F) +
         (dptr[-3] + dptr[3]) * ( 8.0F / 255.0F) +
         (dptr[-2] + dptr[2]) * (28.0F / 255.0F) +
         (dptr[-1] + dptr[1]) * (56.0F / 255.0F) +
         dptr[0]              * (70.0F / 255.0F)
         );
      break;
    case 3:
      dst[srs + loadIdx] =
        (dptr[-4]             * ( 1.0F / 247.0F) +
         dptr[-3]             * ( 8.0F / 247.0F) +
         (dptr[-2] + dptr[2]) * (28.0F / 247.0F) +
         (dptr[-1] + dptr[1]) * (56.0F / 247.0F) +
         dptr[0]              * (70.0F / 247.0F)
         );
      break;
    case 2:
      dst[srs + loadIdx] =
        (dptr[-4]             * ( 1.0F / 219.0F) +
         dptr[-3]             * ( 8.0F / 219.0F) +
         dptr[-2]             * (28.0F / 219.0F) +
         (dptr[-1] + dptr[1]) * (56.0F / 219.0F) +
         dptr[0]              * (70.0F / 219.0F)
         );
      break;
    case 1:
      dst[srs + loadIdx] =
        (dptr[-4]             * ( 1.0F / 163.0F) +
         dptr[-3]             * ( 8.0F / 163.0F) +
         dptr[-2]             * (28.0F / 163.0F) +
         dptr[-1]             * (56.0F / 163.0F) +
         dptr[ 0]             * (70.0F / 163.0F)
         );
      break;
    }
  }
}

// ######################################################################
__global__ void cuda_global_lowpass_9_y(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width, int tile_height)
{
  // Data cache
  float *data = (float *) shared_data; // size of tile_width * tile_height
  float *border = (float *) &data[tile_width*tile_height]; // size of tile_width * 8

  const int sy = threadIdx.y; // source pixel row within source tile

  const int sts = IMUL(blockIdx.y, tile_height); // tile start for source, in rows
  const int ste = sts + tile_height; // tile end for source, in rows
  const int scs = IMUL(blockIdx.x, tile_width) + threadIdx.x;  // Current column index

  // Clamp tile and apron limits by image borders
  const int stec = min(ste, h);




  // only process columns that are actually within image bounds:
  if (scs < w && sts+sy < stec)
  {
    // Shared and global memory indices for current column
    int smemPos = IMUL(sy, tile_width) + threadIdx.x;
    int gmemPos = IMUL(sts + sy, w) + scs;

    data[smemPos] = src[gmemPos];

    if (sy < 4 && gmemPos > IMUL(4,w))
      border[smemPos] = src[gmemPos-IMUL(4,w)];

    int bordOff = 8+sy-tile_height;

    if (sy >= tile_height-4)
    {
      if(ste+4 <= h)
        border[threadIdx.x+IMUL(bordOff,tile_width)] = src[gmemPos+IMUL(4,w)];
      else
        // This is explicitly handling the case where the second to last tile is so close to the edge of the image, that the post border
        // can't fit in the remaining space, maybe we should be handling this as yet another special case, but there are a lot of special
        // cases already in this function (handling smaller than 9 imgs, handling all cases of small last tile)
        border[threadIdx.x+IMUL(bordOff,tile_width)] = 0;
    }

    // Ensure the completness of loading stage because results emitted
    // by each thread depend on the data loaded by other threads:
    __syncthreads();

    // Shared and global memory indices for current column
    smemPos = IMUL(sy, tile_width) + threadIdx.x;
    gmemPos = IMUL(sts + sy, w) + scs;


    // Setup the offsets to get to the correct smem points in the arrays for both the data and the border
    float *dptr = data + smemPos;
    const int sw = tile_width, sw2 = sw + sw, sw3 = sw2 + sw, sw4 = sw3 + sw;
    const int nsw = -sw, nsw2 = nsw - sw, nsw3 = nsw2 - sw, nsw4 = nsw3 - sw;
    const int bn4 = threadIdx.x, bn3 = bn4 + tile_width, bn2 = bn3 + tile_width, bn1 = bn2 + tile_width;
    const int bp1 = bn1+tile_width, bp2 = bp1 + tile_width, bp3 = bp2 + tile_width, bp4 = bp3 + tile_width;

    if(h < 9)
    {
      // This is not at all efficient, just here to ensure filter behaves properly in small image cases
      const int numAhead = max(0,min(h-1-(sts+sy),4));
      const int numBehind = max(0,min(sts+sy,4));
      const int situation = numBehind*10+numAhead;
      switch(situation)
      {
      case 0: // 00
        dst[gmemPos] = dptr[0];
        break;
      case 1: // 01
        dst[gmemPos] = (dptr[0] * (70.0F / 126.0F) +
                        dptr[sw] * (56.0F / 126.0F));
        break;
      case 2: // 02
        dst[gmemPos] = (dptr[0]   * (70.0F / 154.0F) +
                        dptr[sw]  * (56.0F / 154.0F) +
                        dptr[sw2] * (28.0F / 154.0F));
        break;
      case 3:
        dst[gmemPos] = (dptr[0]   * (70.0F / 162.0F) +
                        dptr[sw]  * (56.0F / 162.0F) +
                        dptr[sw2] * (28.0F / 162.0F) +
                        dptr[sw3] * ( 8.0F / 162.0F));
        break;
      case 4:
        dst[gmemPos] = (dptr[0]   * (70.0F / 163.0F) +
                        dptr[sw]  * (56.0F / 163.0F) +
                        dptr[sw2] * (28.0F / 163.0F) +
                        dptr[sw3] * ( 8.0F / 163.0F) +
                        dptr[sw4] * ( 1.0F / 163.0F));
        break;
      case 10:
        dst[gmemPos] = (dptr[0] * (70.0F / 126.0F) +
                        dptr[nsw] * (56.0F / 126.0F));
        break;
      case 11:
        dst[gmemPos] = (dptr[0]                * (70.0F / 182.0F) +
                        (dptr[nsw] + dptr[sw]) * (56.0F / 182.0F));
        break;
      case 12:
        dst[gmemPos] = (dptr[0]                * (70.0F / 210.0F) +
                        (dptr[nsw] + dptr[sw]) * (56.0F / 210.0F) +
                        dptr[sw2]              * (28.0F / 210.0F));
        break;
      case 13:
        dst[gmemPos] = (dptr[0]                 * (70.0F / 218.0F) +
                        (dptr[nsw] + dptr[sw])  * (56.0F / 218.0F) +
                        dptr[sw2]               * (28.0F / 218.0F) +
                        dptr[sw3]               * ( 8.0F / 218.0F));
        break;
      case 14:
        dst[gmemPos] = (dptr[0]                 * (70.0F / 219.0F) +
                        (dptr[nsw] + dptr[sw])  * (56.0F / 219.0F) +
                        dptr[sw2]               * (28.0F / 219.0F) +
                        dptr[sw3]               * ( 8.0F / 219.0F) +
                        dptr[sw4]               * ( 1.0F / 219.0F));
        break;
      case 20:
        dst[gmemPos] = (dptr[0]    * (70.0F / 154.0F) +
                        dptr[nsw]  * (56.0F / 154.0F) +
                        dptr[nsw2] * (28.0F / 154.0F));
        break;
      case 21:
        dst[gmemPos] = (dptr[0]                 * (70.0F / 210.0F) +
                        (dptr[nsw] + dptr[sw])  * (56.0F / 210.0F) +
                        dptr[nsw2]              * (28.0F / 210.0F));
        break;
      case 22:
        dst[gmemPos] = (dptr[0]                  * (70.0F / 238.0F) +
                        (dptr[nsw] + dptr[sw])   * (56.0F / 238.0F) +
                        (dptr[nsw2] + dptr[sw2]) * (28.0F / 238.0F));
        break;
      case 23:
        dst[gmemPos] = (dptr[0]                  * (70.0F / 246.0F) +
                        (dptr[nsw] + dptr[sw])   * (56.0F / 246.0F) +
                        (dptr[nsw2] + dptr[sw2]) * (28.0F / 246.0F) +
                        dptr[sw3]                * ( 8.0F / 246.0F));
        break;
      case 24:
        dst[gmemPos] = (dptr[0]                  * (70.0F / 247.0F) +
                        (dptr[nsw] + dptr[sw])   * (56.0F / 247.0F) +
                        (dptr[nsw2] + dptr[sw2]) * (28.0F / 247.0F) +
                        dptr[sw3]                * ( 8.0F / 247.0F) +
                        dptr[sw4]                * ( 1.0F / 247.0F));
        break;
      case 30:
        dst[gmemPos] = (dptr[0]    * (70.0F / 162.0F) +
                        dptr[nsw]  * (56.0F / 162.0F) +
                        dptr[nsw2] * (28.0F / 162.0F) +
                        dptr[nsw3] * ( 8.0F / 162.0F));
        break;
      case 31:
        dst[gmemPos] = (dptr[0]                 * (70.0F / 218.0F) +
                        (dptr[nsw] + dptr[sw])  * (56.0F / 218.0F) +
                        dptr[nsw2]              * (28.0F / 218.0F) +
                        dptr[nsw3]              * ( 8.0F / 218.0F));
        break;
      case 32:
        dst[gmemPos] = (dptr[0]                  * (70.0F / 246.0F) +
                        (dptr[nsw] + dptr[sw])   * (56.0F / 246.0F) +
                        (dptr[nsw2] + dptr[sw2]) * (28.0F / 246.0F) +
                        dptr[nsw3]               * ( 8.0F / 246.0F));
        break;
      case 33:
        dst[gmemPos] = (dptr[0]                  * (70.0F / 254.0F) +
                        (dptr[nsw] + dptr[sw])   * (56.0F / 254.0F) +
                        (dptr[nsw2] + dptr[sw2]) * (28.0F / 254.0F) +
                        (dptr[nsw3] + dptr[sw3]) * ( 8.0F / 254.0F));
        break;
      case 34:
        dst[gmemPos] = (dptr[0]                  * (70.0F / 255.0F) +
                        (dptr[nsw] + dptr[sw])   * (56.0F / 255.0F) +
                        (dptr[nsw2] + dptr[sw2]) * (28.0F / 255.0F) +
                        (dptr[nsw3] + dptr[sw3]) * ( 8.0F / 255.0F) +
                        dptr[sw4]                * ( 1.0F / 255.0F));
        break;
      case 40:
        dst[gmemPos] = (dptr[0]    * (70.0F / 163.0F) +
                        dptr[nsw]  * (56.0F / 163.0F) +
                        dptr[nsw2] * (28.0F / 163.0F) +
                        dptr[nsw3] * ( 8.0F / 163.0F) +
                        dptr[nsw4] * ( 1.0F / 163.0F));
        break;
      case 41:
        dst[gmemPos] = (dptr[0]                 * (70.0F / 219.0F) +
                        (dptr[nsw] + dptr[sw])  * (56.0F / 219.0F) +
                        dptr[nsw2]              * (28.0F / 219.0F) +
                        dptr[nsw3]              * ( 8.0F / 219.0F) +
                        dptr[nsw4]              * ( 1.0F / 219.0F));
        break;
      case 42:
        dst[gmemPos] = (dptr[0]                  * (70.0F / 247.0F) +
                        (dptr[nsw] + dptr[sw])   * (56.0F / 247.0F) +
                        (dptr[nsw2] + dptr[sw2]) * (28.0F / 247.0F) +
                        dptr[nsw3]               * ( 8.0F / 247.0F) +
                        dptr[nsw4]               * ( 1.0F / 247.0F));
        break;
      case 43:
        dst[gmemPos] = (dptr[0]                  * (70.0F / 255.0F) +
                        (dptr[nsw] + dptr[sw])   * (56.0F / 255.0F) +
                        (dptr[nsw2] + dptr[sw2]) * (28.0F / 255.0F) +
                        (dptr[nsw3] + dptr[sw3]) * ( 8.0F / 255.0F) +
                        dptr[nsw4]               * ( 1.0F / 255.0F));
        break;
      }
    }
    // Are we in the top 4 rows of the whole image
    else if(sts + sy < 4)
    {
      switch(sts+sy)
      {
      case 0:
        dst[gmemPos] =
          (dptr[0]   * (70.0F / 163.0F) +
           dptr[sw]  * (56.0F / 163.0F) +
           dptr[sw2] * (28.0F / 163.0F) +
           dptr[sw3] * ( 8.0F / 163.0F) +
           dptr[sw4] * ( 1.0F / 163.0F)
           );
        break;
      case 1:
        dst[gmemPos] =
          (dptr[0]                * (70.0F / 219.0F) +
           (dptr[nsw] + dptr[sw]) * (56.0F / 219.0F) +
           dptr[sw2]              * (28.0F / 219.0F) +
           dptr[sw3]              * ( 8.0F / 219.0F) +
           dptr[sw4]              * ( 1.0F / 219.0F)
           );
        break;
      case 2:
        dst[gmemPos] =
          (dptr[0]                  * (70.0F / 247.0F) +
           (dptr[nsw] + dptr[sw])   * (56.0F / 247.0F) +
           (dptr[nsw2] + dptr[sw2]) * (28.0F / 247.0F) +
           dptr[sw3]                * ( 8.0F / 247.0F) +
           dptr[sw4]                * ( 1.0F / 247.0F)
           );
        break;
      case 3:
        dst[gmemPos] =
          (dptr[0]                  * (70.0F / 255.0F) +
           (dptr[nsw] + dptr[sw])   * (56.0F / 255.0F) +
           (dptr[nsw2] + dptr[sw2]) * (28.0F / 255.0F) +
           (dptr[nsw3] + dptr[sw3]) * ( 8.0F / 255.0F) +
           dptr[sw4]                * ( 1.0F / 255.0F)
           );
        break;
      }
    }
    else if(sy < 4 && sts+sy<h-4) // If not top 4 in the whole image, are we in the top 4 rows of this tile
    {
      switch(sy)
      {
      case 0:
        dst[gmemPos] =
          (dptr[0]                   * (70.0F / 256.0F) +
           (border[bn1] + dptr[sw])  * (56.0F / 256.0F) +
           (border[bn2] + dptr[sw2]) * (28.0F / 256.0F) +
           (border[bn3] + dptr[sw3]) * ( 8.0F / 256.0F) +
           (border[bn4] + dptr[sw4]) * ( 1.0F / 256.0F)
           );
        break;
      case 1:
        dst[gmemPos] =
          (dptr[0]                   * (70.0F / 256.0F) +
           (dptr[nsw] + dptr[sw])    * (56.0F / 256.0F) +
           (border[bn1] + dptr[sw2]) * (28.0F / 256.0F) +
           (border[bn2] + dptr[sw3]) * ( 8.0F / 256.0F) +
           (border[bn3] + dptr[sw4]) * ( 1.0F / 256.0F)
           );
        break;
      case 2:
        dst[gmemPos] =
          (dptr[0]                   * (70.0F / 256.0F) +
           (dptr[nsw] + dptr[sw])    * (56.0F / 256.0F) +
           (dptr[nsw2] + dptr[sw2])  * (28.0F / 256.0F) +
           (border[bn1] + dptr[sw3]) * ( 8.0F / 256.0F) +
           (border[bn2] + dptr[sw4]) * ( 1.0F / 256.0F)
           );
        break;
      case 3:
        dst[gmemPos] =
          (dptr[0]                   * (70.0F / 256.0F) +
           (dptr[nsw] + dptr[sw])    * (56.0F / 256.0F) +
           (dptr[nsw2] + dptr[sw2])  * (28.0F / 256.0F) +
           (dptr[nsw3] + dptr[sw3])  * ( 8.0F / 256.0F) +
           (border[bn1] + dptr[sw4]) * ( 1.0F / 256.0F)
           );
        break;
      }
    }
    else if(sy >= 4 && sy <tile_height-4 && sts+sy<h-4) // Are we in the middle of the tile
    {
        dst[gmemPos] =
          ((dptr[nsw4] + dptr[sw4]) * ( 1.0F / 256.0F) +
           (dptr[nsw3] + dptr[sw3]) * ( 8.0F / 256.0F) +
           (dptr[nsw2] + dptr[sw2]) * (28.0F / 256.0F) +
           (dptr[nsw] + dptr[sw])   * (56.0F / 256.0F) +
           dptr[0]                  * (70.0F / 256.0F)
           );
    }
    else if(sy >= 4 && sts + sy < h-4) // Are we not at the bottom of the image, but bottom 4 of the tile
    {
      switch(tile_height-sy)
      {
      case 4:
        dst[gmemPos] =
          (dptr[0]                    * (70.0F / 256.0F) +
           (dptr[nsw] + dptr[sw])     * (56.0F / 256.0F) +
           (dptr[nsw2] + dptr[sw2])   * (28.0F / 256.0F) +
           (dptr[nsw3] + dptr[sw3])   * ( 8.0F / 256.0F) +
           (dptr[nsw4] + border[bp1]) * ( 1.0F / 256.0F)
           );
        break;
      case 3:
        dst[gmemPos] =
          (dptr[0]                    * (70.0F / 256.0F) +
           (dptr[nsw] + dptr[sw])     * (56.0F / 256.0F) +
           (dptr[nsw2] + dptr[sw2])   * (28.0F / 256.0F) +
           (dptr[nsw3] + border[bp1]) * ( 8.0F / 256.0F) +
           (dptr[nsw4] + border[bp2]) * ( 1.0F / 256.0F)
           );
        break;
      case 2:
        dst[gmemPos] =
          (dptr[0]                    * (70.0F / 256.0F) +
           (dptr[nsw] + dptr[sw])     * (56.0F / 256.0F) +
           (dptr[nsw2] + border[bp1]) * (28.0F / 256.0F) +
           (dptr[nsw3] + border[bp2]) * ( 8.0F / 256.0F) +
           (dptr[nsw4] + border[bp3]) * ( 1.0F / 256.0F)
           );
        break;
      case 1:
        dst[gmemPos] =
          (dptr[0]                    * (70.0F / 256.0F) +
           (dptr[nsw] +  border[bp1]) * (56.0F / 256.0F) +
           (dptr[nsw2] + border[bp2]) * (28.0F / 256.0F) +
           (dptr[nsw3] + border[bp3]) * ( 8.0F / 256.0F) +
           (dptr[nsw4] + border[bp4]) * ( 1.0F / 256.0F)
           );
        break;
      }
    }
    else if(sy < 4 && sts + sy >= h-4) // We are in the top of a tile close to the bottom of the image
    {
      switch(sy)
      {
      case 0:
        switch(h-(sts+sy))
        {
          case 4:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 255.0F) +
               (border[bn1] + dptr[sw])   * (56.0F / 255.0F) +
               (border[bn2] + dptr[sw2]) * (28.0F / 255.0F) +
               (border[bn3] + dptr[sw3]) * ( 8.0F / 255.0F) +
               border[bn4]               * ( 1.0F / 255.0F)
               );
            break;
          case 3:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 247.0F) +
               (border[bn1] + dptr[sw])   * (56.0F / 247.0F) +
               (border[bn2] + dptr[sw2]) * (28.0F / 247.0F) +
               border[bn3]               * ( 8.0F / 247.0F) +
               border[bn4]               * ( 1.0F / 247.0F)
               );
            break;
          case 2:
            dst[gmemPos] =
              (dptr[0]                * (70.0F / 219.0F) +
               (border[bn1] + dptr[sw]) * (56.0F / 219.0F) +
               border[bn2]             * (28.0F / 219.0F) +
               border[bn3]             * ( 8.0F / 219.0F) +
               border[bn4]             * ( 1.0F / 219.0F)
               );
            break;
          case 1:
            dst[gmemPos] =
              (dptr[0]    * (70.0F / 163.0F) +
               border[bn1]  * (56.0F / 163.0F) +
               border[bn2] * (28.0F / 163.0F) +
               border[bn3] * ( 8.0F / 163.0F) +
               border[bn4] * ( 1.0F / 163.0F)
               );
            break;
        }
        break;
      case 1:
        switch(h-(sts+sy))
        {
          case 4:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 255.0F) +
               (dptr[nsw] + dptr[sw])   * (56.0F / 255.0F) +
               (border[bn1] + dptr[sw2]) * (28.0F / 255.0F) +
               (border[bn2] + dptr[sw3]) * ( 8.0F / 255.0F) +
               border[bn3]               * ( 1.0F / 255.0F)
               );
            break;
          case 3:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 247.0F) +
               (dptr[nsw] + dptr[sw])   * (56.0F / 247.0F) +
               (border[bn1] + dptr[sw2]) * (28.0F / 247.0F) +
               border[bn2]               * ( 8.0F / 247.0F) +
               border[bn3]               * ( 1.0F / 247.0F)
               );
            break;
          case 2:
            dst[gmemPos] =
              (dptr[0]                * (70.0F / 219.0F) +
               (dptr[nsw] + dptr[sw]) * (56.0F / 219.0F) +
               border[bn1]             * (28.0F / 219.0F) +
               border[bn2]             * ( 8.0F / 219.0F) +
               border[bn3]             * ( 1.0F / 219.0F)
               );
            break;
          case 1:
            dst[gmemPos] =
              (dptr[0]    * (70.0F / 163.0F) +
               dptr[nsw]  * (56.0F / 163.0F) +
               border[bn1] * (28.0F / 163.0F) +
               border[bn2] * ( 8.0F / 163.0F) +
               border[bn3] * ( 1.0F / 163.0F)
               );
            break;
        }
        break;
      case 2:
        switch(h-(sts+sy))
        {
          case 4:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 255.0F) +
               (dptr[nsw] + dptr[sw])   * (56.0F / 255.0F) +
               (dptr[nsw2] + dptr[sw2]) * (28.0F / 255.0F) +
               (border[bn1] + dptr[sw3]) * ( 8.0F / 255.0F) +
               border[bn2]               * ( 1.0F / 255.0F)
               );
            break;
          case 3:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 247.0F) +
               (dptr[nsw] + dptr[sw])   * (56.0F / 247.0F) +
               (dptr[nsw2] + dptr[sw2]) * (28.0F / 247.0F) +
               border[bn1]               * ( 8.0F / 247.0F) +
               border[bn2]               * ( 1.0F / 247.0F)
               );
            break;
          case 2:
            dst[gmemPos] =
              (dptr[0]                * (70.0F / 219.0F) +
               (dptr[nsw] + dptr[sw]) * (56.0F / 219.0F) +
               dptr[nsw2]             * (28.0F / 219.0F) +
               border[bn1]             * ( 8.0F / 219.0F) +
               border[bn2]             * ( 1.0F / 219.0F)
               );
            break;
          case 1:
            dst[gmemPos] =
              (dptr[0]    * (70.0F / 163.0F) +
               dptr[nsw]  * (56.0F / 163.0F) +
               dptr[nsw2] * (28.0F / 163.0F) +
               border[bn1] * ( 8.0F / 163.0F) +
               border[bn2] * ( 1.0F / 163.0F)
               );
            break;
        }
        break;
      case 3:
        switch(h-(sts+sy))
        {
          case 4:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 255.0F) +
               (dptr[nsw] + dptr[sw])   * (56.0F / 255.0F) +
               (dptr[nsw2] + dptr[sw2]) * (28.0F / 255.0F) +
               (dptr[nsw3] + dptr[sw3]) * ( 8.0F / 255.0F) +
               border[bn1]               * ( 1.0F / 255.0F)
               );
            break;
          case 3:
            dst[gmemPos] =
              (dptr[0]                  * (70.0F / 247.0F) +
               (dptr[nsw] + dptr[sw])   * (56.0F / 247.0F) +
               (dptr[nsw2] + dptr[sw2]) * (28.0F / 247.0F) +
               dptr[nsw3]               * ( 8.0F / 247.0F) +
               border[bn1]               * ( 1.0F / 247.0F)
               );
            break;
          case 2:
            dst[gmemPos] =
              (dptr[0]                * (70.0F / 219.0F) +
               (dptr[nsw] + dptr[sw]) * (56.0F / 219.0F) +
               dptr[nsw2]             * (28.0F / 219.0F) +
               dptr[nsw3]             * ( 8.0F / 219.0F) +
               border[bn1]             * ( 1.0F / 219.0F)
               );
            break;
          case 1:
            dst[gmemPos] =
              (dptr[0]    * (70.0F / 163.0F) +
               dptr[nsw]  * (56.0F / 163.0F) +
               dptr[nsw2] * (28.0F / 163.0F) +
               dptr[nsw3] * ( 8.0F / 163.0F) +
               border[bn1] * ( 1.0F / 163.0F)
               );
            break;
        }
        break;
      }
    }
    else  // We must be at the bottom 4 of the image
    {
      switch(h-(sts+sy))
      {
      case 4:
        dst[gmemPos] =
          (dptr[0]                  * (70.0F / 255.0F) +
           (dptr[nsw] + dptr[sw])   * (56.0F / 255.0F) +
           (dptr[nsw2] + dptr[sw2]) * (28.0F / 255.0F) +
           (dptr[nsw3] + dptr[sw3]) * ( 8.0F / 255.0F) +
           dptr[nsw4]               * ( 1.0F / 255.0F)
           );
        break;
      case 3:
        dst[gmemPos] =
          (dptr[0]                  * (70.0F / 247.0F) +
           (dptr[nsw] + dptr[sw])   * (56.0F / 247.0F) +
           (dptr[nsw2] + dptr[sw2]) * (28.0F / 247.0F) +
           dptr[nsw3]               * ( 8.0F / 247.0F) +
           dptr[nsw4]               * ( 1.0F / 247.0F)
           );
        break;
      case 2:
        dst[gmemPos] =
          (dptr[0]                * (70.0F / 219.0F) +
           (dptr[nsw] + dptr[sw]) * (56.0F / 219.0F) +
           dptr[nsw2]             * (28.0F / 219.0F) +
           dptr[nsw3]             * ( 8.0F / 219.0F) +
           dptr[nsw4]             * ( 1.0F / 219.0F)
           );
        break;
      case 1:
        dst[gmemPos] =
          (dptr[0]    * (70.0F / 163.0F) +
           dptr[nsw]  * (56.0F / 163.0F) +
           dptr[nsw2] * (28.0F / 163.0F) +
           dptr[nsw3] * ( 8.0F / 163.0F) +
           dptr[nsw4] * ( 1.0F / 163.0F)
           );
        break;
      }
    }
  }
}

__global__ void cuda_global_lowpass_5_x(const float *src,  const unsigned int w, const unsigned int h, float* dst, int tile_width)
{
  // Data cache in shared memory
  //__shared__ float data[CUDA_1D_TILE_W];
  //__shared__ float border[4];
  //tile_width=CUDA_1D_TILE_W;

  // Save 4 pixels for the border
  float *data = (float *) shared_data; // size of tile_width
  float *border = (float *) &data[tile_width]; // size of 4

  const int sx = threadIdx.x;                    // source pixel within source tile
  const int sts = IMUL(blockIdx.x, tile_width);  // tile start for source, relative to row start
  const int srs = IMUL(blockIdx.y, w);           // Row start index in source data

  // Load global memory values into our data cache:
  const int loadIdx = sts + sx;  // index of one pixel value to load
  if (loadIdx < w) data[sx] = src[srs + loadIdx];

  // Load beginning border
  if (sx < 2 && sts > 0)
    border[sx] = src[srs + loadIdx - 2];
  // Load ending border
  else if(sx >= tile_width-2 && sts+tile_width < w-2)
    border[4+sx-tile_width] = src[srs + sts + sx + 2];

  // Ensure the completness of loading stage because results emitted
  // by each thread depend on the data loaded by other threads:
  __syncthreads();


  if ( loadIdx < w ) {
    const int writeIdx = sts + sx; // write index relative to row start
    const float *dptr = data + sx;
    // [ 1 4 (6) 4 1 ] / 16.0

    // If we are smaller than the Gaussian filter we are using, special case this
    // this is not very efficient, just making sure it can handle small images
    if(w < 5)
    {
      int numAhead = max(0,min(w-1-(sts+sx),2));
      int numBehind = max(0,min(sts+sx,2));
      int situation = numBehind*10+numAhead;
      switch(situation)
      {
      case 0: // 00
        dst[srs + writeIdx] = dptr[0];
        break;
      case 1: // 01
        dst[srs + writeIdx] = (dptr[0] * (6.0F / 10.0F) +
                               dptr[1] * (4.0F / 10.0F));
        break;
      case 2: // 02
        dst[srs + writeIdx] = (dptr[0] * (6.0F / 11.0F) +
                               dptr[1] * (4.0F / 11.0F) +
                               dptr[2] * (1.0F / 11.0F));
        break;
      case 10:
        dst[srs + writeIdx] = (dptr[0] * (6.0F / 10.0F) +
                               dptr[-1] * (4.0F / 10.0F));
        break;
      case 11:
        dst[srs + writeIdx] = (dptr[0]              * (6.0F / 14.0F) +
                               (dptr[-1] + dptr[1]) * (4.0F / 14.0F));
        break;
      case 12:
        dst[srs + writeIdx] = (dptr[0]              * (6.0F / 15.0F) +
                               (dptr[-1] + dptr[1]) * (4.0F / 15.0F) +
                               dptr[2]              * (1.0F / 15.0F));
        break;
      case 20:
        dst[srs + writeIdx] = (dptr[0] * (6.0F / 11.0F) +
                               dptr[-1] * (4.0F / 11.0F) +
                               dptr[-2] * (1.0F / 11.0F));
        break;
      case 21:
        dst[srs + writeIdx] = (dptr[0]              * (6.0F / 15.0F) +
                               (dptr[-1] + dptr[1]) * (4.0F / 15.0F) +
                               dptr[-2]              * (1.0F / 15.0F));
        break;
      }
    }
    // First set of pixels in the row
    else if(sts+sx < 2)
    {
      switch(sx)
      {
      case 0:
        dst[srs + writeIdx] = (dptr[0] * (6.0F / 11.0F) +
                               dptr[1] * (4.0F / 11.0F) +
                               dptr[2] * (1.0F / 11.0F));
        break;
      case 1:
        dst[srs + writeIdx] = (dptr[0]              * (6.0F / 15.0F) +
                               (dptr[-1] + dptr[1]) * (4.0F / 15.0F) +
                               dptr[2]              * (1.0F / 15.0F));
        break;
      }
    }
    // First two pixels in tile
    else if(sx < 2 && sts+sx < w-2)
    {
      switch(sx)
      {
      case 0:
        dst[srs + writeIdx] = (dptr[0]               * (6.0F / 16.0F) +
                               (border[1] + dptr[1]) * (4.0F / 16.0F) +
                               (border[0] + dptr[2]) * (1.0F / 16.0F));
        break;
      case 1:
        dst[srs + writeIdx] = (dptr[0]               * (6.0F / 16.0F) +
                               (dptr[-1] + dptr[1])  * (4.0F / 16.0F) +
                               (border[1] + dptr[2]) * (1.0F / 16.0F));
        break;
      }
    }
    // In the middle of the tile
    else if(sx < tile_width-2 && sts+sx < w-2)
    {
      dst[srs + writeIdx] = (dptr[0]              * (6.0F / 16.0F) +
                             (dptr[-1] + dptr[1]) * (4.0F / 16.0F) +
                             (dptr[-2] + dptr[2]) * (1.0F / 16.0F));
    }
    // Last two pixels of the tile
    else if(sts+sx < w-2)
    {
      switch(tile_width-sx)
      {
      case 2:
        dst[srs + writeIdx] = (dptr[0]                * (6.0F / 16.0F) +
                               (dptr[-1] + dptr[1])   * (4.0F / 16.0F) +
                               (dptr[-2] + border[2]) * (1.0F / 16.0F));
        break;
      case 1:
        dst[srs + writeIdx] = (dptr[0]                * (6.0F / 16.0F) +
                               (dptr[-1] + border[2]) * (4.0F / 16.0F) +
                               (dptr[-2] + border[3]) * (1.0F / 16.0F));
        break;
      }
    }
    // Last two pixels of the row
    else
    {
      switch(w-(sts+sx))
      {
      case 2:
        dst[srs + writeIdx] = (dptr[0]              * (6.0F / 15.0F) +
                               (dptr[-1] + dptr[1]) * (4.0F / 15.0F) +
                               (dptr[-2])           * (1.0F / 15.0F));
        break;
      case 1:
        dst[srs + writeIdx] = (dptr[0]    * (6.0F / 11.0F) +
                               (dptr[-1]) * (4.0F / 11.0F) +
                               (dptr[-2]) * (1.0F / 11.0F));
        break;
      }
    }
  }
}

__global__ void cuda_global_lowpass_5_y(const float *src,  const unsigned int w, const unsigned int h, float* dst, int tile_width, int tile_height)
{
  // Data cache
  //__shared__ float data[CUDA_TILE_W*CUDA_TILE_H];
  //__shared__ float border[CUDA_TILE_W*4];
  //const int tile_width=CUDA_TILE_W;
  //const int tile_height= CUDA_TILE_H;

  // Save 4 rows for the border
  float *data = (float *) shared_data; //tile_width * tile_height size
  float *border = (float *) &data[tile_width*tile_height]; // size of tile_width*4

  const int sy = threadIdx.y; // source pixel row within source tile

  const int sts = IMUL(blockIdx.y, tile_height); // tile start for source, in rows
  const int ste = sts + tile_height; // tile end for source, in rows

  // Clamp tile and apron limits by image borders
  const int stec = min(ste, h);

  // Current column index
  const int scs = IMUL(blockIdx.x, tile_width) + threadIdx.x;

  int smemPos = IMUL(sy, tile_width) + threadIdx.x;
  int gmemPos = IMUL(sts + sy, w) + scs;

  // only process columns that are actually within image bounds:
  if (scs < w && sts+sy < h) {
    // Shared and global (source) memory indices for current column

    // Load data
    data[smemPos] = src[gmemPos];

    // Load border
    if (sy < 2 && gmemPos > IMUL(2,w))
      border[smemPos] = src[gmemPos-IMUL(2,w)];

    int bordOff = 4+sy-tile_height;

    if (sy >= tile_height-2 && ste+2 < h)
      border[threadIdx.x+IMUL(bordOff,tile_width)] = src[gmemPos+IMUL(2,w)];

    // Ensure the completness of loading stage because results emitted
    // by each thread depend on the data loaded by other threads:
    __syncthreads();

    // Cycle through the tile body, clamped by image borders
    // Calculate and output the results
    float *dptr = data + smemPos;
    const int sw = tile_width, sw2 = sw + sw;
    const int nsw = -sw, nsw2 = nsw - sw;
    const int bn2 = threadIdx.x, bn1 = bn2 + tile_width;
    const int bp1 = bn1+tile_width, bp2 = bp1 + tile_width;

    //  [ 1 4 (6) 4 1 ] / 16

    // If we are smaller than the Gaussian filter we are using, special case this
    // this is not very efficient, just making sure it can handle small images
    if(h < 5)
      {
        int numAhead = max(0,min(h-1-(sts+sy),2));
        int numBehind = max(0,min(sts+sy,2));
        int situation = numBehind*10+numAhead;
        switch(situation)
          {
          case 0: // 00
            dst[gmemPos] = dptr[0];
            break;
          case 1: // 01
            dst[gmemPos] = (dptr[0] * (6.0F / 10.0F) +
                            dptr[sw] * (4.0F / 10.0F));
            break;
          case 2: // 02
            dst[gmemPos] = (dptr[0] * (6.0F / 11.0F) +
                            dptr[sw] * (4.0F / 11.0F) +
                            dptr[sw2] * (1.0F / 11.0F));
            break;
          case 10:
            dst[gmemPos] = (dptr[0] * (6.0F / 10.0F) +
                            dptr[nsw] * (4.0F / 10.0F));
            break;
          case 11:
            dst[gmemPos] = (dptr[0]              * (6.0F / 14.0F) +
                            (dptr[nsw] + dptr[sw]) * (4.0F / 14.0F));
            break;
          case 12:
            dst[gmemPos] = (dptr[0]              * (6.0F / 15.0F) +
                            (dptr[nsw] + dptr[sw]) * (4.0F / 15.0F) +
                            dptr[sw2]              * (1.0F / 15.0F));
            break;
          case 20:
            dst[gmemPos] = (dptr[0] * (6.0F / 11.0F) +
                            dptr[nsw] * (4.0F / 11.0F) +
                            dptr[nsw2] * (1.0F / 11.0F));
            break;
          case 21:
            dst[gmemPos] = (dptr[0]              * (6.0F / 15.0F) +
                            (dptr[nsw] + dptr[sw]) * (4.0F / 15.0F) +
                            dptr[nsw2]              * (1.0F / 15.0F));
            break;
          }
      }
    // Are we in the top 2 rows of the whole image
    else if(sts + sy < 2)
      {
        switch(sts+sy)
          {
          case 0:
            dst[gmemPos] =
              (dptr[0]   * (6.0F / 11.0F) +
               dptr[sw]  * (4.0F / 11.0F) +
               dptr[sw2] * (1.0F / 11.0F)
               );
            break;
          case 1:
            dst[gmemPos] =
              (dptr[0]                * (6.0F / 15.0F) +
               (dptr[nsw] + dptr[sw]) * (4.0F / 15.0F) +
               dptr[sw2]              * (1.0F / 15.0F)
               );
            break;
          }
      }
    else if(sy < 2 && sts+sy<h-2) // If not top 2 in the whole image, are we in the top 2 rows of this tile
      {
        switch(sy)
          {
          case 0:
            dst[gmemPos] =
              (dptr[0]                   * (6.0F / 16.0F) +
               (border[bn1] + dptr[sw])  * (4.0F / 16.0F) +
               (border[bn2] + dptr[sw2]) * (1.0F / 16.0F)
               );
            break;
          case 1:
            dst[gmemPos] =
              (dptr[0]                   * (6.0F / 16.0F) +
               (dptr[nsw] + dptr[sw])    * (4.0F / 16.0F) +
               (border[bn1] + dptr[sw2]) * (1.0F / 16.0F)
               );
            break;
          }
      }
    else if(sy <tile_height-2 && sts+sy<h-2) // Are we in the middle of the tile
      {
        dst[gmemPos] =
          ((dptr[nsw2] + dptr[sw2]) * (1.0F / 16.0F) +
           (dptr[nsw] + dptr[sw])   * (4.0F / 16.0F) +
           dptr[0]                  * (6.0F / 16.0F)
           );
      }
    else if(sts + sy < h-2) // Are we not at the bottom of the image, but bottom 4 of the tile
      {
        switch(tile_height-sy)
          {
          case 2:
            dst[gmemPos] =
              (dptr[0]                    * (6.0F / 16.0F) +
               (dptr[nsw] + dptr[sw])     * (4.0F / 16.0F) +
               (dptr[nsw2] + border[bp1]) * (1.0F / 16.0F)
               );
            break;
          case 1:
            dst[gmemPos] =
              (dptr[0]                    * (6.0F / 16.0F) +
               (dptr[nsw] +  border[bp1]) * (4.0F / 16.0F) +
               (dptr[nsw2] + border[bp2]) * (1.0F / 16.0F)
               );
            break;
          }
      }
    else // We must be at the bottom 4 of the image
      {
        switch(h-(sts+sy))
          {
          case 2:
            dst[gmemPos] =
              (dptr[0]                * (6.0F / 15.0F) +
               (dptr[nsw] + dptr[sw]) * (4.0F / 15.0F) +
               dptr[nsw2]             * (1.0F / 15.0F)
               );
            break;
          case 1:
            dst[gmemPos] =
              (dptr[0]    * (6.0F / 11.0F) +
               dptr[nsw]  * (4.0F / 11.0F) +
               dptr[nsw2] * (1.0F / 11.0F)
               );
            break;
          }
      }

  }

}



__global__ void cuda_global_lowpass_3_x(const float *src,  const unsigned int w, const unsigned int h, float* dst, int tile_width)
{

  // Save 2 pixels for the border
  float *data = (float *) shared_data; // size of tile_width
  float *border = (float *) &data[tile_width]; // size of 2

  const int sx = threadIdx.x;                    // source pixel within source tile
  const int sts = IMUL(blockIdx.x, tile_width);  // tile start for source, relative to row start
  const int srs = IMUL(blockIdx.y, w);           // Row start index in source data

  // Load global memory values into our data cache:
  const int loadIdx = sts + sx;  // index of one pixel value to load
  if (loadIdx < w) data[sx] = src[srs + loadIdx];

  // Load beginning border
  if (sx < 1 && sts > 0)
    border[sx] = src[srs + loadIdx - 1];
  // Load ending border
  else if(sx >= tile_width-1 && sts+tile_width < w-1)
    border[2+sx-tile_width] = src[srs + sts + sx + 1];

  // Ensure the completness of loading stage because results emitted
  // by each thread depend on the data loaded by other threads:
  __syncthreads();


  if ( loadIdx < w ) {
    const int writeIdx = sts + sx; // write index relative to row start
    const float *dptr = data + sx;
    // [ 1 (2) 1 ] / 4.0

    // If we are smaller than the Gaussian filter we are using, special case this
    // this is not very efficient, just making sure it can handle small images
    if(w < 3)
    {
      int numAhead = max(0,min(w-1-(sts+sx),1));
      int numBehind = max(0,min(sts+sx,1));
      int situation = numBehind*10+numAhead;
      switch(situation)
      {
      case 0: // 00
        dst[srs + writeIdx] = dptr[0];
        break;
      case 1: // 01
        dst[srs + writeIdx] = (dptr[0] * (2.0F / 3.0F) +
                               dptr[1] * (1.0F / 3.0F));
        break;
      case 10:
        dst[srs + writeIdx] = (dptr[0] * (2.0F / 3.0F) +
                               dptr[-1] * (1.0F / 3.0F));
        break;
      case 11:
        dst[srs + writeIdx] = (dptr[0]              * (2.0F / 4.0F) +
                               (dptr[-1] + dptr[1]) * (1.0F / 4.0F));
        break;
      }
    }
    // First set of pixels in the row
    else if(sts+sx < 1)
    {
      switch(sx)
      {
      case 0:
        dst[srs + writeIdx] = (dptr[0] * (2.0F / 3.0F) +
                               dptr[1] * (1.0F / 3.0F));
        break;
      }
    }
    // First pixel in tile
    else if(sx < 1 && sts+sx < w-1)
    {
      switch(sx)
      {
      case 0:
        dst[srs + writeIdx] = (dptr[0]               * (2.0F / 4.0F) +
                               (border[0] + dptr[1]) * (1.0F / 4.0F));
        break;
      }
    }
    // In the middle of the tile
    else if(sx < tile_width-1 && sts+sx < w-1)
    {
      dst[srs + writeIdx] = (dptr[0]              * (2.0F / 4.0F) +
                             (dptr[-1] + dptr[1]) * (1.0F / 4.0F));
    }
    // Last pixel of the tile
    else if(sts+sx < w-1)
    {
      switch(tile_width-sx)
      {
      case 1:
        dst[srs + writeIdx] = (dptr[0]                * (2.0F / 4.0F) +
                               (dptr[-1] + border[1]) * (1.0F / 4.0F));
        break;
      }
    }
    // Last pixel of the row
    else
    {
      switch(w-(sts+sx))
      {
      case 1:
        dst[srs + writeIdx] = (dptr[0]    * (2.0F / 3.0F) +
                               (dptr[-1]) * (1.0F / 3.0F));
        break;
      }
    }
  }
}

__global__ void cuda_global_lowpass_3_y(const float *src,  const unsigned int w, const unsigned int h, float* dst, int tile_width, int tile_height)
{
  // Data cache
  //__shared__ float data[CUDA_TILE_W*CUDA_TILE_H];
  //__shared__ float border[CUDA_TILE_W*4];
  //const int tile_width=CUDA_TILE_W;
  //const int tile_height= CUDA_TILE_H;

  // Save 2 rows for the border
  float *data = (float *) shared_data; //tile_width * tile_height size
  float *border = (float *) &data[tile_width*tile_height]; // size of tile_width*2

  const int sy = threadIdx.y; // source pixel row within source tile

  const int sts = IMUL(blockIdx.y, tile_height); // tile start for source, in rows
  const int ste = sts + tile_height; // tile end for source, in rows

  // Clamp tile and apron limits by image borders
  const int stec = min(ste, h);

  // Current column index
  const int scs = IMUL(blockIdx.x, tile_width) + threadIdx.x;

  // only process columns that are actually within image bounds:
  if (scs < w && sts+sy < h) {
    // Shared and global (source) memory indices for current column
    int smemPos = IMUL(sy, tile_width) + threadIdx.x;
    int gmemPos = IMUL(sts + sy, w) + scs;

    // Load data
    data[smemPos] = src[gmemPos];

    // Load border
    if (sy < 1 && gmemPos > w)
      border[smemPos] = src[gmemPos-w];

    int bordOff = 2+sy-tile_height;

    if (sy >= tile_height-1 && ste+1 < h)
      border[threadIdx.x+IMUL(bordOff,tile_width)] = src[gmemPos+w];

    // Ensure the completness of loading stage because results emitted
    // by each thread depend on the data loaded by other threads:
    __syncthreads();


    // Cycle through the tile body, clamped by image borders
    // Calculate and output the results
    float *dptr = data + smemPos;
    const int sw = tile_width;
    const int nsw = -sw;
    const int bn1 = threadIdx.x;
    const int bp1 = bn1+tile_width;

    //  [ 1 (2) 1 ] / 4

    // If we are smaller than the Gaussian filter we are using, special case this
    // this is not very efficient, just making sure it can handle small images
    if(h < 3)
      {
        int numAhead = max(0,min(h-1-(sts+sy),1));
        int numBehind = max(0,min(sts+sy,1));
        int situation = numBehind*10+numAhead;
        switch(situation)
          {
          case 0: // 00
            dst[gmemPos] = dptr[0];
            break;
          case 1: // 01
            dst[gmemPos] = (dptr[0] * (2.0F / 3.0F) +
                            dptr[sw] * (1.0F / 3.0F));
            break;
          case 10:
            dst[gmemPos] = (dptr[0] * (2.0F / 3.0F) +
                            dptr[nsw] * (1.0F / 3.0F));
            break;
          case 11:
            dst[gmemPos] = (dptr[0]              * (2.0F / 4.0F) +
                            (dptr[nsw] + dptr[sw]) * (1.0F / 4.0F));
            break;
          }
      }
    // Are we in the top row of the whole image
    else if(sts + sy < 1)
      {
        switch(sts+sy)
          {
          case 0:
            dst[gmemPos] =
              (dptr[0]   * (2.0F / 3.0F) +
               dptr[sw]  * (1.0F / 3.0F));
            break;
          }
      }
    else if(sy < 1 && sts+sy<h-1) // If not top in the whole image, are we in the top row of this tile
      {
        switch(sy)
          {
          case 0:
            dst[gmemPos] =
              (dptr[0]                   * (2.0F / 4.0F) +
               (border[bn1] + dptr[sw])  * (1.0F / 4.0F));
            break;
          }
      }
    else if(sy <tile_height-1 && sts+sy<h-1) // Are we in the middle of the tile
      {
        dst[gmemPos] =
          (dptr[0]                  * (2.0F / 4.0F) +
           (dptr[nsw] + dptr[sw])   * (1.0F / 4.0F));
      }
    else if(sts + sy < h-1) // Are we not at the bottom of the image, but bottom row of the tile
      {
        switch(tile_height-sy)
          {
          case 1:
            dst[gmemPos] =
              (dptr[0]                    * (2.0F / 4.0F) +
               (dptr[nsw] +  border[bp1]) * (1.0F / 4.0F));
            break;
          }
      }
    else // We must be at the bottom row of the image
      {
        switch(h-(sts+sy))
          {
          case 1:
            dst[gmemPos] =
              (dptr[0]    * (2.0F / 3.0F) +
               dptr[nsw]  * (1.0F / 3.0F));
            break;
          }
      }

  }

}


#include "CUDA/cutil.h"

texture<float, 2, cudaReadModeElementType> texRef;


__global__ void cuda_global_lowpass_texture_9_x_dec_x(const float *src, int w, int h, float *dst, int dw, int dh, int tile_width, int tile_height)
{
  // w and h are from the original source image
  float *data = (float *) shared_data; //(tile_width+8) * tile_height size
  const int yResPos = IMUL(blockIdx.y, tile_height) + threadIdx.y; // Y position
  const int xResPos = IMUL(blockIdx.x,tile_width) + threadIdx.x; // index of one pixel value to load
  const int ySrcPos = yResPos; // Y position (unchanged in this function)
  const int xSrcPos = xResPos<<1; // X position
  const int ctrOff = 4;
  const int shrIdx = threadIdx.x+ctrOff;
  // [ 1 8 28 56 (70) 56 28 8 1 ]
  if(xResPos < dw && yResPos < dh)
    {
      data[shrIdx] = tex2D(texRef,xSrcPos,ySrcPos);
      if(threadIdx.x < ctrOff)
        data[shrIdx-ctrOff] = tex2D(texRef,xSrcPos-ctrOff,ySrcPos);
      else if(threadIdx.x > tile_width - ctrOff)
        data[shrIdx+ctrOff] = tex2D(texRef,xSrcPos+ctrOff,ySrcPos);
      __syncthreads();
      dst[IMUL(yResPos,dw) + xResPos] =
        ((data[shrIdx-4] + data[shrIdx+4]) * ( 1.0F / 256.0F) +
         (data[shrIdx-3] + data[shrIdx+3]) * ( 8.0F / 256.0F) +
         (data[shrIdx-2] + data[shrIdx+2]) * (28.0F / 256.0F) +
         (data[shrIdx-1] + data[shrIdx+1]) * (56.0F / 256.0F) +
         data[shrIdx]              * (70.0F / 256.0F)
         );
    }
}

void cuda_c_lowpass_texture_9_x_dec_x(const float *src, int w, int h, float *dst, int dw, int dh, int tile_width, int tile_height)
{
  //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  texRef.addressMode[0] = cudaAddressModeClamp;
  texRef.addressMode[1] = cudaAddressModeClamp;
  texRef.filterMode = cudaFilterModePoint;
  texRef.normalized = false;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>(); //cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
  CUDA_SAFE_CALL_NO_SYNC(cudaBindTexture2D(NULL,texRef,src,channelDesc,w,h,w*sizeof(float)));
  dim3 blockGridRows(iDivUp(dw, tile_width), iDivUp(dh, tile_height));
  dim3 threadBlockRows(tile_width, tile_height);
  cuda_global_lowpass_texture_9_x_dec_x<<<blockGridRows, threadBlockRows, (tile_width+8)*tile_height*sizeof(float)>>>(src, w, h, dst, dw, dh, tile_width, tile_height);
  CUDA_SAFE_CALL_NO_SYNC(cudaUnbindTexture(texRef));
}

__global__ void cuda_global_lowpass_texture_9_y_dec_y(const float *src, int w, int h, float *dst, int dw, int dh, int tile_width, int tile_height)
{
  // w and h are from the original source image
  float *data = (float *) shared_data; //tile_width * (tile_height+8) size
  const int yResPos = IMUL(blockIdx.y, tile_height) + threadIdx.y; // Y position
  const int xResPos = IMUL(blockIdx.x,tile_width) + threadIdx.x; // index of one pixel value to load
  const int ySrcPos = yResPos<<1; // Y position
  const int xSrcPos = xResPos; // X position (unchanged in this function)
  const int ctrOff = 4;
  const int shrIdx = threadIdx.y+ctrOff;
  // [ 1 8 28 56 (70) 56 28 8 1 ]
  if(xResPos < dw && yResPos < dh)
    {
      data[shrIdx] = tex2D(texRef,xSrcPos,ySrcPos);
      if(threadIdx.y < ctrOff)
        data[shrIdx-ctrOff] = tex2D(texRef,xSrcPos,ySrcPos-ctrOff);
      else if(threadIdx.y > tile_height - ctrOff)
        data[shrIdx+ctrOff] = tex2D(texRef,xSrcPos,ySrcPos+ctrOff);
      __syncthreads();
      dst[IMUL(yResPos,dw) + xResPos] =
        ((data[shrIdx-4] + data[shrIdx+4]) * ( 1.0F / 256.0F) +
         (data[shrIdx-3] + data[shrIdx+3]) * ( 8.0F / 256.0F) +
         (data[shrIdx-2] + data[shrIdx+2]) * (28.0F / 256.0F) +
         (data[shrIdx-1] + data[shrIdx+1]) * (56.0F / 256.0F) +
         data[shrIdx]              * (70.0F / 256.0F)
         );
    }
}

void cuda_c_lowpass_texture_9_y_dec_y(const float *src, int w, int h, float *dst, int dw, int dh, int tile_width, int tile_height)
{
  //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  texRef.addressMode[0] = cudaAddressModeClamp;
  texRef.addressMode[1] = cudaAddressModeClamp;
  texRef.filterMode = cudaFilterModePoint;
  texRef.normalized = false;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>(); //cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
  CUDA_SAFE_CALL_NO_SYNC(cudaBindTexture2D(NULL,texRef,src,channelDesc,w,h,w*sizeof(float)));
  dim3 blockGridRows(iDivUp(dw, tile_width), iDivUp(dh, tile_height));
  dim3 threadBlockRows(tile_width, tile_height);
  cuda_global_lowpass_texture_9_y_dec_y<<<blockGridRows, threadBlockRows, tile_width*(tile_height+8)*sizeof(float)>>>(src, w, h, dst, dw, dh, tile_width, tile_height);
  CUDA_SAFE_CALL_NO_SYNC(cudaUnbindTexture(texRef));
}


#endif
