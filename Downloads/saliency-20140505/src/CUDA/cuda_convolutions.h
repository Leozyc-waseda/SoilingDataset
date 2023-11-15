/*!@file CUDA/cuda-convolutions.h CUDA/GPU convolution methods */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/cuda_convolutions.h $
// $Id: cuda_convolutions.h 12962 2010-03-06 02:13:53Z irock $
//

#ifndef CUDA_CONVOLUTIONS_H_DEFINED
#define CUDA_CONVOLUTIONS_H_DEFINED

#include <cuda.h>
#include "CUDA/cutil.h"
#include "cudadefs.h"



__global__ void cuda_global_convolveHmaxHelper(float *res, const float *src, const int src_w, const int src_h,
                                                const float *f, const int Nx, const int Ny, const int tile_width, const int tile_height)
{
  const int y_pos = IMUL(blockIdx.y, tile_height) + threadIdx.y;
  const int x_pos = IMUL(blockIdx.x, tile_width) + threadIdx.x;
  const int ld_idx = IMUL(threadIdx.y,tile_width) + threadIdx.x;
  const int res_idx = IMUL(y_pos,src_w)+x_pos;

  // Load the filter into shared memory
  float *data = (float *) shared_data; // size of tile_width
  const int tile_sz = IMUL(tile_height,tile_width);
  const int fil_sz = IMUL(Nx,Ny);

  for(int i=0;ld_idx+i<fil_sz;i+=tile_sz)
  {
    const int curld_idx = ld_idx+i;
    data[curld_idx] = f[curld_idx];
  }
  __syncthreads();

  if(y_pos < src_h && x_pos < src_w)
  {
    const int Nx2 = (Nx - 1) / 2;
    const int Ny2 = (Ny - 1) / 2;
    float sum = 0.0F;
    float sumw = 0.0F;
    float im;
    for(int i=0;i<Ny;i++)
      {
        const int s_y=y_pos-Ny2+i;
        const int f_y=Ny-i-1;
        if(s_y >= 0 && s_y < src_h)
        {
          for(int j=0;j<Nx;j++)
          {
            const int s_x=x_pos-Nx2+j;
            const int f_x=Nx-j-1;
            const int flt_idx = IMUL(f_y,Nx)+f_x;
            const int src_idx = IMUL(s_y,src_w)+s_x;
            if(s_x >=0 && s_x < src_w)
              {
                im = src[src_idx];
                //sum += im*data[flt_idx];
                sum = __fadd_rn(__fmul_rn(im,data[flt_idx]),sum);
                //sumw += im*im;
                sumw = __fadd_rn(__fmul_rn(im,im),sumw);
              }
          }
        }
      }
    if(sumw > 0.0F)
      res[res_idx] = fabs(sum)/sqrt(sumw);
    else
      res[res_idx] = sum;
  }

}


__global__ void cuda_global_convolveZeroHelper(float *res, const float *src, const int src_w, const int src_h,
                                               const float *f, const int Nx, const int Ny, const int tile_width, const int tile_height)
{
  const int y_pos = IMUL(blockIdx.y, tile_height) + threadIdx.y;
  const int x_pos = IMUL(blockIdx.x, tile_width) + threadIdx.x;
  const int ld_idx = threadIdx.y*tile_width + threadIdx.x;
  const int res_idx = y_pos*src_w+x_pos;

  // Load the filter into shared memory
  float *data = (float *) shared_data; // size of tile_width
  const int tile_sz = IMUL(tile_height,tile_width);
  const int fil_sz = IMUL(Nx,Ny);

  for(int i=0;ld_idx+i<fil_sz;i+=tile_sz)
  {
    const int curld_idx = ld_idx+i;
    data[curld_idx] = f[curld_idx];
  }

  __syncthreads();

  if(y_pos < src_h && x_pos < src_w)
  {
    const int Nx2 = (Nx - 1) >>1;
    const int Ny2 = (Ny - 1) >>1;
    float sum = 0.0F;
    for(int i=0;i<Ny;i++)
      {
        const int s_y=y_pos-Ny2+i;
        const int f_y=Ny-i-1;
        if(s_y >= 0 && s_y < src_h)
          {
            for(int j=0;j<Nx;j++)
              {
                const int s_x=x_pos-Nx2+j;
                const int f_x=Nx-j-1;
                if(s_x >=0 && s_x < src_w)
                  {
                    const int flt_idx = IMUL(f_y,Nx)+f_x;
                    const int src_idx = IMUL(s_y,src_w)+s_x;
                    //sum += src[src_idx]*data[flt_idx];
                    //sum = __fmaf_rn(src[src_idx],data[flt_idx],sum);
                    sum = __fadd_rn(__fmul_rn(src[src_idx],data[flt_idx]),sum);
                  }
              }
          }
      }
    res[res_idx] = sum;
  }

}

__global__ void cuda_global_convolveCleanHelper(float *res, const float *src, const int src_w, const int src_h,
                                                const float *f, const int Nx, const int Ny, const int tile_width, const int tile_height)
{
  const int y_pos = IMUL(blockIdx.y, tile_height) + threadIdx.y;
  const int x_pos = IMUL(blockIdx.x, tile_width) + threadIdx.x;
  const int ld_idx = threadIdx.y*tile_width + threadIdx.x;
  const int res_idx = y_pos*src_w+x_pos;

  // Load the filter into shared memory
  float *data = (float *) shared_data; // size of tile_width
  const int tile_sz = IMUL(tile_height,tile_width);
  const int fil_sz = IMUL(Nx,Ny);

  for(int i=0;ld_idx+i<fil_sz;i+=tile_sz)
  {
    const int curld_idx = ld_idx+i;
    data[curld_idx] = f[curld_idx];
  }

  __syncthreads();

  if(y_pos < src_h && x_pos < src_w)
  {
    const int Nx2 = (Nx - 1) / 2;
    const int Ny2 = (Ny - 1) / 2;
    float sum = 0.0F;
    float sumf = 0.0F;
    float sumw = 0.0F;
    for(int i=0;i<Ny;i++)
      {
        const int s_y=y_pos-Ny2+i;
        const int f_y=Ny-i-1;
        for(int j=0;j<Nx;j++)
          {
            const int s_x=x_pos-Nx2+j;
            const int f_x=Nx-j-1;
            const int flt_idx = IMUL(f_y,Nx)+f_x;
            const int src_idx = IMUL(s_y,src_w)+s_x;
            if(s_y >= 0 && s_y < src_h  && s_x >=0 && s_x < src_w)
              {
                //sum += src[src_idx]*data[flt_idx];
                sum = __fadd_rn(__fmul_rn(src[src_idx],data[flt_idx]),sum);
                sumw += data[flt_idx];
                sumf += data[flt_idx];
              }
            else
              {
                sumf += data[flt_idx];
              }
          }
      }

    res[res_idx] = sum*sumf/sumw;
  }

}


__global__ void cuda_global_convolveHmaxHelperOptimized(float *res, const float *src, const int src_w, const int src_h,
                                               const float *f, const int Nx, const int Ny, const int tile_width, const int tile_height)
{
  const int y_pos = IMUL(blockIdx.y, tile_height) + threadIdx.y;
  const int x_pos = IMUL(blockIdx.x, tile_width) + threadIdx.x;
  const int ld_idx = IMUL(threadIdx.y,tile_width) + threadIdx.x;
  const int res_idx = IMUL(y_pos,src_w)+x_pos;

  // Load the filter into shared memory
  const int tile_sz = IMUL(tile_height,tile_width);
  const int fil_sz = IMUL(Nx,Ny);

  const int Nx2 = (Nx - 1) >>1;
  const int Ny2 = (Ny - 1) >>1;

  float *filt = (float *) shared_data; // size of filter: fil_sz
  float *data = (float *) &(filt[fil_sz]); // size of tile: (tile_width+Nx)*(tile_height+Ny);
  const int dw = tile_width+Nx;
  //const int dh = tile_height+Ny; // Not used
  const int shr_x = threadIdx.x+Nx2;
  const int shr_y = threadIdx.y+Ny2;
  const int shr_idx = IMUL(shr_y,dw)+shr_x;

  // Load filter
  for(int i=0;ld_idx+i<fil_sz;i+=tile_sz)
    {
      const int curld_idx = ld_idx+i;
      filt[curld_idx] = f[curld_idx];
    }
  if(y_pos < src_h && x_pos < src_w)
    {
      // Load bulk of tile
      data[shr_idx]=src[res_idx];
      const bool topBorder = (y_pos > Ny2 && threadIdx.y < Ny2);
      const bool bottomBorder = (y_pos < src_h-Ny2 && threadIdx.y > tile_height-Ny2-1);
      const bool leftBorder = (x_pos > Nx2 && threadIdx.x < Nx2);
      const bool rightBorder = (x_pos < src_w-Nx2 && threadIdx.x > tile_width-Nx2-1);
      // Load top border
      if(topBorder)
        data[shr_idx-IMUL(Ny2,dw)] = src[res_idx-IMUL(Ny2,src_w)];
      // Load bottom border
      if(bottomBorder)
        data[shr_idx+IMUL(Ny2,dw)] = src[res_idx+IMUL(Ny2,src_w)];
      // Load left border
      if(leftBorder)
        data[shr_idx-Nx2] = src[res_idx-Nx2];
      // Load right border
      if(rightBorder)
        data[shr_idx+Nx2] = src[res_idx+Nx2];
      // Load corners
      if(topBorder && leftBorder)
        data[shr_idx-IMUL(Ny2,dw)-Nx2] = src[res_idx-IMUL(Ny2,src_w)-Nx2];
      if(topBorder && rightBorder)
        data[shr_idx-IMUL(Ny2,dw)+Nx2] = src[res_idx-IMUL(Ny2,src_w)+Nx2];
      if(bottomBorder && leftBorder)
        data[shr_idx+IMUL(Ny2,dw)-Nx2] = src[res_idx+IMUL(Ny2,src_w)-Nx2];
      if(bottomBorder && rightBorder)
        data[shr_idx+IMUL(Ny2,dw)+Nx2] = src[res_idx+IMUL(Ny2,src_w)+Nx2];
      __syncthreads();

      float sum = 0.0F, sumw = 0.0F;
      for(int i=0;i<Ny;i++)
        {
          const int s_y=shr_y-Ny2+i;
          const int y_loc = y_pos-Ny2+i;
          const int f_y=Ny-i-1;
          if(y_loc >= 0 && y_loc < src_h)
            {
              for(int j=0;j<Nx;j++)
                {
                  const int s_x=shr_x-Nx2+j;
                  const int x_loc = x_pos-Nx2+j;
                  const int f_x=Nx-j-1;
                  if(x_loc >=0 && x_loc < src_w)
                    {
                      const int flt_idx = IMUL(f_y,Nx)+f_x;
                      const int dat_idx = IMUL(s_y,dw)+s_x;
                      //sum += im*data[flt_idx];
                      sum = __fadd_rn(__fmul_rn(data[dat_idx],filt[flt_idx]),sum);
                      //sumw += im*im;
                      sumw = __fadd_rn(__fmul_rn(data[dat_idx],data[dat_idx]),sumw);
                    }
                }
            }
        }
      if(sumw > 0.0F)
        res[res_idx] = fabs(sum)/sqrt(sumw);
      else
        res[res_idx] = sum;
    }
}



__global__ void cuda_global_convolveZeroHelperOptimized(float *res, const float *src, const int src_w, const int src_h,
                                               const float *f, const int Nx, const int Ny, const int tile_width, const int tile_height)
{
  const int y_pos = IMUL(blockIdx.y, tile_height) + threadIdx.y;
  const int x_pos = IMUL(blockIdx.x, tile_width) + threadIdx.x;
  const int ld_idx = IMUL(threadIdx.y,tile_width) + threadIdx.x;
  const int res_idx = IMUL(y_pos,src_w)+x_pos;

  // Load the filter into shared memory
  const int tile_sz = IMUL(tile_height,tile_width);
  const int fil_sz = IMUL(Nx,Ny);

  const int Nx2 = (Nx - 1) >>1;
  const int Ny2 = (Ny - 1) >>1;

  float *filt = (float *) shared_data; // size of filter: fil_sz
  float *data = (float *) &(filt[fil_sz]); // size of tile: (tile_width+Nx)*(tile_height+Ny);
  const int dw = tile_width+Nx;
  //const int dh = tile_height+Ny; // Not used
  const int shr_x = threadIdx.x+Nx2;
  const int shr_y = threadIdx.y+Ny2;
  const int shr_idx = IMUL(shr_y,dw)+shr_x;

  // Load filter
  for(int i=0;ld_idx+i<fil_sz;i+=tile_sz)
    {
      const int curld_idx = ld_idx+i;
      filt[curld_idx] = f[curld_idx];
    }
  if(y_pos < src_h && x_pos < src_w)
    {
      // Load bulk of tile
      data[shr_idx]=src[res_idx];
      const bool topBorder = (y_pos > Ny2 && threadIdx.y < Ny2);
      const bool bottomBorder = (y_pos < src_h-Ny2 && threadIdx.y > tile_height-Ny2-1);
      const bool leftBorder = (x_pos > Nx2 && threadIdx.x < Nx2);
      const bool rightBorder = (x_pos < src_w-Nx2 && threadIdx.x > tile_width-Nx2-1);
      // Load top border
      if(topBorder)
        data[shr_idx-IMUL(Ny2,dw)] = src[res_idx-IMUL(Ny2,src_w)];
      // Load bottom border
      if(bottomBorder)
        data[shr_idx+IMUL(Ny2,dw)] = src[res_idx+IMUL(Ny2,src_w)];
      // Load left border
      if(leftBorder)
        data[shr_idx-Nx2] = src[res_idx-Nx2];
      // Load right border
      if(rightBorder)
        data[shr_idx+Nx2] = src[res_idx+Nx2];
      // Load corners
      if(topBorder && leftBorder)
        data[shr_idx-IMUL(Ny2,dw)-Nx2] = src[res_idx-IMUL(Ny2,src_w)-Nx2];
      if(topBorder && rightBorder)
        data[shr_idx-IMUL(Ny2,dw)+Nx2] = src[res_idx-IMUL(Ny2,src_w)+Nx2];
      if(bottomBorder && leftBorder)
        data[shr_idx+IMUL(Ny2,dw)-Nx2] = src[res_idx+IMUL(Ny2,src_w)-Nx2];
      if(bottomBorder && rightBorder)
        data[shr_idx+IMUL(Ny2,dw)+Nx2] = src[res_idx+IMUL(Ny2,src_w)+Nx2];
      __syncthreads();

      float sum = 0.0F;
      for(int i=0;i<Ny;i++)
        {
          const int s_y=shr_y-Ny2+i;
          const int y_loc = y_pos-Ny2+i;
          const int f_y=Ny-i-1;
          if(y_loc >= 0 && y_loc < src_h)
            {
              for(int j=0;j<Nx;j++)
                {
                  const int s_x=shr_x-Nx2+j;
                  const int x_loc = x_pos-Nx2+j;
                  const int f_x=Nx-j-1;
                  if(x_loc >=0 && x_loc < src_w)
                    {
                      const int flt_idx = IMUL(f_y,Nx)+f_x;
                      const int dat_idx = IMUL(s_y,dw)+s_x;
                      //sum += src[src_idx]*data[flt_idx];
                      //sum = __fmaf_rn(src[src_idx],data[flt_idx],sum);
                      sum = __fadd_rn(__fmul_rn(data[dat_idx],filt[flt_idx]),sum);
                    }
                }
            }
        }
      res[res_idx] = sum;
    }
}




__global__ void cuda_global_optConvolve(float *res, const float *src, const int src_w, const int src_h,
                             const float *f, const int fil_w, const int fil_h, const int tile_width, const int tile_height)
{
  const int y_pos = IMUL(blockIdx.y, tile_height) + threadIdx.y;
  const int x_pos = IMUL(blockIdx.x, tile_width) + threadIdx.x;
  const int ld_idx = threadIdx.y*tile_width + threadIdx.x;
  const int res_idx = y_pos*src_w+x_pos;

  // Load the filter into shared memory
  float *data = (float *) shared_data; // size of tile_width
  const int tile_sz = IMUL(tile_height,tile_width);
  const int fil_sz = IMUL(fil_w,fil_h);

  for(int i=0;i<fil_sz;i+=tile_sz)
  {
    const int curld_idx = ld_idx+IMUL(i,tile_sz);
    if(curld_idx < fil_sz)
      data[curld_idx] = f[curld_idx];
  }
  __syncthreads();

  const int Nx2 = (fil_w - 1) / 2;
  const int Ny2 = (fil_h - 1) / 2;

  //const int srow_skip = src_w-fil_w;

  //for (int dst_y = 0; dst_y < src_h; ++dst_y)
  // Determine if we're safely inside the image in the y-direction:
  const bool y_clean = y_pos >= Ny2 && y_pos <  (src_h - Nx2);

  //for (int dst_x = 0; dst_x < src_w; ++dst_x, ++dptr)
  // Determine if we're safely inside the image in the x-direction:
  const bool x_clean = x_pos >= Nx2 && x_pos <  (src_w - Nx2);

  // Here is where we pick whether we can use the optimized inner
  // loop (in cases where the filter and image patch overlap
  // completely) or whether we must use the inner loop that can
  // handle boundary conditions.

  if (x_clean && y_clean)
    {
      float dst_val = 0.0f;
      for(int i=0;i<fil_h;i++)
        {
          const int s_y=y_pos-Ny2+i;
          const int f_y=fil_h-i-1;
        for(int j=0;j<fil_w;j++)
          {
            const int s_x=x_pos-Nx2+j;
            const int f_x=fil_w-j-1;
            const int flt_idx = IMUL(f_y,fil_w)+f_x;
            const int src_idx = IMUL(s_y,src_w)+s_x;
            dst_val += src[src_idx]*data[flt_idx];
          }
        }
      res[res_idx] = dst_val;
    }
  else if(x_pos < src_w && y_pos < src_h)
    {
      // OK, we're at an image boundary, so what do we do to make
      // up for the missing pixels? The approach here is to
      // compute the average value of the non-missing pixels, and
      // proceed as if the missing pixels had that value. This
      // minimizes the introduction of edge artifacts e.g. when
      // convolving with an oriented filter.
      float dst_val = 0.0f;
      float src_sum = 0.0f;
      int src_cnt = 0;
      float fil_sum_skipped = 0.0f;

      for(int i=0;i<fil_h;i++)
        {
          const int s_y=y_pos-Ny2+i;
          const int f_y=fil_h-i-1;
          if(s_y >= 0 && s_y < src_h)
          {
            for(int j=0;j<fil_w;j++)
            {
              const int s_x=x_pos-Nx2+j;
              const int f_x=fil_w-j-1;
              const int flt_idx = IMUL(f_y,fil_w)+f_x;
              const int src_idx = IMUL(s_y,src_w)+s_x;
              if(s_x >=0 && s_x < src_w)
              {
                float src_val = src[src_idx];
                dst_val += src_val*data[flt_idx];
                src_sum += src_val;
                src_cnt++;
              }
              else
              {
                fil_sum_skipped += data[flt_idx];
              }
            }
          }
          else
          {
              for (int f_x = fil_w-1; f_x >= 0; --f_x)
                fil_sum_skipped += data[IMUL(f_y,fil_w)+f_x];
          }
        }
      const float src_avg = src_sum / src_cnt;
      res[res_idx] = dst_val + (fil_sum_skipped*src_avg);


    }

}



__global__ void cuda_global_xFilterZero(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int shr_len, const int tile_len)
{
  const int hfs2 = (hfs-1) / 2;
  const int y_pos = blockIdx.y;
  const int x_pos = IMUL(blockIdx.x, tile_len) + threadIdx.x;
  const int ld_idx = threadIdx.x;
  const int res_idx = y_pos*src_w+x_pos;

  // Load the filter into shared memory
  float *data = (float *) shared_data; // size of share_len
  float sum = 0;
  const int numIters = IDIVUP(hfs,shr_len);
  // Handle the fact that the filter might be very large, and unable to fit in our shared memory size
  for(int iter=0;iter<numIters;iter++)
    {
      const int filt_beg = IMUL(iter,shr_len);
      const int filt_end = filt_beg+shr_len;
      const int load_max = min(filt_end,hfs)-filt_beg;
      for(int data_idx=ld_idx;data_idx<load_max;data_idx+=tile_len)
        {
          const int filt_idx = data_idx + filt_beg;
          data[data_idx] = f[filt_idx];
        }
      __syncthreads();

      if(y_pos < src_h && x_pos < src_w)
        {

          for(int j=0;j<load_max;j++)
            {
              const int s_x=x_pos+hfs2-(filt_beg+j);
              const int f_x=j;
              if(s_x >=0 && s_x < src_w)
                {
                  const int src_idx = IMUL(y_pos,src_w)+s_x;
                  sum += src[src_idx]*data[f_x];
                }
            }

        }
      // Avoid doing this synch unless we have a very large filter
      if(numIters > 1)
        __syncthreads();

    }
  if(y_pos < src_h && x_pos < src_w)
    {
      res[res_idx] = sum;
    }

}

__global__ void cuda_global_xFilterClean(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int shr_len, const int tile_len)
{
  const int hfs2 = (hfs - 1) / 2;
  const int y_pos = blockIdx.y;
  const int x_pos = IMUL(blockIdx.x, tile_len) + threadIdx.x;
  const int ld_idx = threadIdx.x;
  const int res_idx = y_pos*src_w+x_pos;

  // Load the filter into shared memory
  float *data = (float *) shared_data; // size of tile_width
  float sum = 0, sumw = 0, sumf = 0;
  const int numIters = IDIVUP(hfs,shr_len);
  // Handle the fact that the filter might be very large, and unable to fit in our shared memory size
  for(int iter=0;iter<numIters;iter++)
    {
      const int filt_beg = IMUL(iter,shr_len);
      const int filt_end = filt_beg+shr_len;
      const int load_max = min(filt_end,hfs)-filt_beg;
      for(int data_idx=ld_idx;data_idx<load_max;data_idx+=tile_len)
        {
          const int filt_idx = data_idx + filt_beg;
          data[data_idx] = f[filt_idx];
        }
      __syncthreads();

      if(y_pos < src_h && x_pos < src_w)
        {

          for(int j=0;j<load_max;j++)
            {
              const int s_x=x_pos+hfs2-(filt_beg+j);
              const int f_x=j;
              if(s_x >=0 && s_x < src_w)
                {
                  const int src_idx = IMUL(y_pos,src_w)+s_x;
                  sum += src[src_idx]*data[f_x];
                  sumw += data[f_x];
                }
              else
                {
                  sumf += data[f_x];
                }
            }
        }
      // Avoid doing this synch unless we have a very large filter
      if(numIters > 1)
        __syncthreads();

    }
  if(y_pos < src_h && x_pos < src_w)
    {
      sumf+=sumw;
      res[res_idx] = sum*sumf/sumw;
    }

}

__global__ void cuda_global_xFilterReplicate(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int shr_len, const int tile_len)
{
  const int hfs2 = (hfs - 1) / 2;
  const int y_pos = blockIdx.y;
  const int x_pos = IMUL(blockIdx.x, tile_len) + threadIdx.x;
  const int ld_idx = threadIdx.x;
  const int res_idx = y_pos*src_w+x_pos;

  // Load the filter into shared memory
  float *data = (float *) shared_data; // size of tile_width
  float sum = 0;
  const int numIters = IDIVUP(hfs,shr_len);
  // Handle the fact that the filter might be very large, and unable to fit in our shared memory size
  for(int iter=0;iter<numIters;iter++)
    {
      const int filt_beg = IMUL(iter,shr_len);
      const int filt_end = filt_beg+shr_len;
      const int load_max = min(filt_end,hfs)-filt_beg;
      for(int data_idx=ld_idx;data_idx<load_max;data_idx+=tile_len)
        {
          const int filt_idx = data_idx + filt_beg;
          data[data_idx] = f[filt_idx];
        }
      __syncthreads();

      if(y_pos < src_h && x_pos < src_w)
        {

          for(int j=0;j<load_max;j++)
            {
              const int s_x=x_pos+hfs2-(filt_beg+j);
              const int f_x=j;
              if(s_x < 0)
                {
                  sum += src[IMUL(y_pos,src_w)]*data[f_x];
                }
              else if(s_x >= src_w)
                {
                  sum += src[IMUL(y_pos+1,src_w)-1]*data[f_x];
                }
              else
                {
                  const int src_idx = IMUL(y_pos,src_w)+s_x;
                  sum += src[src_idx]*data[f_x];
                }
            }
        }
      // Avoid doing this synch unless we have a very large filter
      if(numIters > 1)
        __syncthreads();

    }
  if(y_pos < src_h && x_pos < src_w)
    {
      res[res_idx] = sum;
    }


}


__global__ void cuda_global_yFilterZero(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int shr_len, const int tile_len)
{
  const int hfs2 = (hfs - 1) / 2;
  const int y_pos = IMUL(blockIdx.y, tile_len) + threadIdx.y;
  const int x_pos = blockIdx.x;
  const int ld_idx = threadIdx.y;
  const int res_idx = y_pos*src_w+x_pos;

  // Load the filter into shared memory
  float *data = (float *) shared_data; // size of tile_width
  float sum = 0;
  const int numIters = IDIVUP(hfs,shr_len);
  // Handle the fact that the filter might be very large, and unable to fit in our memory size
  for(int iter=0;iter<numIters;iter++)
    {
      const int filt_beg = IMUL((iter),shr_len);
      const int filt_end = filt_beg+shr_len;
      const int load_max = min(filt_end,hfs)-filt_beg;
      for(int data_idx=ld_idx;data_idx<load_max;data_idx+=tile_len)
        {
          const int filt_idx = data_idx + filt_beg;
          data[data_idx] = f[filt_idx];
        }
      __syncthreads();

      if(y_pos < src_h && x_pos < src_w)
        {

          for(int j=0;j<load_max;j++)
            {
              const int s_y=y_pos+hfs2-(filt_beg+j);
              const int f_y=j;
              if(s_y >=0 && s_y < src_h)
                {
                  const int src_idx = IMUL(s_y,src_w)+x_pos;
                  sum += src[src_idx]*data[f_y];
                }
            }
        }
      // Avoid doing this synch unless we have a very large filter
      if(numIters > 1)
        __syncthreads();

    }
  if(y_pos < src_h && x_pos < src_w)
    {
      res[res_idx] = sum;
    }

}

__global__ void cuda_global_yFilterClean(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int shr_len, const int tile_len)
{
  const int hfs2 = (hfs - 1) / 2;
  const int y_pos = IMUL(blockIdx.y, tile_len) + threadIdx.y;
  const int x_pos = blockIdx.x;
  const int ld_idx = threadIdx.y;
  const int res_idx = y_pos*src_w+x_pos;

  // Load the filter into shared memory
  float *data = (float *) shared_data; // size of tile_width
  float sum = 0, sumw = 0, sumf = 0;
  const int numIters = IDIVUP(hfs,shr_len);
  // Handle the fact that the filter might be very large, and unable to fit in our memory size
  for(int iter=0;iter<numIters;iter++)
    {
      const int filt_beg = IMUL((iter),shr_len);
      const int filt_end = filt_beg+shr_len;
      const int load_max = min(filt_end,hfs)-filt_beg;
      for(int data_idx=ld_idx;data_idx<load_max;data_idx+=tile_len)
        {
          const int filt_idx = data_idx + filt_beg;
          data[data_idx] = f[filt_idx];
        }
      __syncthreads();

      if(y_pos < src_h && x_pos < src_w)
        {

          for(int j=0;j<load_max;j++)
            {
              const int s_y=y_pos+hfs2-(filt_beg+j);
              const int f_y=j;
              if(s_y >=0 && s_y < src_h)
                {
                  const int src_idx = IMUL(s_y,src_w)+x_pos;
                  sum += src[src_idx]*data[f_y];
                  sumw += data[f_y];
                }
              else
                {
                  sumf += data[f_y];
                }
            }
        }
      // Avoid doing this synch unless we have a very large filter
      if(numIters > 1)
        __syncthreads();
    }
  if(y_pos < src_h && x_pos < src_w)
    {
      sumf+=sumw;
      res[res_idx] = sum*sumf/sumw;
    }

}

__global__ void cuda_global_yFilterReplicate(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int shr_len, const int tile_len)
{
  const int hfs2 = (hfs - 1) / 2;
  const int y_pos = IMUL(blockIdx.y, tile_len) + threadIdx.y;
  const int x_pos = blockIdx.x;
  const int ld_idx = threadIdx.y;
  const int res_idx = y_pos*src_w+x_pos;


  // Load the filter into shared memory
  float *data = (float *) shared_data; // size of tile_width
  float sum = 0;
  const int numIters = IDIVUP(hfs,shr_len);
  // Handle the fact that the filter might be very large, and unable to fit in our memory size
  for(int iter=0;iter<numIters;iter++)
    {
      const int filt_beg = IMUL((iter),shr_len);
      const int filt_end = filt_beg+shr_len;
      const int load_max = min(filt_end,hfs)-filt_beg;
      for(int data_idx=ld_idx;data_idx<load_max;data_idx+=tile_len)
        {
          const int filt_idx = data_idx + filt_beg;
          data[data_idx] = f[filt_idx];
        }
      __syncthreads();

      if(y_pos < src_h && x_pos < src_w)
        {
          for(int j=0;j<load_max;j++)
            {
              const int s_y=y_pos+hfs2-(filt_beg+j);
              const int f_y=j;
              if(s_y < 0)
                {
                  sum += src[x_pos]*data[f_y];
                }
              else if(s_y >= src_h)
                {
                  sum += src[IMUL(src_h-1,src_w)+x_pos]*data[f_y];
                }
              else
                {
                  const int src_idx = IMUL(s_y,src_w)+x_pos;
                  sum += src[src_idx]*data[f_y];
                }
            }
        }
      // Avoid doing this synch unless we have a very large filter
      if(numIters > 1)
        __syncthreads();
    }
  if(y_pos < src_h && x_pos < src_w)
    {
      res[res_idx] = sum;
    }


}


__global__ void cuda_global_optXFilterZero(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len)
{
  const int hfs2 = (hfs-1) / 2;
  const int y_pos = blockIdx.y;
  const int x_pos = IMUL(blockIdx.x, tile_len) + threadIdx.x;
  const int data_idx = threadIdx.x;
  const int res_idx = y_pos*src_w+x_pos;
  const int ctr_off = hfs2;

  // Load the filter into shared memory
  float *filt = (float *) shared_data; // size of hfs
  float *data = (float *) &(shared_data[hfs]); // size of tile_len+hfs-1

  float sum = 0;

  for(int filt_idx=data_idx;filt_idx<hfs;filt_idx+=tile_len)
    {
      filt[filt_idx] = f[filt_idx];
    }
  data[data_idx+ctr_off] = src[res_idx];
  // Load early border
  if(data_idx < ctr_off && x_pos > ctr_off)
    {
      data[data_idx] = src[res_idx-ctr_off];
    }
  // Load late border
  if(data_idx > tile_len-ctr_off-1 && x_pos < src_w-ctr_off)
    {
      data[data_idx+ctr_off+ctr_off] = src[res_idx+ctr_off];
    }
  __syncthreads();


  if(y_pos < src_h && x_pos < src_w)
    {

      for(int j=0;j<hfs;j++)
        {
          const int s_x=data_idx+ctr_off+hfs2-j;
          const int x_loc=x_pos+hfs2-j;
          const int f_x=j;
          if(x_loc >=0 && x_loc < src_w)
            {
              sum += data[s_x]*filt[f_x];
            }
        }

    }
  if(y_pos < src_h && x_pos < src_w)
    {
      res[res_idx] = sum;
    }

}


__global__ void cuda_global_optXFilterClean(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len)
{
  const int hfs2 = (hfs - 1) / 2;
  const int y_pos = blockIdx.y;
  const int x_pos = IMUL(blockIdx.x, tile_len) + threadIdx.x;
  const int data_idx = threadIdx.x;
  const int res_idx = y_pos*src_w+x_pos;
  const int ctr_off = hfs2;

  // Load the filter into shared memory
  float *filt = (float *) shared_data; // size of hfs
  float *data = (float *) &(shared_data[hfs]); // size of tile_len+hfs-1

  float sum = 0, sumw = 0, sumf = 0;

  for(int filt_idx=data_idx;filt_idx<hfs;filt_idx+=tile_len)
    {
      filt[filt_idx] = f[filt_idx];
    }
  if(y_pos < src_h && x_pos < src_w)
    {

      data[data_idx+ctr_off] = src[res_idx];
      // Load early border
      if(data_idx < ctr_off && x_pos > ctr_off)
        {
          data[data_idx] = src[res_idx-ctr_off];
        }
      // Load late border
      if(data_idx > tile_len-ctr_off-1 && x_pos < src_w-ctr_off)
        {
          data[data_idx+ctr_off+ctr_off] = src[res_idx+ctr_off];
        }
      __syncthreads();


      for(int j=0;j<hfs;j++)
        {
          const int s_x=data_idx+ctr_off+hfs2-j;
          const int x_loc=x_pos+hfs2-j;
          const int f_x=j;
          if(x_loc >=0 && x_loc < src_w)
            {
              sum += data[s_x]*filt[f_x];
              sumw += filt[f_x];
            }
          else
            {
              sumf += filt[f_x];
            }
        }
      sumf+=sumw;
      res[res_idx] = sum*sumf/sumw;
    }

}


__global__ void cuda_global_optXFilterReplicate(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len)
{
  const int hfs2 = (hfs - 1) / 2;
  const int y_pos = blockIdx.y;
  const int x_pos = IMUL(blockIdx.x, tile_len) + threadIdx.x;
  const int data_idx = threadIdx.x;
  const int res_idx = y_pos*src_w+x_pos;
  const int ctr_off = hfs2;

  // Load the filter into shared memory
  float *filt = (float *) shared_data; // size of hfs
  float *data = (float *) &(shared_data[hfs]); // size of tile_len+hfs-1

  float sum = 0;

  for(int filt_idx=data_idx;filt_idx<hfs;filt_idx+=tile_len)
    {
      filt[filt_idx] = f[filt_idx];
    }
  if(y_pos < src_h && x_pos < src_w)
    {

      data[data_idx+ctr_off] = src[res_idx];
      // Load early border
      if(data_idx < ctr_off && x_pos > ctr_off)
        {
          data[data_idx] = src[res_idx-ctr_off];
        }
      // Load late border
      if(data_idx > tile_len-ctr_off-1 && x_pos < src_w-ctr_off)
        {
          data[data_idx+ctr_off+ctr_off] = src[res_idx+ctr_off];
        }
      __syncthreads();

      for(int j=0;j<hfs;j++)
        {
          const int x_loc=x_pos+hfs2-j;
          const int f_x=j;
          if(x_loc < 0)
            {
              // Hold to the leftmost pixel
              sum += data[ctr_off]*filt[f_x];
            }
          else if(x_loc >= src_w)
            {
              // Hold to the rightmost pixel
              const int max_x = src_w-IMUL(blockIdx.x, tile_len)+ctr_off-1;
              sum += data[max_x]*filt[f_x];
            }
          else
            {
              const int s_x=data_idx+ctr_off+hfs2-j;
              sum += data[s_x]*filt[f_x];
            }
        }

      res[res_idx] = sum;
    }

}


__global__ void cuda_global_optYFilterZero(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len)
{
  const int hfs2 = (hfs-1) / 2;
  const int y_pos = IMUL(blockIdx.y, tile_len) + threadIdx.y;
  const int x_pos = blockIdx.x;
  const int data_idx = threadIdx.y;
  const int res_idx = y_pos*src_w+x_pos;
  const int ctr_off = hfs2;

  // Load the filter into shared memory
  float *filt = (float *) shared_data; // size of hfs
  float *data = (float *) &(shared_data[hfs]); // size of tile_len+hfs-1

  float sum = 0;

  for(int filt_idx=data_idx;filt_idx<hfs;filt_idx+=tile_len)
    {
      filt[filt_idx] = f[filt_idx];
    }
  if(y_pos < src_h && x_pos < src_w)
    {
      data[data_idx+ctr_off] = src[res_idx];
      // Load early border
      if(data_idx < ctr_off && y_pos > ctr_off)
        {
          data[data_idx] = src[res_idx-IMUL(ctr_off,src_w)];
        }
      // Load late border
      if(data_idx > tile_len-ctr_off-1 && y_pos < src_h-ctr_off)
        {
          data[data_idx+ctr_off+ctr_off] = src[res_idx+IMUL(ctr_off,src_w)];
        }
      __syncthreads();


      for(int j=0;j<hfs;j++)
        {
          const int s_y=data_idx+ctr_off+hfs2-j;
          const int y_loc=y_pos+hfs2-j;
          const int f_y=j;
          if(y_loc >=0 && y_loc < src_h)
            {
              sum += data[s_y]*filt[f_y];
            }
        }

      res[res_idx] = sum;
    }

}


__global__ void cuda_global_optYFilterClean(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len)
{
  const int hfs2 = (hfs - 1) / 2;
  const int y_pos = IMUL(blockIdx.y,tile_len) + threadIdx.y;
  const int x_pos = blockIdx.x;
  const int data_idx = threadIdx.y;
  const int res_idx = y_pos*src_w+x_pos;
  const int ctr_off = hfs2;

  // Load the filter into shared memory
  float *filt = (float *) shared_data; // size of hfs
  float *data = (float *) &(shared_data[hfs]); // size of tile_len+hfs-1

  float sum = 0, sumw = 0, sumf = 0;

  // Load the filter
  for(int filt_idx=data_idx;filt_idx<hfs;filt_idx+=tile_len)
    {
      filt[filt_idx] = f[filt_idx];
    }
  if(y_pos < src_h && x_pos < src_w)
    {
      // Load the bulk of the tile data
      data[data_idx+ctr_off] = src[res_idx];
      // Load early border
      if(data_idx < ctr_off && y_pos > ctr_off)
        {
          data[data_idx] = src[res_idx-IMUL(ctr_off,src_w)];
        }
      // Load late border
      if(data_idx > tile_len-ctr_off-1 && y_pos < src_h-ctr_off)
        {
          data[data_idx+ctr_off+ctr_off] = src[res_idx+IMUL(ctr_off,src_w)];
        }
      __syncthreads();

      for(int j=0;j<hfs;j++)
        {
          const int s_y=data_idx+ctr_off+hfs2-j;
          const int y_loc=y_pos+hfs2-j;
          const int f_y=j;
          if(y_loc >=0 && y_loc < src_h)
            {
              sum += data[s_y]*filt[f_y];
              sumw += filt[f_y];
            }
          else
            {
              sumf += filt[f_y];
            }
        }
    }
  if(y_pos < src_h && x_pos < src_w)
    {
      sumf+=sumw;
      res[res_idx] = sum*sumf/sumw;
    }

}

__global__ void cuda_global_optYFilterReplicate(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len)
{
  const int hfs2 = (hfs - 1) / 2;
  const int y_pos = IMUL(blockIdx.y, tile_len) + threadIdx.y;
  const int x_pos = blockIdx.x;
  const int data_idx = threadIdx.y;
  const int res_idx = y_pos*src_w+x_pos;
  const int ctr_off = hfs2;

  // Load the filter into shared memory
  float *filt = (float *) shared_data; // size of hfs
  float *data = (float *) &(shared_data[hfs]); // size of tile_len+hfs-1

  float sum = 0;

  for(int filt_idx=data_idx;filt_idx<hfs;filt_idx+=tile_len)
    {
      filt[filt_idx] = f[filt_idx];
    }
  if(y_pos < src_h && x_pos < src_w)
    {
      data[data_idx+ctr_off] = src[res_idx];
      // Load early border
      if(data_idx < ctr_off && y_pos > ctr_off)
        {
          data[data_idx] = src[res_idx-IMUL(ctr_off,src_w)];
        }
      // Load late border
      if(data_idx > tile_len-ctr_off-1 && y_pos < src_h-ctr_off)
        {
          data[data_idx+ctr_off+ctr_off] = src[res_idx+IMUL(ctr_off,src_w)];
        }
      __syncthreads();

      for(int j=0;j<hfs;j++)
        {
          const int y_loc=y_pos+hfs2-j;
          const int f_y=j;
          if(y_loc < 0)
            {
              // Hold to the topmost pixel
              sum += data[ctr_off]*filt[f_y];
            }
          else if(y_loc >= src_h)
            {
              // Hold to the bottommost pixel
              const int max_y = src_h-IMUL(blockIdx.y, tile_len)+ctr_off-1;
              sum += data[max_y]*filt[f_y];
            }
          else
            {
              const int s_y=data_idx+ctr_off+hfs2-j;
              sum += data[s_y]*filt[f_y];
            }
        }
    }
  if(y_pos < src_h && x_pos < src_w)
    {
      res[res_idx] = sum;
    }

}


#endif
