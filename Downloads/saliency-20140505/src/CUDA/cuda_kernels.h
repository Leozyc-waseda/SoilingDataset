/*!@file CUDA/cuda_kernels.h CUDA/GPU convolution kernel generation code  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/cuda_kernels.h $
// $Id: cuda_kernels.h 12962 2010-03-06 02:13:53Z irock $
//


#ifndef CUDA_KERNELS_H_DEFINED
#define CUDA_KERNELS_H_DEFINED

#include <cuda.h>
#include "CUDA/cutil.h"
#include "cudadefs.h"


__global__ void cuda_global_dogFilterHmax(float *dest, const float theta, const float gamma, const int size, const float div, const int tile_width, const int tile_height)
{

  // Note here sz is size along one dimension of this SQUARE filter (so total filter size is sz*sz)

  // change the angles in degree to the those in radian : rotation degree
  float thetaRads = M_PI / 180.0F * theta;

  // calculate constants
  float lambda = size*2.0F/div;
  float sigma = lambda*0.8F;
  float sigq = sigma*sigma;
  int center    = (int)ceil(size/2.0F);
  int filtSizeL = center-1;
  //int filtSizeR = size-filtSizeL-1;
  int ypos = IMUL(blockIdx.y,tile_height) + threadIdx.y;
  int xpos = IMUL(blockIdx.x,tile_width) + threadIdx.x;
  int dst_idx = IMUL(ypos,size) + xpos;

  int x = xpos -filtSizeL;
  int y = ypos -filtSizeL;
  // for DOG operation : to give orientation, it uses omit y-directional
  // component

  if(xpos < size && ypos < size)
    {
      if(sqrt((float)(IMUL(x,x)+IMUL(y,y))) > size/2.0F)
        {
          dest[dst_idx] = 0.0F;
        }
      else
        {
          float rtX =  y * cos(thetaRads) - x * sin(thetaRads);
          float rtY = y * sin(thetaRads) + x * cos(thetaRads);
          dest[dst_idx] = exp(-(rtX*rtX + gamma*gamma*rtY*rtY)/(2.0F*sigq)) *
            cos(2*M_PI*rtX/lambda);
        }
    }
}

__global__ void cuda_global_dogFilter(float *dest, float stddev, float theta, int half_size, int size, int tile_width, int tile_height)
{
  // change the angles in degree to the those in radian : rotation degree
  float thetaRads = M_PI / 180.0F * theta;

  // calculate constants
  float sigq = stddev * stddev;
  int ypos = IMUL(blockIdx.y,tile_height) + threadIdx.y;
  int xpos = IMUL(blockIdx.x,tile_width) + threadIdx.x;
  int dst_idx = IMUL(ypos,size) + xpos;

  int x = xpos - half_size;
  int y = ypos - half_size;
  // for DOG operation : to give orientation, it uses omit y-directional
  // component

  if(xpos < size && ypos < size)
    {
      float rtX =  x * cos(thetaRads) + y * sin(thetaRads);
      float rtY = -x * sin(thetaRads) + y * cos(thetaRads);
      dest[dst_idx] = (rtX*rtX/sigq - 1.0F)/sigq *
        exp(-(rtX*rtX + rtY*rtY)/(2.0F*sigq));
    }

}



__global__ void cuda_global_gaborFilter3(float *kern, const float major_stddev, const float minor_stddev,
                                         const float period, const float phase,
                                         const float theta, const int size, const int tile_len, const int sz)
{

  // change the angles in degree to the those in radians:
  const float psi = M_PI / 180.0F * phase;
  const float rtDeg = M_PI / 180.0F * theta;

  // calculate constants:
  const float omega = (2.0F * M_PI) / period;
  const float co = cos(rtDeg), si = sin(rtDeg);
  const float major_sigq = 2.0F * major_stddev * major_stddev;
  const float minor_sigq = 2.0F * minor_stddev * minor_stddev;

  const int src_idx = blockIdx.x*tile_len + threadIdx.x;

  // compute gabor:
  //for (int y = -size; y <= size; ++y)
  //  for (int x = -size; x <= size; ++x)

  const int y = floorf(src_idx / (size*2+1)) - size;
  const int x = src_idx % (size*2+1) - size;
  const float major = x*co + y*si;
  const float minor = x*si - y*co;
  if(src_idx < sz)
    kern[src_idx] = float(cos(omega * major + psi)
                          * exp(-(major*major) / major_sigq)
                          * exp(-(minor*minor) / minor_sigq));

}

__global__ void cuda_global_gaussian(float *res, float c, float sig22, int hw, int tile_len, int sz)
{
  const int idx = blockIdx.x*tile_len + threadIdx.x;


  if(idx<sz)
    {
      float x = float(idx-hw);
      res[idx] = c*exp(x*x*sig22);
    }
}


#endif
