/*!@file CUDA/CudaImageDisplay.h Interface for CUDA-OpenGL Display */

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
#include "cudadefs.h"

#ifndef IMAGE_DISPLAY_H_DEFINED
#define IMAGE_DISPLAY_H_DEFINED

 #define BLOCKDIM_X 8
 #define BLOCKDIM_Y 8

// CUDA wrapper functions for allocation/freeing texture arrays
extern "C" cudaError_t CUDA_Bind2TextureArray(int index);
extern "C" cudaError_t CUDA_UnbindTexture(int index);
extern "C" cudaError_t CUDA_MallocArray(unsigned int *src, int imageW, int imageH,int index);
extern "C" cudaError_t CUDA_UpdateArray(unsigned int *src, int imageW, int imageH,int index);
extern "C" cudaError_t CUDA_FreeArray();

// CUDA kernel functions
extern "C" void cuda_Copy( unsigned int *d_dst, int imageW, int imageH);
extern "C" void CUDA_convert_float_uint(float3_t* src,unsigned int *dst,int tile_length,int size);
#endif
