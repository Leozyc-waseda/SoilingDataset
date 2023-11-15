/*!@file CUDA/CudaConvolutions.H C++ wrapper for CUDA Convolutions */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaConvolutions.H $
// $Id: CudaConvolutions.H 12962 2010-03-06 02:13:53Z irock $
//


#ifndef CUDACONVOLUTIONS_H_DEFINED
#define CUDACONVOLUTIONS_H_DEFINED

#include "CUDA/CudaImage.H"
#include "Image/Convolutions.H"
#include "CUDA/cudadefs.h"

CudaImage<float> cudaConvolveZeroHelper(const CudaImage<float>& src, const CudaImage<float>& filter,
                                        const int Nx, const int Ny, bool runOptimized=true);

CudaImage<float> cudaConvolveCleanHelper(const CudaImage<float>& src, const CudaImage<float>& filter,
                                         const int Nx, const int Ny, bool runOptimized=true);

CudaImage<float> cudaConvolveHmax(const CudaImage<float>& src, const CudaImage<float>& filter, bool runOptimized=true);

CudaImage<float> cudaConvolve(const CudaImage<float>& src, const CudaImage<float>& filter,
                  const int Nx, const int Ny,
                              ConvolutionBoundaryStrategy boundary, bool runOptimized=true);

inline CudaImage<float> cudaConvolve(const CudaImage<float>& src, const CudaImage<float>& filter,
                                     ConvolutionBoundaryStrategy boundary, bool runOptimized=true)
{
  return cudaConvolve(src, filter, filter.getWidth(), filter.getHeight(), boundary, runOptimized);
}

CudaImage<float> cudaOptConvolve(const CudaImage<float>& src, const CudaImage<float>& f);

CudaImage<float> cudaXFilter(const CudaImage<float>& src, const CudaImage<float>& hFilt, const int hfs,
                             ConvolutionBoundaryStrategy boundary, bool runOptimized=true);

CudaImage<float> cudaYFilter(const CudaImage<float>& src, const CudaImage<float>& hFilt, const int hfs,
                             ConvolutionBoundaryStrategy boundary, bool runOptimized=true);

CudaImage<float> cudaSepFilter(const CudaImage<float>& src, const CudaImage<float>& hFilter,
                   const CudaImage<float>& vFilter,
                               ConvolutionBoundaryStrategy boundary, bool runOptimized=true);

CudaImage<float> cudaSepFilter(const CudaImage<float>& src, const CudaImage<float>& hFilt, const CudaImage<float>& vFilt,
                               const int hfs, const int vfs,
                               ConvolutionBoundaryStrategy boundary, bool runOptimized=true);

#endif
