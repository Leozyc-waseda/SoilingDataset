/*!@file CUDA/CudaKernels.H C++ wrapper for CUDA Kernel generation */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaKernels.H $
// $Id: CudaKernels.H 12962 2010-03-06 02:13:53Z irock $
//



#ifndef CUDAKERNELS_H_DEFINED
#define CUDAKERNELS_H_DEFINED

#include "CUDA/CudaImage.H"
#include "CUDA/cudadefs.h"

// Generate a difference of Gaussian filter as parameterized by HMAX
CudaImage<float> cudaDogFilterHmax(MemoryPolicy mp, int dev, const float theta, const float gamma, const int size, const float div);

// Generate a difference of Gaussian filter
CudaImage<float> cudaDogFilter(MemoryPolicy mp, int dev, const float stddev, const float theta, const int halfsize_in);

// On CUDA device, produces a Gabor kernel with optionally unequal major+minor axis lengths.
CudaImage<float> cudaGaborFilter3(MemoryPolicy mp, int dev, const float major_stddev, const float minor_stddev, const float period, const float phase, const float theta, int size);

// On CUDA device, generate a 1D Gaussian
CudaImage<float> cudaGaussian(MemoryPolicy mp, int dev, const float coeff, const float sigma,
                              const int maxhw, const float threshperc=1.0F);

#endif
