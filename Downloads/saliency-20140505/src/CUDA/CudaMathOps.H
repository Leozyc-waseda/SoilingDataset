/*!@file CUDA/CudaMathOps.H C++ wrapper for CUDA Math operations */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaMathOps.H $
// $Id: CudaMathOps.H 12962 2010-03-06 02:13:53Z irock $
//



#ifndef CUDAMATHOPS_H_DEFINED
#define CUDAMATHOPS_H_DEFINED

#include "CUDA/CudaImage.H"
#include "CUDA/cudadefs.h"
#include "Image/Point2D.H"

void cudaGetMin(const CudaImage<float>& src, CudaImage<float>& minim, CudaImage<float> *buf=0);
void cudaGetMax(const CudaImage<float>& src, CudaImage<float>& maxim, CudaImage<float> *buf=0);
void cudaGetAvg(const CudaImage<float>& src, CudaImage<float>& avgim, CudaImage<float> *buf=0);
void cudaGetMinMax(const CudaImage<float>& src, CudaImage<float>& minim, CudaImage<float>& maxim, CudaImage<float> *buf=0);
void cudaGetMinMaxAvg(const CudaImage<float>& src, CudaImage<float>& minim, CudaImage<float>& maxim, CudaImage<float>& avgim, CudaImage<float> *buf=0);
template <class T> T cudaGetScalar(const CudaImage<T>& src);
void cudaInplaceNormalize(CudaImage<float>& dst, const float nmin, const float nmax);
void cudaInplaceClamp(CudaImage<float>& dst, const float cmin, const float cmax);
void cudaInplaceRectify(CudaImage<float>& dst);
void cudaClear(CudaImage<float>& dst, const float val);
void cudaAbs(CudaImage<float>& src);
void cudaInplaceAddScalar(CudaImage<float>& dst, const CudaImage<float>& offset);
void cudaInplaceSubtractScalar(CudaImage<float>& dst, const CudaImage<float>& offset);
void cudaInplaceMultiplyScalar(CudaImage<float>& dst, const CudaImage<float>& offset);
void cudaInplaceDivideScalar(CudaImage<float>& dst, const CudaImage<float>& offset);
void cudaInplaceAddImages(CudaImage<float>& im1, const CudaImage<float>& im2);
void cudaInplaceSubtractImages(CudaImage<float>& im1, const CudaImage<float>& im2);
void cudaInplaceMultiplyImages(CudaImage<float>& im1, const CudaImage<float>& im2);
void cudaInplaceDivideImages(CudaImage<float>& im1, const CudaImage<float>& im2);
CudaImage<float> cudaGetAvg(const CudaImage<float>& src);
CudaImage<float> cudaGetSum(const CudaImage<float>& src);
CudaImage<float> cudaSquared(const CudaImage<float>& src);
CudaImage<float> cudaSqrt(const CudaImage<float>& src);
CudaImage<float> cudaAddImages(const CudaImage<float>& im1, const CudaImage<float>& im2);
CudaImage<float> cudaSubtractImages(const CudaImage<float>& im1, const CudaImage<float>& im2);
CudaImage<float> cudaMultiplyImages(const CudaImage<float>& im1, const CudaImage<float>& im2);
CudaImage<float> cudaDivideImages(const CudaImage<float>& im1, const CudaImage<float>& im2);
CudaImage<float> cudaTakeMax(const CudaImage<float>& im1, const CudaImage<float>& im2);
CudaImage<float> cudaQuadEnergy(const CudaImage<float>& real, const CudaImage<float>& imag);
void cudaInplaceAttenuateBorders(CudaImage<float>& a, int size);
void cudaFindMin(const CudaImage<float>& src, Point2D<int>& p, float& val);
void cudaFindMax(const CudaImage<float>& src, Point2D<int>& p, float& val);
#endif
