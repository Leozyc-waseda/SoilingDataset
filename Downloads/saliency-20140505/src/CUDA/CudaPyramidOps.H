/*!@file CUDA/CudaPyramidOps.H Free functions operating on pyramid data structures */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaPyramidOps.H $
// $Id: CudaPyramidOps.H 12962 2010-03-06 02:13:53Z irock $
//

#ifndef CUDA_CUDAPYRAMIDOPS_H_DEFINED
#define CUDA_CUDAPYRAMIDOPS_H_DEFINED

#include "CUDA/CudaImageSet.H"
#include "CUDA/CudaImage.H"
#include "Image/PyramidTypes.H"

//! Uses a Gaussian filter of size \a filterSize.
CudaImageSet<float> cudaBuildPyrGaussian(const CudaImage<float>& image,
                                    int firstlevel, int depth,
                                    int filterSize);

//! Uses a Laplacian filter of size \a filterSize.
CudaImageSet<float> cudaBuildPyrLaplacian(const CudaImage<float>& image,
                                    int firstlevel, int depth,
                                    int filterSize);

//! Build an oriented pyramid from a Laplacian pyramid
CudaImageSet<float> cudaBuildPyrOrientedFromLaplacian(const CudaImageSet<float>& laplacian,
                                                      int filterSize, float theta, float intens = 10.0F);

//! Build an oriented pyramid
CudaImageSet<float> cudaBuildPyrOriented(const CudaImage<float>& image, int firstlevel, int depth, int filterSize,
                                         float theta, float intens = 10.0F);

//! Build a local average pyramid
CudaImageSet<float> cudaBuildPyrLocalAvg(const CudaImage<float>& image, int depth);

//! Build a local average 2x2 pyramid
CudaImageSet<float> cudaBuildPyrLocalAvg2x2(const CudaImage<float>& image, int depth);

//! Build a local max pyramid
CudaImageSet<float> cudaBuildPyrLocalMax(const CudaImage<float>& image, int depth);

//! Build a generic pyramid
CudaImageSet<float> cudaBuildPyrGeneric(const CudaImage<float>& image,
                                        int firstlevel, int depth,
                                        const PyramidType typ, const float gabor_theta=0.0F,
                                        const float intens=10.0F);


//! Compute center-surround between images at lev1 and lev2
CudaImage<float> cudaCenterSurround(const CudaImageSet<float>& pyr,
                                    const int lev1, const int lev2,
                                    const bool absol,
                                    const CudaImageSet<float>* clipPyr = 0);

//! Compute center-surround between images, splitting positive and negative
void cudaCenterSurround(const CudaImageSet<float>& pyr,
                        const int lev1, const int lev2,
                        CudaImage<float>& pos, CudaImage<float>& neg,
                        const CudaImageSet<float>* clipPyr = 0);

//! Compute single-opponent center-surround between images at lev1 and lev2
CudaImage<float> cudaCenterSurroundSingleOpponent(const CudaImageSet<float>& cpyr,
                                                  const CudaImageSet<float>& spyr,
                                                  const int lev1, const int lev2,
                                                  const bool absol,
                                                  const CudaImageSet<float>* clipPyr = 0);

//! Compute single-opp center-surr between imgs, splitting pos and neg
void cudaCenterSurroundSingleOpponent(const CudaImageSet<float>& cpyr,
                                      const CudaImageSet<float>& spyr,
                                      const int lev1, const int lev2,
                                      CudaImage<float>& pos, CudaImage<float>& neg,
                                      const CudaImageSet<float>* clipPyr = 0);

//! Compute center-surround between difference images at lev1 and lev2
CudaImage<float> cudaCenterSurroundDiff(const CudaImageSet<float>& pyr1,
                                        const CudaImageSet<float>& pyr2,
                                        const int lev1, const int lev2,
                                        const bool absol,
                                        const CudaImageSet<float>* clipPyr = 0);

//! Compute center-surround differnces, splitting pos & neg values
void cudaCenterSurroundDiff(const CudaImageSet<float>& pyr1,
                            const CudaImageSet<float>& pyr2,
                            const int lev1, const int lev2,
                            CudaImage<float>& pos, CudaImage<float>& neg,
                            const CudaImageSet<float>* clipPyr = 0);

#endif
