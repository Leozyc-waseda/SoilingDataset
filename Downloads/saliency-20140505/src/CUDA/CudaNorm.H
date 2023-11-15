/*!@file CUDA/CudaNorm.H C++ wrapper for CUDA Normalizations */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaNorm.H $
// $Id: CudaNorm.H 12962 2010-03-06 02:13:53Z irock $
//


#ifndef CUDANORM_H_DEFINED
#define CUDANORM_H_DEFINED

#include "CUDA/CudaImage.H"
#include "Image/fancynorm.H"
#include "CUDA/cudadefs.h"

CudaImage<float> cudaMaxNormalize(const CudaImage<float>& src,
                      const float mi, const float ma, const MaxNormType normtyp=VCXNORM_DEFAULT,
                      int nbiter=FANCYITER, const CudaImage<float> *lrexcit=0);
CudaImage<float> cudaMaxNormalizeFancy(const CudaImage<float>& src, const float mi, const float ma,
                           const int nbiter, const float weakness,
                           const CudaImage<float>* lrexcit);

#endif
