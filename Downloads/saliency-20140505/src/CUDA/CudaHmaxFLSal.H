/*!@file CUDA/CudaHmaxFLSal.H Modified from T. Serre's HMAX model with C1 layer input from Saliency */

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
//
//

#ifndef CUDAHMAXFLSAL_H_DEFINED
#define CUDAHMAXFLSAL_H_DEFINED

#include "CUDA/CudaSaliency.H"
#include "CUDA/CudaImage.H"
#include "CUDA/CudaHmax.H"
#include "CUDA/CudaHmaxFL.H"
#include <vector>

#include <cmath>
#include <string>
#include <vector>
#include <dirent.h>
#include "nub/ref.h"

class CudaHmaxFLSal : public CudaHmaxFL {
public:
   //! Default constructor. Need to call init() later.
  CudaHmaxFLSal();

  //!Constructor
  CudaHmaxFLSal(MemoryPolicy mp, int dev, nub::ref<CudaSaliency> sal_in);

   //! Destructor
  ~CudaHmaxFLSal();

   //! Randomly extract a set of C1Patches from positive training images
  void extractRandC1Patches(Image<float> *& posTrainingImages, int numPosTrainImages, std::vector<int> patchSizes, int nPatchesPerSize, int no);

  // Get the C1 level from already computed saliency information(must be public if you need to learn patches)
  void getC1(CudaImage<float>**& c1Res);

  // Calculate the C1 level from salieny for the given image(must be public if you need to learn patches)
  void getC1(const CudaImage<float>& input, CudaImage<float>**& c1Res);


protected:
  nub::ref<CudaSaliency> sal;

};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
