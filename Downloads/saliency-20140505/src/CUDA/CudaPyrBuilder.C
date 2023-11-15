/*!@file CUDA/CudaPyrBuilder.C Classes for building CUDA based dyadic pyramids */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaPyrBuilder.C $
// $Id: CudaPyrBuilder.C 12962 2010-03-06 02:13:53Z irock $
//

#include "CUDA/CudaPyrBuilder.H"

#include "CUDA/CudaImage.H"
#include "CUDA/CudaImageSet.H"
#include "CUDA/CudaCutPaste.H"
#include "Image/Pixels.H"
#include "CUDA/CudaPyramidCache.H"
#include "CUDA/CudaPyramidOps.H"
#include "rutz/trace.h"


// ######################################################################
// ##### PyrBuilder functions:
// ######################################################################

// ######################################################################
template <class T>
CudaPyrBuilder<T>::CudaPyrBuilder()
{

}

// ######################################################################
template <class T>
CudaPyrBuilder<T>::~CudaPyrBuilder()
{

}

// ######################################################################
template <class T>
void CudaPyrBuilder<T>::reset()
{

}


// ######################################################################
// ##### ReichardtPyrBuilder Functions:
// ######################################################################
template <class T> CudaReichardtPyrBuilder<T>::CudaReichardtPyrBuilder() :
  CudaPyrBuilder<T>()
{
}

template <class T> CudaReichardtPyrBuilder<T>::CudaReichardtPyrBuilder(const float dx,
                                            const float dy,
                                            const PyramidType typ,
                                            const float gabor_theta,
                                            const float intens) :
  CudaPyrBuilder<T>(), itsDX(dx), itsDY(dy), itsPtype(typ),
  itsGaborAngle(gabor_theta), itsGaborIntens(intens)
{

}

template <class T> CudaImageSet<T> CudaReichardtPyrBuilder<T>::build(const CudaImage<T>& image,
                                          const int firstlevel,
                                          const int depth,
                                          CudaPyramidCache<T>* cache)
{
  const CudaImageSet<T>* const cached =
    (cache != 0 && itsPtype == Gaussian5)
    ? cache->gaussian5.get(image) // may be null if there is no cached pyramid
    : 0;

  // create a pyramid with the input image
  CudaImageSet<T> upyr =
    cached != 0
    ? *cached
    : cudaBuildPyrGeneric(image, firstlevel, depth, itsPtype,
                      itsGaborAngle, itsGaborIntens);
  // create an empty pyramid
  CudaImageSet<T> spyr(depth);

  // fill the empty pyramid with the shifted version
  for (int i = firstlevel; i < depth; ++i)
    spyr[i] = cudaShiftImage(upyr[i], itsDX, itsDY);

  // store both pyramids in the deques
  unshifted.push_back(upyr);
  shifted.push_back(spyr);

  CudaImageSet<T> result(depth);

  // so, it's our first time? Pretend the pyramid before this was
  // the same as the current one ...
  if (unshifted.size() == 1)
    {
      unshifted.push_back(upyr);
      shifted.push_back(spyr);
    }

  // need to pop off old pyramid?
  if (unshifted.size() == 3)
    {
      unshifted.pop_front();
      shifted.pop_front();
    }

  // compute the Reichardt maps
  for (int i = firstlevel; i < depth; ++i)
    {
      result[i] =
        (unshifted.back()[i] * shifted.front()[i]) -
        (unshifted.front()[i] * shifted.back()[i]);
    }

  return result;
}

// ######################################################################
template <class T> CudaReichardtPyrBuilder<T>* CudaReichardtPyrBuilder<T>::clone() const
{
  return new CudaReichardtPyrBuilder<T>(*this);
}

// ######################################################################
template <class T> void CudaReichardtPyrBuilder<T>::reset()
{
  shifted.clear();
  unshifted.clear();
}

// ######################################################################
// ##### Instantiations
// ######################################################################

#define INSTANTIATE(T) \
template class CudaPyrBuilder< T >; \
template class CudaReichardtPyrBuilder< T >; \

INSTANTIATE(float);
