/*!@file Image/ImageSetOps.C Free functions operating on sets of images */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaImageSetOps.C $
// $Id: CudaImageSetOps.C 12962 2010-03-06 02:13:53Z irock $
//

#include "CUDA/CudaImageSetOps.H"

#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/MathOps.H"
#include "Image/Range.H"
#include "Image/ShapeOps.H"
#include "Util/Assert.H"
#include "rutz/compat_cmath.h"

// ######################################################################
// ######################################################################
// ##### ImageSet processing functions
// ######################################################################
// ######################################################################

// ######################################################################
template <class T>
bool cudaIsHomogeneous(const CudaImageSet<T>& x)
{
  if (x.size() == 0) return true;

  const Dims d = x[0].getDims();

  for (uint i = 1; i < x.size(); ++i)
    if (d != x[i].getDims())
      return false;

  return true;
}

// ######################################################################
template <class T>
bool cudaIsDyadic(const CudaImageSet<T>& pyr)
{
  if (pyr.size() == 0) return false;

  for (uint i = 1; i < pyr.size(); ++i)
    {
      const Dims prevdims = pyr[i-1].getDims();
      const Dims curdims = pyr[i].getDims();

      // make sure we don't go below 1
      const int pw2 = std::max(prevdims.w()/2,1);
      const int ph2 = std::max(prevdims.h()/2,1);

      if (curdims.w() != pw2) return false;
      if (curdims.h() != ph2) return false;
    }

  return true;
}

// ######################################################################
template <class T>
CudaImageSet<T> cudaTakeSlice(const CudaImageSet<T>* sets, uint nsets, uint level)
{
  CudaImageSet<T> result(nsets);

  for (uint i = 0; i < nsets; ++i)
    {
      result[i] = sets[i][level];
    }

  return result;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
