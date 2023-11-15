/*!@file CUDA/CudaImageSetOps.H Free functions operating on sets of images
 */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaImageSetOps.H $
// $Id: CudaImageSetOps.H 12962 2010-03-06 02:13:53Z irock $
//

#ifndef CUDAIMAGESET_OPS_H_DEFINED
#define CUDAIMAGESET_OPS_H_DEFINED

#include "Util/Types.H" // for uint

class Dims;
template <class T> class CudaImage;
template <class T> class CudaImageSet;
template <class T> class Range;

// ############################################################
// ############################################################
/*! @name CudaImageSet processing functions

    Not all of these functions are templates, simply because alternate
    versions aren't currently needed.
*/
//@{

//! Returns true if all images in the set are the same size.
/*! Also returns true for the empty set. */
template <class T>
bool cudaIsHomogeneous(const CudaImageSet<T>& x);

//! Check whether the pyramid is dyadic.
/*! A dyadic pyramid is one in which each level is one half the width
    and one half the height of the preceding level. NOTE that this
    function is defined here, in CudaImageSetOps.H, rather than in
    Pyramid_Ops.H, in order to avoid a cyclic dependency between those
    two modules. CudaImageSetOps needs isDyadic() in order to implement
    makeImageArray(). */
template <class T>
bool cudaIsDyadic(const CudaImageSet<T>& pyr);


//! Make an CudaImageSet from the s'th level from each of the N input sets.
/*! This is useful, for example, in transforming an array of CudaImageSets
    from multiple-scales-per-set/one-orientation-per-set into the
    reverse, multiple-orientations-per-set/one-scale-per-set.
*/
template <class T>
CudaImageSet<T> cudaTakeSlice(const CudaImageSet<T>* sets, uint N, uint s);


#endif // !CUDAIMAGESET_OPS_H_DEFINED

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
