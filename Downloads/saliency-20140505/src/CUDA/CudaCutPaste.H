/*!@file CUDA/CudaCutPaste.H Cut+paste operations from/to CudaImage subregions */

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
// Primary maintainer for this file: Rob Peters <rjpeters@klab.caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaCutPaste.H $
// $Id: CudaCutPaste.H 12962 2010-03-06 02:13:53Z irock $
//

#ifndef CUDA_CUDACUTPASTE_H_DEFINED
#define CUDA_CUDACUTPASTE_H_DEFINED

#include "Image/PixelsTypes.H"

class Dims;
template <class T> class Point2D;
class Rectangle;
template <class T> class CudaImage;

//! Extract a subimage from the input image.
/*! The upper left corner of the subimage is given by a Point2D<int>, and the
    size of the subimage is given by the Dims. If zerofill is false, then
    the rectangle given by Point2D<int>+Dims must be wholly contained within the
    bounds of the input image. If zerofill is true, then any output region
    that lies outside the bounds of the input will be zero-filled (where
    zero means the default value for type T). */
CudaImage<float> cudaCrop(const CudaImage<float>& src,
                     const Point2D<int>& upperLeft, const Dims& size,
                     const bool zerofill = false);

//! Overload of crop() using a Rectangle instead of a Point2D<int>+Dims.
CudaImage<float> cudaCrop(const CudaImage<float>& src,
                     const Rectangle& rect, const bool zerofill = false);

//! Shift an image by a non even amount of pixels, need to interpolate
CudaImage<float> cudaShiftImage(const CudaImage<float>& src, const float dx, const float dy);

//! Paste an image onto an existing image
void cudaInplacePaste(CudaImage<float>& dst,
                      const CudaImage<float>& img, const Point2D<int>& pos);

//! Paste a PixRGB image onto an existing image
void cudaInplacePaste(CudaImage<PixRGB<float> >& dst,
                      const CudaImage<PixRGB<float> >& img, const Point2D<int>& pos);

//! Overlay an image onto an existing image, only overwriting an old pixel if new pixel is nonzero
void cudaInplaceOverlay(CudaImage<float>& dst, const CudaImage<float>& img, const Point2D<int>& pos);

//! Overlay a PixRGB image onto an existing image, only overwriting an old pixel if new pixel is nonzero
void cudaInplaceOverlay(CudaImage<PixRGB<float> >& dst, const CudaImage<PixRGB<float> >& img, const Point2D<int>& pos);

#endif
