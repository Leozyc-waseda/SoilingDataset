/*!@file Matlab/mexConverts.H conversions from ans to MEX arrays
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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Matlab/mexConverts.H $
// $Id: mexConverts.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef MEXCONVERTS_H_DEFINED
#define MEXCONVERTS_H_DEFINED

#include "Util/Types.H"
#include "Image/Point2D.H"
#include "Image/ImageSet.H"
#include <mex.h>
#include <vector>

template <class T> class PixRGB;
template <class T> class Image;

//! convert a mxArray into a grayscale image
template <class T>
Image<T> mexArray2Image(const mxArray* arrptr);

//! convert a mxArray into a RGB image
template <class T>
Image< PixRGB<T> > mexArray2RGBImage(const mxArray* arrptr);

//! convert a grayscale image into a mxArray
template <class T>
mxArray* Image2mexArray(const Image<T>& img);

//! convert a grayscale image into a mxArray of class 'uint8'
/** We take a pointer rather than a reference to ensure that there are
    no implicit Image type conversions in calls to this function -- we
    want exactly an Image<byte>, and not an Image<float> or some other
    pixel type. */
mxArray* Image2mexArrayUint8(const Image<byte>* img);

//! convert a RGB image into a mxArray of class 'double'
template <class T>
mxArray* RGBImage2mexArray(const Image< PixRGB<T> >& img);

/// convert a RGB image into a mxArray of class 'uint8'
/** We take a pointer rather than a reference to ensure that there are
    no implicit Image type conversions in calls to this function -- we
    want exactly an Image<PixRGB<byte>>, and not an
    Image<PixRGB<float>> or some other pixel type. */
mxArray* RGBImage2mexArrayUint8(const Image< PixRGB<byte> >* img);

//! convert the first num_elements elements of a mxArray into a std::vector
template <class T>
std::vector<T> mexArr2Vector(const mxArray* vecptr, uint num_elements);

//! convert a mxArray vector into a std::vector
template <class T>
std::vector<T> mexArr2Vector(const mxArray* vecptr);

//! convert the first num_elements elements of a std::vector into a mxArray
template <class T>
mxArray* Vector2mexArr(const std::vector<T>& vect, uint num_elements);

//! convert a std::vector into a mxArray vector
template <class T>
mxArray* Vector2mexArr(const std::vector<T>& vect);

//! convert a 2 x N mxArray into a std::vector<Point2D<int> >
std::vector<Point2D<int> > mexArr2Point2DVec(const mxArray* arrptr);

//! convert a std::vector<Point2D<int> > into a 2 x N mxArray
mxArray* Point2DVec2mexArr(const std::vector<Point2D<int> >& vect);

//! convert a vector of pointers to images of equal size into a 3D mxArray
template <class T>
mxArray* ImgVec2mexArr(const ImageSet<T>& vect);

//! convert a 3D mxArray into a vector of pointers to images of equal size
template <class T>
ImageSet<T> mexArr2ImgVec(const mxArray* arrptr);

//! convert an ImageSet into a cell array of images
template <class T>
mxArray* ImageSet2cellArr(const ImageSet<T>& imset);

//! convert a cell array of images into an ImageSet
template <class T>
ImageSet<T> cellArr2ImageSet(const mxArray* arrptr);

//! convert a vector of ImageSet into a 2d cell array
template <class T>
mxArray* ImageSetVec2cellArr(const std::vector< ImageSet<T> >& vec);

//! convert a 2d cell array into a vector of ImageSet
template <class T>
std::vector< ImageSet<T> > cellArr2ImageSetVec(const mxArray* arrptr);

// ######################################################################
inline double conv(const double* src, double* target)
{ *target = *src; return *target; }

inline byte conv(const double* src, byte* target)
{ *target = (byte)(*src * 255.0); return *target; }

inline double conv(const byte* src, double* target)
{ *target = (double)*src / 255.0; return *target; }

inline byte conv(const byte* src, byte* target)
{ *target = *src; return *target; }

inline float conv(const double* src, float* target)
{ *target = (float)*src; return *target; }

inline double conv(const float* src, double* target)
{ *target = (double)*src; return *target; }

inline float conv(const byte* src, float* target)
{ *target = (float)*src / 255.0; return *target; }


// ######################################################################
//! Class to help manage return values in mex files.
class MexReturn
{
public:
  //! Constructor takes arguments from the mexFunction call
  MexReturn(int nlhs, mxArray *plhs[]);

  //! Destructor checks that all arguments are assigned
  virtual ~MexReturn();

  //! returns the number of return arguments
  int numArgs();

  //! test if 0 <= num < numArgs()
  bool isNumOK(int num);

  //! store a mxArray as return argumnt num; true if successful
  bool store(int num, mxArray *val);

  //! store  double scalar
  bool store(int num, double val);

  //! store a vector of numbers
  template <class T>
  bool store(int num, const std::vector<T>& val);

  //! store a grayscale image
  template <class T>
  bool store(int num, const Image<T>& val);

  //! store an RGB image
  template <class T>
  bool store(int num, const Image< PixRGB<T> >& val);

 private :
  int itsNlhs;
  mxArray **itsPlhs;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
