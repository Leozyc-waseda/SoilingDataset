/*!@file Matlab/mexConverts.C conversions from and to MEX arrays
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Matlab/mexConverts.C $
// $Id: mexConverts.C 15468 2013-04-18 02:18:18Z itti $
//

#include "Matlab/mexConverts.H"

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Matlab/mexLog.H"
#include "Util/sformat.H"
#include "rutz/shared_ptr.h"
#include <matrix.h> // matlab

// ######################################################################
template <class T>
Image<T> mexArray2Image(const mxArray* arrptr)
{
  const int* intptr = mxGetDimensions(arrptr);
  const int num_dim = mxGetNumberOfDimensions(arrptr);
  const int height = *intptr++;
  const int width = *intptr++;

  // check whether we actually have a 2D array
  if (num_dim != 2)
    mexFatal(sformat("Wrong number of dimensions for gray scale image: %i "
                     "- should be 2.",num_dim));

  // convert UINT8 or double array into an Image object
  Image<T> img(width, height, NO_INIT);

  if (mxIsUint8(arrptr))
    {
      const byte *src_ptr, *src_ptr2;
      src_ptr = (byte*)mxGetPr(arrptr);
      typename Image<T>::iterator tgt_ptr = img.beginw();
      for (int y = 0; y < height; y++)
        {
          src_ptr2 = src_ptr;
          for (int x = 0; x < width; x++)
            {
              conv(src_ptr2,&*(tgt_ptr++));
              src_ptr2 += height;
            }
          src_ptr++;
        }
    }
  else if (mxIsDouble(arrptr))
    {
      const double *src_ptr, *src_ptr2;
      src_ptr = (double*)mxGetPr(arrptr);
      typename Image<T>::iterator tgt_ptr = img.beginw();
      for (int y = 0; y < height; y++)
        {
          src_ptr2 = src_ptr;
          for (int x = 0; x < width; x++)
            {
              conv(src_ptr2,&*(tgt_ptr++));
              src_ptr2 += height;
            }
          src_ptr++;
        }
    }
  else mexFatal("Image data type must be uint8 or double.");

  return img;
}


// ######################################################################
template <class T>
Image< PixRGB<T> > mexArray2RGBImage(const mxArray* arrptr)
{
  const int* intptr = mxGetDimensions(arrptr);
  const int num_dim = mxGetNumberOfDimensions(arrptr);
  const int height = *intptr++;
  const int width = *intptr++;
  const int size = width * height;
  int RGBcount = 0;

  // check whether we have RGB or grayscale image
  if (num_dim > 2)
    {
      if (*intptr == 3) RGBcount = size;
      else if (*intptr != 1)
        mexFatal(sformat("Wrong number of dimensions for image: %d "
                         "- should be 2 for grayscale or 3 for RGB",num_dim));
    }
  int RGBcount2 = RGBcount + RGBcount;

  // convert UINT8 or double array into an Image object
  Image< PixRGB<T> > img(width, height, NO_INIT);
  if (mxIsUint8(arrptr))
    {
      const byte *src_ptr, *src_ptr2;
      src_ptr = (byte*)mxGetPr(arrptr);
      T dummy;
      typename Image<PixRGB<T> >::iterator tgt_ptr = img.beginw();
      for (int y = 0; y < height; y++)
        {
          src_ptr2 = src_ptr;
          for (int x = 0; x < width; x++)
            {
              tgt_ptr->setRed  (conv(src_ptr2, &dummy));
              tgt_ptr->setGreen(conv(src_ptr2 + RGBcount, &dummy));
              tgt_ptr->setBlue (conv(src_ptr2 + RGBcount2, &dummy));
              tgt_ptr++;
              src_ptr2 += height;
            }
          src_ptr++;
        }
    }
  else if (mxIsDouble(arrptr))
    {
      const double *src_ptr, *src_ptr2;
      src_ptr = (double*)mxGetPr(arrptr);
      T dummy;
      typename Image<PixRGB<T> >::iterator tgt_ptr = img.beginw();
      for (int y = 0; y < height; y++)
        {
          src_ptr2 = src_ptr;
          for (int x = 0; x < width; x++)
            {
              tgt_ptr->setRed  (conv(src_ptr2, &dummy));
              tgt_ptr->setGreen(conv(src_ptr2 + RGBcount, &dummy));
              tgt_ptr->setBlue (conv(src_ptr2 + RGBcount2, &dummy));
              tgt_ptr++;
              src_ptr2 += height;
            }
          src_ptr++;
        }
    }
  else mexFatal("Image data type must be uint8 or double.");

  return img;
}

// ######################################################################
template <class T>
mxArray* Image2mexArray(const Image<T>& img)
{
  if (img.getSize() == 0) return mxCreateDoubleScalar(0);
  double *tgt_ptr, *tgt_ptr2;
  int height = img.getHeight(), width = img.getWidth();
  mxArray* arrptr = mxCreateDoubleMatrix(height, width, mxREAL);
  tgt_ptr = (double*)mxGetPr(arrptr);
  typename Image<T>::const_iterator src_ptr = img.begin();
  for (int y = 0; y < height; y++)
    {
      tgt_ptr2 = tgt_ptr;
      for (int x = 0; x < width; x++)
        {
          conv(&*(src_ptr++),tgt_ptr2);
          tgt_ptr2 += height;
        }
      tgt_ptr++;
    }
  return arrptr;
}

// ######################################################################
mxArray* Image2mexArrayUint8(const Image<byte>* img)
{
  const int height = img->getHeight();
  const int width = img->getWidth();
  mxArray* arrptr =
    mxCreateNumericMatrix(height, width, mxUINT8_CLASS, mxREAL);
  byte* tgt_ptr = reinterpret_cast<byte*>(mxGetPr(arrptr));
  Image<byte>::const_iterator src_ptr = img->begin();
  for (int y = 0; y < height; ++y)
    {
      byte* tgt_ptr2 = tgt_ptr;
      for (int x = 0; x < width; ++x)
        {
          *tgt_ptr2 = *src_ptr++;
          tgt_ptr2 += height;
        }
      ++tgt_ptr;
    }
  return arrptr;
}

// ######################################################################
template <class T>
mxArray* RGBImage2mexArray(const Image< PixRGB<T> >& img)
{
  if (img.getSize() == 0) return mxCreateDoubleScalar(0);
  double *tgt_ptr, *tgt_ptr2;
  T r, g, b;
  int dims[3];
  dims[0] = img.getHeight();
  dims[1] = img.getWidth();
  dims[2] = 3;
  int size = dims[0] * dims[1], size2 = size + size;
  mxArray* arrptr = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
  tgt_ptr = (double*)mxGetPr(arrptr);
  typename Image<PixRGB<T> >::const_iterator src_ptr = img.begin();
  for (int y = 0; y < dims[0]; y++)
    {
      tgt_ptr2 = tgt_ptr;
      for (int x = 0; x < dims[1]; x++)
        {
          r = src_ptr->red(); g = src_ptr->green(); b = src_ptr->blue();
          conv(&r,tgt_ptr2); conv(&g,tgt_ptr2 + size); conv(&b, tgt_ptr2 + size2);
          src_ptr++;
          tgt_ptr2 += dims[0];
        }
      tgt_ptr++;
    }
  return arrptr;
}

// ######################################################################
mxArray* RGBImage2mexArrayUint8(const Image< PixRGB<byte> >* img)
{
  const int dims[3] =
    {
      img->getHeight(),
      img->getWidth(),
      3
    };
  const int size = dims[0] * dims[1];
  const int size2 = size + size;
  mxArray* arrptr =
    mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
  byte* tgt_ptr = reinterpret_cast<byte*>(mxGetPr(arrptr));
  Image<PixRGB<byte> >::const_iterator src_ptr = img->begin();

  for (int y = 0; y < dims[0]; ++y)
    {
      byte* tgt_ptr2 = tgt_ptr;
      for (int x = 0; x < dims[1]; ++x)
        {
          tgt_ptr2[0] = src_ptr->red();
          tgt_ptr2[size] = src_ptr->green();
          tgt_ptr2[size2] = src_ptr->blue();
          ++src_ptr;
          tgt_ptr2 += dims[0];
        }
      ++tgt_ptr;
    }

  return arrptr;
}

// ######################################################################
template <class T>
std::vector<T> mexArr2Vector(const mxArray* vecptr, uint num_elements)
{
  if (!mxIsDouble(vecptr))
    mexErrMsgTxt("Can only convert vectors of type double.");
  if((uint)mxGetNumberOfElements(vecptr) < num_elements)
    mexErrMsgTxt("Not enough data to convert to vector.");

  double* src_ptr = (double*)mxGetPr(vecptr);
  std::vector<T> result;
  for (uint i = 0; i < num_elements; i++)
    result.push_back((T)(*src_ptr++));

  return result;
}

// ######################################################################
template <class T>
std::vector<T> mexArr2Vector(const mxArray* vecptr)
{
  return mexArr2Vector<T>(vecptr, (uint)mxGetNumberOfElements(vecptr));
}

// ######################################################################
template <class T>
mxArray* Vector2mexArr(const std::vector<T>& vect, uint num_elements)
{
  if (vect.size() < num_elements)
    mexFatal(sformat("std::vector too small to convert to mxArray: "
                     "%" ZU ", but should be %d.",vect.size(),num_elements));

  if (vect.empty()) return mxCreateDoubleScalar(0);

  mxArray* result = mxCreateDoubleMatrix(1, num_elements, mxREAL);
  double* tgt_ptr = (double*)mxGetPr(result);
  for (uint i = 0; i < num_elements; i++)
    *tgt_ptr++ = (double)vect[i];
  return result;
}

// ######################################################################
template <class T>
mxArray* Vector2mexArr(const std::vector<T>& vect)
{
  return Vector2mexArr(vect, vect.size());
}

// ######################################################################
std::vector<Point2D<int> > mexArr2Point2DVec(const mxArray* arrptr)
{
  if (!mxIsDouble(arrptr))
    mexFatal("Can only convert vectors of type double.");
  if (mxGetNumberOfDimensions(arrptr) != 2)
    mexFatal("mxArray must be 2D to be converted to vector<Point2D<int> >.");
  if (mxGetM(arrptr) != 2)
    mexFatal("mxArray must be of dimensions 2 x N to be converted "
             "to vector<Point2D<int> >");
  int n = mxGetN(arrptr);
  const double* src_ptr = (double*) mxGetPr(arrptr);
  std::vector<Point2D<int> > result;

  for (int i = 0; i < n; i++)
    {
      result.push_back(Point2D<int>((int)src_ptr[0], (int)src_ptr[1]));
      src_ptr += 2;
    }
  return result;
}

// ######################################################################
mxArray* Point2DVec2mexArr(const std::vector<Point2D<int> >& vect)
{
  if (vect.empty()) return mxCreateDoubleScalar(0);
  mxArray* result = mxCreateDoubleMatrix(2, vect.size(), mxREAL);
  double* tgt_ptr = (double*)mxGetPr(result);
  for (uint i = 0; i < vect.size(); i++)
    {
      *tgt_ptr++ = (double)vect[i].i;
      *tgt_ptr++ = (double)vect[i].j;
    }
  return result;
}

// ######################################################################
template <class T>
mxArray* ImgVec2mexArr(const ImageSet<T>& vect)
{
  if (vect.isEmpty()) return mxCreateDoubleScalar(0);

  int dims[3];
  dims[0] = vect.size();
  dims[1] = vect[0].getHeight();
  dims[2] = vect[0].getWidth();
  int inc = dims[0] * dims[1];
  mxArray* result = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
  double *tgt, *tgt2, *tgt3;
  tgt = (double*)mxGetPr(result);

  for (int i = 0; i < dims[0]; i++)
    {
      tgt2 = tgt;
      if (!vect[i].isSameSize(vect[0]))
        mexFatal("All images in an ImageSet must have the same dimensions.");
      typename Image<T>::const_iterator src = vect[i].begin();
      for (int y = 0; y < dims[1]; y++)
        {
          tgt3 = tgt2;
          for (int x = 0; x < dims[2]; x++)
            {
              conv(&*(src++),tgt3);
              tgt3 += inc;
            }
          tgt2 += dims[0];
        }
      tgt++;
    }
  return result;
}

// ######################################################################
template <class T>
ImageSet<T> mexArr2ImgVec(const mxArray* arrptr)
{
  const int num_dim = mxGetNumberOfDimensions(arrptr);
  const int* dims = mxGetDimensions(arrptr);
  int num_img=0, height=0, width=0;

  switch(num_dim)
    {
    case 2: // just a single image -> create just one element of the vector
      {
        num_img = 1;
        height = dims[0];
        width = dims[1];
        break;
      }

    case 3: // multiple images -> how many and how big?
      {
        num_img = dims[0];
        height = dims[1];
        width = dims[2];
        break;
      }

    default:
      {
        mexFatal(sformat("Array must have 2 or 3 dimensions to be converted to "
                         "a grayscale image vector, but not %d.",num_dim));
        break;
      }
    }
  const int inc = num_img * height;
  ImageSet<T> result(num_img, Dims(dims[2], dims[1]), NO_INIT);

  if (mxIsUint8(arrptr))
    {
      const byte *src, *src2, *src3;
      src = (byte*)mxGetPr(arrptr);
      for (int img_no = 0; img_no < num_img; ++img_no)
        {
          src2 = src;
          typename Image<T>::iterator tgt = result[img_no].beginw();
          for (int y = 0; y < height; ++y)
            {
              src3 = src2;
              for (int x = 0; x < width; ++x)
                {
                  conv(src3,&*(tgt++));
                  src3 += inc;
                }
              src2 += num_img;
            }
          src++;
        }
    }
  else if (mxIsDouble(arrptr))
    {
      const double *src, *src2, *src3;
      src = (double*)mxGetPr(arrptr);
      for (int img_no = 0; img_no < num_img; ++img_no)
        {
          src2 = src;
          typename Image<T>::iterator tgt = result[img_no].beginw();
          for (int y = 0; y < height; ++y)
            {
              src3 = src2;
              for (int x = 0; x < width; ++x)
                {
                  conv(src3,&*(tgt++));
                  src3 += inc;
                }
              src2 += num_img;
            }
          src++;
        }
    }
  else mexFatal("Image data type must be uint8 or double.");

  return result;
}

// ######################################################################
template <class T>
mxArray* ImageSet2cellArr(const ImageSet<T>& imset)
{
  int num = imset.size();
  mxArray *result = mxCreateCellArray(1,&num);

  for (int i = 0; i < num; ++i)
    mxSetCell(result,i,Image2mexArray(imset[i]));

  return result;
}

// ######################################################################
template <class T>
ImageSet<T> cellArr2ImageSet(const mxArray* arrptr)
{
  if (!mxIsCell(arrptr)) mexFatal("cellArr2ImageSet: expected cell array");

  int num = mxGetNumberOfElements(arrptr);
  ImageSet<T> result(num);

  for (int i = 0; i < num; ++i)
    result[i] = mexArray2Image<T>(mxGetCell(arrptr,i));

  return result;
}

// ######################################################################
template <class T>
mxArray* ImageSetVec2cellArr(const std::vector< ImageSet<T> >& vec)
{
  int dims[2],idx[2];
  dims[0] = vec.size();
  dims[1] = vec.size();

  mxArray *result = mxCreateCellArray(2,dims);

  for (idx[0] = 0; idx[0] < dims[0]; ++idx[0])
    for (idx[1] = 0; idx[1] < dims[1]; ++idx[1])
      {
        int sub = mxCalcSingleSubscript(result,2,idx);
        mxSetCell(result,sub,Image2mexArray(vec[idx[0]][idx[1]]));
      }
  return result;
}

// ######################################################################
template <class T>
std::vector< ImageSet<T> > cellArr2ImageSetVec(const mxArray* arrptr)
{
  if (!mxIsCell(arrptr)) mexFatal("arrPtr needs to be a 2D cell array.");

  if (mxGetNumberOfDimensions(arrptr) == 1)
    {
      std::vector< ImageSet<T> > result(1);
      result[0] = cellArr2ImageSet<T>(arrptr);
      return result;
    }
  else if (mxGetNumberOfDimensions(arrptr) != 2)
    mexFatal("arrPtr needs to be a 2D cell array.");

  int numSets = mxGetM(arrptr);
  int numImgs = mxGetN(arrptr);
  int idx[2];

  std::vector< ImageSet<T> > result(numSets,ImageSet<T>(uint(numImgs)));

  for (idx[0] = 0; idx[0] < numSets; ++idx[0])
    for (idx[1] = 0; idx[1] < numImgs; ++idx[1])
      {
        int sub = mxCalcSingleSubscript(arrptr,2,idx);
        result[idx[0]][idx[1]] = mexArray2Image<T>(mxGetCell(arrptr,sub));
      }
  return result;
}

// ######################################################################
// ###### member functions for class MexReturn
// ######################################################################
MexReturn::MexReturn(int nlhs, mxArray **plhs)
  : itsNlhs(nlhs),
    itsPlhs(plhs)
{}

// ######################################################################
MexReturn::~MexReturn()
{
  bool okay = true;
  for(int i = 0; i < itsNlhs; ++i)
    if (!itsPlhs[i])
      {
        mexError(sformat("Return argument %i is not assigned.",i+1));
        okay = false;
      }

  if (!okay) mexFatal("There were unassigned return arguments.");
}

// ######################################################################
int MexReturn::numArgs()
{ return itsNlhs; }

// ######################################################################
bool MexReturn::isNumOK(int num)
{ return ((num >= 0) && (num < numArgs())); }

// ######################################################################
bool MexReturn::store(int num, mxArray *val)
{
  if (!isNumOK(num)) return false;
  itsPlhs[num] = val;
  return true;
}

// ######################################################################
bool MexReturn::store(int num, double val)
{  return store(num,mxCreateDoubleScalar(val)); }

// ######################################################################
template <class T>
bool MexReturn::store(int num, const std::vector<T>& val)
{
  if (!isNumOK(num)) return false;
  return store(num,Vector2mexArr(val));
}

// ######################################################################
template <class T>
bool MexReturn::store(int num, const Image<T>& val)
{
  if (!isNumOK(num)) return false;
  return store(num,Image2mexArray(val));
}

// ######################################################################
template <class T>
bool MexReturn::store(int num, const Image< PixRGB<T> >& val)
{
  if (!isNumOK(num)) return false;
  return store(num,RGBImage2mexArray(val));
}

// ######################################################################


#define INSTANTIATE(T) \
template Image<T> mexArray2Image(const mxArray* arrptr); \
template Image< PixRGB<T> > mexArray2RGBImage(const mxArray* arrptr); \
template mxArray* Image2mexArray(const Image<T>& img); \
template mxArray* RGBImage2mexArray(const Image< PixRGB<T> >& img); \
template mxArray* ImgVec2mexArr(const ImageSet<T>& vect); \
template ImageSet<T> mexArr2ImgVec(const mxArray* arrptr); \
template mxArray* ImageSet2cellArr(const ImageSet<T>& imset); \
template ImageSet<T> cellArr2ImageSet(const mxArray* arrptr); \
template mxArray* ImageSetVec2cellArr(const std::vector< ImageSet<T> >& vec); \
template std::vector< ImageSet<T> > cellArr2ImageSetVec(const mxArray* arrptr); \
template bool MexReturn::store(int num, const Image<T>& val); \
template bool MexReturn::store(int num, const Image< PixRGB<T> >& val); \


INSTANTIATE(byte);
INSTANTIATE(float);

// Also instantiate these for doubles:
template mxArray* Image2mexArray(const Image<double>& img);

#define INSTANTIATE2(T) \
template std::vector<T> mexArr2Vector(const mxArray* vecptr, uint num_elements); \
template std::vector<T> mexArr2Vector(const mxArray* vecptr); \
template mxArray* Vector2mexArr(const std::vector<T>& vect, uint num_elements); \
template mxArray* Vector2mexArr(const std::vector<T>& vect); \
template bool MexReturn::store(int num, const std::vector<T>& val); \

INSTANTIATE2(byte);
INSTANTIATE2(float);
INSTANTIATE2(int);
INSTANTIATE2(double);


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
