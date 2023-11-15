/*!@file CUDA/CudaImage.H An image template class meant for interacting with
  CUDA devices */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaImage.H $
// $Id: CudaImage.H 12962 2010-03-06 02:13:53Z irock $
//

#ifndef CUDAIMAGE_H_DEFINED
#define CUDAIMAGE_H_DEFINED

#include "Util/Assert.H"
#include "Util/Promotions.H"
#include "Image/ArrayData.H"
#include "Image/Point2D.H"
#include "Image/Rectangle.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "CUDA/wrap_c_cuda.h"
#include <algorithm> // for std::min

//! CudaImage template class
/*! This is a image template class for CUDA devices that can handle grayscale
    as well as color or multispectral images. CudaImage methods should
    be instantiable for any type T that has the basic arithmetic
    operators. However, our CUDA processing is inherently float based, so
    conversions may occur.  Note that some CudaImage functions will only
    work with float types, and other will only work for composite types
    such as PixRGB.
*/

template <class T>
class CudaImage
{
public:

  // ############################################################
  /*! @name Constructors, destructors, assignment */
  //@{

  //! Construct from C array using memory policy
  /*! Build from C array; an internal copy of the C array will be
      allocated, so the C array can (and should) be freed without
      affecting the Image. */
  inline CudaImage(const T* inarray, int width, int height, const MemoryPolicy mp, const int dev, const MemoryPolicy srcmp, const int srcdev);

  //! Construct from C array using memory policy
  /*! Build from C array; an internal copy of the C array will be
      allocated, so the C array can (and should) be freed without
      affecting the CudaImage. */
  inline CudaImage(const T* inarray, const Dims& dims, const MemoryPolicy mp, const int dev, const MemoryPolicy srcmp, const int srcdev);

  //! Allocates memory for given size, optionally zero-clear that memory, and set the memory policy
  inline CudaImage(int width, int height, InitPolicy init, const MemoryPolicy mp, const int dev);

  //! Constructor that only allocates memory for given size
  inline explicit CudaImage(const Dims& dims, const MemoryPolicy mp, const int dev);

  //! Constructor that only allocates memory for given size and init type
  inline explicit CudaImage(const Dims& dims, InitPolicy init, const MemoryPolicy mp, const int dev);

  //! Construct an empty (0-by-0) image (useful for arrays of CudaImages).
  inline CudaImage();

  //! Copy constructor
  /*! e.g.:
      \code
      CudaImage<byte> im(other);
      // or
      CudaImage<byte> im = other; // with other also of type CudaImage<byte>
      \endcode
  */
  inline CudaImage(const CudaImage<T>& A);

  //! Copy constructor to particular memory
  /*! e.g.:
      \code
      CudaImage<byte> im(other);
      // or
      CudaImage<byte> im = other; // with other also of type CudaImage<byte>
      \endcode
  */
  inline CudaImage(const CudaImage<T>& A, const MemoryPolicy mp, const int dev);

  //! Import copy constructor
  /*! e.g.:
      \code
      CudaImage<byte> im(other);
      // or
      CudaImage<byte> im = other; // with other of type CudaImage<float>
      \endcode
  */
  inline CudaImage(const Image<T>& A, const MemoryPolicy mp, const int dev);

  //! Assigment operator.
  /*! e.g.:
      \code
      CudaImage<byte> im1, im2; im2 = im1;
      \endcode
  */
  inline CudaImage<T>& operator=(const CudaImage<T>& A);

  //! Destructor
  inline ~CudaImage();

  //! Free memory and switch to uninitialized state.
  /*! Note that it is \b NOT necessary to call this function to ensure
      proper cleanup, that will be done in the destructor by
      default. Rather, freeMem() is offered just as a performance
      optimization, to allow you to release a potentially large chunk
      of memory when you are finished using it. */
  inline void freeMem();

  //@}

  // ############################################################
  /*! @name Math operators */
  //! Math operators
  //@{

  //! Addition inplace by a host provided scalar
  inline CudaImage<float>& operator+=(const float val);

  //! Subtraction inplace by a host provided scalar
  inline CudaImage<float>& operator-=(const float val);

  //! Multiplication inplace by a host provided scalar
  inline CudaImage<float>& operator*=(const float val);

  //! Division inplace by a host provided scalar
  inline CudaImage<float>& operator/=(const float val);

  //! Addition inplace by a device image or scalar (a 1x1 CudaImage is treated as a scalar)
  inline CudaImage<float>& operator+=(const CudaImage<float>& im);

  //! Subtraction inplace by a device image or scalar (a 1x1 CudaImage is treated as a scalar)
  inline CudaImage<float>& operator-=(const CudaImage<float>& im);

  //! Multiplication inplace by a device image or scalar (a 1x1 CudaImage is treated as a scalar)
  inline CudaImage<float>& operator*=(const CudaImage<float>& im);

  //! Division inplace by a device image or scalar (a 1x1 CudaImage is treated as a scalar)
  inline CudaImage<float>& operator/=(const CudaImage<float>& im);

  //! Addition by a host provided scalar
  inline CudaImage<float> operator+(const float val) const;

  //! Subtraction by a host provided scalar
  inline CudaImage<float> operator-(const float val) const;

  //! Multiplcation by a host provided scalar
  inline CudaImage<float> operator*(const float val) const;

  //! Division by a host provided scalar
  inline CudaImage<float> operator/(const float val) const;

  //! Addition by a device image or scalar (a 1x1 CudaImage is treated as a scalar)
  inline CudaImage<float> operator+(const CudaImage<float>& im) const;

  //! Subtraction by a device image or scalar (a 1x1 CudaImage is treated as a scalar)
  inline CudaImage<float> operator-(const CudaImage<float>& im) const;

  //! Multiplication by a device image or scalar (a 1x1 CudaImage is treated as a scalar)
  inline CudaImage<float> operator*(const CudaImage<float>& im) const;

  //! Division by a device image or scalar (a 1x1 CudaImage is treated as a scalar)
  inline CudaImage<float> operator/(const CudaImage<float>& im) const;
  //@}


  // ############################################################
  /*! @name Memory management functions */
  //@{

  //! Swap the contents of two images
  inline void swap(CudaImage<T>& other);


  //! Return a new image object with a deep copy of the underlying data
  /*! This function is necessary for safe use of
      attach()/detach(). That is, unfortunately attach()/detach() are
      not safe for use with shared image objects -- consider the
      following code:

      \code
      double d[4] = { 0.0, 1.0, 2.0, 3.0};

      // create an CudaImage that is attach()'ed to the double array
      CudaImage<double> a;
      a.attach(&d[0], 2, 2);

      const CudaImage<double> b = a;
      // now 'b' thinks it has a safe lock on some const values:

      d[0] = -1.0;
      // OOPS! By changing values in the 'd' array directly, we'll now
      // have changed things to that b[0] == -1.0, even though 'b' was
      // declared as 'const'
      \endcode

      The solution to this problem is to prohibit the copy done in
      'b=a' above (this triggers an LFATAL() in
      ArrayData::acquire()). That assures us that any ArrayData that
      has a StoragePolicy of WRITE_THRU will be un-shareable.

      So, back to the point -- the correct way to write the code above
      would be to use deepcopy():

      \code
      double d[4] = { 0.0, 1.0, 2.0, 3.0};

      CudaImage<double> a;
      a.attach(&d[0], 2, 2);

      const CudaImage<double> b = a.deepcopy();

      d[0] = -1.0;
      // Now, 'b' is insulated from any changes to 'd' since we've
      // done a deep copy, so even now we'll still have b[0]==0.0
      \endcode
  */
  inline CudaImage<T> deepcopy() const;

  //! Return a new image w/ deep copy of underlying data using memory policy
  /*! This function is needed to allow new image generation to any
      memory area in the system, which is going to have to be a deep copy
   */
  inline CudaImage<T> deepcopy(const MemoryPolicy mp, const int dev) const;

  inline Image<T> exportToImage() const;

  //! Free mem and realloc new array (array contents are lost).
  /*! Use rescale() instead if you want to preserve image contents. */
  inline void resize(const Dims& dims, const bool clear = false);

  //! Free mem and realloc new array (array contents are lost).
  /*! Use rescale() instead if you want to preserve image contents. */
  inline void resize(const int width, const int height,
                     const bool clear = false);

  //@}

  // ############################################################
  /*! @name Access functions*/
  //@{

  //! Check whether image is non-empty (i.e., non-zero height and width).
  inline bool initialized() const;

  //! Get image size (width * height)
  inline int getSize() const;

  //! Get image size (width * height)
  inline uint size() const;

  //! Get image width
  inline int getWidth() const;

  //! Get image height
  inline int getHeight() const;

  //! Get image width+height in Dims struct
  inline const Dims& getDims() const;

  //! Get image bounds as a rectangle with upper-left point at (0,0) and dims matching the image dims
  inline Rectangle getBounds() const;

  //! Check if *this is the same size as the other thing
  /*! The other thing can be any type that exposes getHeight() and
      getWidth() */
  template <class C>
  inline bool isSameSize(const C& other) const;

  //! Check if the image is 1D, i.e., width == 1 or height == 1
  inline bool is1D() const;

  //! Check if the image is a vector, i.e., width == 1
  inline bool isVector() const;

  //! Check if the image is a transposed vector, i.e., height == 1
  inline bool isTransposedVector() const;

  //! Check if the image is square, i.e., width == height
  inline bool isSquare() const;

  //! Returns read-only (const) pointer to internal image array
  inline const T* getCudaArrayPtr() const;

  //! Returns read/write (non-const) pointer to internal image array
  inline T* getCudaArrayPtr();

  //! Test whether point falls inside array boundaries
  inline bool coordsOk(const Point2D<int>& P) const;

  //! Test whether point falls inside array boundaries
  inline bool coordsOk(const int i, const int j) const;

  //! Test whether point falls inside array boundaries
  /*! This test is intended to be used before you attempt a getValInterp() */
  inline bool coordsOk(const Point2D<float>& p) const;

  //! Test whether point falls inside array boundaries
  /*! This test is intended to be used before you attempt a getValInterp() */
  inline bool coordsOk(const float i, const float j) const;

  //! Test whether rectangle fits in image
  inline bool rectangleOk(const Rectangle& rect) const;

  //! Return memory policy of underlying array data
  inline MemoryPolicy getMemoryPolicy() const;

  //! Return memory policy of underlying array data
  inline int getMemoryDevice() const;

  //@}

  //! clear contents (or set to given value)
  inline void clear(const char& val = 0);

  //@}

  // ############################################################
  /*! @name Functions for testing/debugging only */
  //@{

  //! For testing/debugging only.
  bool hasSameData(const CudaImage<T>& b) const;

  //! For testing/debugging only.
  /*! Returns the current reference count. */
  long refCount() const throw();

  //! For testing/debugging only.
  /*! Check if the ArrayHandle is shared. */
  bool isShared() const throw();

  //@}

private:
  // ############################################################
  // ##### Data:
  // ############################################################
  ArrayHandle<T> itsHdl;
  inline const ArrayData<T>& impl() const;
  inline ArrayData<T>& uniq();

};

// ######################################################################
// ######################################################################
// ######################################################################
// FREE FUNCTIONS:
// ######################################################################
// ######################################################################
// ######################################################################

//! Return a new Image with the same data, but interpreted as a different shape
/*! NOTE that this is very different from resize()! With reshape(), we
    are not changing the number of elements in the Image; rather we
    are just changing the way that the internal 1-D memory array is
    mapped to a logical 2-D Image. For example, you could reshape a
    10x5 Image to a 50x1 Image, or a 25x2 Image, or vice versa. You
    need to be aware the fact that Image uses a row-major storage
    format internally (i.e., as you step through the memory, you first
    traverse all the pixels in row 1, then all the pixels in row 2,
    etc.). So if you reshape a WxH Image to a WHx1 Image, your new 1-D
    Image will appear to have the rows of the original Image arranged
    end-to-end.

    Programmer note: In principle, we could support a reshape()
    operation that wouldn't require copying any data; in order to do
    that we'd need to split the Dims management out of ArrayData and
    ArrayHandle, so that a single ArrayData object could be used in
    multiple Image objects, each with potentially different Dims (but
    all with the same total number of pixels). However, for now we
    just use a simpler implementation which copies the data into the
    new Image.
*/
template <class T>
inline CudaImage<T> reshape(const CudaImage<T>& orig, const Dims& newdims)
{
  ASSERT(orig.getDims().sz() == newdims.sz());
  return CudaImage<T>(orig.getCudaArrayPtr(), newdims, orig.getMemoryPolicy(), orig.getMemoryDevice(), orig.getMemoryPolicy(), orig.getMemoryDevice());
}

// ######################################################################
// ######################################################################
// ######################################################################
// INLINE FUNCTIONS:
// ######################################################################
// ######################################################################
// ######################################################################

// ######################################################################
// ##### Constructors & Destructors:
// ######################################################################

// ######################################################################
template <class T> inline
CudaImage<T>::CudaImage(const T* inarray, int width, int height, MemoryPolicy trgmp, int trgdev, const MemoryPolicy srcmp, const int srcdev) :
  itsHdl(new ArrayData<T>(Dims(width, height), inarray, trgmp, trgdev, srcmp, srcdev))
{}

// ######################################################################
template <class T> inline
CudaImage<T>::CudaImage(const T* inarray, const Dims& dims, MemoryPolicy trgmp, int trgdev, const MemoryPolicy srcmp, const int srcdev) :
  itsHdl(new ArrayData<T>(dims, inarray, trgmp, trgdev, srcmp, srcdev))
{}

// ######################################################################
template <class T> inline
CudaImage<T>::CudaImage(int width, int height, InitPolicy init, MemoryPolicy mp, int dev) :
  itsHdl(new ArrayData<T>(Dims(width, height), init, mp, dev))
{}

// ######################################################################
template <class T> inline
CudaImage<T>::CudaImage(const Dims& dims, const MemoryPolicy mp, const int dev) :
  itsHdl(new ArrayData<T>(dims, NO_INIT, mp, dev))
{}

// ######################################################################
template <class T> inline
CudaImage<T>::CudaImage(const Dims& dims, InitPolicy init,  const MemoryPolicy mp, const int dev) :
  itsHdl(new ArrayData<T>(dims, init, mp, dev))
{}

// ######################################################################
template <class T> inline
CudaImage<T>::CudaImage() :
  itsHdl(new ArrayData<T>())
{}

// ######################################################################
template <class T> inline
CudaImage<T>::CudaImage(const CudaImage<T>& A) :
  itsHdl(A.itsHdl)
{}

// ######################################################################
template <class T> inline
CudaImage<T>::CudaImage(const CudaImage<T>& A, const MemoryPolicy mp, const int dev) :
  itsHdl(new ArrayData<T>(A.getDims(), NO_INIT, mp,dev))
{
  // Find out where src is sitting
  const MemoryPolicy srcmp = A.getMemoryPolicy();
  const int srcdev = A.getMemoryDevice();
  // How should we get the actual number of elements? dist from stop to aptr or size()?
  //int n = stop-aptr;
  int n = size();
  ArrayHelper<T>::copy_initialize(getCudaArrayPtr(),n,A.getCudaArrayPtr(),mp,dev,srcmp,srcdev);
}

template <class T> inline
CudaImage<T>::CudaImage(const Image<T>& A, const MemoryPolicy mp, const int dev) :
  itsHdl(new ArrayData<T>(A.getDims(), NO_INIT, mp,dev))
{
  // How should we get the actual number of elements? dist from stop to aptr or size()?
  //int n = stop-aptr;
  int n = size();
  ArrayHelper<T>::copy_initialize(getCudaArrayPtr(),n,A.getArrayPtr(),mp,dev,HOST_MEMORY,CUDA_HOST_DEVICE_NUM);
}

// ######################################################################
template <class T> inline
CudaImage<T>& CudaImage<T>::operator=(const CudaImage<T>& A)
{
  CudaImage<T> A_copy( A );
  this->swap(A_copy);
  return *this;
}

template <class T> inline
CudaImage<float>& CudaImage<T>::operator+=(const float val)
{
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  cuda_c_inplaceAddHostScalar(this->getCudaArrayPtr(), val, tile.sz(), this->size());
  return *this;
}

template <class T> inline
CudaImage<float>& CudaImage<T>::operator-=(const float val)
{
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  cuda_c_inplaceSubtractHostScalar(this->getCudaArrayPtr(), val, tile.sz(), this->size());
  return *this;
}

template <class T> inline
CudaImage<float>& CudaImage<T>::operator*=(const float val)
{
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  cuda_c_inplaceMultiplyHostScalar(this->getCudaArrayPtr(), val, tile.sz(), this->size());
  return *this;
}

template <class T> inline
CudaImage<float>& CudaImage<T>::operator/=(const float val)
{
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  cuda_c_inplaceDivideHostScalar(this->getCudaArrayPtr(), val, tile.sz(), this->size());
  return *this;
}

// For math operations with CudaImage arguments, size()==1 CudaImages are treated as scalars to allow for
// device side values to be applied as scalars

template <class T> inline
CudaImage<float>& CudaImage<T>::operator+=(const CudaImage<float>& im)
{
  ASSERT(im.initialized());
  ASSERT(this->getMemoryDevice() == im.getMemoryDevice());
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  if(im.getSize() == 1)
  {
    cuda_c_inplaceAddScalar(this->getCudaArrayPtr(), im.getCudaArrayPtr(), tile.sz(), this->size());
  }
  else
  {
    ASSERT(this->isSameSize(im));
    cuda_c_inplaceAddImages(this->getCudaArrayPtr(), im.getCudaArrayPtr(), tile.sz(), this->size());
  }
  return *this;
}

template <class T> inline
CudaImage<float>& CudaImage<T>::operator-=(const CudaImage<float>& im)
{
  ASSERT(im.initialized());
  ASSERT(this->getMemoryDevice() == im.getMemoryDevice());
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  if(im.getSize() == 1)
  {
    cuda_c_inplaceSubtractScalar(this->getCudaArrayPtr(), im.getCudaArrayPtr(), tile.sz(), this->size());
  }
  else
  {
    ASSERT(this->isSameSize(im));
    cuda_c_inplaceSubtractImages(this->getCudaArrayPtr(), im.getCudaArrayPtr(), tile.sz(), this->size());
  }
  return *this;
}

template <class T> inline
CudaImage<float>& CudaImage<T>::operator*=(const CudaImage<float>& im)
{
  ASSERT(im.initialized());
  ASSERT(this->getMemoryDevice() == im.getMemoryDevice());
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  if(im.getSize() == 1)
  {
    cuda_c_inplaceMultiplyScalar(this->getCudaArrayPtr(), im.getCudaArrayPtr(), tile.sz(), this->size());
  }
  else
  {
    ASSERT(this->isSameSize(im));
    cuda_c_inplaceMultiplyImages(this->getCudaArrayPtr(), im.getCudaArrayPtr(), tile.sz(), this->size());
  }
  return *this;
}

template <class T> inline
CudaImage<float>& CudaImage<T>::operator/=(const CudaImage<float>& im)
{
  ASSERT(im.initialized());
  ASSERT(this->getMemoryDevice() == im.getMemoryDevice());
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);

  if(im.getSize() == 1)
  {
    cuda_c_inplaceDivideScalar(this->getCudaArrayPtr(), im.getCudaArrayPtr(), tile.sz(), this->size());
  }
  else
  {
    ASSERT(this->isSameSize(im));
    cuda_c_inplaceDivideImages(this->getCudaArrayPtr(), im.getCudaArrayPtr(), tile.sz(), this->size());
  }
  return *this;
}


template <class T> inline
CudaImage<float> CudaImage<T>::operator+(const float val) const
{
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> result(this->getDims(),NO_INIT,this->getMemoryPolicy(),dev);

  cuda_c_addHostScalar(this->getCudaArrayPtr(), val, result.getCudaArrayPtr(), tile.sz(), this->size());

  return result;
}

template <class T> inline
CudaImage<float> CudaImage<T>::operator-(const float val) const
{
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> result(this->getDims(),NO_INIT,this->getMemoryPolicy(),dev);

  cuda_c_subtractHostScalar(this->getCudaArrayPtr(), val, result.getCudaArrayPtr(), tile.sz(), this->size());

  return result;
}

template <class T> inline
CudaImage<float> CudaImage<T>::operator*(const float val) const
{
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> result(this->getDims(),NO_INIT,this->getMemoryPolicy(),dev);

  cuda_c_multiplyHostScalar(this->getCudaArrayPtr(), val, result.getCudaArrayPtr(), tile.sz(), this->size());

  return result;
}

template <class T> inline
CudaImage<float> CudaImage<T>::operator/(const float val) const
{
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> result(this->getDims(),NO_INIT,this->getMemoryPolicy(),dev);

  cuda_c_divideHostScalar(this->getCudaArrayPtr(), val, result.getCudaArrayPtr(), tile.sz(), this->size());

  return result;
}


// For math operations with CudaImage arguments, size()==1 CudaImages are treated as scalars to allow for
// device side values to be applied as scalars

template <class T> inline
CudaImage<float> CudaImage<T>::operator+(const CudaImage<float>& im) const
{
  ASSERT(im.initialized());
  ASSERT(this->getMemoryDevice() == im.getMemoryDevice());
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> result(this->getDims(),NO_INIT,this->getMemoryPolicy(),dev);
  if(im.getSize() == 1)
  {
    cuda_c_addScalar(this->getCudaArrayPtr(), im.getCudaArrayPtr(), result.getCudaArrayPtr(), tile.sz(), this->size());
  }
  else
  {
    ASSERT(this->isSameSize(im));
    cuda_c_addImages(this->getCudaArrayPtr(), im.getCudaArrayPtr(), result.getCudaArrayPtr(), tile.sz(), this->size());
  }
  return result;
}

template <class T> inline
CudaImage<float> CudaImage<T>::operator-(const CudaImage<float>& im) const
{
  ASSERT(im.initialized());
  ASSERT(this->getMemoryDevice() == im.getMemoryDevice());
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> result(this->getDims(),NO_INIT,this->getMemoryPolicy(),dev);
  if(im.getSize() == 1)
  {
    cuda_c_subtractScalar(this->getCudaArrayPtr(), im.getCudaArrayPtr(), result.getCudaArrayPtr(), tile.sz(), this->size());
  }
  else
  {
    ASSERT(this->isSameSize(im));
    cuda_c_subtractImages(this->getCudaArrayPtr(), im.getCudaArrayPtr(), result.getCudaArrayPtr(), tile.sz(), this->size());
  }
  return result;
}

template <class T> inline
CudaImage<float> CudaImage<T>::operator*(const CudaImage<float>& im) const
{
  ASSERT(im.initialized());
  ASSERT(this->getMemoryDevice() == im.getMemoryDevice());
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> result(this->getDims(),NO_INIT,this->getMemoryPolicy(),dev);
  if(im.getSize() == 1)
  {
    cuda_c_multiplyScalar(this->getCudaArrayPtr(), im.getCudaArrayPtr(), result.getCudaArrayPtr(), tile.sz(), this->size());
  }
  else
  {
    ASSERT(this->isSameSize(im));
    cuda_c_multiplyImages(this->getCudaArrayPtr(), im.getCudaArrayPtr(), result.getCudaArrayPtr(), tile.sz(), this->size());
  }
  return result;
}

template <class T> inline
CudaImage<float> CudaImage<T>::operator/(const CudaImage<float>& im) const
{
  ASSERT(im.initialized());
  ASSERT(this->getMemoryDevice() == im.getMemoryDevice());
  const int dev = this->getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  CudaImage<float> result(this->getDims(),NO_INIT,this->getMemoryPolicy(),dev);
  if(im.getSize() == 1)
  {
    cuda_c_divideScalar(this->getCudaArrayPtr(), im.getCudaArrayPtr(), result.getCudaArrayPtr(), tile.sz(), this->size());
  }
  else
  {
    ASSERT(this->isSameSize(im));
    cuda_c_divideImages(this->getCudaArrayPtr(), im.getCudaArrayPtr(), result.getCudaArrayPtr(), tile.sz(), this->size());
  }
  return result;
}

// ######################################################################
template <class T> inline
CudaImage<T>::~CudaImage()
{ /* memory deallocation is handled by ArrayData's destructor */ }

// ######################################################################
template <class T> inline
void CudaImage<T>::freeMem()
{
  CudaImage<T> empty;
  this->swap(empty);
}

// ######################################################################
// ##### Memory management functions:
// ######################################################################

// ######################################################################
template <class T> inline
void CudaImage<T>::swap(CudaImage<T>& other)
{
  itsHdl.swap(other.itsHdl);
}

// ######################################################################
template <class T> inline
CudaImage<T> CudaImage<T>::deepcopy(const MemoryPolicy mp, const int dev) const
{
  // see comment in class definition for why deepcopy() might be
  // needed
  return CudaImage<T>(this->getCudaArrayPtr(), this->getDims(), mp, dev, this->getMemoryPolicy(), this->getMemoryDevice());
}


template <class T> inline
Image<T> CudaImage<T>::exportToImage() const
{
  if(getMemoryPolicy() == HOST_MEMORY)
    return Image<T>(getCudaArrayPtr(), getDims());
  else
  {
    Image<T> ret(getDims(),NO_INIT);
    ArrayHelper<T>::copy_initialize(ret.getArrayPtr(),size(),getCudaArrayPtr(),HOST_MEMORY,CUDA_HOST_DEVICE_NUM,getMemoryPolicy(),getMemoryDevice());
    return ret;
  }
}

// ######################################################################
template <class T> inline
void CudaImage<T>::resize(const Dims& dims, const bool do_clear)
{
  // This algorithm is slightly more convoluted than may appear necessary
  // at first, in order to be most efficient in the context of
  // copy-on-write. If the requested size is different than the current
  // size, it's very simple: we just make a new image of the right size,
  // clearing it if so requested. Otherwise, if we're keeping the same
  // size, we don't necessarily have to make a new data block; instead we
  // just delegate to clear(), which does the right thing depending on
  // whether we currently have a shared or unshared data block.

  if (dims != getDims())
    {
      ArrayHandle<T> resized(new ArrayData<T>(dims, do_clear ? ZEROS:NO_INIT, this->getMemoryPolicy(), this->getMemoryDevice()));
      itsHdl.swap(resized);
    }
  else // we're keeping the same size, so just clear() if necessary
    {
      if (do_clear) clear( 0 );
    }
}

// ######################################################################
template <class T> inline
void CudaImage<T>::resize(const int width, const int height, const bool do_clear)
{
  resize(Dims(width, height), do_clear);
}

// ######################################################################
// ##### Access functions:
// ######################################################################

// ######################################################################
template <class T> inline
bool CudaImage<T>::initialized() const
{ return getWidth() > 0 && getHeight() > 0; }

// ######################################################################
template <class T> inline
int CudaImage<T>::getSize() const
{ return getDims().sz(); }

// ######################################################################
template <class T> inline
uint CudaImage<T>::size() const
{ return getDims().sz(); }

// ######################################################################
template <class T> inline
int CudaImage<T>::getWidth() const
{ return itsHdl.get().w(); }

// ######################################################################
template <class T> inline
int CudaImage<T>::getHeight() const
{ return itsHdl.get().h(); }

// ######################################################################
template <class T> inline
const Dims& CudaImage<T>::getDims() const
{ return itsHdl.get().dims(); }

// ######################################################################
template <class T> inline
Rectangle CudaImage<T>::getBounds() const
{ return Rectangle(Point2D<int>(0,0), itsHdl.get().dims()); }

// ######################################################################
template <class T> template <class C> inline
bool CudaImage<T>::isSameSize(const C& other) const
{ return getWidth() == other.getWidth() && getHeight() == other.getHeight(); }

// ######################################################################
template <class T> inline
bool CudaImage<T>::is1D() const
{ return (getWidth() == 1) || (getHeight() == 1); }

// ######################################################################
template <class T> inline
bool CudaImage<T>::isVector() const
{ return (getWidth() == 1); }

// ######################################################################
template <class T> inline
bool CudaImage<T>::isTransposedVector() const
{ return (getHeight() == 1); }

// ######################################################################
template <class T> inline
bool CudaImage<T>::isSquare() const
{ return (getWidth() == getHeight()); }

// ######################################################################
template <class T> inline
const T* CudaImage<T>::getCudaArrayPtr() const
{
  return impl().data();
}

// ######################################################################
template <class T> inline
T* CudaImage<T>::getCudaArrayPtr()
{
  return uniq().dataw();
}

// ######################################################################
template <class T> inline
bool CudaImage<T>::coordsOk(const Point2D<int>& P) const
{
  return (P.i >= 0 && P.j >= 0 && P.i < getWidth() && P.j < getHeight());
}

// ######################################################################
template <class T> inline
bool CudaImage<T>::coordsOk(const int i, const int j) const
{
  return (i >= 0 && j >= 0 && i < getWidth() && j < getHeight());
}

// ######################################################################
template <class T> inline
bool CudaImage<T>::coordsOk(const Point2D<float>& p) const
{
  return this->coordsOk(p.i, p.j);
}

// ######################################################################
template <class T> inline
bool CudaImage<T>::coordsOk(const float i, const float j) const
{
  return (i >= 0.0F && j >= 0.0F &&
          i < float(getWidth() - 1) && j < float(getHeight() - 1));
}

// ######################################################################
template <class T> inline
bool CudaImage<T>::rectangleOk(const Rectangle& rect) const
{
  return (rect.left() < getWidth() && rect.rightI() < getWidth() &&
          rect.top() < getHeight() && rect.bottomI() < getHeight() &&
          rect.left() >= 0 && rect.rightI() >= 0 &&
          rect.top() >= 0 && rect.bottomI() >= 0);
}

// ######################################################################
template <class T> inline
MemoryPolicy CudaImage<T>::getMemoryPolicy() const
{
  return impl().getMemoryPolicy();
}

// ######################################################################
template <class T> inline
int CudaImage<T>::getMemoryDevice() const
{
  return impl().getMemoryDevice();
}


// ######################################################################
template <class T> inline
void CudaImage<T>::clear(const char &val)
{
  // Check if we have a shared implementation... if yes, then for
  // efficiency we should release our copy before doing the iterative
  // loop, which would otherwise unnecessarily make a unique copy of
  // the shared data for us, which we would then ceremoniously ignore
  // as we clear it to a new value.
  if (itsHdl.isShared())
    *this = CudaImage<T>(getDims(), NO_INIT, this->getMemoryPolicy(), this->getMemoryDevice());

  CudaDevices::memset(this->getCudaArrayPtr(),val,size()*sizeof(T),this->getMemoryDevice());
}

// ######################################################################
// ##### Functions for testing/debugging only:
// ######################################################################

template <class T> inline
bool CudaImage<T>::hasSameData(const CudaImage<T>& b) const
{ return itsHdl.hasSameData(b.itsHdl); }

// ######################################################################
template <class T> inline
long CudaImage<T>::refCount() const throw() { return itsHdl.refCount(); }

// ######################################################################
template <class T> inline
bool CudaImage<T>::isShared() const throw() { return itsHdl.isShared(); }

// ######################################################################
// ##### Private methods:
// ######################################################################

template <class T> inline
const ArrayData<T>& CudaImage<T>::impl() const
{ return itsHdl.get(); }

// ######################################################################
template <class T> inline
ArrayData<T>& CudaImage<T>::uniq()
{ return itsHdl.uniq(this->getMemoryPolicy(),this->getMemoryDevice()); }

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif
