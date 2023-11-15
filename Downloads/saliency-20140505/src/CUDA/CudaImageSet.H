/*!@file CUDA/CudaImageSet.H */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaImageSet.H $
// $Id: CudaImageSet.H 12962 2010-03-06 02:13:53Z irock $
//

#ifndef CUDAIMAGESET_H_DEFINED
#define CUDAIMAGESET_H_DEFINED

#include "CUDA/CudaImage.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Util/Assert.H"

#include <deque>

// ######################################################################
//! This class implements a set of images, often used as a dyadic pyramid.
template <class T>
class CudaImageSet
{
public:
  //! Construct with a given number of empty images.
  inline CudaImageSet(const uint n=0);

  //! Construct with a given number of images of the given size.
  inline CudaImageSet(const uint n, const Dims& dims, const MemoryPolicy mp, const int dev);

  //! Construct with a given number of images, size and type of initialization
  inline CudaImageSet(const uint n, const Dims& dims, const InitPolicy init, const MemoryPolicy mp, const int dev);

  inline CudaImageSet(const ImageSet<T>& cp, const MemoryPolicy mp, const int dev);

  // default desructor OK

  //! Copy Constructor
  inline CudaImageSet(const CudaImageSet<T>& cp);

  //! Copy Constructor with different memory location
  inline CudaImageSet(const CudaImageSet<T>& cp, const MemoryPolicy mp, const int dev);

  //! Swap contents with another ImageSet
  inline void swap(CudaImageSet<T>& that);

  //! Keep the same number of images, but make all images empty (0-by-0 size).
  /*! This returns the ImageSet to a state where isEmpty() is true. */
  inline void clear();

  //! Reset to a new number of images, all images will be empty (0-by-0 size).
  inline void reset(uint newDepth = 0);

  //! Return number of images in image set.
  inline uint size() const;

  //! Get image from a given level.
  inline const CudaImage<T>& getImage(const uint lev) const;

  //! Get image from a given level (shorthand for getImage()).
  inline const CudaImage<T>& operator[](const uint lev) const;

  //! Get mutable image from a given level.
  inline CudaImage<T>& getImageMut(const uint lev);

  //! Get mutable image from a given level (shorthand for getImageMut()).
  inline CudaImage<T>& operator[](const uint lev);

  //! push an image to the back of the set
  inline void push_back(CudaImage<T>& img);

  //! push an image to the front of the set
  inline void push_front(CudaImage<T>& img);

  //! return the last image in the set and pop it off
  inline CudaImage<T> pop_back();

  //! return the first image in the set and pop it off
  inline CudaImage<T> pop_front();

  //! return the last image in the set (without popping it off)
  inline CudaImage<T> front() const;

  //! return the first image in the set (without popping it off)
  inline CudaImage<T> back() const;

  //! Return true if the pyramid has no non-empty images (width*height > 0).
  inline bool isEmpty() const;

  //! Return true if the pyramid has any non-empty images (width*height > 0).
  inline bool isNonEmpty() const;

  //! Return a new ImageSet with images in the half-open range [a,b[
  inline CudaImageSet<T> subSet(const uint a, const uint b) const;

private:
  std::deque< CudaImage<T> > itsLevels;
};



// ######################################################################
// ######################################################################
// ##### INLINED FUNCTIONS:
// ######################################################################
// ######################################################################

// ######################################################################
template <class T> inline
CudaImageSet<T>::CudaImageSet(const uint n)
  : itsLevels(n, CudaImage<T>())
{}

// ######################################################################
template <class T> inline
CudaImageSet<T>::CudaImageSet(const uint n, const Dims& dims, const MemoryPolicy mp, const int dev)
: itsLevels(n, CudaImage<T>(dims,mp,dev))
{}

// ######################################################################
template <class T> inline
CudaImageSet<T>::CudaImageSet(const ImageSet<T>& cp, const MemoryPolicy mp, const int dev)
  : itsLevels(cp.size())
{
  for (uint i = 0; i < cp.size(); ++i)
    this->itsLevels[i] = CudaImage<T>(cp[i],mp,dev);
}


// ######################################################################
template <class T> inline
CudaImageSet<T>::CudaImageSet(const uint n, const Dims& dims, const InitPolicy init, const MemoryPolicy mp, const int dev)
: itsLevels(n, CudaImage<T>(dims,init,mp,dev))
{}

// ######################################################################
template <class T> inline
CudaImageSet<T>::CudaImageSet(const CudaImageSet<T>& cp)
: itsLevels(cp.size())
{
  for (uint i = 0; i < cp.size(); ++i)
    this->itsLevels[i] = CudaImage<T>(cp.itsLevels[i]);
}

// ######################################################################
template <class T> inline
CudaImageSet<T>::CudaImageSet(const CudaImageSet<T>& cp, const MemoryPolicy mp, const int dev)
: itsLevels(cp.size())
{
  for (uint i = 0; i < cp.size(); ++i)
    this->itsLevels[i] = CudaImage<T>(cp.itsLevels[i],mp,dev);
}

// ######################################################################
template <class T> inline
void CudaImageSet<T>::swap(CudaImageSet<T>& that)
{
  this->itsLevels.swap(that.itsLevels);
}

// ######################################################################
template <class T> inline
void CudaImageSet<T>::clear()
{
  // Reset to an empty vector of given size by swapping with a temp vector
  std::deque< CudaImage<T> >(itsLevels.size()).swap(itsLevels);
}

// ######################################################################
template <class T> inline
void CudaImageSet<T>::reset(uint newDepth)
{
  // Reset to an empty vector of given size by swapping with a temp vector
  std::deque< CudaImage<T> >(newDepth).swap(itsLevels);
}

// ######################################################################
template <class T> inline
uint CudaImageSet<T>::size() const { return itsLevels.size(); }

// ######################################################################
template <class T> inline
const CudaImage<T>& CudaImageSet<T>::getImage(const uint lev) const
{
  ASSERT(lev < itsLevels.size());
  return itsLevels[lev];
}

// ######################################################################
template <class T> inline
const CudaImage<T>& CudaImageSet<T>::operator[](const uint lev) const
{ return getImage(lev); }

// ######################################################################
template <class T> inline
CudaImage<T>& CudaImageSet<T>::getImageMut(const uint lev)
{
  ASSERT(lev < itsLevels.size());
  return itsLevels[lev];
}

// ######################################################################
template <class T> inline
CudaImage<T>& CudaImageSet<T>::operator[](const uint lev)
{ return getImageMut(lev); }

// ######################################################################
template <class T> inline
void CudaImageSet<T>::push_back(CudaImage<T>& img)
{
  itsLevels.push_back(img);
}

// ######################################################################
template <class T> inline
void CudaImageSet<T>::push_front(CudaImage<T>& img)
{
  itsLevels.push_front(img);
}

// ######################################################################
template <class T> inline
CudaImage<T> CudaImageSet<T>::pop_back()
{
  ASSERT(isNonEmpty());
  CudaImage<T> tmp = itsLevels.back();
  itsLevels.pop_back();
  return tmp;
}

// ######################################################################
template <class T> inline
CudaImage<T> CudaImageSet<T>::pop_front()
{
  ASSERT(isNonEmpty());
  CudaImage<T> tmp = itsLevels.front();
  itsLevels.pop_front();
  return tmp;
}

// ######################################################################
template <class T> inline
CudaImage<T> CudaImageSet<T>::front() const
{
  ASSERT(isNonEmpty());
  return itsLevels.front();
}

// ######################################################################
template <class T> inline
CudaImage<T> CudaImageSet<T>::back() const
{
  ASSERT(isNonEmpty());
  return itsLevels.back();
}

// ######################################################################
template <class T> inline
bool CudaImageSet<T>::isEmpty() const { return !isNonEmpty(); }

// ######################################################################
template <class T> inline
bool CudaImageSet<T>::isNonEmpty() const
{
  for (uint i = 0; i < itsLevels.size(); ++i)
    if (itsLevels[i].initialized()) return true;

  return false;
}

// ######################################################################
template <class T> inline
CudaImageSet<T> CudaImageSet<T>::subSet(const uint a, const uint b) const
{
  ASSERT(b >= a);
  ASSERT(b <= this->size());

  CudaImageSet<T> result(b-a);
  for (uint i = a; i < b; ++i)
    result.itsLevels[i-a] = this->itsLevels[i];

  return result;
}

// Explicit Instantiations

template class CudaImageSet<float>;

#endif // !CUDAIMAGESET_H_DEFINED

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
