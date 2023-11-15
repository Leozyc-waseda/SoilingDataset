/*!@file Beobot/ImageSpring.H derived from the image template class */
// all the pixels are linked to theirs neighbors with springs


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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/ImageSpring.H $
// $Id: ImageSpring.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef IMAGE_SPRING_H_DEFINED
#define IMAGE_SPRING_H_DEFINED

#include "Image/Image.H"
#include "Util/Assert.H"

template <class T> class PixRGB;

//! This class is some sort of self organizing map used for clustering
/*! All the pixels are linked by springs whose stiffness depends on the
    difference between them
*/
template<class T>
class ImageSpring : public Image<T>
{
public:
  //! Constructor that only allocates memory for given size
  inline ImageSpring(const int width, const int height,
                     const int nbNeighbors);

  //! Constructor for uninitialized image (useful for arrays of Images).
  /*! Use init() later to initialize */
  inline ImageSpring();

  //! Copy constructor
  inline ImageSpring(const ImageSpring<T>& A);

  //! init function
  inline void init(const int width, const int height,
                   const int nbNeighbors);

  //! init function, to copy another ImageSpring
  inline void init(const T* a, const float* posX, const float* posY,
                   const float** stiff,
                   const int width, const int height,
                   const int nbNeighbors);

  //! Destructor
  inline ~ImageSpring();

  //! Free memory and switch to uninitialized state
  inline void freeMem();

  //! initialize 'stiff' and 'pos'
  void initClustering(bool initPosMasses);

  //! compute the stiffness of the spring between the masses
  /*! based on the color-distance between pixels */
  void computeStiff(void);

  //! updates the positions of the masses
  /*! based on simple mechanics */
  void computePos(const float dt);

  //! computes mean and stdev (**heuristic**)
  /*! this function is a heuristic because it will only
    consider nbHeuristic pixels at random */
  void getStats(void);

  //! computes meanDist and stdevDist (**heuristic**)
  /*! weight must be defined
    this function is an heuristic because it will only
    consider nbHeuristic pairs of pixels at random */
  void getStatsDist(void);

  //! returns (X,Y) given an index=X+w*Y (see posX)
  inline void getXY(const int index, Point2D<int> &pt) const;

  //! returns the index=X+w*Y (see posX)
  inline void getIndex(const Point2D<int> point, int &index) const;

  //! returns the number of neighbors
  inline int getNbNeighbors(void) const;

  //! distance between the 2 masses
  float getDistanceMasses(const int index1, const int index2) const;

  //! computes the index of the nth neighbor of "index"
  /*! returns true iif the neighbor is in the image */
  inline bool getNeighbor(const int index, const int n,
                          int& indexNeighbor) const;

  //! Get position of masses and strength of springs (for debug)
  void getPositions(Image< PixRGB<byte> >& img, const int zoom = 16);


  //! get clustered image and track centroid position
  void getClusteredImage(const Image< PixRGB<byte> > &scene,
                         Image< PixRGB<byte> > &clusteredImage,
                         Point2D<int> &supposedTrackCentroid,
                         const Point2D<int>& previousTrackCentroid);

private:
  //! subfunction find n-connex graph
  /*! n connex sub-graph are the subgraph whose nodes are connected to
   one another by at least n links. The nodes already marked have non 0
  values in 'marked', the current node is 'currentPoint' and the current label
  (i.e. number of the group) is 'color'. 'begin' is used to initialize the
  algorithm because if we apply n-connexity from the beginng then obviously
  no node will qualify as the first one we see has 0 neighbors... */
  void goGraph(Image<int32> &marked, const int currentIndex,
               const int32 color, const int begin = 3);

  //! defines the neighborhhod topology (5, 9 or 25)
  /*! see "getNeighbor" for indexing */
  int nbNeighbors;

  //! mean and stdev of the image
  /*! not necessarily scalar, don't confuse with meanDist, stdevDist */
  T mean, stdev;

  //! mean and stdev of the distance between meighbor pixels
  /*! always scalar by definition of  adistance...
    don't be confused, it's NOT the mean and stdev of the image */
  double meanDist, stdevDist;

  //! to define weightedDistance() between pixels
  /*! this is normally stdev.inverse(), this way the system is
      completly independent from the dynamic range of the low
      level features */
  T weight;

  //! physical position of the "masses" at time t
  /*! 2D array represented as 1D array: index = x+y*nX */
  float *posX, *posY;

  //! physical position of the "masses" at time t-dt.
  float *oldPosX, *oldPosY;

  /*! stiff[index][n] is the stiffness of the spring between index and
    its nth neighbor. Note: we consider that length is 0 for all
    springs ! */
  float **stiff;

  //! computes weight given mean & stdev
  void computeWeight(void);

  //! sets the "physical" position of mass at (x,y)
  inline void setPos(const Point2D<int> mass, const float x, const float y);

  //! sets the stiffness of the spring
  inline void setStiff(const Point2D<int> mas, int neighboor, float stiff);

  //! the moments will be estimated with nbHeuristic outcomes
  const static int nbHeuristic = 500;
};

// ######################################################################
// ######################################################################
// ######################################################################
// INLINE FUNCTIONS:
// ######################################################################
// ######################################################################
// ######################################################################

// ######################################################################
template<class T> inline
ImageSpring<T>::ImageSpring(const int width, const int height,
                            const int nbN) :
  Image<T>(width, height, NO_INIT)
{
  init(width, height, nbN);
}

// ######################################################################
template<class T> inline
ImageSpring<T>::ImageSpring() : Image<T>()
{ posX = NULL; posY = NULL; oldPosX = NULL; oldPosY = NULL; stiff = NULL; }

// ######################################################################
template<class T> inline
ImageSpring<T>::ImageSpring(const ImageSpring<T>& A) : Image<T>(A)
{
  ASSERT(A.initialized());
  posX = NULL; posY = NULL; oldPosX = NULL; oldPosY = NULL; stiff = NULL;

  init(A.getArrayPtr(), posX, posY, const_cast<const float**>(A.stiff),
       A.getWidth(), A.getHeight(), A.nbNeighbors);
}

// ######################################################################
template<class T> inline
ImageSpring<T>::~ImageSpring()
{ freeMem(); }

// ######################################################################
template<class T> inline
void ImageSpring<T>::freeMem()
{
  if (stiff) {
    for (int index = 0;index < this->getSize(); index++) delete [] stiff[index];
    delete [] stiff; stiff = NULL;
  }
  if (posX) { delete [] posX; posX = NULL; }
  if (posY) { delete [] posY; posY = NULL; }
  if (oldPosX) { delete [] oldPosX; oldPosX = NULL; }
  if (oldPosY) { delete [] oldPosY; oldPosY = NULL; }
  Image<T>::freeMem();
}

// ######################################################################
template<class T> inline
void ImageSpring<T>::init(const T* inarray,
                          const float* inPosX, const float* inPosY,
                          const float** inStiff,
                          const int width, const int height,
                          const int nbN)
{
  freeMem();  // delete any allocated memory
  this->resize(width, height);  // allocate underlying Image<T>
  nbNeighbors = nbN; int size = this->getSize();

  // allocate memory for the mass positions:
  posX = new float[size]; memcpy(posX, inPosX, size * sizeof(float));
  posY = new float[size]; memcpy(posY, inPosY, size * sizeof(float));
  oldPosX = new float[size]; memcpy(oldPosX, inPosX, size * sizeof(float));
  oldPosY = new float[size]; memcpy(oldPosY, inPosY, size * sizeof(float));

  // the neighbor #0 is the grid point, so there are nbN+1 neighbors
  typedef float* floatptr;
  stiff = new floatptr[size];
  for (int index = 0; index < size; index++)
    {
      stiff[index] = new float[nbN + 1];
      memcpy(stiff[index], inStiff[index], (nbN + 1) * sizeof(float));
    }
}

// ######################################################################
template<class T> inline
void ImageSpring<T>::init(const int width, const int height,
                          const int nbN)
{
  freeMem();  // delete any allocated memory
  this->resize(width, height);  // allocate underlying Image<T>
  nbNeighbors = nbN; int size = this->getSize();

  // allocate memory for the mass positions:
  posX = new float[size]; posY = new float[size];
  oldPosX = new float[size]; oldPosY = new float[size];

  typedef float* floatptr;
  stiff = new floatptr[size];
  for (int index = 0; index < size; index++) stiff[index] = new float[nbN + 1];
}

// ######################################################################
template<class T> inline
void ImageSpring<T>::getXY( const int index, Point2D<int> &pt ) const
{ pt.i = index % this->getWidth(); pt.j = index / this->getWidth(); }

// ######################################################################
template<class T> inline
void ImageSpring<T>::getIndex( const Point2D<int> point, int &index ) const
{ index = point.i + point.j * this->getWidth(); }

// ######################################################################
template<class T> inline
bool ImageSpring<T>::getNeighbor( const int index, const int n,
                                  int& indexNeighbor ) const
{
  ASSERT(n > 0 && n <= nbNeighbors);

  Point2D<int> pt; getXY(index, pt);

  switch(n)
    {
    case 0 : break;   // to the grid

      // 4-neighbors
    case 1 : pt.i--;         break;
    case 2 :         pt.j--; break;
    case 3 : pt.i++;         break;
    case 4 :         pt.j++; break;

      // 8-neighbors
    case 5 : pt.i--; pt.j--; break;
    case 6 : pt.i++; pt.j--; break;
    case 7 : pt.i--; pt.j++; break;
    case 8 : pt.i++; pt.j++; break;

      // 24-neighbors
    case 9 : pt.i-=2; pt.j+=2; break;
    case 10: pt.i-=1; pt.j+=2; break;
    case 11:          pt.j+=2; break;
    case 12: pt.i+=1; pt.j+=2; break;
    case 13: pt.i+=2; pt.j+=2; break;
    case 14: pt.i+=2; pt.j+=1; break;
    case 15: pt.i+=2;          break;
    case 16: pt.i+=2; pt.j-=1; break;
    case 17: pt.i+=2; pt.j-=2; break;
    case 18: pt.i+=1; pt.j-=2; break;
    case 19:          pt.j-=2; break;
    case 20: pt.i-=1; pt.j-=2; break;
    case 21: pt.i-=2; pt.j-=2; break;
    case 22: pt.i-=2; pt.j-=1; break;
    case 23: pt.i-=2;          break;
    case 24: pt.i-=2; pt.j+=1; break;

      // larger than that is unsupported:
    default:
      LFATAL("Neighbor out or range");
    }

  getIndex( pt, indexNeighbor );

  return this->coordsOk( pt );
}

// ######################################################################
template<class T> inline
int ImageSpring<T>::getNbNeighbors( void ) const
{ return nbNeighbors; }

// ######################################################################
template<class T> inline
void ImageSpring<T>::setPos(const Point2D<int> mass, const float x, const float y)
{
  ASSERT(this->initialized() && this->coordsOk(mass));
  int index; getIndex(mass, index);
  posX[index] = x; posY[index] = y;
}

// ######################################################################
template<class T>
void ImageSpring<T>::setStiff(const Point2D<int> mass, int neighbor, float stif)
{
  ASSERT(this->initialized() && this->coordsOk(mass) && neighbor <= nbNeighbors);
  int index; getIndex(mass, index);
  stiff[index][neighbor] = stif;
}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif
