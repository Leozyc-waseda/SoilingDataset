/*!@file MBARI/FOEestimator.H a class for estimating the focus of expansion
  in a video
 */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/FOEestimator.H $
// $Id: FOEestimator.H 4663 2005-06-23 17:47:28Z rjpeters $
//

#ifndef FOEESTIMATOR_H_DEFINED
#define FOEESTIMATOR_H_DEFINED

#include "Image/ImageCache.H"
#include "MBARI/Geometry2D.H"
#include "Util/Types.H"

// ######################################################################
//! compute the focus of expansion (FOE) from the pixel-based optical flow
class FOEestimator
{
public:

  //! constructor
  /*!@param numAvg the number of frames used for averaging to get a smooth estimate
    @param pyramidLevel the subsampling level at which to compute the optical flow*/
  FOEestimator(int numAvg = 20, int pyramidLevel = 0);

  // default =, copy and destr. okay

  //! update with a new frame of the movie sequence
  /*!@return if the cache contains less than three frames, an invalid Vector2D
    is returned, otherwise the estimated location of the FOE is returned*/
  Vector2D updateFOE(const Image<byte>& img, const Rectangle region);

  //! update with a new frame of the movie sequence
  /*!@return if the cache contains less than three frames, an invalid Vector2D
    is returned, otherwise the estimated location of the FOE is returned*/
  Vector2D updateFOE(const Image<byte>& img);

  //! returns the last estimate of the FOA
  Vector2D getFOE();

private:
  float getZeroCrossing(const Image<float>& vec);

  const int itsPyrLevel;
  ImageCache<byte> itsFrames;
  ImageCacheAvg<float> itsXvectors, itsYvectors;
  Vector2D itsFOE;
};


#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
