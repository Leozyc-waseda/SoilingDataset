/*!@file RCBot/Motion/MotionEnergy.H detect motion in an image stream */

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
// Primary maintainer for this file: Lior Elazary <lelazary@yahoo.com>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/Motion/MotionEnergy.H $
// $Id: MotionEnergy.H 8429 2007-05-25 18:29:40Z rjpeters $
//

#ifndef MotionEnergy_H
#define MotionEnergy_H

#include "Image/PyramidOps.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/DrawOps.H"
#include "Image/PyramidTypes.H"
#include "Image/PyrBuilder.H"
#include "Image/MathOps.H"

#include <math.h>

// ######################################################################
//! This class implements Spatiotemporal Energy motion detections
/*! The class uses the filter results of pyramids of other types
  to implement Spatiotemporal Energy motion detection. The motion is extracted
  by finding edge orientations in the time and space domain. The edge
  orientation are computed using sobel edge dectors. */

template <class T>
class MotionEnergyPyrBuilder : public PyrBuilder<T>
{
public:
  // ######################################################################
  // #### Constructors
  // ######################################################################
  MotionEnergyPyrBuilder(const PyramidType typ,
                         const float gabor_theta = 0.0F,
                         const float intens = 10.0F,
                         const int timeDomainSize = 5,
                         const float magThreshold = 1500);

  //! default copy constructor and assignment are okay
  //! std::vector and std::deque handle the copying of the individual images

  //! cloning constructor
  virtual MotionEnergyPyrBuilder<T>* clone() const;

  //! Builds a Motion Energy pyramid based on a image
  virtual ImageSet<T> build(const Image<T>& image,
                            const int firstlevel,
                            const int depth,
                            PyramidCache<T>* cache = 0);

  void updateMotion(const Image<T>& image, const int depth);

  Image<float> buildHorizontalMotionLevel(int lev);
  Image<float> buildVerticalMotionLevel(int lev);

  ImageSet<float> buildHorizontalMotion();
  ImageSet<float> buildVerticalMotion();

  //! reset the stored queues - overrides PyrBuilder<T>::reset()
  virtual void reset();

  //! show the motion image as vectors (use for debuging)
  //! return the average motion
  float DrawVectors(Image<T> &img, Image<float> &motion);

private:

  //! get the motion from the image by using sobel grad detector
  //! to detect edge orientations in the space and time
  void getMotion(const Image<T> &img, Image<float> &motion, int y_pos);

  std::deque< ImageSet<T> > imgPyrQ; // hold the time domain pyramid
  PyramidType itsPtype;              // Gaussian, Laplacian, Gabor, etc.
  float itsGaborAngle;               // orientation tuning for Gabor pyr
  float itsGaborIntens;              // filter strength for Gabor pyr
  unsigned int itsTimeDomainSize;    // the number of frames to hold in imgPyrQ
  float itsMagThreshold;             // only accept mag over this threshold
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
