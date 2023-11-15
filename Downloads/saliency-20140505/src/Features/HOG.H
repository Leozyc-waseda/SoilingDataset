/*!@file Features/HOG.H  Histogram Of Gradients */
//Code derived from http://www.cs.uchicago.edu/~pff/latent/
//author = "Felzenszwalb, P. F. and Girshick, R. B. and McAllester, D.",
//title = "Discriminatively Trained Deformable Part Models, Release 4",
//howpublished = "http://people.cs.uchicago.edu/~pff/latent-release4/"}

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
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
// Primary maintainer for this file: Lior Elazary
// $HeadURL$
// $Id$
//

#ifndef HOG_H_DEFINED
#define HOG_H_DEFINED

#include "Util/Types.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ImageSet.H"

#include <vector>
#include <string>

//TODO: clean this up, and support more ori
// unit vectors used to compute gradient orientation
const double itsUU[9] = {1.0000, 
  0.9397, 
  0.7660, 
  0.500, 
  0.1736, 
  -0.1736, 
  -0.5000, 
  -0.7660, 
  -0.9397};
const double itsVV[9] = {0.0000, 
  0.3420, 
  0.6428, 
  0.8660, 
  0.9848, 
  0.9848, 
  0.8660, 
  0.6428, 
  0.3420};

class HOG
{
  
public:
  //! Constructor
  HOG();

  //! Destructor
  virtual ~HOG();

  //! Compute HOG features on an RGB image
  ImageSet<float> getFeatures(const Image<PixRGB<byte> >& img, int numBins);

  //! Compute the gradient on a color img by taking the max gradient
  //! If numOrientations > 0 then snap to the best orientation
  void getMaxGradient(const Image<PixRGB<byte> >& img,
      Image<float>& mag, Image<float>& ori,
      int numOrientations = -1);

  ImageSet<float> getOriHistogram(const Image<float>& mag,
      const Image<float>& ori, int numOrientations, int numBins);

 
  ImageSet<double> computeFeatures(const ImageSet<float>& hist);
  
  Image<double> getHistogramEnergy(const ImageSet<float>& hist);
 
  
  Image<PixRGB<byte> > getHistogramImage(const ImageSet<float>& hist, const int binSize=20);
  
  
protected:

};


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
