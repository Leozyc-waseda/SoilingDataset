/*!@file Features/HistogramOfGradients.H  */


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
// Primary maintainer for this file: Daniel Parks <danielfp@usc.edu>
// $HeadURL$
// $Id$
//

#ifndef HistogramOfGradients_H_DEFINED
#define HistogramOfGradients_H_DEFINED

#include "Util/Types.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ImageSet.H"
#include "SIFT/FeatureVector.H"

#include <vector>
#include <string>

class JunctionHOG;
class HistogramOfGradients
{

private:
  // No static initializations are allowed for this class, use references instead (protect against slicing)
  HistogramOfGradients & operator=(const HistogramOfGradients & other);

public:
  //! Constructor
  HistogramOfGradients(bool normalize=true, Dims cellDims = Dims(8,8), bool fixedCells = false, int numOrientations=9, bool signedOrient=false);

  virtual HistogramOfGradients *clone () { return new HistogramOfGradients(*this);}

  //! Destructor
  virtual ~HistogramOfGradients();

  //! Calculate the gradient of 1-3 chanels of images
  virtual void calculateGradient(const Image<float>& img, const Image<float>& img2, const Image<float>& img3, Image<float>& gradmag, Image<float>&gradang);

  //! Calculate histogram using a gradient
  virtual std::vector<float> createHistogramFromGradient(const Image<float>& gradmag, const Image<float>& gradang);

  //! Create histogram of max of gradient across 1-3 channels of an image
  virtual std::vector<float> createHistogram(const Image<float>& img, const Image<float>& img2=Image<float>(), const Image<float>& img3=Image<float>());

  //! Normalize block using nearby blocks
  virtual std::vector<float> normalizeFeatureVector(FeatureVector featureVector);

  //! Force angle to conform to oriSigned option and add to histogram
  virtual void addToBin(const float xf, const float yf, const float ang_in, const float mag, const bool oriSigned, FeatureVector &fv);

protected:
  //! Whether to normalize columns in the histogram
  bool itsNormalize;
  //! If fixed dims is true, itsCellDims is the number of cells in the histogram whose cell size depends on the input image
  //! If fixed dims is false, itsCellDims is the size of an individual cell, and the number of cells depends on the input image
  Dims itsCellDims;  
  //! If true, fix the size of the histogram no matter the image size, if false, fix the size of the cell no matter the image size
  bool itsFixedDims;
  //! Number of bins to categorize orientations
  int  itsOriBins;
  bool itsOriSigned;
  float itsEpsilon;

};


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
