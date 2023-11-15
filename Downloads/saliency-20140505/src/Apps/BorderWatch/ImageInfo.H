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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Apps/BorderWatch/ImageInfo.H $
// $Id: ImageInfo.H 12962 2010-03-06 02:13:53Z irock $
//


#ifndef IMAGEINFO_H_DEFINED
#define IMAGEINFO_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelOptionDef.H"
#include "Component/ModelParam.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/PixelsTypes.H"
#include "Image/Point2D.H"
#include "Neuro/VisualCortexConfigurator.H"
#include "Simulation/SimModule.H"

#include <vector>

class SimEventQueue;

//! Monitor an image over time, to detect interesting events
class ImageInfo : public SimModule
{
public:
  //! Various statistical measures of interestingness returned by this object
  struct ImageStats
  {
    Image<float> smap;
    float saliency;
    Point2D<int> salpoint;
    float energy;
    float uniqueness;
    float entropy;
    float rand;
    float KLsurprise;
    float MSDsurprise;
    Image<float> belief1;
    Image<float> belief2;

    float score;  // compound score combining all above metrics
  };

  //! Constructor
  ImageInfo(OptionManager& mgr,
            const std::string& descrName = "ImageInfo", const std::string& tagName = "ImageInfo");

  //! Destructor
  virtual ~ImageInfo();

  // Recompute the score of the chip by passing it a new image to process
  ImageStats update(nub::ref<SimEventQueue>& q, const Image<PixRGB<byte> >& img, const int frameID);

private:
  // Apply a kalman filter to the input image and set a mu and sigma belief image
  float integrateData(const Image<float> &data, const float R, const float Q,
                      Image<float>& bel_mu, Image<float>& bel_sig);

  Image<float> itsBelief1Mu;    // Mean and variance maps over two timescales uses to compute
  Image<float> itsBelief1Sig;   //    the probability of a target
  Image<float> itsBelief2Mu;    //
  Image<float> itsBelief2Sig;   //

  // Compute a mean square difference between the previous saliency map and the current one
  float updateMSDiff(const Image<PixRGB<byte> >& img, Image<float> salMap);

  nub::ref<VisualCortexConfigurator> itsVcc;          // The visual cortex component

  Image<float> itsPrevSmap; // The previous saliency map - used by Bruce's MSDiff algorithm

  OModelParam<float> itsRandGain;         // The final score is computed as a sum of products of the
  OModelParam<float> itsEntropyGain;      //    individual metrics and these coefficients
  OModelParam<float> itsSaliencyGain;
  OModelParam<float> itsUniquenessGain;
  OModelParam<float> itsMSDSurpriseGain;
  OModelParam<float> itsKLSurpriseGain;
};

#endif


