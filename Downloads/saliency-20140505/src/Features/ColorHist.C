/*!@file Features/ColorHist.C  */


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

#include "Features/ColorHist.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Kernels.H"
#include "Image/CutPaste.H"
#include "Image/ColorOps.H"
#include "Image/FilterOps.H"
#include "Image/ShapeOps.H"
#include "SIFT/FeatureVector.H"


ColorHist::ColorHist()
{
}

ColorHist::~ColorHist()
{
}

std::vector<float> ColorHist::createFeatureVector(const Image<PixRGB<byte> >& img)
{

  if (!img.initialized())
    return std::vector<float>();

  Image<float>  lum,rg,by;
  getLAB(img, lum, rg, by);

  //Smooth the image
  Image<float> kernel = gaussian<float>(0.0F, 1.4, lum.getWidth(),1.0F);
  Image<float> kernel2 = gaussian<float>(0.0F, 1.4, lum.getWidth(),1.0F);
  Image<float> kernel3 = gaussian<float>(0.0F, 1.4, lum.getWidth(),1.0F);
  //// do the convolution:
  lum = sepFilter(lum, kernel, kernel, CONV_BOUNDARY_CLEAN);
  rg = sepFilter(rg, kernel2, kernel2, CONV_BOUNDARY_CLEAN);
  by = sepFilter(by, kernel3, kernel3, CONV_BOUNDARY_CLEAN);


  std::vector<float> lumFv = createFeatureVector(lum);
  std::vector<float> rgFv = createFeatureVector(rg);
  std::vector<float> byFv = createFeatureVector(by);

  std::vector<float> fv = lumFv;
  fv.insert(fv.end(), rgFv.begin(), rgFv.end());
  fv.insert(fv.end(), byFv.begin(), byFv.end());
  
  return fv;

}

std::vector<float> ColorHist::createFeatureVector(const Image<float>& img)
{
  FeatureVector fv;
  // check this scale
  const int radius = std::min(img.getWidth()/2, img.getHeight()/2);
  //const float gausssig = float(radius); // 1/2 width of descript window
  //const float gaussfac = -0.5f/(gausssig * gausssig);

  const int xi = img.getWidth()/2;
  const int yi = img.getHeight()/2;

  Image<float> gradmag, gradorie;
  gradientSobel(img, gradmag, gradorie);

  // Scan a window of diameter 2*radius+1 around the point of
  // interest, and we will cumulate local samples into a 4x4 grid
  // of bins, with interpolation. NOTE: rx and ry loop over a
  // square that is assumed centered around the point of interest.
  for (int ry = -radius; ry < radius; ry++)
    for (int rx = -radius; rx < radius; rx++)
    {
      // get the coords in the image frame of reference:
      const float orgX = rx + float(xi);
      const float orgY = ry + float(yi);

      if (! gradmag.coordsOk(orgX, orgY)) // outside image
        continue; // forget this coordinate

      // find the fractional coords of the corresponding bin
      // (we subdivide our window into a 4x4 grid of bins):
      const float xf = 2.0f + 2.0f * float(rx)/float(radius);
      const float yf = 2.0f + 2.0f * float(ry)/float(radius);

      // find the Gaussian weight from distance to center and
      // get weighted gradient magnitude:
      const float gaussFactor = 1; //expf((rx*rx + ry*ry) * gaussfac);
      const float weightedMagnitude =
        gaussFactor * gradmag.getValInterp(orgX, orgY);

      // get the gradient orientation relative to the keypoint
      // orientation and scale it for 8 orientation bins:
      float gradAng = gradorie.getValInterp(orgX, orgY);
      gradAng=fmod(gradAng, 2*M_PI); //bring the range from 0 to M_PI

      //convert from -M_PI to M_PI
      if (gradAng < 0.0)
        gradAng += 2*M_PI; //convert to -M_PI to M_PI
      if (gradAng >= M_PI)
        gradAng -= 2*M_PI;
      //split to eight bins
      const float orient = (gradAng + M_PI) * 8 / (2 * M_PI);

      // will be interpolated into 2 x 2 x 2 bins:
      fv.addValue(xf, yf, orient, weightedMagnitude);
    }

  return fv.getFeatureVector();

}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
