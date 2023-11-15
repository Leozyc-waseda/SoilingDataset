/*!@file Features/HistogramOfGradients.C  */


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

#include "Features/HistogramOfGradients.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Kernels.H"
#include "Image/CutPaste.H"
#include "Image/ColorOps.H"
#include "Image/FilterOps.H"
#include "Image/ShapeOps.H"


HistogramOfGradients::HistogramOfGradients(bool normalize, Dims cellDims, bool fixedCells, int numOrientations, bool signedOrient) :
itsNormalize(normalize),
itsCellDims(cellDims),
itsFixedDims(fixedCells),
itsOriBins(numOrientations),
itsOriSigned(signedOrient),
itsEpsilon(0.0001F)
{
  
}

HistogramOfGradients::~HistogramOfGradients()
{
}

std::vector<float> HistogramOfGradients::createHistogram(const Image<float>& img, const Image<float>& img2, const Image<float>& img3)
{
  Image<float> gradmag, gradang;
  calculateGradient(img,img2,img3,gradmag,gradang);
  return createHistogramFromGradient(gradmag,gradang);
}


void HistogramOfGradients::calculateGradient(const Image<float>& img, const Image<float>& img2, const Image<float>& img3, Image<float>& gradmag, Image<float>&gradang)
{

  gradientSobel(img, gradmag, gradang);

  if(img2.initialized())
    {
      Image<float> gradmag2, gradang2;
      // Calculate Sobel for image 2
      gradientSobel(img2,gradmag2,gradang2);
      // Take element-wise max of magnitudes and use that to choose angles, store into last arguments 
      takeLinkedMax(gradmag,gradmag2,gradang,gradang2,gradmag,gradang);
    }

  if(img3.initialized())
    {
      Image<float> gradmag3, gradang3;
      // Calculate Sobel for image 2
      gradientSobel(img3,gradmag3,gradang3);
      // Determine maximum filter per orientation
      takeLinkedMax(gradmag,gradmag3,gradang,gradang3,gradmag,gradang);
    }
  
}

std::vector<float> HistogramOfGradients::createHistogramFromGradient(const Image<float>& gradmag, const Image<float>& gradang)
{
  // Determine the number of cells
  
  Dims cells;
  if(itsFixedDims)
    cells = itsCellDims;
  else
    cells = Dims(int(round(float(gradmag.getWidth())/itsCellDims.w())),int(round(float(gradmag.getHeight())/itsCellDims.h())));

  FeatureVector fv = FeatureVector(cells.w(),cells.h(),itsOriBins);

 // Scan image and we will cumulate local samples into a "cells" sized grid
  // of bins, with interpolation.
  for (int rx=1; rx<gradmag.getWidth()-1; rx++)
    for (int ry=1; ry<gradmag.getHeight()-1; ry++)
      {
        if(!gradmag.coordsOk(rx,ry)) // outside image
          continue;
        // Get bin fractions
        const float xf = float(rx)/float(gradmag.getWidth())*cells.w();
        const float yf = float(ry)/float(gradmag.getHeight())*cells.h();

      float gradMag = gradmag.getValInterp(rx, ry);
      float gradAng = gradang.getValInterp(rx, ry);

      // will be interpolated into cells.w x cells.h x itsOriBins:
      addToBin(xf,yf,gradAng,gradMag,itsOriSigned,fv);

      }

  std::vector<float> returnVec;
  // normalize bins
  if(itsNormalize)
    {
      LINFO("Normalizing feature vector");
      returnVec = normalizeFeatureVector(fv);
    }
  else
    returnVec = fv.getFeatureVector();
  LINFO("Calculated HOG of size %Zu",returnVec.size());
  return returnVec;

}

std::vector<float> HistogramOfGradients::normalizeFeatureVector(FeatureVector featureVector)
{
  std::vector<float> fv = featureVector.getFeatureVector();
  const int xsize = featureVector.getXSize();
  const int ysize = featureVector.getYSize();
  const int zsize = featureVector.getZSize();
  // Output is shrunk by number of cells in the block used in normalization
  const int newxsize = xsize - 2;
  const int newysize = ysize - 2;
  // If we have too small of a histogram, we can't normalize
  if(newxsize < 1 || newysize < 1)
    return fv;
  std::vector<float>::const_iterator ofv=fv.begin();
  std::vector<float> distfv=std::vector<float>(xsize*ysize);
  std::vector<float>::iterator dfv=distfv.begin();
  // Calculate L2 Norm ||v||^2 which will be used to update vector v like so: v=sqrt(||v||^2+epsilon)
  for(int x=0;x<xsize;x++)
    for(int y=0;y<ysize;y++)
      {
	float dSq=0;
	for(int z=0;z<zsize;z++)
	  {
	    // Increment original vector
	    dSq += *ofv * *ofv;
	    ofv++;
	  }
	*(dfv++)=dSq;
      }
  dfv=distfv.begin();
  ofv=fv.begin();
  std::vector<float> newfv=std::vector<float>(newxsize*newysize*zsize*4);
  std::vector<float>::iterator nfv=newfv.begin();
  const int h=newysize;
  for(int x=0;x<xsize;x++)
    for(int y=0;y<ysize;y++)
      {
        if(x<newxsize && y<newysize)
          {
            // Upper left normalization block
            float n1 = sqrt(*dfv + *(dfv+h) + *(dfv+1) + *(dfv+h+1) + itsEpsilon);
            // Upper right normalization block
            float n2 = sqrt(*(dfv+h) + *(dfv+2*h) + *(dfv+h+1) + *(dfv+2*h+1) + itsEpsilon);
            // Lower left normalization block
            float n3 = sqrt(*(dfv+1) + *(dfv+h+1) + *(dfv+2) + *(dfv+h+2) + itsEpsilon);
            //Lower right normalization block
            float n4 = sqrt(*(dfv+h+1) + *(dfv+2*h+1) + *(dfv+h+2) + *(dfv+2*h+2) + itsEpsilon);
            
            for(int z=0;z<zsize;z++)
              {
                //Increment new vector
                *(nfv++)=*ofv/n1;
                *(nfv++)=*ofv/n2;
                *(nfv++)=*ofv/n3;
                *(nfv++)=*ofv/n4;
                ofv++;
              }
            // Increment distance vector
            dfv++;
          }
        else
          {
            ofv+=zsize;
            dfv++;
          }
      }
  
  return newfv;
}

void HistogramOfGradients::addToBin(const float xf, const float yf, const float ang_in, const float mag, const bool oriSigned, FeatureVector &fv)
{
  float ang=ang_in;
  float oriBin;
  int numBins = fv.getZSize();
  if(oriSigned)
    {
      // ensure that angle is within -2*M_PI to 2*M_PI
      ang=fmod(ang,2*M_PI);
      // convert from -2*M_PI:2*M_PI to 0:2*M_PI
      if (ang < 0.0)
	ang += 2*M_PI; 
      oriBin = ang / 2.0 / M_PI * float(numBins);
    }
  else
    {
      // ensure that angle is within -M_PI to M_PI
      ang=fmod(ang,M_PI);
      // convert from -M_PI:M_PI to 0:M_PI
      if (ang < 0.0)
	ang += M_PI; 
      oriBin = ang / M_PI * float(numBins);
    }
  fv.addValue(xf, yf, oriBin, mag);
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
