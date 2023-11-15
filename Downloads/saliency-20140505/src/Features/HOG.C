/*!@file Features/HOG.C */


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

#include "Features/HOG.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Kernels.H"
#include "Image/CutPaste.H"
#include "Image/ColorOps.H"
#include "Image/FilterOps.H"
#include "Image/ShapeOps.H"
#include "SIFT/FeatureVector.H"
#include "GUI/DebugWin.H"

#include <stdio.h>

HOG::HOG()
{
}

HOG::~HOG()
{
}

ImageSet<float> HOG::getFeatures(const Image<PixRGB<byte> >& img, int numBins)
{
  int itsNumOrientations = 18;

  Image<float> mag, ori;
  getMaxGradient(img, mag, ori, itsNumOrientations);

  ImageSet<float> histogram = getOriHistogram(mag, ori, itsNumOrientations, numBins);

  ImageSet<double> features = computeFeatures(histogram);

  return features;
}

Image<PixRGB<byte> > HOG::getHistogramImage(const ImageSet<float>& hist, const int lineSize)
{
  if (hist.size() == 0)
    return Image<PixRGB<byte> >();


  Image<float> img(hist[0].getDims()*lineSize, ZEROS);
  //Create a one histogram with the maximum features for the 9 orientations
  //TODO: features need to be separated
  for(uint feature=0; feature<9; feature++)
  {
    float ori = (float)feature/M_PI + (M_PI/2);
    for(int y=0; y<hist[feature].getHeight(); y++)
    {
      for(int x=0; x<hist[feature].getWidth(); x++)
      {
        float histVal = hist[feature].getVal(x,y);

        //TODO: is this redundant since the first 9 features are
        //contained in the signed 18 features?
        if (hist[feature+9].getVal(x,y) > histVal)
          histVal = hist[feature+9].getVal(x,y);
        if (hist[feature+18].getVal(x,y) > histVal)
          histVal = hist[feature+18].getVal(x,y);
        if (histVal < 0) histVal = 0; //TODO: do we want this?

        drawLine(img, Point2D<int>((lineSize/2) + x*lineSize,
                                   (lineSize/2) + y*lineSize),
                                   -ori, lineSize,
                                   histVal);
      }
    }
  }

  inplaceNormalize(img, 0.0F, 255.0F);

  return toRGB(img);
}

ImageSet<double> HOG::computeFeatures(const ImageSet<float>& hist)
{

  // compute energy in each block by summing over orientations
  Image<double> norm = getHistogramEnergy(hist);

  const int w = norm.getWidth();
  const int h = norm.getHeight();
  
  const int numFeatures = hist.size() +   //Contrast-sensitive features
                    hist.size()/2 + //contrast-insensitive features
                    4 +             //texture features
                    1;              //trancation feature (this is zero map???)

  const int featuresW = std::max(w-2, 0);
  const int featuresH = std::max(h-2, 0);

  ImageSet<double> features(numFeatures, Dims(featuresW, featuresH), ZEROS);

  Image<double>::const_iterator normPtr = norm.begin();
  Image<double>::const_iterator ptr;

  // small value, used to avoid division by zero
  const double eps = 0.0001;

  for(int y=0; y<featuresH; y++)
    for(int x=0; x<featuresW; x++)
    {

      //Combine the norm values of neighboring bins
      ptr = normPtr + (y+1)*w + x+1;
      const double n1 = 1.0 / sqrt(*ptr + *(ptr+1) +
                                   *(ptr+w) + *(ptr+w+1) +
                                   eps);
      ptr = normPtr + y*w + x+1;
      const double n2 = 1.0 / sqrt(*ptr + *(ptr+1) +
                                   *(ptr+w) + *(ptr+w+1) +
                                   eps);
      ptr = normPtr + (y+1)*w + x;
      const double n3 = 1.0 / sqrt(*ptr + *(ptr+1) +
                                   *(ptr+w) + *(ptr+w+1) +
                                   eps);
      ptr = normPtr + y*w + x;      
      const double n4 = 1.0 / sqrt(*ptr + *(ptr+1) +
                                   *(ptr+w) + *(ptr+w+1) +
                                   eps);

      //For texture features
      double t1 = 0, t2 = 0, t3 = 0, t4 = 0;

      // contrast-sensitive features
      uint featureId = 0;
      for(uint ori=0; ori < hist.size(); ori++)
      {
        Image<float>::const_iterator histPtr = hist[ori].begin();
        const float histVal = histPtr[(y+1)*w + x+1];
        double h1 = std::min(histVal * n1, 0.2);
        double h2 = std::min(histVal * n2, 0.2);
        double h3 = std::min(histVal * n3, 0.2);
        double h4 = std::min(histVal * n4, 0.2);

        t1 += h1; t2 += h2; t3 += h3; t4 += h4;

        Image<double>::iterator featuresPtr = features[featureId++].beginw();
        featuresPtr[y*featuresW + x] = 0.5 * (h1 + h2 + h3 + h4);
      }

      // contrast-insensitive features
      int halfOriSize = hist.size()/2;
      for(int ori=0; ori < halfOriSize; ori++)
      {
        Image<float>::const_iterator histPtr1 = hist[ori].begin();
        Image<float>::const_iterator histPtr2 = hist[ori+halfOriSize].begin();
        const double sum = histPtr1[(y+1)*w + x+1] + histPtr2[(y+1)*w + x+1];
        double h1 = std::min(sum * n1, 0.2);
        double h2 = std::min(sum * n2, 0.2);
        double h3 = std::min(sum * n3, 0.2);
        double h4 = std::min(sum * n4, 0.2);

        Image<double>::iterator featuresPtr = features[featureId++].beginw();
        featuresPtr[y*featuresW + x] = 0.5 * (h1 + h2 + h3 + h4);
      }

      // texture features
      Image<double>::iterator featuresPtr = features[featureId++].beginw();
      featuresPtr[y*featuresW + x] = 0.2357 * t1;

      featuresPtr = features[featureId++].beginw();
      featuresPtr[y*featuresW + x] = 0.2357 * t2;
      
      featuresPtr = features[featureId++].beginw();
      featuresPtr[y*featuresW + x] = 0.2357 * t3;

      featuresPtr = features[featureId++].beginw();
      featuresPtr[y*featuresW + x] = 0.2357 * t4;

      // truncation feature
      // This seems to be just 0, do we need it?
      featuresPtr = features[featureId++].beginw();
      featuresPtr[y*featuresW + x] = 0;

    }



  return features;

}

Image<double> HOG::getHistogramEnergy(const ImageSet<float>& hist)
{
  if (hist.size() == 0)
    return Image<double>();

  Image<double> norm(hist[0].getDims(), ZEROS);

  //TODO: check for overflow
  int halfOriSize = hist.size()/2;
  // compute energy in each block by summing over orientations
  for(int ori=0; ori<halfOriSize; ori++)
  {
    Image<float>::const_iterator src1Ptr = hist[ori].begin();
    Image<float>::const_iterator src2Ptr = hist[ori+halfOriSize].begin();

    Image<double>::iterator normPtr = norm.beginw();
    Image<double>::const_iterator normPtrEnd = norm.end();

    while(normPtr < normPtrEnd)
    {
      *(normPtr++) += (*src1Ptr + *src2Ptr) * (*src1Ptr + *src2Ptr);
      src1Ptr++;
      src2Ptr++;
    }
  }

  return norm;
}

ImageSet<float> HOG::getOriHistogram(const Image<float>& mag, const Image<float>& ori, int numOrientations, int numBins)
{
  Dims blocksDims = Dims(
      (int)round((double)mag.getWidth()/double(numBins)),
      (int)round((double)mag.getHeight()/double(numBins)));

  ImageSet<float> hist(numOrientations, blocksDims, ZEROS);

  Image<float>::const_iterator magPtr = mag.begin(), oriPtr = ori.begin();
  //Set the with an height to a whole bin numbers. 
  //If needed replicate the data when summing the bins
  int w = blocksDims.w()*numBins; 
  int h = blocksDims.h()*numBins;
  int magW = mag.getWidth(); 
  int magH = mag.getHeight();
  int histWidth = blocksDims.w(); 
  int histHeight = blocksDims.h();

  for (int y = 1; y < h-1; y ++)
    for (int x = 1; x < w-1; x ++)
    {
      // add to 4 histograms around pixel using linear interpolation
      double xp = ((double)x+0.5)/(double)numBins - 0.5;
      double yp = ((double)y+0.5)/(double)numBins - 0.5;
      int ixp = (int)floor(xp);
      int iyp = (int)floor(yp);
      double vx0 = xp-ixp;
      double vy0 = yp-iyp;
      double vx1 = 1.0-vx0;
      double vy1 = 1.0-vy0;
      

      //If we are outside out mag/ori data, then use the last values in it
      int magX = std::min(x, magW-2);
      int magY = std::min(y, magH-2);
      double mag = magPtr[magY*magW  + magX];
      int ori = int(oriPtr[magY*magW + magX]);

      Image<float>::iterator histPtr = hist[ori].beginw();

      if (ixp >= 0 && iyp >= 0)
        histPtr[iyp*histWidth + ixp] += vx1*vy1*mag;

      if (ixp+1 < histWidth && iyp >= 0)
        histPtr[iyp*histWidth + ixp+1] += vx0*vy1*mag;

      if (ixp >= 0 && iyp+1 < histHeight) 
        histPtr[(iyp+1)*histWidth + ixp] += vx1*vy0*mag;

      if (ixp+1 < histWidth && iyp+1 < histHeight) 
        histPtr[(iyp+1)*histWidth + ixp+1] += vx0*vy0*mag;
    }

  return hist;
}

void HOG::getMaxGradient(const Image<PixRGB<byte> >& img,
    Image<float>& mag, Image<float>& ori,
    int numOrientations)
{
  if (numOrientations != 0 &&
      numOrientations > 18)
    LFATAL("Can only support up to 18 orientations for now.");
  
  mag.resize(img.getDims()); ori.resize(img.getDims());

  Image<PixRGB<byte> >::const_iterator src = img.begin();
  Image<float>::iterator mPtr = mag.beginw(), oPtr = ori.beginw();
  const int w = mag.getWidth(), h = mag.getHeight();

  float zero = 0;

  // first row is all zeros:
  for (int i = 0; i < w; i ++) { *mPtr ++ = zero; *oPtr ++ = zero; }
  src += w;

  // loop over inner rows:
  for (int j = 1; j < h-1; j ++)
    {
      // leftmost pixel is zero:
      *mPtr ++ = zero; *oPtr ++ = zero; ++ src;

      // loop over inner columns:
      for (int i = 1; i < w-1; i ++)
        {
          PixRGB<int> valx = src[1] - src[-1];
          PixRGB<int> valy = src[w] - src[-w];

          //Mag
          double mag1 = (valx.red()*valx.red()) + (valy.red()*valy.red());
          double mag2 = (valx.green()*valx.green()) + (valy.green()*valy.green());
          double mag3 = (valx.blue()*valx.blue()) + (valy.blue()*valy.blue());

          double mag = mag1;
          double dx = valx.red();
          double dy = valy.red();

          //Get the channel with the strongest gradient
          if (mag2 > mag)
          {
            dx = valx.green();
            dy = valy.green();
            mag = mag2;
          }
          if (mag3 > mag)
          {
            dx = valx.blue();
            dy = valy.blue();
            mag = mag3;
          }

          *mPtr++ = sqrt(mag);
          if (numOrientations > 0)
          {
            //Snap to num orientations
            double bestDot = 0;
            int bestOri = 0;
            for (int ori = 0; ori < numOrientations/2; ori++) {
              double dot = itsUU[ori]*dx + itsVV[ori]*dy;
              if (dot > bestDot) {
                bestDot = dot;
                bestOri = ori;
              } else if (-dot > bestDot) {
                bestDot = -dot;
                bestOri = ori+(numOrientations/2);
              }
            }
            *oPtr++ = bestOri;

          } else {
            *oPtr++ = atan2(dy, dx);
          }
          ++ src;
        }

      // rightmost pixel is zero:
      *mPtr ++ = zero; *oPtr ++ = zero; ++ src;
    }

  // last row is all zeros:
  for (int i = 0; i < w; i ++) { *mPtr ++ = zero; *oPtr ++ = zero; }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
