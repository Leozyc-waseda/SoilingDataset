/*!@file BeoSub/test-EdgeDetect.C */

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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-EdgeDetect.C $
// $Id: test-EdgeDetect.C 7452 2006-11-15 22:49:56Z ilab13 $
//

#include "GUI/XWindow.H"
//CAMERA STUFF
#include "Image/Image.H"
#include "Image/Pixels.H"

#include "Component/ModelManager.H"
#include "Devices/FrameGrabberFactory.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Image/ColorOps.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Transforms.H"
#include "Raster/Raster.H"
#include "BeoSub/hysteresis.H"
#include "VFAT/segmentImageTrackMC.H"
#include "rutz/shared_ptr.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream> //needed for segmentImageTrackMC!
#include <math.h>

#include "Image/Convolver.H"
//END CAMERA STUFF

////////////////////////////////////////////////////
// Canny Edge Detection
// Randolph Voorhies
// 11/15/05
///////////////////////////////////////////////////


#define PI 3.14159265
#define TRACE_CONTOUR_DEPTH_LIMIT 3


//Depth limited search to trace contours
bool traceContour(const Image<float> &magnitudeImage, const int x, const int y, const int loThresh, const int hiThresh, const int from, const int steps) {
 if(magnitudeImage.getVal(x,y) >= hiThresh)
  return true;
 if(magnitudeImage.getVal(x,y) < loThresh)
  return false;
 if(x == 0 || x >= magnitudeImage.getWidth() - 1 || y == 0 || y >= magnitudeImage.getHeight() - 1)
  return false;

 if(steps >= TRACE_CONTOUR_DEPTH_LIMIT)
  return false;

 if(magnitudeImage.getVal(x-1,y) > loThresh && from != 1)
  if(traceContour(magnitudeImage, x-1, y, loThresh, hiThresh, 5, steps+1))
   return true;
 if(magnitudeImage.getVal(x-1,y-1) > loThresh && from != 2)
  if(traceContour(magnitudeImage, x-1, y-1, loThresh, hiThresh, 6, steps+1))
   return true;
 if(magnitudeImage.getVal(x-1,y+1) > loThresh && from != 8)
  if(traceContour(magnitudeImage, x-1, y+1, loThresh, hiThresh, 4, steps+1))
   return true;
 if(magnitudeImage.getVal(x,y-1) > loThresh && from != 3)
  if(traceContour(magnitudeImage, x, y-1, loThresh, hiThresh, 7, steps+1))
   return true;
 if(magnitudeImage.getVal(x,y+1) > loThresh && from != 7)
  if(traceContour(magnitudeImage, x, y+1, loThresh, hiThresh, 3, steps+1))
   return true;
 if(magnitudeImage.getVal(x+1,y-1) > loThresh && from != 4)
  if(traceContour(magnitudeImage, x+1, y-1, loThresh, hiThresh, 8, steps+1))
   return true;
 if(magnitudeImage.getVal(x+1,y) > loThresh && from != 5)
  if(traceContour(magnitudeImage, x+1, y, loThresh, hiThresh, 1, steps+1))
   return true;
 if(magnitudeImage.getVal(x+1,y+1) > loThresh && from != 6)
  if(traceContour(magnitudeImage, x+1, y+1, loThresh, hiThresh, 2, steps+1))
   return true;

return false;
}

void nonMaxSuppressAndContTrace(Image<byte> &edgeImage, const Image<float> magnitudeImage, const Image<float> orientationImage, const int loThresh, const int hiThresh) {
 float mag, pa=0, pb=0, orientDeg;

 for(int x = 1; x < magnitudeImage.getWidth() - 1; x++)
  for(int y = 1; y < magnitudeImage.getHeight() - 1; y++) {
   mag = magnitudeImage.getVal(x,y);
   orientDeg = orientationImage.getVal(x,y)*(180/PI);

   if((orientDeg >= 0 && orientDeg <= 45) || (orientDeg > -180 && orientDeg <= -135)) {
     pa = (magnitudeImage.getVal(x+1,y) + magnitudeImage.getVal(x+1,y-1))/2;
     pb = (magnitudeImage.getVal(x-1,y) + magnitudeImage.getVal(x-1,y+1))/2;
   }
   else if((orientDeg > 45 && orientDeg <= 90) || (orientDeg > -135 && orientDeg <= -90)) {
     pa = (magnitudeImage.getVal(x+1,y-1) + magnitudeImage.getVal(x,y-1))/2;
     pb = (magnitudeImage.getVal(x-1,y+1) + magnitudeImage.getVal(x,y+1))/2;
   }
   else if((orientDeg > 90 && orientDeg <= 135) || (orientDeg > -90 && orientDeg <= -45)) {
     pa = (magnitudeImage.getVal(x,y-1) + magnitudeImage.getVal(x-1,y-1))/2;
     pb = (magnitudeImage.getVal(x,y+1) + magnitudeImage.getVal(x+1,y+1))/2;
   }
   else if((orientDeg > 135 && orientDeg <= 180) || (orientDeg > -45 && orientDeg < 0)) {
     pa = (magnitudeImage.getVal(x-1,y-1) + magnitudeImage.getVal(x-1,y))/2;
     pb = (magnitudeImage.getVal(x+1,y+1) + magnitudeImage.getVal(x+1,y))/2;
   }

   if(mag > pa && mag > pb) {
     if(mag < loThresh)
      continue;
     else if(mag > hiThresh)
      edgeImage.setVal(x,y,255);
     else if(traceContour(magnitudeImage,x,y,loThresh,hiThresh, -1, 0)) {
      edgeImage.setVal(x,y,255);
     }
    }
   }
}

//TODO: Templates not working correctly: sending a Image<float> as a source breaks it.
template <class T_or_RGB>
Image<byte> cannyEdgeDetect(const Image<T_or_RGB> source, const float sigma, const int loThresh, const int hiThresh) {

 Image<byte> edgeImage(source.getDims(), ZEROS);            //Image to return
 Image<float> magnitudeImage(source.getDims(), ZEROS);      //First Derivative Magnitude Image
 Image<float> orientationImage(source.getDims(), ZEROS);    //First Derivative Orientation Image

 //gaussian blurred image -- to reduce noise
 Image<float> fblurImage = luminance(convGauss(source, sigma, sigma,10));

 //Find the magnitude and orientation
 gradient(fblurImage, magnitudeImage, orientationImage);

 //Perform non-maximul suppression and contour tracing to determine edges
 nonMaxSuppressAndContTrace(edgeImage, magnitudeImage, orientationImage, loThresh, hiThresh);

 return edgeImage;
}



//good tlo and thi values are 10, 16, with a sigma of 2
int main(int argc, char* argv[]) {

  int loThresh, hiThresh;
  float sigma;
  Image< PixRGB<byte> > cameraImage;
  Image<float> floatImage;

  if(argc != 4) {
   std::cout << "USAGE: bin/EdgeDetect sigma Tlow Thigh " << std::endl;
   std::cout << "       *Thigh and Tlow are thresholds for the canny algorithm" << std::endl;
   std::cout << "       *sigma is used in gaussian mask" << std::endl;
   exit(0);
  }

  sigma = (float)atoi(argv[1]);
  loThresh = atoi(argv[2]);
  hiThresh = atoi(argv[3]);

  XWindow* xwin;
  XWindow* xwin2;

  ModelManager camManager("ColorTracker Tester");
  nub::soft_ref<FrameIstream> gb(makeIEEE1394grabber(camManager, "colorcam", "cocam"));
  camManager.addSubComponent(gb);
  //camManager.loadConfig("camconfig.pmap");
  gb->setModelParamVal("FrameGrabberSubChan", 0);
  camManager.start();

  cameraImage = gb->readRGB();

  xwin = new XWindow(cameraImage.getDims());
  xwin2 = new XWindow(cameraImage.getDims());

  while(1) {
    cameraImage = gb->readRGB();
    //floatImage = cameraImage;
    xwin->drawImage(cameraImage);
    //xwin2->drawImage(cannyEdgeDetect(floatImage, sigma, loThresh, hiThresh));
    xwin2->drawImage(cannyEdgeDetect(cameraImage, sigma, loThresh, hiThresh));
  }
return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
