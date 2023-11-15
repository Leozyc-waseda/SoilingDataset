/*!@file BeoSub/test-Skeleton.C */

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

int main() {
  Image< PixRGB<byte> > cameraImage;
  Image<byte> bwImage;

  XWindow *xwin;

  ModelManager camManager("ColorTracker Tester");
  nub::soft_ref<FrameIstream> gb(makeIEEE1394grabber(camManager, "colorcam", "cocam"));
  camManager.addSubComponent(gb);
  //camManager.loadConfig("camconfig.pmap");
  gb->setModelParamVal("FrameGrabberSubChan", 0);
  camManager.start();

  cameraImage = gb->readRGB();

  xwin = new XWindow(cameraImage.getDims());

/*
Accessor Functions for Image Class
Set:
cameraImage.setVal(x,y,Value)

Get:
cameraImage.getVal(x,y)

*/
int i=0;
int j=0;
cameraImage = gb->readRGB();
bwImage = luminance(cameraImage);

  while(1) {
    xwin->drawImage(bwImage);

    if(i<cameraImage.getWidth()-1) i++;
    else {
        i=0;
        if(j<cameraImage.getHeight()-1) j++;
    }

    bwImage.setVal(i,j,255);
   }


return 0;
}
