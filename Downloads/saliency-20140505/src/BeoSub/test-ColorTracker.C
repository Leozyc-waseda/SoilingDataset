/*!@file BeoSub/test-ColorTracker.C Test ColorTracker test module */

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
// Primary maintainer for this file: Zack Gossman <gossman@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-ColorTracker.C $
// $Id: test-ColorTracker.C 14376 2011-01-11 02:44:34Z pez $
//

#ifndef TESTCOLORTRACKER_H_DEFINED
#define TESTCOLORTRACKER_H_DEFINED

#include "BeoSub/ColorTracker.H"

//CAMERA STUFF
#include "Image/Image.H"
#include "Image/Pixels.H"

#include "Component/ModelManager.H"
#include "Devices/FrameGrabberFactory.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>
//END CAMERA STUFF

int main(int argc, char **argv)
{
  float x = 0.0;
  float y = 0.0;
  float mass = 0;

  //Parse the command line options
  char *infilename = NULL;  //Name of the input image
  char *colorArg = NULL;    //Color for tracking

  if(argc < 3){
    fprintf(stderr,"\n<USAGE> %s image color \n",argv[0]);
    fprintf(stderr,"\n      image:      An image to process. Must be in PGM format.\n");
    fprintf(stderr,"                  Type 'none' for camera input.\n");
    fprintf(stderr,"      color:       Color to track\n");
    fprintf(stderr,"                  Candidates: Blue, Yellow, Green, Orange, Red, Brown\n");
    exit(1);
  }
  infilename = argv[1];
  colorArg = argv[2];

  printf("READ: 1: %s 2: %s\n", infilename, colorArg);

  // instantiate a model manager (for camera input):
  ModelManager camManager("ColorTracker Tester");
  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameIstream>
    gb(makeIEEE1394grabber(camManager, "colorcam", "cocam"));

  if(!strcmp(infilename, "none")){
    //GRAB image from camera to be tested
    camManager.addSubComponent(gb);

    //Load in config file for camera FIX: put in a check whether config file exists!
    //camManager.loadConfig("camconfig.pmap");

    // set the camera number (in IEEE1394 lingo, this is the
    // "subchannel" number):
    //    gb->setModelParamVal("FrameGrabberSubChan", 0);
  }
  camManager.start();

  // instantiate a model manager for the color tracker module:
  ModelManager manager("ColorTracker Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<ColorTracker> test(new ColorTracker(manager));
  manager.addSubComponent(test);

  manager.start();

  Image< PixRGB<byte> > Img;


  if(!strcmp(infilename, "none")){
  }
  else{
    //TO TEST FROM FILE
    Img = Raster::ReadRGB(infilename);
  }

  while(1){
    //Get image to be matched
    //TO TEST FROM CAMERA
    if(!strcmp(infilename, "none")){
      Img = gb->readRGB();
    }

    //run the matching code
    test->setupTracker(colorArg, Img, true);
    test->runTracker(25.0, x, y, mass);
    //printf("x:%f, y:%f, mass:%f\n",x,y, mass);
  }
  return 0;
}

#endif
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
