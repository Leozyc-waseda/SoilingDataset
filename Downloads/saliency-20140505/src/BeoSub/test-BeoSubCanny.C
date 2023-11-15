/*!@file BeoSub/test-BeoSubCanny.C Test BeoSubColor exclusion and shape detection */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoSubCanny.C $
// $Id: test-BeoSubCanny.C 6990 2006-08-11 18:13:51Z rjpeters $
//

#ifndef TESTBEOSUBCANNY_H_DEFINED
#define TESTBEOSUBCANNY_H_DEFINED

#include "BeoSub/BeoSubCanny.H"

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
//test stuff
//SHAPE STUFF
#include "CannyModel.H"
//END SHAPE STUFF

int main(int argc, char **argv)
{
  if(MYLOGVERB) printf("This should never print\n");
  //Parse the command line options
  char *infilename = NULL;  //Name of the input image
  char *shapeArg = NULL;    //Shape for recognition
  char *colorArg = NULL;    //Color for tracking

  if(argc < 3){
    fprintf(stderr,"\n<USAGE> %s image shape color \n",argv[0]);
    fprintf(stderr,"\n      image:      An image to process. Must be in PGM format.\n");
    fprintf(stderr,"                  Type 'none' for camera input.\n");
    fprintf(stderr,"      shape:      Shape on which to run recognition\n");
    fprintf(stderr,"                  Candidates: Rectangle, Square, Circle, Octagon.\n");
    fprintf(stderr,"      color:       Color to track\n");
    fprintf(stderr,"                  Candidates: Blue, Yellow, none (for no color tracking).\n");
    exit(1);
  }
  infilename = argv[1];
  shapeArg = argv[2];
  colorArg = argv[3];

  printf("READ: 1: %s 2: %s 3: %s\n", infilename, shapeArg, colorArg);


  // instantiate a model manager (for camera input):
  ModelManager manager("Canny Tester");
  // Instantiate our various ModelComponents:

  nub::soft_ref<FrameIstream>
    gb(makeIEEE1394grabber(manager, "cannycam", "cc"));

  if(!strcmp(infilename, "none")){//if the camera is to be used

    //GRAB image from camera to be tested
    manager.addSubComponent(gb);

  }

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeoSubCanny> test(new BeoSubCanny(manager));
  manager.addSubComponent(test);

  if(!strcmp(infilename, "none")){
    //Load in config file for camera FIX: put in a check whether config file exists!
    manager.loadConfig("camconfig.pmap");
  }

  manager.start();

  //Test with a circle
  rutz::shared_ptr<ShapeModel> shape;

  Image< PixRGB<byte> > Img;

  double* p;


  //Set up shape to be matched
    if(!strcmp(shapeArg, "Rectangle")){
      //rectangle
      p = (double*)calloc(6, sizeof(double));
      p[1] = 150.0; //Xcenter
      p[2] = 120.0; //Ycenter
      p[4] = 80.0f; // Width
      p[5] = 80.0f; // Height
      p[3] = (3.14159/4.0); //alpha
      shape.reset(new RectangleShape(120.0, p, true));
    }
    else if(!strcmp(shapeArg, "Square")){
      //square
      p = (double*)calloc(5, sizeof(double));
      p[1] = 150.0; //Xcenter
      p[2] = 120.0; //Ycenter
      p[3] = 100.0; // Height
      p[4] = (3.14159/4.0); // alpha
      shape.reset(new SquareShape(100.0, p, true));
    }
    else if(!strcmp(shapeArg, "Octagon")){
      //octagon
      p = (double*)calloc(5, sizeof(double));
      p[1] = 150.0; //Xcenter
      p[2] = 120.0; //Ycenter
      p[3] = 60.0f; // Height
      p[4] = (3.14159/4.0); // alpha
      shape.reset(new OctagonShape(80.0, p, true));
    }
    else if(!strcmp(shapeArg, "Circle")){
      //circle
      p = (double*)calloc(4, sizeof(double));
      p[1] = 150.0; //Xcenter
      p[2] = 120.0; //Ycenter
      p[3] = 50.0; // Radius
      shape.reset(new CircleShape(40.0, p, true));
    }
    else if(!strcmp(shapeArg, "Parallel")){
      //Parallel lines
      p = (double*)calloc(6, sizeof(double));
      p[1] = 150.0; //Xcenter
      p[2] = 120.0; //Ycenter
      p[4] = 120.0f; // Width
      p[5] = 50.0f; // Height
      p[3] = (3.14159/4.0); //alpha
      shape.reset(new ParallelShape(120.0, p, true));
    }
    else{
      printf("Cannot run shape recognition without a shape to recognize! Returning...\n");
      p = (double*)calloc(1, sizeof(double));
      shape.reset(new CircleShape(9999.0, p, false));
      return(false);
    }



  while(1){
    //Get image to be matched
    //TO TEST FROM CAMERA
    if(!strcmp(infilename, "none")){
      Img = gb->readRGB();
    }
    else{
      //TO TEST FROM FILE (shouldn't really be in loop):
      Img = Raster::ReadRGB(infilename);
    }

    shape->setDimensions(p);

    //run the matching code
    test->setupCanny(colorArg, Img, true);

    //Middle
    //p[1] = 150.0;
    //p[2] = 120.0;
    //shape->setDimensions(p);
    bool shapeFound = test->runCanny(shape);

    if(!shapeFound){
      //stupid compiler, breaking on stupid warnings
    }

    //NOTE: Uncomment the following code to test using multiple starting points

    if(!shapeFound){ //Upper left
      p[1] = 60.0; //Xcenter
      p[2] = 180.0; //Ycenter
      shape->setDimensions(p);
      shapeFound = test->runCanny(shape);
    }
    if(!shapeFound){ //Upper right
      p[1] = 260.0; //Xcenter
      p[2] = 180.0; //Ycenter
      shape->setDimensions(p);
      shapeFound = test->runCanny(shape);
    }
    if(!shapeFound){ //Lower left
      p[1] = 60.0; //Xcenter
      p[2] = 60.0; //Ycenter
      shape->setDimensions(p);
      shapeFound = test->runCanny(shape);
    }
    if(!shapeFound){ //Lower right
      p[1] = 260.0; //Xcenter
      p[2] = 60.0; //Ycenter
      shape->setDimensions(p);
      shapeFound = test->runCanny(shape);
    }

  }
  return 0;
}

#endif
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
