/*!@file BeoSub/test-ComplexObject.C ComplexObject test module */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-ComplexObject.C $
// $Id: test-ComplexObject.C 14376 2011-01-11 02:44:34Z pez $
//



//NOTE: new stuff seems to have made recognition occure much less often. Why? Do NOT commit until this is FIXed!
#ifndef TESTCOMPLEXOBJECT_H_DEFINED
#define TESTCOMPLEXOBJECT_H_DEFINED

#include "BeoSub/ComplexObject.H"

//CAMERA STUFF
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/Transforms.H"
#include "Image/MathOps.H"
#include "GUI/XWindow.H"
#include "Component/ModelManager.H"
#include "Devices/FrameGrabberFactory.H"
#include "Raster/Raster.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "BeoSub/BeoSubCanny.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

//END CAMERA STUFF

int main(int argc, char **argv)
{
  Image< PixRGB<byte> > outputImg;

  Point2D<int> tl, tr, br, bl;
  MYLOGVERB=LOG_CRIT;
  bool hasSetup = false;
  //Parse the command line options
  const char *infilename = NULL;  //Name of the input image
  //char *objectfile = NULL; //File of the ComplexObject
  std::string objectfile="";
  int matchcount = 0;
  const char *showMatches = NULL; //input to determine number of displayed matches
  bool showAllMatches = false;

  //int count = 0;

  const char* objectdbfilename = NULL;
  std::string line = "";

  std::vector < rutz::shared_ptr<VisualObjectMatch> > matches;
  Image< PixRGB<byte> > fusedImg;
  Image< PixRGB<byte> > keypointImg;

  std::vector < rutz::shared_ptr<ComplexObject> > cov;
  std::vector < rutz::shared_ptr<ComplexObject> >::iterator coIter, coIterStop;
  rutz::shared_ptr<ComplexObject> co;

  rutz::shared_ptr<XWindow> wini;



  if(argc < 3){
    fprintf(stderr,"\n<USAGE> %s database_filename image showall\n",argv[0]);
    fprintf(stderr,"\n      database_filename:   The path and filename of the object database.\n");
    fprintf(stderr,"\n      image:             An image to process. Must be in PGM format.\n");
    fprintf(stderr,"                           Type 'none' for camera input.\n");
    fprintf(stderr,"\n      showall(Y/N):     Whether one match (N) or all matches (Y)\n");
    fprintf(stderr,"\n                         should be found.\n");
    exit(1);
  }

  //NOTE: the file of the database_filename is simple
  //it is a list of .obj files, with one file per line
  //be careful not to have spaces


  objectdbfilename = argv[1];
  std::string fname = objectdbfilename;

  std::string filepathstr = fname.substr(0, fname.find_last_of('/')+1);


   std::ifstream is(objectdbfilename);
  if (is.is_open() == false)
     { LERROR("Cannot open '%s' -- USING EMPTY", objectdbfilename); exit(0); }

  while(!is.eof()) {
    getline(is, line);
    objectfile = line;

    if (strcmp(objectfile.c_str(), "") == 0) {
      continue;
    }

    printf("Loading: %s\n", objectfile.c_str());

    //    co.reset(new ComplexObject("MyObject", (char*)(filepathstr + objectfile).c_str()));
    co.reset(new ComplexObject((char*)(filepathstr + objectfile).c_str()));
    cov.push_back(co);
  }
  // is>>showMatches;

  //objectfile = argv[1];

  infilename = argv[2];

  showMatches = argv[3];


  if(!strcmp(showMatches, "Y") || !strcmp(showMatches, "y")){
    showAllMatches = true;
  }





  // instantiate a model manager (for camera input):
  ModelManager manager("ComplexObject Tester");
  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameIstream>
    gb(makeIEEE1394grabber(manager, "COcam", "cocam"));

  if(!strcmp(infilename, "none")){
    //GRAB image from camera to be tested
    manager.addSubComponent(gb);

    // set the camera number (in IEEE1394 lingo, this is the
    // "subchannel" number):
    gb->setModelParamVal("FrameGrabberSubChan", 0);
    gb->setModelParamVal("FrameGrabberBrightness", 128);
    gb->setModelParamVal("FrameGrabberHue", 180);
  }
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
      Img = gb->readRGB(); //grab();
    }


    coIter = cov.begin();
    coIterStop = cov.end();
    //rutz::shared_ptr<VisualObject> vo(new VisualObject("mypic", "mypicfilename", Img));

  rutz::shared_ptr<VisualObject>
    vo(new VisualObject("mypic", "mypicfilename", Img,
                        Point2D<int>(-1,-1), std::vector<float>(),
                        std::vector< rutz::shared_ptr<Keypoint> >(), false));
  outputImg = Img;


    while(coIter != coIterStop) {
         matchcount = (*coIter)->matchKeypoints(showAllMatches, vo, VOMA_SIMPLE, matches, keypointImg, fusedImg);



    if(!hasSetup){

      wini.reset(new XWindow(Img.getDims(), -1, -1, "input window"));
      wini->setPosition(0, 0);

      Dims ndims(Img.getDims().w(), (Img.getDims().h()*2));

      hasSetup = true;
    }


   // for (uint i = 0; i < matches.size(); i++) {
    if (matches.size() > 0) {
     // count = 1;
    LINFO("\nMATCHCOUNT :: %d\tmatches.size() :: %d\n", matchcount, int(matches.size()));
      matches[0]->getTransfTestOutline(tl,tr,br,bl);
   // }

     // if(count-- > 0 ) {
      //drawRect(outputImg, Rectangle::tlbrI(tl.j, tl.i, br.j, br.i), PixRGB<byte>(255, 0, 0));
      int size = 3;

      Point2D<int> *tmp = NULL;
      for(int i = 0; i < 4; i++) {
      switch(i) {
        case 0:
          tmp = &tl;
          break;

        case 1:
          tmp = &tr;
          break;

        case 2:
          tmp = &bl;
          break;

        case 3:
          tmp = &br;
          break;
      }


      if (tmp->i <= 3) {
        tmp->i = 4;
      }
      else if (tmp->i >=  outputImg.getWidth()-3) {
        tmp->i =  outputImg.getWidth() - 4;
      }

      if (tmp->j <= 3) {
        tmp->j = 4;
      }
      else if (tmp->j >=  outputImg.getHeight()-3) {
        tmp->j =  outputImg.getHeight() - 4;
      }
      }


      drawLine(outputImg, tl, tr, PixRGB<byte>(255, 0, 0), size);
      drawLine(outputImg, tr, br, PixRGB<byte>(255, 0, 0), size);
      drawLine(outputImg, br, bl, PixRGB<byte>(255, 0, 0), size);
      drawLine(outputImg, bl, tl, PixRGB<byte>(255, 0, 0), size);

      Point2D<int> com = (tl+tr+bl+br)/4;
      drawCross(outputImg, com, PixRGB<byte>(255, 0, 0), 10, size);
    }



    //show images to screen

    *coIter++;
    }
    wini->drawImage(outputImg);
   }
  return 0;

}


#endif
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
