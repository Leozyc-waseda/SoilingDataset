/*!@file BeoSub/ColorTracker.C I method to check for the existence of a defined color in an image */

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
// Primary maintainer for this file:  Zack Gossman <gossman@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/ColorTracker.C $
// $Id: ColorTracker.C 9412 2008-03-10 23:10:15Z farhan $

#include "BeoSub/ColorTracker.H"
#include "BeoSub/ColorDef.H"

// ######################################################################
ColorTracker::ColorTracker(OptionManager& mgr,
                         const std::string& descrName,
                         const std::string& tagName):
  ModelComponent(mgr, descrName, tagName)
{
  color.resize(4,0.0F);

  debugmode = true;
  MYLOGVERB = 0;
  hasSetup = false;
  hasRun = false;
}

// ######################################################################
ColorTracker::~ColorTracker(){

}

// ######################################################################
void ColorTracker::setupTracker(const char* colorArg, Image< PixRGB<byte> > image, bool debug){

  debugmode = debug;//turns output on or off

  //NOTE: the following group will output a large amount of extranous text to the console. However, it is necessary to keep it here in order to avoid a 320x240 hardcoded size. TRY doing a default setup like in the CannyModel class. FIX!
  if(!hasSetup){
    segmenter = new segmentImageTrackMC<float,unsigned int,4> (image.getWidth()*image.getHeight());
    segmenter->SITsetCircleColor(0,255,0);
    segmenter->SITsetBoxColor(255,255,0,0,255,255);
    segmenter->SITsetUseSmoothing(true,10);
    segmenter->SITtoggleCandidateBandPass(false);
    segmenter->SITtoggleColorAdaptation(false);

    int ww = image.getWidth()/4;
    int hh = image.getHeight()/4;
    segmenter->SITsetFrame(&ww,&hh);
  }

  //COLORTRACKING DECS: note that much of these may not need to be in this file
  //Colors MUST use H2SV2 pixel values! use test-sampleH2SV2 to sample needed values! --Z

ColorDef();

  //0 = H1 (0-1), 1=H2 (0-1), 2=S (0-1), 3=V (0-1)
  if(!strcmp(colorArg, "Red")){
        color=red;
  }

  else if(!strcmp(colorArg, "Green")){
        color=green;
  }

  else if(!strcmp(colorArg, "Orange")){
        color=orange;
  }
  else if(!strcmp(colorArg, "Blue")){
        color=blue;
  }
  else if(!strcmp(colorArg, "Yellow")){
        color=yellow;
  }
else if(!strcmp(colorArg, "White")){
        color=white;
  }
  else if(!strcmp(colorArg, "Black")){
        color=black;
  }
  else if (!strcmp(colorArg, "Brown")) {
        //BROWN
        color=brown;
  }
  else{
    printf("Color argument not recognized.\n");
  }


  //! +/- tollerance value on mean for track
  std::vector<float> std(4,0.0F);
  //NOTE that the saturation tolerance is important (if it gos any higher thn this, it will nearly always recognize white!)
  std[0] = 0.20000; std[1] = 0.40000; std[2] = 0.44500; std[3] = 0.65000;

  //! normalizer over color values (highest value possible)
  std::vector<float> norm(4,0.0F);
  norm[0] = 1.0F; norm[1] = 1.0F; norm[2] = 1.0F; norm[3] = 1.0F;

  //! how many standard deviations out to adapt, higher means less bias
  //The lower these are, the more strict recognition will be in subsequent frames
  //TESTED AND PROVED do NOT change unless you're SURE
  std::vector<float> adapt(4,0.0F);
  //adapt[0] = 3.5F; adapt[1] = 3.5F; adapt[2] = 3.5F; adapt[3] = 3.5F;
  adapt[0] = 5.0F; adapt[1] = 5.0F; adapt[2] = 5.0F; adapt[3] = 5.0F;

  //! highest value for color adaptation possible (hard boundry)
  std::vector<float> upperBound(4,0.0F);
  upperBound[0] = color[0] + 0.45F; upperBound[1] = color[1] + 0.45F; upperBound[2] = color[2] + 0.55F; upperBound[3] = color[3] + 0.55F;

  //! lowest value for color adaptation possible (hard boundry)
  std::vector<float> lowerBound(4,0.0F);
  lowerBound[0] = color[0] - 0.45F; lowerBound[1] = color[1] - 0.45F; lowerBound[2] = color[2] - 0.55F; lowerBound[3] = color[3] - 0.55F;
  //END DECS

 if(!strcmp(colorArg, "White") || !strcmp(colorArg, "Black")){
    adapt[0] = 25.0F; adapt[1] = 25.0F; adapt[2] = 0.1F; adapt[3] = 0.1F;

    std[0] = 1.0F; std[1] = 1.0F; std[2] = 0.1F; std[3] = 0.1F;

  }
  //Read image from file and display
  colorImg = image;

  //color tracking stuff
  segmenter->SITsetTrackColor(&color,&std,&norm,&adapt,&upperBound,&lowerBound);

  if(debugmode){
    if(!hasSetup){
      xwin1.reset(new XWindow(colorImg.getDims(), -1, -1, "input window"));
      xwin1->setPosition(0, 0);
    }
    xwin1->drawImage(colorImg);
  }
  hasSetup = true;

}

// ######################################################################
bool ColorTracker::runTracker(float threshold, float &xpos, float &ypos, float &mass){//xpos and ypos and mass are reference containers for final x and y positions
  bool colorFound = false;

  Image< PixH2SV2<float> > H2SVimage = colorImg;

  //junk images to make the segmenter happy
  Image< PixRGB<byte> > display;
  Image<PixRGB<byte> > Aux;
  segmenter->SITtrackImageAny(H2SVimage,&display,&Aux,true);

  // Retrieve our output image
  outputImg =  (Image<byte>)quickInterpolate(segmenter->SITreturnCandidateImage(),4);
  float foundMass = 0.0;
  int whichBlob = 0;
  int xWhere = 0;
  int yWhere = 0;

  //identify the largest blob found
  for(unsigned int i = 0; i < segmenter->SITnumberBlobs(); i++){
    if(4.0*(segmenter->SITgetBlobMass(i)) > foundMass){
      whichBlob = (int)i;
      foundMass = 4.0*(segmenter->SITgetBlobMass(i));
    }
  }

  //Check whether largest blob exceeds threshold size
  if(foundMass > threshold){
    mass = foundMass;
    xpos = xWhere = 4*(segmenter->SITgetBlobPosX(whichBlob));
    ypos = yWhere = 4*(segmenter->SITgetBlobPosY(whichBlob));
    if(debugmode){
      printf("\n\nLargest blob found with mass %f at %d, %d\n\n", foundMass, xWhere, yWhere);
    }
    colorFound = true;
  }
  else if(debugmode){
    printf("\n\nColor not found in appreciable quantities. Largest blob is %f\n\n", foundMass);
  }
  if(debugmode){
    //point to center of mass
    drawDisk(outputImg, Point2D<int>(xWhere, yWhere), 2, PixRGB<byte>(225, 20, 20));
    drawCircle(outputImg, Point2D<int>(xWhere, yWhere), (int)(sqrt(foundMass)),  PixRGB<byte>(20, 20, 255), 2);

    if(!hasRun){
      xwin2.reset(new XWindow(outputImg.getDims(), -1, -1, "output window"));
      xwin2->setPosition(outputImg.getWidth()+10, 0);
    }
    xwin2->drawImage(outputImg);
  }

  hasRun = true;
  return colorFound;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

