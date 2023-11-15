/*!@file BeoSub/BeoSubTaskDecoder.C */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// Primary maintainer for this file
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubTaskDecoder.C $
// $Id: BeoSubTaskDecoder.C 10794 2009-02-08 06:21:09Z itti $
//

#include "BeoSub/BeoSubTaskDecoder.H"

#include "VFAT/segmentImageTrackMC.H"
#include <iostream>

// ######################################################################
BeoSubTaskDecoder::BeoSubTaskDecoder(OptionManager& mgr,
                         const std::string& descrName,
                         const std::string& tagName) :
  ModelComponent(mgr, descrName, tagName)
{

  debugmode = false;
  setup = false;
  color.resize(4,0.0F);
  std.resize(4,0.0F);
  norm.resize(4,0.0F);
  adapt.resize(4,0.0F);
  lowerBound.resize(4,0.0F);
  upperBound.resize(4,0.0F);
  candidate_color = "none";
  Hz = 0.0F;
  fps = 0.0F;
  res = 0.0F;

  //NOTE that hardcoding these dimensions may be harmful and thus have to be FIXed

  width = 320;
  height = 240;

  segmenter = new segmentImageTrackMC<float,unsigned int, 4> (width*height);

  int wi = 320/4;
  int hi = 240/4;

  segmenter->SITsetFrame(&wi,&hi);

  segmenter->SITsetCircleColor(0,255,0);
  segmenter->SITsetBoxColor(255,255,0,0,255,255);
  segmenter->SITsetUseSmoothing(false,10);


  segmenter->SITtoggleCandidateBandPass(false);
  segmenter->SITtoggleColorAdaptation(false);

}

BeoSubTaskDecoder::~BeoSubTaskDecoder() {}

void BeoSubTaskDecoder::setupRed()
{
 //-----------------------------------------------
 //! Mean color to track red blinking bike  LED
        // H1 - H2 - S - V

  colorConf.openFile("colortrack.conf", true);

  PixRGB<float> P1(colorConf.getItemValueF("RED_R"),
                   colorConf.getItemValueF("RED_G"),
                   colorConf.getItemValueF("RED_B"));

  PixH2SV1<float> RED(P1);

  color[0] = RED.H1(); color[1] = RED.H2(); color[2] = RED.S(); color[3] = RED.V();

  printf("h1: %f, h2: %f, sat: %f, value: %f", color[0], color[1], color[2], color[3]);

 //! +/- tollerance value on mean for track
  std[0] = colorConf.getItemValueF("RED_std0");
  std[1] = colorConf.getItemValueF("RED_std1");
  std[2] = colorConf.getItemValueF("RED_std2");
  std[3] = colorConf.getItemValueF("RED_std3");

  //! normalizer over color values (highest value possible)
  norm[0] = colorConf.getItemValueF("RED_norm0");
  norm[1] = colorConf.getItemValueF("RED_norm1");
  norm[2] = colorConf.getItemValueF("RED_norm2");
  norm[3] = colorConf.getItemValueF("RED_norm3");

  //! how many standard deviations out to adapt, higher means less bias
  adapt[0] = colorConf.getItemValueF("RED_adapt0");
  adapt[1] = colorConf.getItemValueF("RED_adapt1");
  adapt[2] = colorConf.getItemValueF("RED_adapt2");
  adapt[3] = colorConf.getItemValueF("RED_adapt3");

  //! highest value for color adaptation possible (hard boundry)
  upperBound[0] = color[0] + colorConf.getItemValueF("RED_up0");
  upperBound[1] = color[1] + colorConf.getItemValueF("RED_up1");
  upperBound[2] = color[2] + colorConf.getItemValueF("RED_up2");
  upperBound[3] = color[3] + colorConf.getItemValueF("RED_up3");

  //! lowest value for color adaptation possible (hard boundry)
  lowerBound[0] = color[0] - colorConf.getItemValueF("RED_lb0");
  lowerBound[1] = color[1] - colorConf.getItemValueF("RED_lb1");
  lowerBound[2] = color[2] - colorConf.getItemValueF("RED_lb2");
  lowerBound[3] = color[3] - colorConf.getItemValueF("RED_lb3");
}

void BeoSubTaskDecoder::setupGreen() {
  /*
  colorConf.openFile("colortrack.conf", true);

  PixRGB<float> P1(colorConf.getItemValueF("GREEN_R"),
                   colorConf.getItemValueF("GREEN_G"),
                   colorConf.getItemValueF("GREEN_B"));

  PixH2SV1<float> GREEN(P1);

  color[0] = GREEN.H1(); color[1] = GREEN.H2(); color[2] = GREEN.S(); color[3] = GREEN.V();

  //! +/- tollerance value on mean for track
  std[0] = colorConf.getItemValueF("GREEN_std0");
  std[1] = colorConf.getItemValueF("GREEN_std1");
  std[2] = colorConf.getItemValueF("GREEN_std2");
  std[3] = colorConf.getItemValueF("GREEN_std3");

  //! normalizer over color values (highest value possible)
  norm[0] = colorConf.getItemValueF("GREEN_norm0");
  norm[1] = colorConf.getItemValueF("GREEN_norm1");
  norm[2] = colorConf.getItemValueF("GREEN_norm2");
  norm[3] = colorConf.getItemValueF("GREEN_norm3");

  //! how many standard deviations out to adapt, higher means less bias
  adapt[0] = colorConf.getItemValueF("GREEN_adapt0");
  adapt[1] = colorConf.getItemValueF("GREEN_adapt1");
  adapt[2] = colorConf.getItemValueF("GREEN_adapt2");
  adapt[3] = colorConf.getItemValueF("GREEN_adapt3");

  //! highest value for color adaptation possible (hard boundry)
  upperBound[0] = color[0] + colorConf.getItemValueF("GREEN_up0");
  upperBound[1] = color[1] + colorConf.getItemValueF("GREEN_up1");
  upperBound[2] = color[2] + colorConf.getItemValueF("GREEN_up2");
  upperBound[3] = color[3] + colorConf.getItemValueF("GREEN_up3");

  //! lowest value for color adaptation possible (hard boundry)
  lowerBound[0] = color[0] - colorConf.getItemValueF("GREEN_lb0");
  lowerBound[1] = color[1] - colorConf.getItemValueF("GREEN_lb1");
  lowerBound[2] = color[2] - colorConf.getItemValueF("GREEN_lb2");
  lowerBound[3] = color[3] - colorConf.getItemValueF("GREEN_lb3");*/
}


void BeoSubTaskDecoder::setupDecoder(const char* inputColor, bool debug) {
  debugmode = debug;
  candidate_color = inputColor;

  if(!strcmp(candidate_color, "Green")){
    setupGreen();
  }
  else if(!strcmp(candidate_color, "Red")){
    setupRed();
  }
  else{
    printf("Cannot setup decoder without a color to decode! exiting...\n");
    return;
  }

  segmenter->SITsetTrackColor(&color,&std,&norm,&adapt,&upperBound,&lowerBound);

  if(debugmode){
    wini.reset(new XWindow(Dims(width, height), 0, 0, "test-input window"));
    wini->setPosition(0, 0);
    wino.reset(new XWindow(Dims(width, height), 0, 0, "test-output window"));
    wino->setPosition(width+10, 0);
  }

  setup = true;
}

// ######################################################################
bool BeoSubTaskDecoder::runDecoder(ImageSet< PixRGB<byte> > images, float framerate) {

  if(!setup){
    printf("Must setup decoder with a color and debug mode before running!\n");
  }
  imgList = images;
  fps = framerate;


  frameCounter.clear(); //added this line to prevent it from doing an average over time --Kevin
  // DECODE
  res = 0;
  float on_res = 0;
  float off_res = 0;
  int lastBlink = 0;

  for(uint j = 0; j < imgList.size(); j++) {

    Image<PixRGB<byte> > Aux;
    Aux.resize(100,450,true);

    /******************************************************************/
    // SEGMENT IMAGE ON EACH INPUT FRAME
    H2SVimage = imgList[j];
    display = imgList[j];

    segmenter->SITtrackImageAny(H2SVimage,&display,&Aux,true);

    /* Retrieve and Draw all our output images */
    Image<byte> temp = quickInterpolate(segmenter->SITreturnCandidateImage(),4);

    if(debugmode){
      wini->drawImage(display);
      wino->drawImage(temp);
    }

    /****************************************************************/
    // ADD EACH FRAME TO A VECTOR INDICATING IF LIGHT WAS ON/OFF
    if(!segmenter->SITreturnLOT()) {
      frameCounter.push_back(true);
      lastBlink = frameCounter.size();
      if(frameCounter[frameCounter.size() - 2] == false)
        on_res++;
    }
    else {
      frameCounter.push_back(false);
      off_res++;
    }


 //      // if the camera caught a remant of the light ON from the previous frame, ignore it
//       if(frameCounter.size() > 2 && frameCounter[frameCounter.size() - 1] == true
//          && frameCounter[frameCounter.size() - 2] == false)
//         frameCounter.push_back(false);
//       else
//         frameCounter.push_back(true);

//     }
//     else
//       frameCounter.push_back(false);

  }

  ///  if(lastBlink < (int)frameCounter.size())
  ///  off_res - frameCounter.size() + lastBlink;

  frameCounter.resize(lastBlink);
  res = off_res / on_res;
  printf("off_res: %f\n", off_res);
  printf("on_res: %f\n", on_res);
  printf("res: %f\n", res);
  //for(uint j = 1; j < frameCounter.size(); j++) {

    // look at the blinker as a clock, we just like for the rising edge.
    //if(frameCounter[j] == true && frameCounter[j-1] == false)

  //calculateHz();
  //order();
  return(true); //junk return, cuz it balks otherwise

}

float BeoSubTaskDecoder::calculateHz() {

  int numBlinks = 0;
  int last_on = -1;

  if(frameCounter[0])
    numBlinks++;

  std::cout << "frameCounter[0]: " << frameCounter[0] << '\n';

  for(uint j = 1; j < frameCounter.size(); j++) {

    std::cout << "frameCounter[" << j << "]: " << frameCounter[j] << '\n';


    if(frameCounter[j]) {
      last_on = j;
      // fills in small holes, if color tracker did not pick up entire length of blink.
      // depends on resolution
      if(res > 1 && last_on >= 0 && last_on == (int)j - 2)
        frameCounter[j - 1] = true;

    //rising edge
      if(!frameCounter[j-1])
        numBlinks++;
    }
  }


  float secs = frameCounter.size() / fps;

  Hz = (float)numBlinks / secs;

  printf("res: %f\n", res);
  printf("numBlinks: %d\n", numBlinks);
  printf("fps: %f\n", fps);
  printf("secs: %f\n", secs);

  // unsigned int traverse = 0;

//   while(traverse < frameCounter.size() && frameCounter[traverse] == false)
//     traverse++;


//   float denominator = 0;
//   float numerator = traverse;

//   for(unsigned int i = traverse; i < frameCounter.size(); i++) {
//     std::cout << i << ": ";

//     if(frameCounter[i] == true) {
//       if(debugmode)
//         std::cout << "true\n";

//       denominator++;

//       if(frameCounter.size() > 2 && fps > 15 && frameCounter[i - 2] == true)
//         denominator--;
//     }
//     else {
//       if(debugmode)
//         std::cout << "false\n";

//       numerator++;
//     }

//   }

//   Hz = denominator / ((denominator + numerator) / fps);

  if(debugmode){
   // std::cout << "there are " << denominator << " trues\n";
  //  std::cout << "there are " << numerator << " falses\n";
    std::cout << "frequency is about " << Hz << "Hz\n";
  }

  float ones = (int)Hz;
  float tenths = (int)((Hz - ones) * 10 + 0.5);

  float final_Hz = ones + tenths/10;
  return(final_Hz);

}

void BeoSubTaskDecoder::order() {
   if(strcmp(candidate_color, "Red") == 0 && Hz == 5.0F) {}
     //this is the order
   else {}
}
