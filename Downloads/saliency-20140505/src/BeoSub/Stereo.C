/*!@file BeoSub/Stereo.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/Stereo.C $
// $Id: Stereo.C 14376 2011-01-11 02:44:34Z pez $
//

#include "BeoSub/CannyEdge.H"
#include "BeoSub/HoughTransform.H"
#include "Component/ModelManager.H"
#include "Devices/FrameGrabberFactory.H"
#include "GUI/XWindow.H"
#include "Image/ColorOps.H"
#include "Image/DrawOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/MorphOps.H"     // for closeImg()
#include "Image/Pixels.H"
#include "Image/Transforms.H"
#include "MBARI/Geometry2D.H"
#include "Raster/Raster.H"
#include "SIFT/VisualObject.H"
#include "SIFT/VisualObjectMatch.H"
#include "SIFT/VisualObjectMatchAlgo.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "rutz/shared_ptr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream> //needed for segmentImageTrackMC!
#include <math.h>
#include <sstream>
#include <string>
#include <vector>

#define YTHRESH 20
////////////////////////////////////////////////////
// Stereo Vision
// Randolph Voorhies & Andre Rosa
// 01/15/06
///////////////////////////////////////////////////

using namespace std;


struct SweepPoint {
  unsigned int currentIntensity, targetIntensity;
  int xpos, xpos2, ypos;
  float disparity;
};

string IntToString(int num)
{
  ostringstream myStream; //creates an ostringstream object
  myStream << num << flush;

  /*
   * outputs the number into the string stream and then flushes
   * the buffer (makes sure the output is put into the stream)
   */

  return(myStream.str()); //returns the string form of the stringstream object
}

void drawMark(Image< PixRGB<byte> > &cameraImage, int x, int y, int size, float dist) {

  drawCross(cameraImage, Point2D<int>(x,y), PixRGB<byte> (255,0,0), size/2);

  //writeText(cameraImage, Point2D<int>(x,y), (const char*)IntToString(dist).c_str());
}


int isolateRed(const Image< PixRGB<byte> > &inputImage,  Image< PixRGB<byte> > &outputImage) {
  int blackCount = 0;
  for(int j = 0; j < inputImage.getHeight(); j++ ) {
    for(int i = 0; i < inputImage.getWidth(); i++ )
    {

     /* float h1 = inputImage.getVal(i, j).H1();
      float h2 = inputImage.getVal(i, j).H2();
      float s = inputImage.getVal(i, j).S();
      float v = inputImage.getVal(i, j).V();*/


      float avgR = inputImage.getVal(i, j).red();
      float avgG = inputImage.getVal(i, j).green();
      float avgB = inputImage.getVal(i, j).blue();

      float avg = (avgR+avgG+avgB)/3.0;
      float sigma = pow((double)(avgR - avg), (double)2.) + pow((double)(avgG - avg), (double)2.) + pow((double)(avgB - avg), (double)2.);
      float stdDev = sqrt( (1./3.) * sigma );

      if (avgR > avgG && avgR > avgB && stdDev > 25 && outputImage.coordsOk(i,j)) {
        outputImage.setVal(i, j, 255);
      }
      else {
        if(outputImage.coordsOk(i,j)){
          outputImage.setVal(i, j, 0);
          blackCount++;
        }
      }

    }
  }

  return (blackCount*100)/(inputImage.getHeight() * inputImage.getWidth());
}


int main(int argc, char* argv[]) {

  Image< PixRGB<byte> > cameraImage;
  Image< PixRGB<byte> > cameraImage2;
  Image< PixRGB<byte> > fusedImg;

  std::vector<KeypointMatch> matchVec;
  matchVec.clear();

  float xMatchDist = 0;
  xMatchDist = 0;

  XWindow* xwin;
  XWindow* xwin2;

  ModelManager camManager("Stereo Tester");
  nub::ref<FrameIstream> gb(makeIEEE1394grabber(camManager, "left", "leftcam"));
  nub::ref<FrameIstream> gb2(makeIEEE1394grabber(camManager, "right", "rightcam"));

  camManager.addSubComponent(gb);
  camManager.addSubComponent(gb2);

  camManager.loadConfig("camconfig.pmap");
  // play with later camManager.loadConfig("camconfig.pmap");
  gb->setModelParamVal("FrameGrabberSubChan", 0);
  gb->setModelParamVal("FrameGrabberBrightness", 128);
  gb->setModelParamVal("FrameGrabberHue", 180);
  gb->setModelParamVal("FrameGrabberFPS", 15.0F);

  gb2->setModelParamVal("FrameGrabberSubChan", 1);
  gb2->setModelParamVal("FrameGrabberBrightness", 128);
  gb2->setModelParamVal("FrameGrabberHue", 180);
  gb2->setModelParamVal("FrameGrabberFPS", 15.0F);

  camManager.start();


  cameraImage = gb->readRGB();
  cameraImage2 = gb2->readRGB();

  Image<byte> depthMap(cameraImage2.getDims(), ZEROS);

  xwin = new XWindow(cameraImage2.getDims(), 0, 0, "left");
  xwin2 = new XWindow(cameraImage.getDims(), cameraImage.getWidth() + 10, 0, "right");
  new XWindow(Dims(320,480), 0, cameraImage.getHeight() + 30, "together at last");
  new XWindow(cameraImage2.getDims(), cameraImage.getWidth() + 10, cameraImage.getHeight() + 30, "one love");
  new XWindow(depthMap.getDims(), cameraImage.getWidth() + 10, cameraImage.getHeight()*2 + 60, "output");

  rutz::shared_ptr<VisualObject>
    voLeft(new VisualObject("left", "leftfilename", cameraImage2,
                            Point2D<int>(-1,-1), std::vector<float>(),
                            std::vector< rutz::shared_ptr<Keypoint> >(), true));
  rutz::shared_ptr<VisualObject>
    voRight(new VisualObject("right", "rightfilename", cameraImage,
                             Point2D<int>(-1,-1), std::vector<float>(),
                             std::vector< rutz::shared_ptr<Keypoint> >(), true));
  rutz::shared_ptr<VisualObjectMatch> matches(new VisualObjectMatch(voLeft, voRight, VOMA_SIMPLE, (uint)10));

  /*
  while(1) {
    cameraImage = gb->readRGB();
    cameraImage2 = gb2->readRGB();

    xwin->drawImage(cameraImage);
    xwin2->drawImage(cameraImage2);

    ///////////////////////////////////////////////////////////////////////
    voLeft.reset(new VisualObject("left", "leftfilename", cameraImage2));
    voRight.reset(new VisualObject("right", "rightfilename", cameraImage));
    matches.reset(new VisualObjectMatch(voLeft, voRight, VOMA_SIMPLE, (uint)20));


    cout << "cameraImage width: " << cameraImage.getWidth() << endl
         << "cameraImage Height: " << cameraImage.getHeight() << endl;


    matches->prune();

    float min = 15, max = 0;

    if(matches->checkSIFTaffine()) {

      fusedImg = matches->getMatchImage(1.0F);

      xwin3->drawImage(fusedImg);
      xwin4->drawImage(matches->getFusedImage());

      matchVec = matches->getKeypointMatches();



    ////////////////////////////////////////////////////////

      for(int i=0; i< (int)matchVec.size(); i++) {
        cout << "x1:" << matchVec[i].refkp->getX() << " y1: " << matchVec[i].refkp->getY();
        cout << "  x2:" << matchVec[i].tstkp->getX() << " y2: " << matchVec[i].tstkp->getY() << std::endl;

        xMatchDist = matchVec[i].refkp->getX() - matchVec[i].tstkp->getX();
        xMatchDist = abs(xMatchDist);
        // distance = 3 3/4 in = 0.3125
        // focal length = 419.5509189 pixel units
        float depth = (0.3125 * 419.5509189) / xMatchDist;
        // df / (xl - xr)
        //a mathematical model to reduce the change in error
        cout << "depth: " << depth << endl;
        float error = 0.1 * exp(0.275 * depth);

        cout << "error: " << error << endl;
        if(error <= 4)
          depth -= error;

        if(depth > 15 || depth < 0)
          depth = 15;

        int pixelDepth = (int)(255 - depth * 13);

        if(depth > max && depth < 15)
          max = depth;

        if(depth < min)
          min = depth;

        cout << "new depth: " << depth << endl;
        cout << "pixel depth: " << pixelDepth << endl;


        for(int j = -5; j <= 5; j++)
          for(int k = -5; k <= 5; k++)
          depthMap.setVal((int)matchVec[i].refkp->getX() + j, (int)matchVec[i].refkp->getY() + k, (int)(255 - depth * 13));



        drawMark(cameraImage2, (int)matchVec[i].refkp->getX(), (int)matchVec[i].refkp->getY(), (int)xMatchDist, (255 - depth * 13));
      }
    }

    cout << "max: " << max << " min: " << min << endl;
    xwin5->drawImage(depthMap);
    depthMap.clear();



    Raster::waitForKey();


    }*/
  ////////////////////////////////////////////////// disparity by color isolation ///////////////////////////////
  while(1) {
    cameraImage = gb->readRGB();
    cameraImage2 = gb2->readRGB();


   //Isolate green and white pixels in image, and union the results
    isolateRed(cameraImage, cameraImage);

    //Fill in any small holes in the image
    Image<byte> se(5, 5, ZEROS); se += 255;
    Image<byte> closed = closeImg(luminance(cameraImage), se);

    //Find the largest white area
    Image<byte> mask(closed.getDims(), ZEROS);
    Image<byte> camlargest; int largest_n = 0;
    Image<byte> msk(closed.getDims(), ZEROS);
    for (int j = 0; j < closed.getHeight(); j ++) {
      for (int i = 0; i < closed.getWidth(); i ++) {
        if (closed.getVal(i, j) == 255 && mask.getVal(i, j) == 0)
        {
          //LINFO("Flooding (%d, %d)...", i, j);
          msk.clear();
          int n = floodClean(closed, msk, Point2D<int>(i, j), byte(128), byte(255), 4);
          if (n > largest_n) { largest_n = n; camlargest = msk; }
          mask += msk;
        }
      }
    }


    Image<byte> lineImage(camlargest.getDims(), ZEROS);


    /*int xcount, ycount;
    xcount = ycount = 0;
    int xavg, yavg;
    xavg = yavg = 0;
    int pixcount = 0;
    for(int j = 0; j < camlargest.getHeight(); j++) {
      for(int i = 0; i < camlargest.getWidth(); i++) {
        if(camlargest.getVal(i, j) == 255) {
          pixcount++;
          ycount+=j;
          xcount+=i;
        }
      }
    }

    xavg = xcount / pixcount;
    yavg = ycount / pixcount;*/

    //drawCross(cameraImage, Point2D<int>(xavg, yavg), PixRGB<byte>(0, 255, 0), 10);

    //xwin->drawImage(cameraImage);

    //////////////////////////// isolate color ///////////////////////////////////////
  //Isolate green and white pixels in image, and union the results
    isolateRed(cameraImage2, cameraImage2);

    //Fill in any small holes in the image
    Image<byte> cam2largest;
    Image<byte> si(5, 5, ZEROS); si += 255;
    closed = closeImg(luminance(cameraImage2), si);

    //Find the largest white area
    mask = *(new Image<byte>(closed.getDims(), ZEROS));
    largest_n = 0;
    msk = *(new Image<byte>(closed.getDims(), ZEROS));
    for (int j = 0; j < closed.getHeight(); j ++) {
      for (int i = 0; i < closed.getWidth(); i ++) {
        if (closed.getVal(i, j) == 255 && mask.getVal(i, j) == 0)
        {
          //LINFO("Flooding (%d, %d)...", i, j);
          msk.clear();
          int n = floodClean(closed, msk, Point2D<int>(i, j), byte(128), byte(255), 4);
          if (n > largest_n) { largest_n = n; cam2largest = msk; }
          mask += msk;
        }
      }
    }

    //////////////////////////////////////////////////////////////////

    //apply hough transform ////////////////////////////////////
    //lineImage = *(new Image<byte>(largest.getDims(), ZEROS));

    ////////////////////// This is a little more complicated in that I need to see the camera outputs since Hough transform is not consistent.
    //////////////////////// The pixel method is probably what we just need for Task B. And I suggest we use the SIFT method for Task A.
    ////////////////// I will complete this at competition, when I have access to cameras, and if it seems necessary after thorough testing of the other
    /////////////// algorithmns. I am concerned whether SIFT will work for Task A, but we will just have to see. Maybe the pixel method will work for Task A.
    ////////////// But here is an outline of how this algorithmn should work:
    /* Select one line from the lines in the left camera Image. Check to see if it is the left or right edge, based on whether a black pixel resides to the left
       or right of it. Find a similar line in the right Image. If you do. Find the equations of both lines. Find the x values of each line when y = 0.
       Do a disparity calculation, and get the distance between those interpolated points.
       If you do not find the same line. Find a different line in the left camera Image, and repeat. The problem I have with this method is that it is very
       specific to Task B. I think with the pixel method, we can generalize it enough for other things. */
    /*vector<LineSegment2D> lines, lines2;
    unsigned char* edge, *edge2;
    char* file1 = NULL;

    canny(camlargest.getArrayPtr(), camlargest.getHeight(), camlargest.getWidth(), 0.7, 0.2, 0.97, &edge, file1);
    canny(cam2largest.getArrayPtr(), cam2largest.getHeight(), cam2largest.getWidth(), 0.7, 0.2, 0.97, &edge2, file1);

    Image<unsigned char> edgeImage(edge, camlargest.getWidth(), camlargest.getHeight());
    Image<unsigned char> edgeImage2(edge2, cam2largest.getWidth(), cam2largest.getHeight());

    lines = houghTransform(edgeImage, PI/180, 1, 50, cameraImage);
    lines2 = houghTransform(edgeImage2, PI/180, 1, 50, cameraImage2);
    SweepPoint pair, pair2;
    for(int i = 0; i < lines.size(); i++) {
      Point2D<int> testPoint = lines[i].point1();
      if(camlargest.coordsOK(testPoint.i + 5, testPoint.j)) {
        pair.currentIntensity = camlargest.getVal(testPoint.i + 5, testPoint.j);
        break;
      }
    }

    for(int i = 0; i < lines.size(); i++) {
      Point2D<int> testPoint = lines[i].point1();
      if(cam2largest.coordsOK(testPoint.i + 5, testPoint.j) AND pair.currentIntensity == cam2largest.getVal(testPoint.i + 5, testPoint.j)) {
        // solve for equation. same line found.
      }
      else{ // look for another line
      }
    }*/
    ///////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////
    //pixel approach
    // look for valid pixels, ie we will only sweep through the y positions that have valid pixels,
    // which are same color on both left and right camera.
    vector<SweepPoint> validPixels;
    for(int j = 0; j < camlargest.getHeight(); j++) {
      if(camlargest.getVal(0, j) == cam2largest.getVal(0, j)) {
        SweepPoint valid;
        valid.currentIntensity = camlargest.getVal(0, j);
        valid.currentIntensity == 255 ? valid.targetIntensity = 0 : valid.targetIntensity = 255;
        valid.ypos = j;
        valid.xpos = -1;
        valid.xpos2 = -1;
        validPixels.push_back(valid);
      }
    }


    // sweep through both images and find change in pixel intensity and save x positions found at in each image, we have found the edge of the object
    // note: for optimiatzation we should either stop when we reached the end of the image OR when all x positions have been found
    for(int i = 0; i < camlargest.getWidth(); i++) {
      for(unsigned int j = 0; j < validPixels.size(); j++) {
        if(validPixels[j].xpos == -1 && camlargest.getVal(i, validPixels[j].ypos) == validPixels[j].targetIntensity)
          validPixels[j].xpos = i;

        if(validPixels[j].xpos2 == -1 && cam2largest.getVal(i, validPixels[j].ypos) == validPixels[j].targetIntensity)
          validPixels[j].xpos2 = i;

      }
    }

    //int xavg, xavg2;
    float disparityAvg = 0.0;
    float sum = 0.0, mean = 0.0;
    Image< PixRGB<byte> > cross(camlargest);
    Image< PixRGB<byte> > cross2(cam2largest);

    // find the disparity and the standard deviation
    if(!validPixels.empty()) {
      vector<SweepPoint>::iterator iter = validPixels.begin();
      while(iter != validPixels.end()) {
        if((*iter).xpos != -1 && (*iter).xpos2 != -1) {
          xMatchDist = (*iter).xpos2 - (*iter).xpos;
          sum += (*iter).disparity = abs(xMatchDist);
          iter++;
        }
        else // if disparity not possible, remove it from the list
          validPixels.erase(iter);

      }

      mean = sum / validPixels.size();

      float stdsum = 0.0, stdsqr = 0.0;
      for (uint i = 0; i < validPixels.size(); i++) {
        stdsum = validPixels[i].disparity - mean;
        stdsqr+=(stdsum*stdsum);
      }

      float stddev = stdsqr / validPixels.size();
      stddev = (float)sqrt((double)stddev);

      // kick out those disparites that do not fit, and average those that do
      iter = validPixels.begin();
      while(iter != validPixels.end()) {
        if((*iter).disparity > (mean + stddev) || (*iter).disparity < (mean - stddev))
          validPixels.erase(iter);
        else {
          drawCross(cross, Point2D<int>((*iter).xpos, (*iter).ypos), PixRGB<byte>(0, 255, 0), 3);
          drawCross(cross2, Point2D<int>((*iter).xpos2, (*iter).ypos), PixRGB<byte>(0, 255, 0), 3);
          disparityAvg = (float)(*iter).disparity;
          iter++;
        }
      }

      // averaged disparity, depth is calculated later below
      disparityAvg = disparityAvg / validPixels.size();

      //xavg = validPixels[0].xpos;
      //xavg2 = validPixels[0].xpos2;

    }
    else //dummy values, no edge had been found.
      disparityAvg = -1.0;

    xwin->drawImage(cross);
    xwin2->drawImage(cross2);
    /*    lineImage = *(new Image<byte>(largest.getDims(), ZEROS));



    int xavg2, yavg2;
    int xcount2, ycount2;
    int pixcount2;
    xcount2 = ycount2 = 0;
    xavg2 = yavg2 = 0;
    pixcount2 = 0;
    for(int j = 0; j < largest.getHeight(); j++) {
      for(int i = 0; i < largest.getWidth(); i++) {
        if(largest.getVal(i, j) == 255) {
          pixcount2++;
          ycount2+=j;
          xcount2+=i;
        }
      }
    }

    xavg2 = xcount2 / pixcount2;
    yavg2 = ycount2 / pixcount2;

    */
    // drawCross(cameraImage2, Point2D<int>(xavg2, yavg2), PixRGB<byte>(0, 255, 0), 10);

    //xwin2->drawImage(cameraImage2);
    //int xavg2 = 0;


    float max, min;
    max = 15;
    min = 0;

    //xMatchDist = xavg - xavg2;
    //xMatchDist = abs(xMatchDist);
    // distance = 3 3/4 in = 0.3125
    // focal length = 419.5509189 pixel units
    float depth = (0.3125 * 419.5509189) / disparityAvg;
    // df / (xl - xr)
        //a mathematical model to reduce the change in error
    cout << "depth: " << depth << endl;
    float error = 0.1 * exp(0.275 * depth);

    cout << "error: " << error << endl;
    if(error <= 4)
      depth -= error;

    if(depth > 15 || depth < 0)
      depth = 15;

    int pixelDepth = (int)(255 - depth * 13);

    if(depth > max && depth < 15)
      max = depth;

    if(depth < min)
      min = depth;

    cout << "new depth: " << depth << endl;
    cout << "pixel depth: " << pixelDepth << endl;


    /*    for(int j = -5; j <= 5; j++)
      for(int k = -5; k <= 5; k++)
      depthMap.setVal(xavg + j, yavg + k, (int)(255 - depth * 13));*/



    //drawMark(cameraImage2, xavg, yavg, (int)xMatchDist, (255 - depth * 13));

    Raster::waitForKey();
  } ////////////////////////////////////////////-- end disparity by color isolation //////////////////////////////

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
