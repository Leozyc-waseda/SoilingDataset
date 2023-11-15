/*!@file Demo/test-tracking.C Test tracking  */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Demo/test-tracking.C $
// $Id: test-tracking.C 9412 2008-03-10 23:10:15Z farhan $
//

//
#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "GUI/XWindow.H"
#include "Image/DrawOps.H"
#include "Image/CutPaste.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/MathOps.H"
#include "Neuro/EnvVisualCortex.H"
#include "Media/FrameSeries.H"
#include "Media/MediaOpts.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Util/Timer.H"
#include "Util/log.H"
#include "Util/MathFunctions.H"
#include "Learn/Bayes.H"
#include "Learn/BackpropNetwork.H"
#include "Envision/env_image_ops.h"
#include "Neuro/BeoHeadBrain.H"

#include "GUI/DebugWin.H"
#include <ctype.h>
#include <deque>
#include <iterator>
#include <stdlib.h> // for atoi(), malloc(), free()
#include <string>
#include <vector>
#include <map>

void display(Image<PixRGB<byte> > &leftImg,
    const Image<byte> &leftSmap,
    const Point2D<int> &leftWinner,
    const byte maxVal,
    const Point2D<int> &targetLoc);
void display(const Image<PixRGB<byte> > &img,
    const Image<PixRGB<byte> > &smap, Point2D<int> &winner, Rectangle &rect);

//ModelManager *mgr;
XWinManaged *xwin;
Timer timer;
Image<PixRGB<byte> > disp;
byte SmaxVal = 0;
int smap_level = -1;

bool debug = 0;
bool init_points = true;


int main(int argc, const char **argv)
{
  // Instantiate a ModelManager:
  ModelManager *mgr = new ModelManager("USC Robot Head");

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(*mgr));
  mgr->addSubComponent(ifs);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
  mgr->addSubComponent(ofs);

  nub::ref<EnvVisualCortex> leftEvc(new EnvVisualCortex(*mgr));
  mgr->addSubComponent(leftEvc);

  nub::soft_ref<BeoHeadBrain> beoHeadBrain(new BeoHeadBrain(*mgr));
  mgr->addSubComponent(beoHeadBrain);


  mgr->exportOptions(MC_RECURSE);
  mgr->setOptionValString(&OPT_EvcMaxnormType, "None");
  mgr->setOptionValString(&OPT_EvcLevelSpec, "3,4,3,4,3");

  mgr->setOptionValString(&OPT_InputFrameSource, "V4L");
  mgr->setOptionValString(&OPT_FrameGrabberMode, "RGB24");
  mgr->setOptionValString(&OPT_FrameGrabberDims, "320x240");
  mgr->setOptionValString(&OPT_FrameGrabberFPS, "30");

  //leftEvc->setFweight(255);
  //leftEvc->setMweight(255);
  //leftEvc->setIweight(100);
  //leftEvc->setCweight(100);
  //leftEvc->setOweight(100);

  // Parse command-line:
  if (mgr->parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // do post-command-line configs:
  Dims imageDims = ifs->peekDims();

  xwin = new XWinManaged(Dims(imageDims.w()*2,imageDims.h()*2+20),
      -1, -1, "ILab Robot Head Demo");
  disp = Image<PixRGB<byte> >(imageDims.w(),imageDims.h()+20, ZEROS);

  // let's get all our ModelComponent instances started:
  mgr->start();

  smap_level = leftEvc->getMapLevel();

  //start streaming
  ifs->startStream();

  timer.reset();

  byte leftMaxVal;
  Point2D<int> leftMaxPos;

  int frame = 0;

  beoHeadBrain->init(imageDims);

  while(1) {

    Image< PixRGB<byte> > leftImg;
    const FrameState is = ifs->updateNext();
    if (is == FRAME_COMPLETE)
      break;

    //grab the images
    GenericFrame input = ifs->readFrame();
    if (!input.initialized())
      break;
    leftImg = input.asRgb();

    leftEvc->input(leftImg);

    ofs->writeRGB(leftImg, "input", FrameInfo("Copy of input", SRC_POS));

    Image<float> leftVcxmap = leftEvc->getVCXmap();
    inplaceNormalize(leftVcxmap, 0.0F, 255.0F);
    Image<byte> leftSmap = leftVcxmap;
    findMax(leftSmap, leftMaxPos, leftMaxVal);

    Image<byte> grey = luminance(leftImg);

    Point2D<int> clickPos = xwin->getLastMouseClick();
    //int key = xwin->getLastKeyPress();

    Point2D<int> targetLoc;
    if (clickPos.isValid())
    {
      beoHeadBrain->setTarget(clickPos, grey);
    } else {
      targetLoc = beoHeadBrain->trackObject(grey);
    }
    display(leftImg, leftSmap, leftMaxPos,
        SmaxVal, targetLoc);

    frame++;
  }


  // stop all our ModelComponents
  mgr->stop();

  // all done!
  return 0;
}

void display(Image<PixRGB<byte> > &leftImg,
    const Image<byte> &leftSmap,
    const Point2D<int> &leftWinner,
    const byte maxVal,
    const Point2D<int> &targetLoc)
{
  static int avgn = 0;
  static uint64 avgtime = 0;
  static double fps = 0;
  char msg[255];

  //Left Image
  drawCircle(leftImg,
      Point2D<int>(leftWinner.i *(1<<smap_level), leftWinner.j*(1<<smap_level)),
      30, PixRGB<byte>(255,0,0));
  drawCross(leftImg, Point2D<int>(leftImg.getWidth()/2, leftImg.getHeight()/2),
          PixRGB<byte>(0,255,0));
  sprintf(msg, "%i", maxVal);
  writeText(leftImg,
      Point2D<int>(leftWinner.i *(1<<smap_level), leftWinner.j*(1<<smap_level)),
        msg, PixRGB<byte>(255), PixRGB<byte>(127));

  if (targetLoc.isValid())
    drawCircle(leftImg, targetLoc, 3, PixRGB<byte>(0,255,0));

  xwin->drawImage(leftImg, 0, 0);
  Image<PixRGB<byte> > leftSmapDisp = toRGB(quickInterpolate(leftSmap, 1 << smap_level));
  //xwin->drawImage(leftSmapDisp, 0, leftImg.getHeight());


  //calculate fps
  avgn++;
  avgtime += timer.getReset();
  if (avgn == 20)
  {
    fps = 1000.0F / double(avgtime) * double(avgn);
    avgtime = 0;
    avgn = 0;
  }


  Image<PixRGB<byte> > infoImg(leftImg.getWidth()*2, 20, NO_INIT);
  writeText(infoImg, Point2D<int>(0,0), msg,
        PixRGB<byte>(255), PixRGB<byte>(127));
  //xwin->drawImage(infoImg, 0, leftImg.getHeight()*2);

}

void display(const Image<PixRGB<byte> > &img, const Image<PixRGB<byte> > &out,
    Point2D<int> &winner, Rectangle &rect)
{
  static int avgn = 0;
  static uint64 avgtime = 0;
  static double fps = 0;
  char msg[255];

  inplacePaste(disp, img, Point2D<int>(0,0));
  //drawCircle(disp, Point2D<int>(winner.i, winner.j), 30, PixRGB<byte>(255,0,0));
  drawRect(disp, rect, PixRGB<byte>(255,0,0));
  inplacePaste(disp, out, Point2D<int>(img.getWidth(), 0));

  //calculate fps
  avgn++;
  avgtime += timer.getReset();
  if (avgn == 20)
  {
    fps = 1000.0F / double(avgtime) * double(avgn);
    avgtime = 0;
    avgn = 0;
  }

  sprintf(msg, "%.1ffps ", fps);

  writeText(disp, Point2D<int>(0,img.getHeight()), msg,
        PixRGB<byte>(255), PixRGB<byte>(127));

  xwin->drawImage(disp);

}

