/*!@file Demo/test-Head.C Test Head  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Demo/test-Head.C $
// $Id: test-Head.C 9412 2008-03-10 23:10:15Z farhan $
//

//
#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/FrameGrabberFactory.H"
#include "GUI/XWindow.H"
#include "Image/DrawOps.H"
#include "Image/CutPaste.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/MathOps.H"
#include "Neuro/EnvVisualCortex.H"
#include "Util/Timer.H"
#include "Util/log.H"
#include "Learn/Bayes.H"
#include "Devices/BeoHead.H"
#include "Envision/env_image_ops.h"

#include "GUI/DebugWin.H"
#include <ctype.h>
#include <deque>
#include <iterator>
#include <stdlib.h> // for atoi(), malloc(), free()
#include <string>
#include <vector>
#include <map>

#define NumFeatureNames 7
const char *featureNames[7] = {
        "red/green",
        "blue/yellow",
        "intensity",
        "steerable(01/04)",
        "steerable(02/04)",
        "steerable(03/04)",
        "steerable(04/04)"
        };


void display(Image<PixRGB<byte> > &leftImg,
    const Image<byte> &leftSmap,
    const Point2D<int> &leftWinner,
    const byte maxVal);
void display(const Image<PixRGB<byte> > &img,
    const Image<PixRGB<byte> > &smap, Point2D<int> &winner, Rectangle &rect);
Rectangle getBoundBox();
void showHist(std::vector<double> &hist, int pos);
void showProb(int pos);
void showFeatures(std::vector<double> &fv, int pos);
void findMinMax(std::vector<double> &vec, double &min, double &max);

ModelManager *mgr;
XWinManaged *xwin;
XWinManaged *debugXWin;
Timer timer;
Image<PixRGB<byte> > disp;
Point2D<int> curWinner;
Point2D<int> learnPos;
Dims imageDims;
Bayes* bayesNet;
byte SmaxVal = 0;

struct BiasParam
{
  double mean;
  double var;
};

#define NBINS 16
#define NFEATURES NBINS
#define FSIZE 75

std::map<const std::string, std::vector<double> > featuresHist;
std::vector<double> featuresVal(NumFeatureNames,0);


enum State {TRAIN, BIAS, MOVE, NO_BIAS};
State state = NO_BIAS;
Point2D<int> lastWinner;
int smap_level = -1;

void normalize(std::vector<double> &hist)
{
  double sum = 0;
  for(uint i=0; i<hist.size(); i++)
  {
    sum += hist[i];
  }

  for(uint i=0; i<hist.size(); i++)
  {
    if (sum > 0)
      hist[i] *= (1/sum);
    else
      hist[i] = 0;
  }
}

//distance based on the Bhattacharvva Coefficient
double compBhattHist(const std::vector<double> &a_hist,
    const std::vector<double> &b_hist)
{
      double bsum = 0;
      for(uint i=0; i<a_hist.size(); i++)
      {
        bsum += sqrt(a_hist[i]*b_hist[i]);
      }
      return (1-sqrt(1-bsum));
}

//distance based on equlicdean
double compEquHist(const std::vector<double> &a_hist,
    const std::vector<double> &b_hist)
{
      double bsum = 0;
      for(uint i=0; i<a_hist.size(); i++)
      {
        bsum += (a_hist[i]-b_hist[i])*(a_hist[i]-b_hist[i]);
      }
      return (1-sqrt(bsum));
}

//distance based on prob
double compProbHist(const std::vector<double> &a_hist,
    const std::vector<double> &b_hist)
{
      double bsum = 0;
      for(uint i=0; i<a_hist.size(); i++)
      {
        double mean = bayesNet->getMean(0, i);
        double var = bayesNet->getStdevSq(0, i);
        double val = a_hist[i];
        if (var != 0)
        {
          double delta = -(val - mean) * (val - mean);
          double newVal = log(exp(delta/(2*var))/sqrt(2*M_PI*var));
          //if (newVal < -1000) newVal = -1000;
          //src[i] = (intg32)newVal + 1000; //10000-abs(src[i]-featuresValues[std::string(uname)]);
          //printf("%f:%f %f:%f = %f\n", mean,var, val, b_hist[i], newVal);
          bsum += newVal;
        }
      }
      //printf("--%f\n\n", bsum);
      return (exp(bsum)/a_hist.size());
}

int cMapLearnProc(const char *tagName, struct env_image *cmap)
{
  intg32* src = cmap->pixels;
  int winX = learnPos.i >> smap_level; //shift the winner to the submap level
  int winY = learnPos.j >> smap_level; //shift the winner to the submap level

  int foveaWidth = (FSIZE >> smap_level)/2;
  int foveaHeight = (FSIZE >> smap_level)/2;

  //normalize the values from 0 to 255;
  env_max_normalize_none_inplace(cmap, 0, 255);

  int featureIdx = -1;
  for(int i=0; i<NumFeatureNames; i++)
  {
          if (!(strcmp(featureNames[i], tagName))){
                featureIdx = i;
                break;
        }
  }

  //std::vector<double> hist(NBINS,1);
  double maxVal = 0; int i= 0;
  for(int y=winY-foveaHeight; y<winY+foveaHeight; y++)
    for(int x=winX-foveaWidth; x<winX+foveaWidth; x++)
    {
      byte val = src[y*cmap->dims.w+x];
      //if (val > maxVal) maxVal = val;
                        maxVal += val;
                        i++;
      //hist[(int)val/(256/NBINS)]++;
    }
                maxVal = maxVal; ///i;

 // normalize(hist);*/

  /*featuresHist[std::string(tagName)] = hist;*/

  featuresVal[featureIdx] = maxVal; //src[winY*cmap->dims.w+winX];

  return 0;
}

int cMapLeftBiasProc(const char *tagName, struct env_image *cmap)
{
  intg32* src = cmap->pixels;
  //const env_size_t sz = env_img_size(cmap);
  uint winX = learnPos.i >> smap_level; //shift the winner to the submap level
  uint winY = learnPos.j >> smap_level; //shift the winner to the submap level

  uint foveaWidth = (FSIZE >> smap_level)/2;
  uint foveaHeight = (FSIZE >> smap_level)/2;

  //normalize the values from 0 to 255;

  int featureIdx = -1;
  for(int i=0; i<NumFeatureNames; i++)
  {
          if (!(strcmp(featureNames[i], tagName))){
                featureIdx = i;
                break;
        }
  }
  if (featureIdx != -1)
  {
                struct env_image *outImg = new struct env_image;
                env_img_init(outImg, cmap->dims);
                intg32* outSrc = outImg->pixels;

          env_max_normalize_none_inplace(cmap, 0, 255);
          double targetMean = bayesNet->getMean(0, featureIdx);
          double targetVar = bayesNet->getStdevSq(0, featureIdx);

          featuresVal[featureIdx] = src[winY*cmap->dims.w+winX];

  for(uint winY=0; winY<cmap->dims.h; winY++)
    for(uint winX=0; winX<cmap->dims.w; winX++)

    {

      if (winY>=foveaHeight && winY<cmap->dims.h-foveaHeight &&
          winX>=foveaWidth && winX<cmap->dims.w-foveaWidth)
      {
                                        double maxVal = 0; int i= 0;
                                        for(uint fy=winY-foveaHeight; fy<winY+foveaHeight; fy++)
                                                        for(uint fx=winX-foveaWidth; fx<winX+foveaWidth; fx++)
                                                        {
                                                                        byte val = src[fy*cmap->dims.w+fx];
                                                                        maxVal += val;
                                                                        i++;
                                                        }
                                        maxVal = maxVal; ///i;

                          double val = maxVal; //src[y*cmap->dims.w+x];
                          double delta = -(val - targetMean) * (val - targetMean);
                          double newVal = (exp(delta/(2*targetVar))/sqrt(2*M_PI*targetVar))*1000;
                          //if (newVal < -1000) newVal = -1000;
        outSrc[(winY)*cmap->dims.w+(winX)] = (intg32)newVal;
                        } else {
        outSrc[(winY)*cmap->dims.w+(winX)] = 0;
                  }
                }
        env_img_swap(cmap, outImg);
        env_img_make_empty(outImg);

        }

  return 0;
}

int cMapRightBiasProc(const char *tagName, struct env_image *cmap)
{
  intg32* src = cmap->pixels;

  uint foveaWidth = (FSIZE >> smap_level)/2;
  uint foveaHeight = (FSIZE >> smap_level)/2;


  struct env_image *outImg = new struct env_image;
  env_img_init(outImg, cmap->dims);
  intg32* outSrc = outImg->pixels;

  //normalize the values from 0 to 255;
  env_max_normalize_none_inplace(cmap, 0, 255);

  std::vector<double> fhist = featuresHist[std::string(tagName)];

  for(uint winY=0; winY<cmap->dims.h; winY++)
    for(uint winX=0; winX<cmap->dims.w; winX++)

    {

      if (winY>=foveaHeight && winY<cmap->dims.h-foveaHeight &&
          winX>=foveaWidth && winX<cmap->dims.w-foveaWidth)
      {
        std::vector<double> hist(NBINS, 1);
        //build histogram
        for(uint y=winY-foveaHeight; y<winY+foveaHeight; y++)
          for(uint x=winX-foveaWidth; x<winX+foveaWidth; x++)
          {
            byte val = src[y*cmap->dims.w+x];
            hist[(int)val/(256/NBINS)]++;
          }
        normalize(hist);

        //double compVal = compEquHist(hist, fhist);
        double compVal = compBhattHist(hist, fhist);
        //double compVal = compProbHist(hist, fhist);

        outSrc[(winY)*cmap->dims.w+(winX)] = (int)(compVal*256000000);

      } else {
        //zero out the borders
        outSrc[(winY)*cmap->dims.w+(winX)] = 0;
      }
    }

  env_img_swap(cmap, outImg);
  env_img_make_empty(outImg);

  //showHist(winHist, 256);
  return 0;
}

bool debug = 0;
bool training = false;
bool keepTraining = false;
bool tracking = false;
int  trackingTime = 0;

int main(int argc, const char **argv)
{
  // Instantiate a ModelManager:
  mgr = new ModelManager("USC Robot Head");

  nub::ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(*mgr));
  mgr->addSubComponent(gbc);

  nub::ref<EnvVisualCortex> leftEvc(new EnvVisualCortex(*mgr));
  mgr->addSubComponent(leftEvc);

  nub::soft_ref<BeoHead> beoHead(new BeoHead(*mgr));
  mgr->addSubComponent(beoHead);

  mgr->exportOptions(MC_RECURSE);
  mgr->setOptionValString(&OPT_EvcMaxnormType, "None");
  //mgr->setOptionValString(&OPT_EvcLevelSpec, "3,4,3,4,3");
  //mgr->setOptionValString(&OPT_EvcLevelSpec, "2,4,3,4,3");
  //cmin, cmac, smin,smax
  mgr->setOptionValString(&OPT_EvcLevelSpec, "2,4,1,1,3");

  leftEvc->setFweight(0);
  leftEvc->setMweight(0);
  //leftEvc->setCweight(0);
 //leftEvc->setOweight(0);

  // Parse command-line:
  if (mgr->parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  nub::soft_ref<FrameIstream> gbLeft(makeV4Lgrabber(*mgr));
  gbLeft->setModelParamVal("FrameGrabberDevice",std::string("/dev/video0"));
  gbLeft->setModelParamVal("FrameGrabberChannel",1);
  mgr->addSubComponent(gbLeft);

 // mgr->removeSubComponent(*gbc);
 // gbc.reset(NULL); // fully de-allocate the object and its children


  // do post-command-line configs:
  if (gbLeft.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
        "--fg-type=XX command-line option for this program "
        "to be useful");
  imageDims = Dims(gbLeft->getWidth(), gbLeft->getHeight());

  xwin = new XWinManaged(Dims(imageDims.w()*2,imageDims.h()*2+20), -1, -1, "ILab Robot Head Demo");
  debugXWin = new XWinManaged(Dims(255+10,255), -1, -1, "Debug Win");
  disp = Image<PixRGB<byte> >(imageDims.w(),imageDims.h()+20, ZEROS);

  // let's get all our ModelComponent instances started:
  mgr->start();

  smap_level = leftEvc->getMapLevel();

  //init the bayes net
 // bayesNet = new Bayes(NFEATURES, 1);
  bayesNet = new Bayes(NumFeatureNames, 1);

  //start streaming
  gbLeft->startStream();

  Image<double> featureValuesImg(72,1,NO_INIT);

  timer.reset();

  byte leftMaxVal;
  Point2D<int> leftMaxPos;
  double currentLeftPanPos = 0;
  double currentLeftTiltPos = 0;

  int frame = 0;

  beoHead->relaxNeck();

  while(1) {
    //grab the images
    Image< PixRGB<byte> > leftImg = gbLeft->readRGB();

    //pass the images to the vc

    leftEvc->input(leftImg);

    Image<float> leftVcxmap = leftEvc->getVCXmap();

    //make the right eye dominate
    //TODO: need to bias based on the eppolar line
    if (lastWinner.isValid())
    {
      int w = leftVcxmap.getWidth();
      int i = 0;
      for (Image<float>::iterator itr = leftVcxmap.beginw(), stop = leftVcxmap.endw();
          itr != stop; ++itr, i++) {
        int x = i % w;
        int y = i / w;
        float dist = lastWinner.distance(Point2D<int>(x,y));
        *itr = *itr - (1*dist);
      }
    }


    inplaceNormalize(leftVcxmap, 0.0F, 255.0F);
    Image<byte> leftSmap = leftVcxmap;

    findMax(leftSmap, leftMaxPos, leftMaxVal);

    lastWinner = leftMaxPos;

    Point2D<int> clickPos = xwin->getLastMouseClick();

    if (state == BIAS || state == MOVE)
      curWinner = Point2D<int>(leftMaxPos.i *(1<<smap_level),
          leftMaxPos.j*(1<<smap_level));

    if (state == TRAIN) //bias after training
    {
    //  state = BIAS;
   //   leftEvc->setSubmapPostProc(&cMapLeftBiasProc);
    //  lastWinner.i = -1; lastWinner.j = -1;
    }

    if (clickPos.isValid())
    {
      switch(state)
      {
        case TRAIN:
          state = BIAS;
          printf("Biasing smap\n");
         // leftEvc->setSubmapPreProc(&biasProc);
         //rightEvc->setSubmapPreProc(&biasProc);
          leftEvc->setSubmapPostProc(&cMapLeftBiasProc);
          //leftEvc->setSubmapPostProc(NULL);
          //curWinner = clickPos;
          lastWinner.i = -1; lastWinner.j = -1;
          //evc->setSubmapPreProc(&learnProc);
          break;
        case BIAS:
          printf("Moving\n");
          printf("TC: %i %i %f %f\n", leftMaxPos.i, leftMaxPos.j,
             currentLeftPanPos, currentLeftTiltPos);
          //state = MOVE;
          state = NO_BIAS;
          break;
        case MOVE:
          printf("Next pos\n");
          state = BIAS;
          /*state = NO_BIAS;
          curWinner.i = -1;
          curWinner.j = -1;
          leftEvc->setSubmapPreProc(NULL);
          leftEvc->setSubmapPostProc(NULL);*/
          break;
        case NO_BIAS:
          state = TRAIN;
          learnPos = clickPos;
          //learnPos = curWinner;
          curWinner = clickPos;
          //leftEvc->setSubmapPreProc(&learnProc);
          leftEvc->setSubmapPostProc(&cMapLearnProc);
          printf("Learning from %ix%i\n", learnPos.i, learnPos.j);
          break;
      }
    }

    if (state == BIAS && frame > 200)
    {
        printf("Moving\n");
        printf("TC: %i %i %f %f\n", leftMaxPos.i, leftMaxPos.j,
                            currentLeftPanPos, currentLeftTiltPos);
       // state = MOVE;
    }

    /*for(int i=0; i<20; i++)
    {
      printf("%f:%f ",
          bayesNet->getMean(0, i),
          bayesNet->getStdevSq(0, i));
    }
    printf("\n");*/

    //if (featuresHist["intensity"].size() > 0 && state==TRAIN)
    if (state==TRAIN)
    {
      bayesNet->learn(featuresVal, (uint)0);
    }


    //move head
    if (state == MOVE && frame > 20)
    {
     // printf("(%ix%i) p=%f t=%f\n",
     //     clickPos.i, clickPos.j,
     //     currentLeftPanPos, currentLeftTiltPos);

      double leftPanErr = (leftSmap.getWidth()/2)-leftMaxPos.i;
      //double leftPanErr = (leftImg.getWidth()/2)-clickPos.i;

      double leftTiltErr = (leftSmap.getHeight()/2)-leftMaxPos.j;
      //double leftTiltErr = (leftImg.getHeight()/2)-clickPos.j;

      if (fabs(leftPanErr) > 1.0 ||
          fabs(leftTiltErr) > 1.0)
      {
          //printf("TC: 0 0 %f %f\n", currentLeftPanPos, currentLeftTiltPos);
          //state = BIAS;


          //currentLeftPanPos += 0.01*leftPanErr;
          //currentLeftTiltPos += 0.01*leftTiltErr;

              currentLeftPanPos += (-0.0597*leftMaxPos.i+1.16);
              currentLeftTiltPos += (-0.0638*leftMaxPos.j + 0.942);
              //printf("%i %f %f\n", leftMaxPos.i, currentLeftPanPos, (-0.0598*leftMaxPos.i+1.16));
              //state = BIAS;



              beoHead->setLeftEyePan(currentLeftPanPos);
              beoHead->setLeftEyeTilt(currentLeftTiltPos);

              beoHead->setRightEyePan(currentLeftPanPos);
              beoHead->setRightEyeTilt(currentLeftTiltPos);

      }

      frame = 0;
    }

    int key = xwin->getLastKeyPress();

    if (key != -1)
    {
      switch(key)
      {
        case 98: //up
          currentLeftTiltPos += 0.05;
          if (currentLeftTiltPos > 1.0) currentLeftTiltPos = 1.0;
          break;
        case 104: //down
          currentLeftTiltPos -= 0.05;
          if (currentLeftTiltPos < -1.0) currentLeftTiltPos = -1.0;
          break;
        case 100: //left;
          currentLeftPanPos -= 0.05;
          if (currentLeftPanPos < -1.0) currentLeftPanPos = -1.0;
          break;
        case 102: //right
          currentLeftPanPos += 0.05;
          if (currentLeftPanPos > 1.0) currentLeftPanPos = 1.0;
          break;
      }

      printf("%f %f\n", currentLeftPanPos, currentLeftTiltPos);
      beoHead->setLeftEyePan(currentLeftPanPos);
      beoHead->setLeftEyeTilt(currentLeftTiltPos);

    }

   // showHist(featuresHist["intensity"], 0);
 //   if (state==TRAIN)
       showFeatures(featuresVal, 0);
 //   else
 //      showProb(0);

    display(leftImg, leftSmap, leftMaxPos,
        SmaxVal);
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
    const byte maxVal)
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

  xwin->drawImage(leftImg, 0, 0);
  Image<PixRGB<byte> > leftSmapDisp = toRGB(quickInterpolate(leftSmap, 1 << smap_level));
  xwin->drawImage(leftSmapDisp, 0, leftImg.getHeight());


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
  switch (state)
  {
    case TRAIN: sprintf(msg, "%s Train       ", msg); break;
    case BIAS: sprintf(msg, "%s Bias         ", msg); break;
    case MOVE: sprintf(msg, "%s Move         ", msg); break;
    case NO_BIAS: sprintf(msg, "%s No Bias       ", msg); break;
  }


  Image<PixRGB<byte> > infoImg(leftImg.getWidth()*2, 20, NO_INIT);
  writeText(infoImg, Point2D<int>(0,0), msg,
        PixRGB<byte>(255), PixRGB<byte>(127));
  xwin->drawImage(infoImg, 0, leftImg.getHeight()*2);

}

void display(const Image<PixRGB<byte> > &img, const Image<PixRGB<byte> > &out,
    Point2D<int> &winner, Rectangle &rect)
{
  static int avgn = 0;
  static uint64 avgtime = 0;
  static double fps = 0;
  char msg[255];

  inplacePaste(disp, img, Point2D<int>(0,0));
  drawCircle(disp, Point2D<int>(winner.i, winner.j), 30, PixRGB<byte>(255,0,0));
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

Rectangle getBoundBox()
{
  return Rectangle::tlbrI(130,130,190,190);
}

void showHist(std::vector<double> &hist, int loc)
{
  int w = 255, h = 255;

  if (hist.size() == 0) return;

  int dw = w / hist.size();
  Image<byte> res(w, h, ZEROS);

  // draw lines for 10% marks:
  for (int j = 0; j < 10; j++)
    drawLine(res, Point2D<int>(0, int(j * 0.1F * h)),
             Point2D<int>(w-1, int(j * 0.1F * h)), byte(64));
  drawLine(res, Point2D<int>(0, h-1), Point2D<int>(w-1, h-1), byte(64));

  double minii, maxii;
  findMinMax(hist, minii, maxii);

   // uniform histogram
  if (maxii == minii) minii = maxii - 1.0F;

  double range = maxii - minii;

  for (uint i = 0; i < hist.size(); i++)
    {
      int t = abs(h - int((hist[i] - minii) / range * double(h)));

      // if we have at least 1 pixel worth to draw
      if (t < h-1)
        {
          for (int j = 0; j < dw; j++)
            drawLine(res,
                     Point2D<int>(dw * i + j, t),
                     Point2D<int>(dw * i + j, h - 1),
                     byte(255));
          //drawRect(res, Rectangle::tlbrI(t,dw*i,h-1,dw*i+dw-1), byte(255));
        }
    }
  debugXWin->drawImage(res,loc,0);
}

void showFeatures(std::vector<double> &fv, int loc)
{
  int w = 255, h = 255;

  if (fv.size() == 0) return;

  int dw = w / fv.size();
  Image<byte> res(w, h, ZEROS);

  // draw lines for 10% marks:
  for (int j = 0; j < 10; j++)
    drawLine(res, Point2D<int>(0, int(j * 0.1F * h)),
             Point2D<int>(w-1, int(j * 0.1F * h)), byte(64));
  drawLine(res, Point2D<int>(0, h-1), Point2D<int>(w-1, h-1), byte(64));

  double minii = 0, maxii = 255;
  //findMinMax(fv, minii, maxii);


   // uniform histogram
  if (maxii == minii) minii = maxii - 1.0F;

  double range = maxii - minii;

  for (uint i = 0; i < fv.size(); i++)
    {
      int t = abs(h - int((fv[i] - minii) / range * double(h)));

      // if we have at least 1 pixel worth to draw
      if (t < h-1)
        {
          for (int j = 0; j < dw; j++)
            drawLine(res,
                     Point2D<int>(dw * i + j, t),
                     Point2D<int>(dw * i + j, h - 1),
                     byte(255));
          //drawRect(res, Rectangle::tlbrI(t,dw*i,h-1,dw*i+dw-1), byte(255));
        }
    }
  debugXWin->drawImage(res,loc,0);


}

void showProb(int loc)
{
  int w = 255, h = 255;
  uint size = NumFeatureNames;

  if (size == 0) return;

  int dw = w / size;
  Image<PixRGB<byte> > res(w, h, ZEROS);

  // draw lines for 10% marks:
  for (int j = 0; j < 10; j++)
    drawLine(res, Point2D<int>(0, int(j * 0.1F * h)),
             Point2D<int>(w-1, int(j * 0.1F * h)), PixRGB<byte>(64, 64, 64));
  drawLine(res, Point2D<int>(0, h-1), Point2D<int>(w-1, h-1), PixRGB<byte>(64, 64, 64));

  double minii = 0, maxii = 255;
  //findMinMax(fv, minii, maxii);


   // uniform histogram
  if (maxii == minii) minii = maxii - 1.0F;

  double range = maxii - minii;

  for (uint i = 0; i < size; i++)
    {
      double mean = bayesNet->getMean(0, i);
 //     double var = sqrt(bayesNet->getStdevSq(0, i));

      int t = abs(h - int((mean - minii) / range * double(h)));

      // if we have at least 1 pixel worth to draw
      if (t < h-1)
        {
          for (int j = 0; j < dw; j++)
            drawLine(res,
                     Point2D<int>(dw * i + j, t),
                     Point2D<int>(dw * i + j, h - 1),
                     PixRGB<byte>(255, 0, 0));

          //drawRect(res, Rectangle::tlbrI(t,dw*i,h-1,dw*i+dw-1), byte(255));
        }
    }
  debugXWin->drawImage(res,loc,0);


}

void findMinMax(std::vector<double> &vec, double &min, double &max)
{
  max = vec[0];
  min = max;
  for (uint n = 1 ; n < vec.size() ; n++)
  {
    if (vec[n] > max) max = vec[n];
    if (vec[n] < min) min = vec[n];
  }
}

