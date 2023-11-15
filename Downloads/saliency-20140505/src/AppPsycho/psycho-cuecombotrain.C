/*!@file AppPsycho/psycho-cuecombo.C Psychophysics display for a search for a
  target that is presented to the observer prior to the search */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-cuecombo.C $
// $Id: psycho-search.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "Component/ModelManager.H"
#include "Image/ColorOps.H" // for luminance/toRGB()
#include "Image/CutPaste.H" // for inplacePaste()
#include "Image/DrawOps.H" // for drawDisk()
#include "Image/Image.H"
#include "Image/LowPass.H"
#include "Image/MathOps.H"  // for addPowerNoise()
#include "Image/ShapeOps.H"
#include "Image/Range.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Raster/Raster.H"
#include "Util/MathFunctions.H"

#include <ctype.h>
#include <vector>

#define NMASK 5 /* number of different masks, doesn't matter much */
#define STIMRAD 160 /* radius of circle */
#define STIMMOD 24 /* amplitude of waves */
#define STIMTHICK 3 /* half the thickness of stimuli () */
#define NOISESIZE 400 /* size of the surrounding noise */
#define CONTRAST 0.6 /* contrast of circle, 1 is full black */ 
#define NOISECONTRAST 1.0 /* contrast of noise, 1 is full */

// times in milliseconds
#define SOA 250 /* stimulus presentation time */
#define ISI 500 /* stimulus + mask presentation time */
#define TIMEOUT 1500 /* total amount of response time */
#define FEEDBACK 250 /* amount of time for feedback */
#define TR 2000 /* total amount of time between trials */
#define FRAMETIME 30 /* approximate time between Vsyncs */

#define NOISECLARITY 4 /* integer, larger means more distinctly oriented */
#define NBLOCKS 20 /* number of blocks */
#define NTRANSFER 0 /* number of transfer blocks */
#define MAXTRIALS 60 /* number of maximum trials in a block */  
#define NREVERSALS 8

// TODO: 

char keyleft='j', keyright='k';

namespace CircleMorph
{
  // one category is a circle, the other is a six-petaled curve
  // these are free to rotate
  void draw(Image<PixRGB<byte> > &img, const double frac, byte const cg) {
    // find center
    Dims d(img.getDims());
    Point2D<int> center(d.w()/2,d.h()/2);
    
    float const radius = STIMRAD;
    float const maxModulation = STIMMOD;
    int const numCycles = 6;
    
    float const phase = randomDouble();
    float const dT = M_PI/120;

    PixRGB<byte> const col(cg,cg,cg);
    for(float T = -M_PI; T <= M_PI; T += dT) {
      float r = radius + maxModulation * frac * cos(phase + numCycles * T);
      Point2D<int> pp(center.i + r*cos(T), center.j + r*sin(T));
      drawDisk(img, pp, STIMTHICK, col);
    }
  }
}

namespace NullMorph
{ 
  // both categories have a circle at the center and a line through the center
  // categories vary in orientation    
  // the lines are free to vary in size
  void draw(Image<PixRGB<byte> > &img, const double frac, byte const cg) {

    // set a fixed boundary for the angle
    static const float angleOffset = M_PI * randomDouble();

    // find center
    Dims d(img.getDims());
    Point2D<int> center(d.w()/2,d.h()/2);
    
    PixRGB<byte> const col(cg,cg,cg);
    // draw the circle
    drawCircle(img, center, STIMRAD, col, STIMTHICK);

    // get the angle 
    float const angle = angleOffset + M_PI * frac;

    // set a random length
    float const len = STIMRAD + (randomDouble() * 2.0 - 1.0) * STIMMOD;
    drawLine(img, center, angle, 2*len, col, STIMTHICK);
  }
}

class stairParameter {
  public:
    stairParameter(double initStim = 0.0) : stimLevel(initStim), nCorrect(3), nIncorrect(1), nReversals(NREVERSALS),
                                            decRateBegin(0.5), incRate(0.25), decRate(0.125), initLevel(initStim)
    {reset();}
    void registerResult(bool thisResult) {
      if (currResult == thisResult) {
        currTrend++;
      }  else {
        currTrend = 1;
        currResult = thisResult;
        if(trialNumber != 0) currReversal++;
      }
      if(currResult && currTrend == nCorrect) {
        if(currReversal == 0) stimLevel *= (1 - decRateBegin);
        else stimLevel *= (1 - decRate);
        currTrend = 0;
      } else if (!currResult && currTrend == 1) {
        stimLevel *= (1 + incRate);
        currTrend = 0;
      }
      stimLevel = std::min(stimLevel, initLevel); // the stimulus level cannot exceed this
      trialNumber++;
    }
    
    int getNumTrials() const { return trialNumber;}
    void reset() {currResult = true; currTrend = 0; currReversal = 0; trialNumber = 0; stimLevel = initLevel;}
    bool done() const {return (currReversal == nReversals) || trialNumber >= MAXTRIALS;}
    double getStimLevel() const {return stimLevel;}

    // generate stimuli in interval [stimLevel/2,stimLevel)
    double generateStim() const {return randomDouble()*stimLevel/2 + stimLevel/2;}
    
  private:
    double stimLevel;
    int const nCorrect;
    int const nIncorrect;
    int const nReversals;
    double const decRateBegin;
    double const incRate;
    double const decRate;
    double const initLevel; // the stimulus level cannot be higher than this
    int currTrend;
    int currReversal;
    bool currResult;
    int trialNumber;
};

Image<PixRGB<byte> > drawFilteredNoise(Dims dims, float degreeAngle, byte maxAmp)
{
  // NB: this is pretty inefficient, but simple to code - faster may be a rotated non-separable kernel depending on the size
  Dims rotDims(dims.w()*fabs(cos(degreeAngle)) + dims.h()*fabs(sin(degreeAngle)), 
               dims.w()*fabs(sin(degreeAngle)) + dims.h()*fabs(cos(degreeAngle)));
  
  // create a much bigger image, filter it in the x dimension
  Image<float> imgx(rotDims,NO_INIT); //, imgy(dims, NO_INIT);
  for (Image<float>::iterator itr = imgx.beginw(); itr != imgx.endw(); itr++) *itr = randomDouble();
  imgx = lowPassX(pow(2,NOISECLARITY)+1,imgx);
  
  // rotate the large image
  Image<float> dblImg = rotate(imgx, rotDims.w()/2, rotDims.h()/2,degreeAngle) * maxAmp;
  
  // crop the large image
  Image<float> cropImg = crop(dblImg,
                 Point2D<int>(rotDims.w()/2 - dims.w()/2,rotDims.h()/2 - dims.h()/2), dims);

  return toRGB(Image<byte>(cropImg));
}

void addInPlace(Image<PixRGB<byte> > &src, Image<PixRGB<byte> > dst, Point2D<int> upperLeft)
{
  // check bounds
  Rectangle r(upperLeft, dst.getDims());
  assert(src.rectangleOk(r));

  // normalize around the mean
  int m = mean(luminance(dst));
  for(int i = 0; i < dst.getWidth(); i++)
    for(int j = 0; j < dst.getHeight(); j++) {
      int b = src.getVal(upperLeft.i + i, upperLeft.j + j).red() + dst.getVal(i,j).red() - m;

      src[Point2D<int>(upperLeft.i + i, upperLeft.j + j)] = PixRGB<byte>(b,b,b);
    }
}

// ######################################################################

Image<PixRGB<byte> > textScreen(std::vector<std::string> paragraphs, Dims scrDims)
{
  const uint lineWidth = 80;
  std::vector<std::string> lines;

  for (uint i = 0; i < paragraphs.size(); i++) {
    std::string str = paragraphs[i];
    while(!str.empty()) {
      // split string into good sized lines
      if(str.length() < lineWidth) {lines.push_back(str); str.erase();}
      else {
        int ctr = lineWidth;
        while(str.at(ctr) != ' ') ctr--;
        lines.push_back(str.substr(0,ctr)); // grab the space
        str.erase(0, ctr+1); 
      }
    }
    lines.push_back(""); // newline
  }

  Image<PixRGB<byte> > img(scrDims, ZEROS);
  img += PixRGB<byte>(128,128,128);
  Image<PixRGB<byte> > introTxt = makeMultilineTextBox(scrDims.w()*3/4, &lines[0], lines.size(), PixRGB<byte>(0,0,0), PixRGB<byte>(128,128,128), lineWidth, 11);
  inplacePaste(img, introTxt, img.getBounds().center() - introTxt.getBounds().center());

  return img;
}

// ######################################################################
static int submain(const int argc, char** argv)
{
  std::vector<std::string> introLines(3); 
  introLines.push_back("For this experiment your task will be to identify images as one of two different categories, either category A or category B.  You will not be given any explicit training, but you are expected to discover the categories on your own.  New images may be introduced at each block, but the categorization of an image will never change within this session.");
  
  introLines.push_back(sformat("In each trial, you will see the image briefly, followed by a mask which is not part of the image.  You have one and a half seconds to indicate whether the image is category A or B with a keypress for each category, starting when the image is displayed.  Press '%c' to indicate category A and '%c' to indicate category B.  Please answer as accurately and quickly as possible.  After each trial, you will receive feedback as to whether you were correct or not.  The next trial will then begin automatically.", keyleft, keyright));
  introLines.push_back(sformat("There are %d blocks of trials.  The experiment is designed to become progressively harder within each block (but not between blocks)  Each block has a maximum of %d trials, but may be shorter depending on your performance.  The session should take no more than 30 minutes.  Press any key to continue, and press any key to progress between blocks.", NBLOCKS, MAXTRIALS));

  MYLOGVERB = LOG_INFO;  // suppress debug messages

  initRandomNumbers();

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Search");

  // Instantiate our various ModelComponents:
  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);
  d->setEventLog(el);

  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  et->setEventLog(el);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<ntrials>", 0, -1) == false)
     return(1);

  // EyeLink opens the screen for us, so make sure SDLdisplay is slave:
  if (etc->getModelParamString("EyeTrackerType").compare("EL") == 0)
    d->setModelParamVal("SDLslaveMode", true);

  Dims ddims = d->getDims();

  // let's get all our ModelComponent instances started:
  manager.start();

  // preparing the instruction screen
  SDL_Surface *instruction;

  Image<PixRGB<byte> > intro = textScreen(introLines, ddims);

  LINFO("%s",toStr(intro.getDims()).c_str());
  instruction = d->makeBlittableSurface(intro, false);

  // let's prepare the 1/f mask images:
  SDL_Surface *mask[NMASK]; 
  for (int i = 0; i < NMASK; i++) {
    Image<double> powerNoise(NOISESIZE, NOISESIZE, ZEROS); 
    powerNoise = addPowerNoise(powerNoise, -1.0);
    powerNoise = remapRange(powerNoise, rangeOf(powerNoise), Range<double>(0.0,255.0));
    Image<PixRGB<byte> > byteNoise(ddims, ZEROS); byteNoise.clear(d->getGrey());

    inplacePaste(byteNoise, toRGB(Image<byte>(powerNoise)), 
                 Point2D<int>(ddims.w()/2-NOISESIZE/2, ddims.h()/2 - NOISESIZE/2));

    mask[i] = d->makeBlittableSurface(byteNoise, false);
  }
  
  d->displaySurface(instruction);
  d->waitForKey();

  // pick the A and B response keys in a random fashion, it doesn't really matter.
  if (coinFlip()) std::swap(keyleft, keyright);

  // prep for main (block) loop:
  // prepare two angles for left and right association over all blocks
  double const leftAngle = randomDouble() * M_PI/2;
  double const rightAngle = M_PI/2 + leftAngle;

  // response and experiment variables
  Timer t;
  char key;
  stairParameter stair(1.0);

  // block loop
  for (int block = 0; block < NBLOCKS; block ++) {
    int nCorrect = 0;
    // load up the images and show a fixation cross on a blank screen:
    d->clearScreen();
    
    // prepare the image surfaces for all trials in this block 
    Image< PixRGB<byte> > objimg(d->getDims(), NO_INIT);
    stair.reset();
    SDL_Surface *surf;
    std::vector<bool> isLeft;

    // announce block
    if (block + NTRANSFER < NBLOCKS) 
      el->pushEvent(sformat("Starting Block %d", block + 1));
    else
      el->pushEvent(sformat("Starting Block %d - TRANSFER", block + 1));

    d->displayText(sformat("Press any key to start block #%d/%d.", block + 1, NBLOCKS));
    d->waitForKey();

    // start w/ fixation
    d->clearScreen();
    d->displayFixation();
    bool const inTransfer = (block + NTRANSFER >= NBLOCKS);

    while(!stair.done()) {
      double RT = 0.0;
      // clear the image
      objimg.clear(d->getGrey());
      t.reset();

      // choose the identity - flip two coins, make identity XOR
      bool flip1 = coinFlip();
      bool flip2 = coinFlip();
      bool flip = (flip1 != flip2);

      isLeft.push_back(flip);
      // sample the category value from the staircase 
      double categoryVal = 0.5 + (flip1 ? -0.5 : 0.5) * stair.generateStim();

      // draw the circle figure according to the seed
      if(!inTransfer)
        CircleMorph::draw(objimg, categoryVal, 128.0 * (1.0 - CONTRAST));
      else
        NullMorph::draw(objimg, categoryVal, 128.0 * (1.0 - CONTRAST));

      // get angle that is random 
      double selAngle = (flip2) ? leftAngle : rightAngle;
      
        // plot the noise atop the circle
      Image<PixRGB<byte> > noisePatch = drawFilteredNoise(Dims(NOISESIZE,NOISESIZE), selAngle, 255 * NOISECONTRAST);
      addInPlace(objimg, noisePatch , Point2D<int>(ddims.w()/2-NOISESIZE/2,ddims.h()/2-NOISESIZE/2));
      surf = d->makeBlittableSurface(objimg, false);

      // flush the keybuffer
      while(d->checkForKey() != -1);

      // ready to go whenever the user is ready:
      d->waitNextRequestedVsync(false, true);

      // show the stimulus:
      d->displaySurface(surf, -1);
      el->pushEvent(sformat("Displaying curved stimuli #%zu...",isLeft.size()));

      // wait for a bit (but for a reliable amount of time, so no usleep here):
      double tt = SOA;
      key = d->waitForKeyTimed(tt);
      if (key == -1) RT += SOA; else RT += tt;

      // show a random 1/f mask:
      d->displaySurface(mask[randomUpToNotIncluding(NMASK)], -1);
      el->pushEvent(sformat("Displaying mask #%zu...",isLeft.size()));

      // wait for a bit (but for a reliable amount of time, so no usleep here):
      if (key == -1) {
        tt = ISI-SOA;
        key = d->waitForKeyTimed(tt);
        if (key == -1) RT += ISI-SOA; else RT += tt;
      }
      // clear the screen
      d->clearScreen();
      
      // get participant input for test, checking if it wasn't already input
      if (key == -1) {
        tt = TIMEOUT-ISI;
        key = d->waitForKeyTimed(tt);
        if (key == -1) RT += TIMEOUT-ISI; else RT += tt;
      }

      while(d->checkForKey() != -1) ;  // flush the buffer

      // evaluate the result
      bool isCorrect = ((key != -1) && ((key == keyleft && flip) || (key == keyright && !flip)) );
      stair.registerResult(isCorrect);

      // log event
      el->pushEvent(sformat("@@@@@@@@@@ (Trial,isCorrect, categoryValue, noiseDirection, RT) %d %d %f %d %f", 
                            stair.getNumTrials(), isCorrect, categoryVal, flip, RT));

      // show feedback
      if(isCorrect) {
        d->displayText("Correct!");
        nCorrect++;
      } else if(key == -1)
        d->displayText("Timed out.");
      else if (key != keyleft && key != keyright)
        d->displayText("Not a response key");
      else
        d->displayText("Incorrect.");

      // show feedback for enough timee 
      d->waitFrames(FEEDBACK/FRAMETIME);
      d->clearScreen();
      d->displayFixation();
      
      // free the SDL surface
      SDL_FreeSurface(surf);

      // wait for the rest of the trial time to pass
      while(t.getMilliSecs() < TR) usleep(5000);
    }

    // display results
    d->displayText(sformat("Results: (%d/%zu)",nCorrect,isLeft.size()));
    d->waitForKey();
    
    el->pushEvent(sformat("Final block results: %d/%zu, stimlevel %f", nCorrect, isLeft.size(), stair.getStimLevel()));
    // log results
  }

  d->clearScreen();
  d->displayText("Experiment complete. Thank you!");
  d->waitForKey();

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
    // simple wrapper around submain() to catch exceptions (because we
  // want to allow PsychoDisplay to shut down cleanly; otherwise if we
  // abort while SDL is in fullscreen mode, the X server won't return
  // to its original resolution)
  try
    {
      return submain(argc, argv);
    }
  catch (...)
    {
      REPORT_CURRENT_EXCEPTION;
    }

  return 1;
}
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

