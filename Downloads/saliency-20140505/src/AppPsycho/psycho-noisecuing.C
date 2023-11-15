/*!@file AppPsycho/psycho-noisecuing.C Psychophysics display for a search for a
  target that is presented in various repeated noise backgrounds */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-searchGabor.C $
// $Id: psycho-searchGabor.C 10794 2009-02-08 06:21:09Z itti $
//

#include "Component/ModelManager.H"
#include "Image/ColorOps.H" // for makeRGB()
#include "Image/CutPaste.H" // for concatX()
#include "Image/DrawOps.H" // for drawLine()
#include "Image/Image.H"
#include "Image/ShapeOps.H" // for rescale()
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/ClassicSearchItem.H"
#include "Psycho/SearchArray.H"
#include "Component/EventLog.H"
#include "Raster/Raster.H"
#include "Util/StringUtil.H"
#include "GUI/GUIOpts.H"

#include <sstream>
#include <ctime>
#include <vector>
#include <string>

using namespace std;

// Trial design for contextual cueing experiments.  May class this up soon. 
// Also may make Trial/Experiment classes as ModelComponents of this style.
// Easier to encapsulate experimental designs as separate from trial content.

// But for now it's easier to keep everything public.
enum NoiseColor { WHITE, PINK, BROWN };
struct trialAgenda
{
  bool repeated; //will the trial have a repeated background?
  NoiseColor color;
  geom::vec2d targetloc;
  uint noiseSeed[3];
  Dims scrsize;
  trialAgenda(const bool r, const NoiseColor c, 
              const geom::vec2d t, Dims d)
  {
    repeated = r;
    color = c;
    targetloc = t;
    scrsize = d;
    randomizeNoise(1); // initialize noise to the same pattern at first
  }
  trialAgenda() //NULL initializer
  { }

  std::string colname() const
  {
    switch(color) {
    case WHITE: return "white";
    case PINK:  return "pink";
    case BROWN: return "brown";
    }
    return "";
  }

  void randomizeNoise(const uint Nbkgds)
  {
    for(uint i = 0; i < 3; i++) noiseSeed[i] = randomUpToNotIncluding(Nbkgds)+1;
  }

  std::string backgroundFile(const uint comp) const //components are R=0,G=1,B=2
  {
    //unfortunate hard-coded path
    std::string stimdir = "/lab/jshen/projects/eye-cuing/stimuli/noiseseeds";
    std::string backFile = sformat("%s/%s%03d.png",stimdir.c_str(),colname().c_str(),noiseSeed[comp]);
    
    // check if file exists
    if(!Raster::fileExists(backFile))
      {
        // stop all our ModelComponents
        //        manager.stop();
        LFATAL("Image file %s not found.", backFile.c_str());
      }
    return backFile;
  }

  Image<PixRGB<byte> > generateBkgd() {
    return colorizedBkgd();
  }
  Image<PixRGB<byte> > colorizedBkgd() {
    std::vector<Image<byte> > comp;
  
    for(uint i = 0; i < 3; i++) {
      //skips file validation step - dangerous
      Image<PixRGB<byte> > tmp = Raster::ReadRGB(backgroundFile(i));
      comp.push_back(getPixelComponentImage(tmp,i));
    }
    Image<PixRGB<byte> > RGBnoise = crop(makeRGB(comp[0],comp[1],comp[2]),Point2D<int>(0,0),scrsize);
  
    //lower contrast by mixing with gray (64,64,64) or white (128,128,128) 
    RGBnoise = RGBnoise/2 + PixRGB<byte>(64,64,64);
    return RGBnoise;
  }
};

std::string convertToString(const trialAgenda& val)
{
  std::stringstream s; 
  s << val.colname() << " noise, ";

  if (val.repeated)
    s << "repeated, seed (";
  else
    s << "random, seed (";
  for(uint i = 0; i < 3; i++) 
    s << val.noiseSeed[i] << (i<2 ? "," : ")");

  // target is printed with respect to origin at center - translating to origin at upper left
  s << ", target @ (" << val.targetloc.x() + val.scrsize.w()/2 << "," << val.targetloc.y() + val.scrsize.h()/2 << ")";
  return s.str();
}

// Utility functions 

// Generate a random integer uniformly in (x,y);
int randomInRange(const int x, const int y)
{
  return randomUpToNotIncluding(y-x-1)+(x+1);
}

// Generate a random point uniformly in d
geom::vec2d randomPtIn(const Rectangle d)
{
  return geom::vec2d(randomInRange(d.left(),d.rightO()),
                     randomInRange(d.top(),d.bottomO()));
}



// ######################################################################
static int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Visual Search - cuing");

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  // get post-command-line configs:
  if (manager.parseCommandLine(argc, argv, "<Nblocks> <setSize>",2,2) == false)
    return(1);

  // hook our various babies up:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  // EyeLink opens the screen for us, so make sure SDLdisplay is slave:
  if (etc->getModelParamString("EyeTrackerType").compare("EL") == 0)
    d->setModelParamVal("SDLslaveMode", true);

  // let's get all our ModelComponent instances started:
  manager.start();
  
  // pick the left and right response keys
  // TODO: allow for arrow keys
  char keyleft, keyright;
  do {
    d->pushEvent("===== key for T pointed left =====");
    d->displayText("Press key for response for T pointed left:");
    keyleft = d->waitForKey();
    d->pushEvent("===== key for T pointed right =====");
    d->displayText("Press key for response for T pointed right:");
    keyright = d->waitForKey();
    if (keyleft == keyright) {
      d->displayText(sformat("You must choose two different keys (%d/%d).",keyleft,keyright));
      d->waitForKey();
    }
  }
  while (keyleft == keyright);

  // ********************* Initial calibration ********************//
  // let's display an ISCAN calibration grid:
 
  if (etc->getModelParamString("EyeTrackerType").compare("EL") == 0) {
	d->pushEventBegin("Calibration");
	et->setBackgroundColor(d);
	et->calibrate(d);
	d->pushEventEnd("Calibration");  
  }
  else if(etc->getModelParamString("EyeTrackerType").compare("ISCAN") == 0) {
    d->clearScreen();
    d->displayISCANcalib();
    d->waitForKey();

  // let's do an eye tracker calibration:
  d->displayText("Press any key to calibrate; other key to skip");
  int c = d->waitForKey();
  if (c == ' ') d->displayEyeTrackerCalibration(3, 3);
  }

  // we are ready to start:
  d->clearScreen();
  d->displayText("Press any key to start the experiment");
  d->waitForKey();

  // **************** Experimental settings *************** //

  // number of images in each block, etc.  
  const uint Nrepeats = 6; //repeats per block
  const uint Nblocks = fromStr<int>(argv[1]);; //blocks per experiment
  const uint Ntrials = 12; //blocks per trial
  const uint Nbreak = 5; //blocks per break

  // time out for each trial in seconds
  const double Ntimeout = 5.0; 

  // number of available noise frames in the stimuli folder
  const uint Nnoises = 256;

  // physical layout of search arrays
  const int itemsize = 90;
  const int setSize = fromStr<int>(argv[2]); // from cmd line
  const double grid_spacing = 120.0, min_spacing = grid_spacing; 
  
  // size of screen - should be no bigger than 1920x1080
  const Dims screenDims = 
    fromStr<Dims>(manager.getOptionValString(&OPT_SDLdisplayDims));
  
  // target/distractor type: we have choice of c o q - t l +
  const std::string Ttype = "T", Dtype = "L";
  const double phiMax = M_PI*5/8, phiMin = M_PI*3/8; //make the rotation easy to see
  
  ClassicSearchItemFactory 
    targetsLeft(SearchItem::FOREGROUND, Ttype, 
                itemsize,
                Range<double>(-phiMax,-phiMin)),
    targetsRight(SearchItem::FOREGROUND, Ttype, 
                 itemsize,
                 Range<double>(phiMin,phiMax)),
    distractors(SearchItem::BACKGROUND, Dtype, 
                itemsize,
                Range<double>(-M_PI/2,M_PI/2));

  // ******************** Trial Design ************************* //
  std::vector<rutz::shared_ptr<trialAgenda> > trials;
  int tIndex[Ntrials]; 
  NoiseColor colors[Ntrials];
  bool rep[Ntrials];

  SearchArray sarray(screenDims, grid_spacing, min_spacing, itemsize);
  const PixRGB<byte> gray(128,128,128);

  // Design and shuffle trials
  initRandomNumbers();
  for (uint i = 0; i < Ntrials; i++) 
    {
      tIndex[i] = i; // for index shuffling
      colors[i] = NoiseColor(i%3);
      rep[i] = (i < Nrepeats);
    }

  for (uint i = 0; i < Ntrials; i++)
    {
      // a random location for each target
      const geom::vec2d pos = randomPtIn(sarray.itemBounds());
      
      rutz::shared_ptr<trialAgenda> myTrial(new trialAgenda(rep[i],colors[i],pos,screenDims));
      myTrial->randomizeNoise(Nnoises); //initialize noise seed
      trials.push_back(myTrial);
    }
  
  char buf[256];
  //******************* block loop:***********************//
  for (uint iblock = 1; iblock <= Nblocks; iblock++) {

    // block design - shuffle blocks
    randShuffle(tIndex,Ntrials);   

    d->displayText(sformat("Beginning block %d",iblock));
    d->waitForKey();

    sprintf(buf,"===== Beginning block %u/%u =====", iblock,Nblocks);
    d->pushEvent(buf);
    LINFO("%s",buf);

    rutz::shared_ptr<trialAgenda> thisTrial;    
    //******************* trial loop:***********************//
    for (uint itrial = 1; itrial <= Ntrials; itrial++) {

      //send trial information to log
      sprintf(buf, "===== Block %u/%u, Trial %u/%u =====",
              iblock, Nblocks, itrial, Ntrials);
      d->pushEvent(buf);
      LINFO("%s",buf);

      // pick up the trial depending on this block's sequence
      thisTrial = trials[tIndex[itrial-1]];

      // clear display
      d->clearScreen();

      // If the background needs randomization, do it here
      if(!(thisTrial->repeated)) 
        thisTrial->randomizeNoise(Nnoises);  

      // get the background image
      Image< PixRGB<byte> > bkgdimg = thisTrial->generateBkgd();   

      // clear the search array of objects
      sarray.clear();

      // ******************** generate search array *********************//
      // pick target location and orientation, and place
      // NB: P is in the center-as-origin coordinate frame
      geom::vec2d P = thisTrial->targetloc;
      const bool isTargetLeft = (randomDouble() < 0.5);
      const std::string tarDir = isTargetLeft ? "left" : "right";
      if (isTargetLeft)
        sarray.addElement(targetsLeft.make(P));
      else
        sarray.addElement(targetsRight.make(P));

      // generate the distractors
      sarray.generateBackground(distractors, 2, false, 5000, false);
      // reduce # of distractors to correct amount
      sarray.pareBackground(setSize);
      
      // generate SDL display
      Image< PixRGB<byte> > img = bkgdimg + sarray.getImage() - gray;
      SDL_Surface *surf1 = d->makeBlittableSurface(img, true);
      
      // give a chance to other processes if single-CPU:
      usleep(200000);

      // drift calibration if we're on EyeLink
      if (etc->getModelParamString("EyeTrackerType").compare("EL") == 0) {
        et->recalibrate(d,-1);
        // start the eyetracker
        d->displayFixation();
        usleep(300000); //longer SOA to hopefully avoid crash
        et->track(true);
        d->displayFixationBlink(-1,-1,2,2); //short blink, 2 v-refreshes
      } 
      else if(etc->getModelParamString("EyeTrackerType").compare("ISCAN") == 0) {
        // display fixation, ready to go whenever the user is ready:
        d->displayFixation();
        d->waitForKey();
        // start the eyetracker 
        et->track(true);
        d->displayFixationBlink(-1,-1,2,2); //short blink, 2 v-refreshes
      }
      else { //no input
        d->displayFixation();
        usleep(100000);
        d->displayFixationBlink(-1,-1,2,2); //short blink, 2 v-refreshes
      }

      //************************ Display array************************//
      sprintf(buf, "===== Showing array: %s, turned %s =====",
              toStr(*thisTrial).c_str(), tarDir.c_str());
      LINFO("%s",buf);

      // actually push the array onto the screen
      d->waitNextRequestedVsync(false, true);
      d->pushEvent(buf);
      d->displaySurface(surf1, 0, true);

      // get response back for key (NB: doesn't always take key input for ISCAN)
      const int resp = d->waitForKeyTimeout(1000*Ntimeout);

      // response processing
      if(resp == -1) 
        sprintf(buf, "trial timed out");
      else if(resp != keyleft && resp != keyright)
        sprintf(buf,"trial mistaken - neither key pressed (%c, %c/%c)",resp, keyleft, keyright);
      else if((resp==keyleft) == isTargetLeft)
        sprintf(buf,"trial correct, correct = %s", tarDir.c_str());
      else
        sprintf(buf,"trial incorrect, correct = %s", tarDir.c_str());
              
      // push to both the queue and psych data file
      d->pushEvent(buf);
      LINFO("%s",buf);
      
      usleep(50000);
      et->track(false);

      // clear screen and free the image from memory
      d->clearScreen();
      SDL_FreeSurface(surf1); 
    } // end trial loop

    //allow for a break after every 5 blocks, then recalibrate    
    if (iblock%Nbreak==0 && iblock!=Nblocks) 
      {
        d->displayText("Please take a break: press any key to continue when ready.");
        d->waitForKey();

        if (etc->getModelParamString("EyeTrackerType").compare("EL") == 0) {
          d->pushEventBegin("Calibration");
          et->setBackgroundColor(d);
          et->calibrate(d);
          d->pushEventEnd("Calibration");  
        }
        else if(etc->getModelParamString("EyeTrackerType").compare("ISCAN") == 0) {
          d->clearScreen();
          d->displayISCANcalib();
          d->waitForKey();
          
          // let's do an eye tracker calibration:
          d->displayText("<SPACE> to calibrate; other key to skip");
          int c = d->waitForKey();
          if (c == ' ') d->displayEyeTrackerCalibration(3, 3);
        }
      }        
  }

  d->clearScreen();
  d->displayText("Search trials complete.");
  d->waitForKey();

  d->displayText("Which background is more familiar?");
  d->waitForKey();

  //************************ Background recall trials************************//
  int iRepeat = 0;
  for(uint i = 0; i < Ntrials; i++) {
    rutz::shared_ptr<trialAgenda> thisTrial = trials[i];
    if(!thisTrial->repeated) continue;
    iRepeat++;

    // NB: if this doesn't work, construct a new one
    trialAgenda randTrial = (*thisTrial); // deep copy? 
    Dims halfDims = screenDims/2;


    // generate two different noise patterns, downscaled by 1/2
    Image<PixRGB<byte> > rpt_bkgd = rescale(thisTrial->generateBkgd(),halfDims);
    randTrial.randomizeNoise(Nnoises); 
    Image<PixRGB<byte> > rand_bkgd = rescale(randTrial.generateBkgd(),halfDims);

    // flip a coin and decide left or right
    const bool isRepeatLeft = (randomDouble() < 0.5);
    const std::string repeatDir = isRepeatLeft ? "left" : "right";

    Image<PixRGB<byte> > two_bkgd;
    if (isRepeatLeft) 
      two_bkgd = concatX(rpt_bkgd, rand_bkgd);
    else
      two_bkgd = concatX(rand_bkgd,rpt_bkgd);
    
    // draw two pixel thick vertical line
    drawLine(two_bkgd, Point2D<int>(halfDims.w()-1, 0), Point2D<int>(halfDims.w()-1, halfDims.h()-1),
             PixRGB<byte>(255, 255, 0), 1);
    drawLine(two_bkgd, Point2D<int>(halfDims.w(), 0), Point2D<int>(halfDims.w(), halfDims.h()-1),
               PixRGB<byte>(255, 255, 0), 1);

    sprintf(buf, "===== Recognition for array: %s, on %s side =====",
            toStr(*thisTrial).c_str(), repeatDir.c_str());
    LINFO("%s",buf);

    d->displayText(sformat("Recognition trial %d",iRepeat));
    d->waitForKey();
    SDL_Surface *surf1 = d->makeBlittableSurface(two_bkgd, true);

    // push the two images onto the screen
    d->waitNextRequestedVsync(false, true);
    d->pushEvent(buf);
    d->displaySurface(surf1, 0, true);

    // get response back for key (NB: doesn't always take key input for ISCAN)
    const int resp = d->waitForKey();

    // response processing
    if(resp != keyleft && resp != keyright)
      sprintf(buf,"recognition trial mistaken - neither key pressed (%c, %c/%c)",resp, keyleft, keyright);
    else if((resp==keyleft) == isRepeatLeft)
      sprintf(buf,"recognition trial correct, correct = %s", repeatDir.c_str());
    else
      sprintf(buf,"recognition trial incorrect, correct = %s", repeatDir.c_str());
              
    // push to both the queue and psych data file
    d->pushEvent(buf);
    LINFO("%s",buf);
  }

  //************************ Target recall trials************************//
  d->displayText("Where would the target be? Click with the mouse.");
  d->waitForMouseClick();

  iRepeat = 0;
  for(uint i = 0; i < Ntrials; i++) {
    rutz::shared_ptr<trialAgenda> thisTrial = trials[i];
    if(!thisTrial->repeated) continue;
    iRepeat++;

    Image<PixRGB<byte> > rpt_bkgd = thisTrial->generateBkgd();


    sprintf(buf, "===== Target retrieval for array: %s =====",
            toStr(*thisTrial).c_str());
    LINFO("%s",buf);

    d->displayText(sformat("Target retrieval trial %d",iRepeat));
    d->waitForMouseClick();
    SDL_Surface *surf1 = d->makeBlittableSurface(rpt_bkgd, true);

    // push the two images onto the screen
    d->waitNextRequestedVsync(false, true);
    d->pushEvent(buf);
    d->displaySurface(surf1, 0, true);

    // get response back from mouse 
    d->showCursor(true);
    SDL_Event event;
    while( SDL_WaitEvent( &event )) {     
      if(event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT ) {
	break;
}	
}	
/*
      switch (event.type) {
      case SDL_MOUSEMOTION:
        printf("Mouse moved by %d,%d to (%d,%d)\n", 
               event.motion.xrel, event.motion.yrel,
               event.motion.x, event.motion.y);
        //break;
      case SDL_MOUSEBUTTONDOWN:
        printf("Mouse button %d pressed at (%d,%d)\n",
               event.button.button, event.button.x, event.button.y);
        break;
      case SDL_MOUSEBUTTONUP:
        printf("Mouse button %d raised at (%d,%d)\n",
               event.button.button, event.button.x, event.button.y);
        //break;

      }
    }*/
    d->showCursor(false);

    Point2D<int> resp(event.button.x,event.button.y);
    // response processing
    sprintf(buf,"target placed at %s",toStr(resp).c_str());
              
    // push to both the queue and psych data file
    d->pushEvent(buf);
    LINFO("%s",buf);
  }

  d->displayText("Experiment complete. Press any key to exit.  Thank you!");
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
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
