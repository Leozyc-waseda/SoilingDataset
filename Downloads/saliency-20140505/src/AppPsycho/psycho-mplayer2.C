/*!@file AppPsycho/psycho-mplayertextsearch.C Movie to text search sequence.
   A set of movie clips and a text file with a set of questions and answers are given at the cmdline.  Each trial consists of a movie clip (played in mplayer with audio), followed by a question and a regular grid of answers.
*/

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
// Primary maintainer for this file: John Shen <shenjohn@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-mplayer2.C $

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/DrawOps.H"
#include "Image/SimpleFont.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Util/Types.H"
#include "Util/StringConversions.H"
#include "Util/StringUtil.H"
#include "Psycho/MPlayerWrapper.H"
#include "Devices/SimpleLED.H"
#include <fstream>

#define HDEG 54.9

typedef struct trial
{
  std::string itsClip;
  std::string itsQuestion;
  std::vector<std::string> itsChoices;
  Image<PixRGB<byte> > itsQimage;
  Image<PixRGB<byte> > itsSimage;
  int itsFamily;
} SearchTrial;

// ######################################################################
int submain(const int argc, char** argv)
{

  // ********************************************************************
  // *** This portion initializes all the components ********************
  // ********************************************************************

  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Text");

  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  nub::soft_ref<MPlayerWrapper> player(new MPlayerWrapper(manager));
  manager.addSubComponent(player);

  nub::soft_ref<SimpleLED> recordLight(new SimpleLED(manager));
  manager.addSubComponent(recordLight);

  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<textfile> visual-angle-of-single-character grid-rows grid-columns",
                               4, 4)==false)
    return(1);

  // create an image frame for each sentence in our text file and store
  // it in a vector before we start the experiment, then we can just
  // present each frame like in psycho still

  //First read the text file and all the sentences
  //load our file
  std::ifstream *itsFile;
  itsFile = new std::ifstream(manager.getExtraArg(0).c_str());

  //error if no file
  if (itsFile->is_open() == false)
    LFATAL("Cannot open '%s' for reading",manager.getExtraArg(0).c_str());

  //some storage variables
  std::string line;
  std::string clipstem = "";
  std::vector<SearchTrial> expt(100);
  uint num_trials = 0, num_stems = 0;
  std::vector<uint> curr_stem_index;

  //loop through lines of file
  while (!itsFile->eof())
    {
      getline(*itsFile, line, '\n');

      //store the sentence and type (question or statement)
        if (line[0] == '>')//video
        {
          line.erase(0,1);
          expt[num_trials].itsClip = line;

          //clip filename has format <stem>[a-z].avi
          if(line.compare(0,line.size()-5,clipstem) != 0) //new stem
            {
              num_stems++;
              clipstem = line.substr(0,line.size()-5);
              curr_stem_index.push_back(num_trials);
            }
          expt[num_trials].itsFamily = num_stems;
          num_trials++;
        }
        else if (line[0] == '#') //question, always one line
        {
          line.erase(0,1);
          expt[num_trials-1].itsQuestion = line;
        }
        else if (line[0] == '!') //choice, first choice
        {
          line.erase(0,1);
          expt[num_trials-1].itsChoices.push_back(line);
        }
        else if (line[0] == '&')//sub for a carriage return
        {
          //not handled yet
        }
        else //choice, subsequent choices
        {
          expt[num_trials-1].itsChoices.push_back(line);
        }
    }
  itsFile->close();

  //now we have stored all of our sentences, lets create our search images
  int w = d->getWidth();//width and height of SDL surface
  int h = d->getHeight();

  double fontsize = fromStr<double>(manager.getExtraArg(1));
  uint fontwidth = uint(fontsize * w / HDEG);
  SimpleFont fnt = SimpleFont::fixedMaxWidth(fontwidth); //font

  //store a grid of equally spaced coordinates in a gridrows x gridcols grid;
  const uint gridrows = fromStr<uint>(manager.getExtraArg(2));
  const uint gridcols = fromStr<uint>(manager.getExtraArg(3));
  const uint gridslots = gridrows*gridcols;
  std::vector<int> x_coords(gridslots);
  std::vector<int> y_coords(gridslots);
  for (uint i = 0; i < gridrows; i++)
    {
    for(uint j = 0; j < gridcols; j++)
      {
        x_coords[gridcols*i+j] = (int( double(w*(j+1)) / (gridcols+1)));
        y_coords[gridcols*i+j] = (int( double(h*(i+1)) / (gridrows+1)));
     }
    }

  Point2D<int> tanchor;
  for (uint i = 0; i < num_trials; i++)
  {
    int space = 0;
    int hanchor = int(h/2) - int(fnt.h()/2); //center character half a height behind
    expt[i].itsQimage.resize(w,h);
    expt[i].itsQimage.clear(d->getGrey());

    space = int( double(w - fnt.w() * expt[i].itsQuestion.size()) / 2.0 );
    tanchor = Point2D<int>(space, hanchor);

    writeText(expt[i].itsQimage,tanchor,expt[i].itsQuestion.c_str(),
                    PixRGB<byte>(0,0,0),
                    d->getGrey(),
                    fnt);

    expt[i].itsSimage.resize(w,h);
    expt[i].itsSimage.clear(d->getGrey());
    for (uint j = 0; j < expt[i].itsChoices.size(); j++)
    {
      //place each choice in its place
      if(j >= gridslots) //if there are too many choices
      {
        LDEBUG("Trial %d, clip %s: Too many answer choices for the grid", i, expt[i].itsClip.c_str());
        break;
      }
      space = x_coords[j] - int( double(fnt.w() * expt[i].itsChoices[j].length()) / 2.0);
      hanchor = y_coords[j] - int(fnt.h()/2);
      tanchor = Point2D<int>(space, hanchor);

      writeText(expt[i].itsSimage,tanchor,expt[i].itsChoices[j].c_str(),
                    PixRGB<byte>(0,0,0),
                    d->getGrey(),
                    fnt);
    }
  }

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);
  player->setEventLog(el);

  // let's get all our ModelComponent instances started:
  manager.start();

  //turn the LED on
  recordLight->turnOn();
  el->pushEvent("===== pilot light on =====");

  // let's do an eye tracker calibration:
  et->calibrate(d);

  d->clearScreen();
  d->displayText("<space> for random order, other key for ordered play.");
  int c = d->waitForKey();
  d->clearScreen();

  int plan[num_trials];
  for (uint i = 0; i < num_trials; i++) plan[i] = i;
  if (c == ' ') //randomize while preserving order of dialogues
    {
      LINFO("Randomizing trials...");

      for(uint i = 0; i < num_trials; i++)
        plan[i] = expt[i].itsFamily-1;
      randShuffle(plan, num_trials);

      for(uint i = 0; i < num_trials; i++)
        plan[i]=curr_stem_index[plan[i]]++;

    }

  //turn the LED off
  recordLight->turnOff();
  el->pushEvent("===== pilot light off =====");

  std::string currvideo;
  // main loop:
  for (uint ii = 0; ii < num_trials; ii ++)
    {
                        currvideo = expt[plan[ii]].itsClip;
      //announce the video playing
      //next step - load clips & timings from text file instead of a prompt
      LDEBUG("Playing '%s'...",currvideo.c_str());
      player->setSourceVideo(currvideo);

      //give a chance to other processes (useful on single-CPU machines):
      sleep(1); if (system("/bin/sync")) LERROR("error in sunc()");

      //turn the LED off
      recordLight->turnOff();
      el->pushEvent("pilot light off");

      // display fixation to indicate that we are ready:
      d->displayFixation();

      // ready to go whenever the user is ready:
      d->waitForKey(true);
      d->displayFixationBlink();

      // start the eye tracker:
      et->track(true);

      //play the movie
      d->waitNextRequestedVsync(false, true);
      el->pushEvent(std::string("===== Playing movie: ") +
                                                                                currvideo + " =====");
      player->runfromSDL(d);

      //turn the LED on
      recordLight->turnOn();
      el->pushEvent("pilot light on");

      et->track(false);
                        // load up the frame and show a fixation cross on a blank screen:
      d->clearScreen();
      d->displayFixation();
      Image<PixRGB<byte> > image;

      //start with the question
      image = expt[plan[ii]].itsQimage;
      SDL_Surface* surf = d->makeBlittableSurface(image, true);
      LINFO("question '%d' ready.", ii);

      // ready to go whenever the user is ready:
      d->waitForKey();
      d->waitNextRequestedVsync(false, true);

      d->pushEvent(std::string("===== Showing Question: question#") +
                     toStr<int>(plan[ii])  + " =====");

      et->track(true);
      // blink the fixation:
      d->displayFixationBlink();

      // show the image:
      d->displaySurface(surf, -2);

      // wait for key:
      c = d->waitForKey();

      // free the image:
      SDL_FreeSurface(surf);

      // make sure display is off before we stop the tracker:
      d->clearScreen();

      //next with the search answers
      image = expt[plan[ii]].itsSimage;
      surf = d->makeBlittableSurface(image, true);

      LINFO("sentence '%d' ready.", plan[ii]);

      d->displayFixation();

      // ready to go whenever the user is ready:
      d->waitForKey();
      d->waitNextRequestedVsync(false, true);

      d->pushEvent(std::string("===== Showing Answer Grid: answer#") +
                     toStr<int>(plan[ii])  + " =====");

      // blink the fixation:
      d->displayFixationBlink();

      // show the image:
      d->displaySurface(surf, -2);

      // wait for key:
      c = d->waitForKey();

      // free the image:
      SDL_FreeSurface(surf);

      // make sure display if off before we stop the tracker:
      d->clearScreen();

      // stop the eye tracker:
      usleep(50000);
      et->track(false);

      // every 15 trials let's do a quickie eye tracker recalibration:
                        et->recalibrate(d,15);
                        d->clearScreen();
    }

  d->clearScreen();
  d->displayText("Experiment complete. Thank you!");
  d->waitForKey();


  //turn the LED off (should be in LED->stop as well)
  recordLight->turnOff();
  el->pushEvent("===== pilot light off =====");

        // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

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
