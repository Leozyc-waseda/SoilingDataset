/*!@file AppPsycho/psycho-movie.C Psychophysics display of movies */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-corner-movie.C $
// $Id: psycho-corner-movie.C 13712 2010-07-28 21:00:40Z itti $
//

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Media/MPEGStream.H"
#include "Media/MediaOpts.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "Video/VideoFrame.H"
#include "rutz/time.h"

#include <deque>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <assert.h>

#define CACHELEN 150

// ######################################################################
static bool cacheFrame(nub::soft_ref<InputMPEGStream>& mp,
                       std::deque<VideoFrame>& cache)
{
  const VideoFrame frame = mp->readVideoFrame();
  if (!frame.initialized()) return false; // end of stream

  cache.push_front(frame);
  return true;
}

// ######################################################################
void Tokenize(const std::string& str,
              std::vector<std::string>& tokens,
              const std::string& delimiters = " ")
{
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  std::string::size_type pos     = str.find_first_of(delimiters, lastPos);
  while (std::string::npos != pos || std::string::npos != lastPos)
  {
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
}

// ######################################################################
static int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Movie");

  // Instantiate our various ModelComponents:
  nub::soft_ref<InputMPEGStream> mp
    (new InputMPEGStream(manager, "Input MPEG Stream", "InputMPEGStream"));
  manager.addSubComponent(mp);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  manager.setOptionValString(&OPT_InputMPEGStreamPreload, "true");
  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<display_loc.txt> <movie1.mpg> ... <movieN.mpg>", 2, -1)==false)
    return(1);

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  // EyeLink opens the screen for us, so make sure SDLdisplay is slave:
  if (etc->getModelParamString("EyeTrackerType").compare("EL") == 0)
    d->setModelParamVal("SDLslaveMode", true);

  // let's get all our ModelComponent instances started:
  manager.start();

  // read the display_loc.txt
  const char * display_loc = manager.getExtraArg(0).c_str();
  std::ifstream in( display_loc );

  if (! in) {
    LINFO("error: unable to open display_loc file: %s", display_loc);
    return -1;
  }

  std::vector< std::vector<std::string> > cdl; //cdl = clip display location
  std::string line;
  while( !getline(in, line).eof()){
    std::vector<std::string> tokens;
    Tokenize(line, tokens);
    cdl.push_back(tokens);
  }
  in.close();

  // setup array of movie indices:
  uint nbmovies = manager.numExtraArgs()-1;
  int index[nbmovies];
  for (uint i = 0; i < nbmovies; i ++) index[i] = i;
  LINFO("Randomizing movies..."); randShuffle(index,nbmovies);

  // calibration
  et->calibrate(d);

  if (etc->getModelParamString("EyeTrackerType").compare("EL") == 0){
                et->closeSDL();
                d->openDisplay();
                LINFO("Switching SDL: EyeLink-->iLab");
        }else{
          d->clearScreen();
        }

  d->displayText("<SPACE> to start watching movies");
  d->waitForKey();

  // get background color
  PixRGB<byte> ori_bgcolor = d->getGrey();
  PixRGB<byte> black_bgcolor = PixRGB<byte>(0,0,0);

  // main loop:
  for (uint i = 0; i < nbmovies; i ++)
  {
      // obtain display location
      int x = 100, y = 100, w = 1280, h = 1024; //w = 1233, h = 856;
      int sw = d->getWidth(), sh = d->getHeight();
      std::string playmovie = manager.getExtraArg(index[i]+1).c_str();
      std::vector< std::vector<std::string> >::iterator iter_ii;
      std::vector<std::string>::iterator iter_jj;
      std::string loc = "default";
      for(iter_ii=cdl.begin(); iter_ii!=cdl.end(); iter_ii++)
      {
          iter_jj = (*iter_ii).begin();
          if(playmovie.find(*iter_jj, 0) != std::string::npos){
              iter_jj++;
              loc = *iter_jj;
              if(loc.compare("0") == 0){ //center
                  x = sw/2 - w/2;
                  y = sh/2 - h/2;
              }else if(loc.compare("1") == 0){ //upper right
                  x = sw - w; y = 0;
              }else if(loc.compare("2") == 0){ //upper left
                  x = 0; y = 0;
              }else if(loc.compare("3") == 0){ //lower right
                  x = sw - w; y = sh - h;
              }else if(loc.compare("4") == 0){ //lower left
                  x = 0; y = sh - h;
              }else{ //default, display at the center
                  x = sw/2 - w/2;
                  y = sh/2 - h/2;
              }
          }
      }

      d->changeBackgroundColor(black_bgcolor);
      // cache initial movie frames:
      d->clearScreen();
      bool streaming = true;
      LINFO("Buffering '%s'...", manager.getExtraArg(index[i]+1).c_str());
      mp->setFileName(manager.getExtraArg(index[i]+1));

      std::deque<VideoFrame> cache;
      for (uint j = 0; j < CACHELEN; j ++)
        {
          streaming = cacheFrame(mp, cache);
          if (streaming == false) break;  // all movie frames got cached!
        }
      LINFO("'%s' ready.", manager.getExtraArg(index[i]).c_str());

      // give a chance to other processes (useful on single-CPU machines):
      sleep(1); if (system("/bin/sync")) LERROR("error in sync");

      // display fixation to indicate that we are ready:
      d->displayRedDotFixation();

      // ready to go whenever the user is ready:
      d->waitForKey(); int frame = 0;
      d->waitNextRequestedVsync(false, true);
      d->pushEvent(std::string("===== Playing movie: ") +
                   manager.getExtraArg(index[i]+1) + " =====");
      d->pushEvent(std::string("movie is displayed at location: ") + loc);

      // start the eye tracker:
      et->track(true);

      // blink the fixation:
      d->clearScreen();
      d->displayRedDotFixationBlink(x+w/2, y+h/2);

      // create an overlay:
      d->createVideoOverlay(VIDFMT_YUV420P, mp->getWidth(), mp->getHeight());

      // play the movie:
      d->changeBackgroundColor(PixRGB<byte>(0,0,0));
      rutz::time start = rutz::time::wall_clock_now();
      while(cache.size())
        {
          // let's first cache one more frame:
          if (streaming) streaming = cacheFrame(mp, cache);

          // get next frame to display and put it into our overlay:
          VideoFrame vidframe = cache.back();
          d->displayVideoOverlay_pos(vidframe, frame,
                                 SDLdisplay::NEXT_VSYNC,
                                 x, y, w, h);
          cache.pop_back();

          ++frame;
        }
      rutz::time stop = rutz::time::wall_clock_now();
      const double secs = (stop-start).sec();
      LINFO("%d frames in %.02f sec (~%.02f fps)", frame, secs, frame/secs);

      // destroy the overlay. Somehow, mixing overlay displays and
      // normal displays does not work. With a single overlay created
      // before this loop and never destroyed, the first movie plays
      // ok but the other ones don't show up:
      d->destroyYUVoverlay();
      d->clearScreen();  // sometimes 2 clearScreen() are necessary

      // stop the eye tracker:
      usleep(50000);
      et->track(false);

                        if(i%10 == 0 && i> 0 && i<nbmovies-1) {
                                // 5 minutes break, then do a full calibration
        d->changeBackgroundColor(ori_bgcolor);
        d->displayText("Please take a break, press <SPACE> to continue");
        d->waitForKey();

        if (etc->getModelParamString("EyeTrackerType").compare("EL") == 0){
                d->closeDisplay();
                et->openSDL();
                et->calibrate(d);
          et->closeSDL();
          d->openDisplay();
          LINFO("Switching SDL for quick calibration");
                                }else{
          d->clearScreen();
                d->displayISCANcalib();
                d->waitForKey();

          d->displayText("<SPACE> for eye-tracker calibration");
                d->waitForKey();
                d->displayEyeTrackerCalibration(3, 3);
                d->clearScreen();
                                }

        d->displayText("<SPACE> to start watching movies");
        d->waitForKey();
        d->changeBackgroundColor(black_bgcolor);
                        }
    }

  d->changeBackgroundColor(ori_bgcolor);
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

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
