/*!@file AppPsycho/psycho-monkey.C Psychophysics display of movies for monkeys */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-monkey.C $
// $Id: psycho-monkey.C 13795 2010-08-17 16:47:41Z beobot $
//

#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Transport/TransportOpts.H"
#include "Image/Image.H"
#include "Image/Range.H"
#include "Media/MPEGStream.H"
#include "Media/MediaOpts.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "GUI/SDLdisplay.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "Util/SimTime.H"
#include "Video/VideoFrame.H"
#include "rutz/rand.h"

#include <deque>
#include <sys/time.h>
#include <unistd.h>

#define CACHELEN 500


// ######################################################################
static bool cacheFrame(nub::soft_ref<InputMPEGStream>& mp,
                       std::deque<VideoFrame>& cache,
                       const bool flip)
{
  const VideoFrame frame = mp->readVideoFrame();
  if (!frame.initialized()) return false; // end of stream

  if (flip) cache.push_front(frame.getFlippedHoriz());
  else cache.push_front(frame);

  return true;
}

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Monkey");
  OModelParam<bool> hflip(&OPT_Hflip, &manager);
  OModelParam<bool> keepfix(&OPT_KeepFix, &manager);
  OModelParam<float> fixsize(&OPT_FixSize, &manager);
  OModelParam<float> ppd(&OPT_Ppd, &manager);
  OModelParam<Range<int> > itsWait(&OPT_TrialWait, &manager);
  OModelParam<bool> testrun(&OPT_Testrun, &manager);
  OModelParam<int> itsPercent(&OPT_GrayFramePrcnt, &manager);
  OModelParam<Range<int> > itsRange(&OPT_GrayFrameRange, &manager);

  // Instantiate our various ModelComponents:
  nub::soft_ref<InputMPEGStream> mp
    (new InputMPEGStream(manager, "Input MPEG Stream", "InputMPEGStream"));
  manager.addSubComponent(mp);

  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  manager.setOptionValString(&OPT_InputMPEGStreamPreload, "true");
  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "DML");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<movie1.mpg> ... <movieN.mpg>", 1, -1)==false)
    return(1);

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  //screen center and fixation point
  int fixrad = int(ppd.getVal() * fixsize.getVal());
  if ((fixrad % 2) != 0)
    --fixrad;
  d->setFixationSize(fixrad*2);

  const int cx = d->getWidth() /2 - 1 - (fixrad/2 -1);
  const int cy = d->getHeight()/2 - 1 - (fixrad/2 -1);

  //create a fixation patch
  Image<PixRGB<byte> > patch(fixrad,fixrad,ZEROS);
  patch += PixRGB<byte>(255,0,0);

  std::deque<VideoFrame> cache;
  initRandomNumbers();

  // let's get all our ModelComponent instances started:
  manager.start();
  d->clearScreen();

  // setup array of movie indices:
  const uint nbmovies = manager.numExtraArgs(); 
  const float percentage = (float)itsPercent.getVal() * 0.01F;
  const float factor = percentage/(1 - percentage);
  const uint graycount = uint(factor * (float)nbmovies);
  const uint nbtotal = nbmovies + graycount;

  LINFO("Randomizing movies...");
  int index[nbtotal];
  std::string fnames[nbtotal];
  for (uint i = 0; i < nbmovies; ++i) index[i] = i;
  for (uint i = nbmovies; i < nbtotal; ++i) index[i] = -1;
  randShuffle(index, nbtotal);
  for (uint i = 0; i < nbtotal; ++i) 
    fnames[i] = (index[i] < 0) ? "Gray image" :  manager.getExtraArg(index[i]);
  
  try {
    // main loop:
    for (uint i = 0; i < nbtotal; i ++)
      {
        // cache initial movie frames:
        d->clearScreen();
        if (cache.size()) LFATAL("ooops, cache not empty?");
        bool streaming = true;
        LINFO("Buffering '%s'...", fnames[i].c_str());

        // NOTE: we now specify --preload-mpeg=true with a
        // setOptionValString() a few lines up from here
        if (index[i] > -1)
          {
            mp->setFileName(fnames[i]);
            for (uint j = 0; j < CACHELEN; j ++)
              {
                streaming = cacheFrame(mp, cache, hflip.getVal());
                if (streaming == false) break;  // all movie frames got cached!
              }
          }

        LINFO("'%s' ready.", fnames[i].c_str());

        //wait for requested time till next trial
        usleep(rutz::rand_range<int>(itsWait.getVal().min(), 
                                     itsWait.getVal().max()) * 1000);


        // give a chance to other processes (useful on single-CPU machines):
        sleep(1); if (system("/bin/sync")) LERROR("Cannot sync()");
        LINFO("/bin/sync complete");

        // eat up any extra fixation codes and keys before we start
        // the next fixation round:
        while(et->isFixating()) ;
        while(d->checkForKey() != -1) ;

        // display fixation to indicate that we are ready:
        d->displayRedDotFixation();

        // give us a chance to abort
        d->checkForKey();

        // ready to go whenever the monkey is ready (pulse on parallel port):
        if (!testrun.getVal())
          while(true)
            {
              if (et->isFixating()) break;
              if (d->checkForKey() != -1) break; // allow force start by key
            }

        int frame = 0;
        if (index[i] > -1) d->waitNextRequestedVsync(false, true);
        d->pushEvent(std::string("===== Playing movie: ") +
                     fnames[i] + " =====");

        // start the eye tracker:
        et->track(true);

        if (index[i] > -1) 
          {
            // create an overlay:
            d->createVideoOverlay(VIDFMT_YUV420P); // mpeg stream uses YUV420P
            
            // play the movie:
            while(cache.size())
              {
                // let's first cache one more frame:
                if (streaming) streaming = cacheFrame(mp, cache, hflip.getVal());
                
                // get next frame to display and put it into our overlay:
                VideoFrame vidframe = cache.back();
                if (keepfix.getVal())
                  d->displayVideoOverlay_patch(vidframe, frame,
                                               SDLdisplay::NEXT_VSYNC, cx, cy,
                                               patch);
                else
                  d->displayVideoOverlay(vidframe, frame,
                                         SDLdisplay::NEXT_VSYNC);
                
                cache.pop_back();
                ++frame;
              }
            
            // destroy the overlay. Somehow, mixing overlay displays and
            // normal displays does not work. With a single overlay created
            // before this loop and never destroyed, the first movie plays
            // ok but the other ones don't show up:
            d->destroyYUVoverlay();
            d->clearScreen();  // sometimes 2 clearScreen() are necessary
          }
        else
          {
            d->clearScreen();  
            usleep(rutz::rand_range<int>(itsRange.getVal().min(), 
                                         itsRange.getVal().max()) * 1000);
          }


        // stop the eye tracker:
        usleep(50000);
        et->track(false);
      }

    d->clearScreen();
    LINFO("Experiment complete");
  }
  catch (...)
    {
      REPORT_CURRENT_EXCEPTION;
    };

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
