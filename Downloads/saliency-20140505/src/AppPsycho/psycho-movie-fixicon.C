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
// Primary maintainer for this file: Po-He Tseng <ptseng@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-movie-fixicon.C $
// $Id: psycho-movie-fixicon.C 14131 2010-10-13 05:59:51Z ilab $

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Raster/Raster.H"
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
                               "<movie1.mpg> ... <movieN.mpg>", 1, -1)==false)
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

  // vision test (generate random letters again if pressing 'r')
	int key;
	do {
		d->displayRandomText(7, 6);
  	key = d->waitForKey(true);
	} while(key == 114 || key == 101 || key == 116 || key == 102);

	// let's do an eye tracker calibration:
	d->pushEventBegin("Calibration");
	et->calibrate(d);
	d->pushEventEnd("Calibration");

	// load fixation icon
	Image< PixRGB<byte> > fixicon =
		Raster::ReadRGB(d->getModelParamString("PsychoDisplayFixationIcon"));

	// switching SDL from ilink to our SDL
	if (etc->getModelParamString("EyeTrackerType").compare("EL") == 0){
		et->closeSDL();
		d->openDisplay();
		LINFO("Switching SDL: Eyelink-->iLab");
	}else{
		d->clearScreen();
	}

  d->displayText("Press any key to start......");
  d->waitForKey(true);

  // setup array of movie indices:
  uint nbmovies = manager.numExtraArgs(); int index[nbmovies];
  for (uint i = 0; i < nbmovies; i ++) index[i] = i;
        LINFO("Randomizing movies...");
        randShuffle(index,nbmovies);

  // main loop:
	int x, y, w, h;
  for (uint i = 0; i < nbmovies; i ++)
    {
      // cache initial movie frames:
      d->clearScreen();
      bool streaming = true;
      LINFO("Buffering '%s'...", manager.getExtraArg(index[i]).c_str());
      mp->setFileName(manager.getExtraArg(index[i]));

			// get video display location
			w = mp->getWidth();
			h = mp->getHeight();
			x = (d->getWidth()-w)/2;
			y = (d->getHeight()-h)/2;

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
      d->displayFixationIcon(fixicon);

      // ready to go whenever the user is ready:
      d->waitForKey(true); int frame = 0;
      d->waitNextRequestedVsync(false, true);
      d->pushEvent(std::string("===== Playing movie: ") +
                   manager.getExtraArg(index[i]) + " =====");

      // start the eye tracker:
      et->track(true);

      // blink the fixation:
      d->displayFixationIconBlink(fixicon);

      // create an overlay:
      d->createVideoOverlay(VIDFMT_YUV420P,mp->getWidth(),mp->getHeight()); // mpeg stream returns YUV420P

      // play the movie:
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
      LINFO("%d frames in %.04f sec (~%.04f fps)", frame, secs, frame/secs);

      // destroy the overlay. Somehow, mixing overlay displays and
      // normal displays does not work. With a single overlay created
      // before this loop and never destroyed, the first movie plays
      // ok but the other ones don't show up:
      d->destroyYUVoverlay();
      d->clearScreen();  // sometimes 2 clearScreen() are necessary

      // stop the eye tracker:
      usleep(50000);
      et->track(false);

			// take a break for every 10 clips, or do a quick drift correction
      if (i<nbmovies-1){
	      if (etc->getModelParamString("EyeTrackerType").compare("EL") == 0){
  	      d->closeDisplay();
          et->openSDL();
					if ((i+1)%10 == 0){
						//10 clips, take a break
  					d->clearScreen();
  					d->displayText("Please Take a Break");
  					d->waitForKey(true);
  					d->displayText("Calibration");
            d->pushEventBegin("Calibration");
            et->calibrate(d);
            d->pushEventEnd("Calibration");
          }else{
            et->recalibrate(d,13);
          }

          //et->closeSDL();
          //d->openDisplay();
          LINFO("Switching SDL for quick calibration");
        }else{
       	 // let's do a quickie eye tracker recalibration:
         et->recalibrate(d,13);
        }
   		}
    }

  d->clearScreen();
  d->displayText("Experiment complete. Thank you!");
  d->waitForKey(true);

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
