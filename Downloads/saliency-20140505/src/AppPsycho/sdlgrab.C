/*!@file AppPsycho/sdlgrab.C Test framegrabber and SDL overlay display */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/sdlgrab.C $
// $Id: sdlgrab.C 7241 2006-10-04 23:13:30Z rjpeters $
//

#include "Component/ComponentOpts.H"
#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Image/Image.H"
#include "Psycho/PsychoDisplay.H"
#include "Raster/GenericFrame.H"
#include "Transport/FrameIstream.H"
#include "Util/AllocAux.H"
#include "Util/Types.H"
#include "Util/csignals.H"
#include "Video/VideoFrame.H"
#include "rutz/time.h"

#include <SDL/SDL.h>
#include <unistd.h> // for sync()

//! Test framebrabber and SDL overlay display
int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // 'volatile' because we will modify this from signal handlers
  volatile int signum = 0;
  catchsignals(&signum);

  // Instantiate a ModelManager:
  ModelManager manager("SDLgrab");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    fgc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(fgc);

  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  d->setEventLog(el);

  // select a V4L grabber in 640x480 YUV420P by default:
  manager.exportOptions(MC_RECURSE);
  manager.setOptionValString(&OPT_FrameGrabberType, "V4L");
  manager.setOptionValString(&OPT_FrameGrabberMode, "UYVY");
  manager.setOptionValString(&OPT_FrameGrabberDims, "640x480");
  manager.setOptionValString(&OPT_DeinterlacerType, "Bob");

  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false)
    return 1;

  // do a few post-command-line configs:
  nub::soft_ref<FrameIstream> gb = fgc -> getFrameGrabber();

  // let's get all our ModelComponent instances started:
  manager.start();

  d->setDesiredRefreshDelayUsec(gb->getNaturalFrameTime().usecs(), 0.2F);

  // main loop:
  while (1)
    {
      // give a chance to other processes (useful on single-CPU machines):
      usleep(250000);
      sync();

      d->clearScreen();
      d->displayText("[SPACE] start/stop - [ESC] quit");

      while (1)
        {
          const char key = d->checkForKey();
          if (key == ' ') break;
          else
            {
              if (signum != 0)
                {
                  LINFO("quitting because %s was caught", signame(signum));
                  return -1;
                }

              usleep(10000);
            }
        }

      // create an overlay:
      d->createVideoOverlay(gb->peekFrameSpec().videoFormat);

      // get the frame grabber to start streaming:
      gb->startStream();

      // grab, display:
      bool keepgoing = true; int frame = 0;
      rutz::time start = rutz::time::wall_clock_now();
      while (keepgoing)
        {
          if (signum != 0)
            {
              LINFO("quitting because %s was caught", signame(signum));
              return -1;
            }

          // display the raw frame buffer as an overlay
          d->displayVideoOverlay(gb->readFrame().asVideo(),
                                 frame, SDLdisplay::NO_WAIT);

          ++frame;

          const char key = d->checkForKey();

          // stop if <SPACE> is pressed, and abort if <ESC> by using
          // the built-in ESC check in PsychoDisplay::checkForKey():
          if (key == ' ') keepgoing = false;
        }
      rutz::time stop = rutz::time::wall_clock_now();

      invt_allocation_show_stats();

      const double secs = (stop-start).sec();
      LINFO("%d frames in %.02f sec (~%.02f fps)", frame, secs, frame/secs);

      // destroy the overlay. Somehow, mixing overlay displays and
      // normal displays does not work. With a single overlay created
      // before this loop and never destroyed, the first video plays
      // ok but the other ones don't show up:
      d->destroyYUVoverlay();
      d->clearScreen();  // sometimes 2 clearScreen() are necessary
      d->clearScreen();  // sometimes 2 clearScreen() are necessary
    }

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

extern "C" int main(const int argc, char** argv)
{
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
