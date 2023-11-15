/*!@file AppPsycho/psycho-video.C Psychophysics display and grabbing of video feeds */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-video.C $
// $Id: psycho-video.C 15310 2012-06-01 02:29:24Z itti $
//

#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/DiskDataStream.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Image/Image.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "GUI/SDLdisplay.H"
#include "GUI/GUIOpts.H"
#include "Raster/GenericFrame.H"
#include "Transport/FrameIstream.H"
#include "Util/FileUtil.H"
#include "Util/StringUtil.H" // for split() and join()
#include "Util/Types.H"
#include "Util/csignals.H"
#include "Util/sformat.H"
#include "Video/VideoFrame.H"

#include <SDL/SDL.h>
#include <iterator> // for back_inserter()
#include <unistd.h> // for sync()
#include <time.h>

namespace
{
  class DiskDataStreamListener : public FrameListener
  {
  public:
    DiskDataStreamListener(nub::ref<DiskDataStream> stream)
      :
      itsStream(stream)
    {}

    virtual ~DiskDataStreamListener() {}

    virtual void onRawFrame(const GenericFrame& frame)
    {
      itsStream->writeFrame(frame, "frame");
    }

  private:
    nub::ref<DiskDataStream> itsStream;
  };
}

//! Psychophysics display and grabbing of video feeds
/*! This displays and grabs video feeds to disk, with added machinery
  for eye-tracking. To save time, we grab raw data (in whatever native
  grab mode has been selected, without converting to RGB) then feed
  them to an SDL YUV overlay and save them to disk. For this to work,
  you need to configure your grabber in YUV420P mode, which is what is
  used for displays. The saved files are pure raw YUV420P data and will
  need to be converted, e.g., to RGB, later and in a different
  program (or you can use the YUV input mode of mpeg_encode to create
  MPEG movies from your saved files). */
int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_DEBUG;  // suppress debug messages

  // 'volatile' because we will modify this from signal handlers
  volatile int signum = 0;
  catchsignals(&signum);

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Video");

  // Instantiate our various ModelComponents:
  nub::ref<FrameGrabberConfigurator>
    fgc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(fgc);

  nub::ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::ref<DiskDataStream> ds(new DiskDataStream(manager));
  manager.addSubComponent(ds);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  // select a V4L grabber in 640x480 YUV420P by default:
  manager.exportOptions(MC_RECURSE);
  manager.setOptionValString(&OPT_FrameGrabberType, "V4L");
  manager.setOptionValString(&OPT_FrameGrabberMode, "UYVY");
  manager.setOptionValString(&OPT_FrameGrabberDims, "640x480");
  manager.setOptionValString(&OPT_DeinterlacerType, "Bob");

  manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

  manager.setOptionValString(&OPT_DiskDataStreamSleepUsecs, "10000");
  manager.setOptionValString(&OPT_SDLdisplayDims, "640x480" );

  // Parse command-line:
  if (manager.parseCommandLine
      (argc, argv, "<dir1:dir2:dir3> <nframes> <subject>", 3, 3)
      == false)
    return(1);

  std::string Sub=manager.getExtraArg(2);
  manager.setOptionValString(&OPT_EventLogFileName, Sub);

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  const std::string dirpath = manager.getExtraArg(0);
  const int maxframe = manager.getExtraArgAs<int>(1);

  std::vector<std::string> stems;
  split(dirpath, ":", std::back_inserter(stems));

  if (stems.size() == 0)
    LFATAL("expected at least one entry in the directory path");

  el->setModelParamString("EventLogFileName",
                          stems[0] + "/" + Sub);

  for (size_t i = 0; i < stems.size(); ++i)
    {
      makeDirectory(stems[i]);
      makeDirectory(stems[i] + "/frames");

      stems[i] = stems[i] + "/frames/";
    }

  ds->setModelParamString("DiskDataStreamSavePath",
                          join(stems.begin(), stems.end(), ","));

  nub::ref<FrameIstream> gb = fgc->getFrameGrabber();

  gb->setListener(rutz::shared_ptr<FrameListener>(new DiskDataStreamListener(ds)));

  // let's get all our ModelComponent instances started:
  manager.start();

  d->setDesiredRefreshDelayUsec(gb->getNaturalFrameTime().usecs(), 0.2F);

  // let's display an ISCAN calibration grid:
  d->clearScreen();
  d->displayISCANcalib();
  d->waitForKey();

  // let's do an eye tracker calibration:
  d->displayText("<SPACE> to calibrate; other key to skip");
  int c = d->waitForKey();
  if (c == ' ') d->displayEyeTrackerCalibration(3, 3, 2);
  d->clearScreen();

  // give a chance to other processes (useful on single-CPU machines):
  sleep(1);
  sync();

  // ready for action:
  d->displayText("<SPACE> to start experiment");
  d->waitForKey();

  // display fixation to indicate that we are ready:
  d->clearScreen();
  d->displayFixation();

  // create an overlay:
  d->createVideoOverlay(gb->peekFrameSpec().videoFormat);



  // ready to go whenever the user is ready:
  d->waitForKey();
  d->waitNextRequestedVsync(false, true);
  d->pushEvent("===== START =====");

  // get wall time:
  struct timeval tv;
  struct timezone tz;
  struct tm *tm;
  gettimeofday(&tv, &tz);
  tm = localtime(&tv.tv_sec); char msg[300];
  snprintf(msg, 300, "== WALL TIME: %d:%02d:%02d.%03" ZU ".%03" ZU " ==",
           tm->tm_hour, tm->tm_min, tm->tm_sec, tv.tv_usec / 1000,
           tv.tv_usec % 1000);
  d->pushEvent(msg);

  d->pushEvent(std::string("===== Playing movie: ") + Sub.c_str() + " =====");

  // start the eye tracker:
  et->track(true);

  // blink the fixation:
  d->displayFixationBlink();

  // grab, display and save:
  int framenum = 0;
  while (framenum < maxframe)
    {
      if (signum != 0)
        {
          LINFO("quitting because %s was caught", signame(signum));
          return -1;
        }

      // grab a raw buffer:
      const VideoFrame frame = gb->readFrame().asVideo();

      // display the frame as an overlay
      d->displayVideoOverlay(frame, framenum, SDLdisplay::NO_WAIT);

      ++framenum;

      // check for a keypress to see if the user wants to quit the
      // experiment; pressing '.' will give a graceful exit and normal
      // shutdown, while pressing <ESC> will trigger an LFATAL() and
      // an urgent shutdown:
      if (d->checkForKey() == '.')
        break;
    }

  // destroy the overlay. Somehow, mixing overlay displays and
  // normal displays does not work. With a single overlay created
  // before this loop and never destroyed, the first movie plays
  // ok but the other ones don't show up:
  d->destroyYUVoverlay();
  d->clearScreen();  // sometimes 2 clearScreen() are necessary
  d->clearScreen();  // sometimes 2 clearScreen() are necessary

  // stop the eye tracker:
  usleep(50000);
  et->track(false);

  d->clearScreen();

  // no need to explicit clear DiskDataStream's queue here; that
  // happens automatically in DiskDataStream::stop(), called from
  // manager.stop()

  d->displayText("Experiment complete. Thank you!");
  d->waitForKey();

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
