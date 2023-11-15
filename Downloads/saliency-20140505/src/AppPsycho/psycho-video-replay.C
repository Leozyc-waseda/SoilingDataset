/*!@file AppPsycho/psycho-video-replay.C */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-video-replay.C $
// $Id: psycho-video-replay.C 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef APPPSYCHO_PSYCHO_VIDEO_REPLAY_C_DEFINED
#define APPPSYCHO_PSYCHO_VIDEO_REPLAY_C_DEFINED

#include "Component/ModelManager.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Media/BufferedInputFrameSeries.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Raster/GenericFrame.H"
#include "Util/FileUtil.H"
#include "Util/StringUtil.H"
#include "Util/Types.H"
#include "Util/csignals.H"
#include "Video/VideoFrame.H"

#include <SDL/SDL.h>
#include <fstream>
#include <iterator> // for back_inserter()
#include <unistd.h> // for sync()
#include <vector>

namespace
{
  struct Question
  {
    Question(const std::string& p, const std::string& a)
      : prompt(p), valid_answers(a)
    {}

    std::string prompt;
    std::string valid_answers;
  };

  void parseQuestions(const std::string& fname,
                      std::vector<Question>& qv)
  {
    std::ifstream questions(fname.c_str());

    if (!questions.is_open())
      LFATAL("couldn't open %s for reading", fname.c_str());

    std::string line;
    while (std::getline(questions, line))
      {
        std::vector<std::string> toks;
        split(line, "|", std::back_inserter(toks));

        if (toks.size() != 2)
          LFATAL("invalid line in %s (expected 2 tokens, got %d:\n%s",
                 fname.c_str(), int(toks.size()), line.c_str());

        qv.push_back(Question(toks[0], toks[1]));
      }
  }

  void doQuestion(PsychoDisplay& d, const Question& q,
                  std::ostream& dest)
  {
    while (d.checkForKey() != -1)
      { /* discard any outstanding keypresses */ }

    const PixRGB<byte> bgcol(0,0,0);
    const PixRGB<byte> msgcol(255,255,255);
    const PixRGB<byte> anscol(0,255,0);

    const Dims dims = d.getDims();

    Image<PixRGB<byte> > screen(d.getDims(), NO_INIT);
    screen.clear(bgcol);

    // write prompt:
    Point2D<int> p; p.i = dims.w() / 2 - (10 * q.prompt.length()) / 2;
    p.j = dims.h() / 2 - 10;
    if (p.i < 0)
      LFATAL("Text '%s' does not fit on screen!", q.prompt.c_str());

    writeText(screen, p, q.prompt.c_str(), msgcol, bgcol);

    d.displayImage(screen);

    // get answer:
    int answer_id;
    std::string answer;
    while (true)
      {
        const int c = toupper(d.waitForKey());

        std::string::size_type p = q.valid_answers.find(char(c));

        if (p != q.valid_answers.npos)
          {
            answer_id = int(p);
            answer = char(c);
            break;
          }
      }

    p.i = dims.w() / 2 - 5;
    p.j += 25;
    writeText(screen, p, answer.c_str(), anscol, bgcol);

    d.displayImage(screen);

    // write answer to file:
    dest << answer_id << " % '" << answer
         << "' of '" << q.valid_answers << "' ("
         << q.prompt << ")\n";

    LINFO("A: %d Q: %s", answer_id, q.prompt.c_str());

    usleep(750000);

    d.SDLdisplay::clearScreen(bgcol);

    usleep(250000);
  }
}

//! Psychophysics display of video frames from disk
/*! This displays video frames that have already been saved to disk,
  with added machinery for eye-tracking. */
int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // 'volatile' because we will modify this from signal handlers
  volatile int signum = 0;
  catchsignals(&signum);

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Video Replay");

  // Instantiate our various ModelComponents:

  nub::ref<BufferedInputFrameSeries> bifs
    (new BufferedInputFrameSeries(manager, 256));
  manager.addSubComponent(bifs);

  nub::ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  manager.exportOptions(MC_RECURSE);

  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<outdir> <questions>",
                               2, 2) == false)
    return(1);

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  const std::string outdir = manager.getExtraArg(0);

  makeDirectory(outdir);
  el->setModelParamString("EventLogFileName",
                          outdir + "/psychodata.psy");

  std::ofstream answers((outdir + "/answers").c_str());
  if (!answers.is_open())
    LFATAL("couldn't open %s/answers for writing", outdir.c_str());

  std::vector<Question> questions;
  parseQuestions(manager.getExtraArg(1), questions);

  // let's get all our ModelComponent instances started:
  manager.start();

  const VideoFormat vf = bifs->peekFrameSpec().videoFormat;

  d->setDesiredRefreshDelayUsec(1000000.0/59.94, 0.2F);

  // let's display an ISCAN calibration grid:
  d->clearScreen();
  d->displayISCANcalib();
  d->waitForKey();

  // let's do an eye tracker calibration:
  d->displayText("<SPACE> to calibrate; other key to skip");
  int c = d->waitForKey();
  if (c == ' ') d->displayEyeTrackerCalibration(3, 3, 3);
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
  d->createVideoOverlay(vf);

  // ready to go whenever the user is ready:
  d->waitForKey();
  d->waitNextRequestedVsync(false, true);
  d->pushEvent("===== START =====");

  // start the eye tracker:
  et->track(true);

  // blink the fixation:
  d->displayFixationBlink();

  // grab, display and save:
  int framenum = 0;
  bool did_underflow = false;
  while (true)
    {
      if (signum != 0)
        {
          LINFO("quitting because %s was caught", signame(signum));
          return -1;
        }

      // check for a keypress to see if the user wants to quit the
      // experiment; pressing '.' will give a graceful exit and normal
      // shutdown, while pressing <ESC> will trigger an LFATAL() and
      // an urgent shutdown:
      if (d->checkForKey() == '.')
        break;

      // grab a raw buffer:
      const GenericFrame frame = bifs->get(&did_underflow);
      if (!frame.initialized())
        break;

      // display the frame as an overlay
      d->displayVideoOverlay(frame.asVideo(), framenum,
                             SDLdisplay::NEXT_FRAMETIME);

      ++framenum;
    }

  LINFO("displayed %d frames", framenum);

  if (did_underflow)
    LERROR("input ended due to premature underflow");

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

  for (size_t i = 0; i < questions.size(); ++i)
    doQuestion(*d, questions[i], answers);

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
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPPSYCHO_PSYCHO_VIDEO_REPLAY_C_DEFINED
