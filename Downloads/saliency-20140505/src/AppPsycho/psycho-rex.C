/*!@file AppPsycho/psycho-rex.C control psycho display directly from Rex at
   Doug Munoz Lab, probably works with other Rex setups too.  */

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
// Primary maintainer for this file: David J Berg <dberg@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-rex.C $
// $Id: psycho-rex.C 13556 2010-06-11 00:51:36Z dberg $
//

#include "Component/ModelManager.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Media/FrameSeries.H"
#include "Media/MediaOpts.H"
#include "Image/Image.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "Video/VideoFrame.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
//#include "Psycho/StimController.H"
#include "rutz/time.h"
#include <deque>

/*
static const ModelOptionDef OPT_exitcode =
  { MODOPT_ARG(char), "exitcode", &MOC_INPUT, OPTEXP_CORE,
    "code to exit stimulus listening.",
    "exitcode", '\0', "<char>", "0" };
*/
// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  //MYLOGVERB = LOG_INFO;  // suppress debug messages
  /*
  // Instantiate a ModelManager:
  ModelManager manager("Psycho Rex");

  // Instantiate our various ModelComponents:
  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<StimController> stc(new StimController(manager));
  stc->initialzed(new StimListenerDML(itsExitCode));
  stc->setDisplay(d);
  stc->pauseDisplay(true);
  manager.addSubComponent(stc);

  manager.setOptionValString(&OPT_InputMPEGStreamPreload, "true");
  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<stim1> ... <stimN>", 1, -1)==false)
    return(1);

  //wait when should this stuff be added
  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);
  stc->setEventLog(el);

  initRandomNumbers();

  // let's get all our ModelComponent instances started:
  manager.start();

  // setup array of indices for stimuli:
  uint nbmovies = manager.numExtraArgs(); int index[nbmovies];
  for (uint i = 0; i < nbmovies; i ++) index[i] = i;
  LINFO("Randomizing Stimuli...");
  randShuffle(index,nbmovies);

  try{
    // main loop:
    for (uint i = 0; i < nbmovies; i++)
      {
        d->clearScreen();
        std::string fname = manager.getExtraArg(index[i]);
        LINFO("Buffering/Loading '%s'...", fname.c_str());

        //reset the frame source to the next movie
        std::string pre = "buff:";
        ifs->setFrameSource(pre + fname);

        //preload stuff into buffer
        ifs->startStream();

        LINFO("'%s' ready.", fname.c_str());

        // give a chance to other processes (useful on single-CPU machines):
        sleep(1); system("/bin/sync");

        // display fixation to indicate that we are ready:
        d->displayRedDotFixation();

        // give us a chance to abort
        d->checkForKey();

        // ready to go whenever the monkey is ready (pulse on parallel port):
        while(true)
          {
            if (et->isFixating()) break;
            if (d->checkForKey() != -1) break; // allow force start by key
          }

        int frame = 0;
        d->pushEvent(std::string("===== Playing movie: ") +
                     fname + " =====");

        // start the eye tracker:
        et->track(true);

        //grab the first frame and set our background
        const FrameState is = ifs->updateNext();
        if (is == FRAME_COMPLETE)
          LFATAL("No frames in stimulus");

        //get our frame and make sure it is valid
        GenericFrame input = ifs->readFrame();
        if (!input.initialized())
          LFATAL("Empty first frame");

        //actually set the background and un-pause. If this is a
        //static image, display of the background will hold until Rex
        //says to display. This command may already be in the the queue,
        //as we have been listening but not displaying commands. When
        //we un-pause, the queue will start popping and displaying. If
        //this is a movie, the first frame will draw now, even though
        //we are paused, anything in the qeue will start to alter
        //display on the next frame (call to setBackgroundImage)
        stc->getDispController()->setBackground(input,frame);
        stc->pauseDisplay(false);

        //start passing frames to our display controller
        while(1)
        {
          //update internal counters in ifs, quite if no more frames
          const FrameState is = ifs->updateNext();
          if (is == FRAME_COMPLETE)
            break;

          //get our frame and make sure it is valid
          GenericFrame input = ifs->readFrame();
          if (!input.initialized())
            break;

          stc->getDispController()->setBackground(input,frame);
        }//done with our current stimulus

        //pause our listener, destroying the overlay and waiting the thread
        stc->pauseDisplay(true);

        d->clearScreen();  // sometimes 2 clearScreen() are necessary

        // stop the eye tracker:
        usleep(50000);
        et->track(false);

      }//end loop over stimuli

  d->displayText("Experiment complete. Thank you!");
  d->waitForKey(true);

  }//end try

  catch (...)
    {
      REPORT_CURRENT_EXCEPTION;
    };

  // stop all our ModelComponents
  manager.stop();
  */
  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
