/*!@file AppPsycho/psycho-mplayer.C simple program to run SDL and MPlayer */

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
// Primary maintainer for this file: John Shen <shenjohn at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-mplayer.C $
// $Id: psycho-mplayer.C 13712 2010-07-28 21:00:40Z itti $
//

#ifndef APPPSYCHO_PSYCHO_MPLAYER_C_DEFINED
#define APPPSYCHO_PSYCHO_MPLAYER_C_DEFINED

#include "Component/ModelManager.H"
#include "Psycho/MPlayerWrapper.H"
#include "Component/ComponentOpts.H"
#include "Media/MediaOpts.H"
#include <stdio.h>
#include "Media/MPEGStream.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include <unistd.h>

int submain(const int argc, char ** argv)
{
  ModelManager manager("Psycho-mplayer");
  // Instantiate our various ModelComponents:
  nub::soft_ref<EventLog> log(new EventLog(manager));
  nub::soft_ref<MPlayerWrapper> player(new MPlayerWrapper(manager));
  nub::soft_ref<PsychoDisplay> display(new PsychoDisplay(manager));
  nub::soft_ref<EyeTrackerConfigurator>
    configurator(new EyeTrackerConfigurator(manager));

  //redundant

  manager.addSubComponent(configurator);
  manager.addSubComponent(player);
  manager.addSubComponent(log);
  manager.addSubComponent(display);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<movie1.mpg> ... <movieN.mpg>", 1, -1)==false)
    return(1);

  //Set input video stream/output file
  //manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> tracker = configurator->getET();
  display->setEyeTracker(tracker);
  display->setEventLog(log);
  tracker->setEventLog(log);
  player->setEventLog(log);

  // EyeLink opens the screen for us, so make sure SDLdisplay is slave:
  if (configurator->getModelParamString("EyeTrackerType").compare("EL") == 0)
    display->setModelParamVal("SDLslaveMode", true);


  manager.start();

  //let's do an eye tracker calibration
  tracker->calibrate(display);
  display->clearScreen();
  display->displayText("<SPACE> for random play; other key for ordered");

  int c;
  c = display->waitForKey(true);

  //setup array of movie indices
  uint nbmovies = manager.numExtraArgs(); int index[nbmovies];
  for (uint i = 0; i < nbmovies; i++) index[i]=i;
  if(c == ' ')
    {
      LINFO("Randomizing movies...");
      randShuffle(index, nbmovies);
    }

  for(uint i = 0; i < nbmovies; i++)
    {
      display->clearScreen();

      LDEBUG("Playing '%s'...",manager.getExtraArg(index[i]).c_str());
      player->setSourceVideo(manager.getExtraArg(index[i]));

      //give a chance to other processes (useful on single-CPUT machines):
      sleep(1); if (system("/bin/sync")) LERROR("error in sync");

      // display fixation to indicate that we are ready:
      display->displayFixation();

      // ready to go whenever the user is ready:
      display->waitForKey(true);
      display->waitNextRequestedVsync(false, true);
      display->displayFixationBlink();

      tracker->track(true);
      player->runfromSDL(display);

      usleep(50000);
      tracker->track(false);

      display->clearScreen();
      //every other trial, let's do a quick eye tracker recalibration
      if(i%2==1 && i != nbmovies-1) tracker->recalibrate(display,13);
    }


  display->clearScreen();
  display->displayText("Experiment complete.  Thank you for participating!");
  display->waitForKey(true);


  // stop all our ModelComponents
  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

int main(const int argc, char **argv)
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
#endif // APPPSYCHO_PSYCHO_MPLAYER_C_DEFINED
