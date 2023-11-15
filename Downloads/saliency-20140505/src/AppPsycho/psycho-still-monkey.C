/*!@file AppPsycho/psycho-still.C Psychophysics display of still images */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-still-monkey.C $
// $Id: psycho-still-monkey.C 13678 2010-07-19 18:16:16Z beobot $
//
#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Image/Image.H"
#include "Image/Range.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Raster/Raster.H"
#include "Util/SimTime.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "rutz/rand.h"

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Still");
  OModelParam<bool> keepfix(&OPT_KeepFix, &manager);
  OModelParam<float> fixsize(&OPT_FixSize, &manager);
  OModelParam<float> ppd(&OPT_Ppd, &manager);
  OModelParam<SimTime> itsDur(&OPT_DisplayTime, &manager);
  OModelParam<Range<int> > itsWait(&OPT_TrialWait, &manager);
  OModelParam<bool> testrun(&OPT_Testrun, &manager);

  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "DML");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<img1.ppm> ... <imgN.ppm>", 1, -1)==false)
    return(1);

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  //fixation point
  int fixrad = int(ppd.getVal() * fixsize.getVal());
  if ((fixrad % 2) != 0)
    --fixrad;
  d->setFixationSize(fixrad*2);


  SDL_Rect rect;
  const int siz2 = (fixrad - 1) / 2; // half cross size
  const int w = d->getDims().w(), h = d->getDims().h();
  const int i = w / 2 - 1;
  const int j = h / 2 - 1;
  rect.x = i - siz2; rect.y = j - siz2; rect.w = fixrad; rect.h = fixrad;

  // let's get all our ModelComponent instances started:
  manager.start();
  d->clearScreen();

  // setup array of movie indices:
  uint nbimgs = manager.numExtraArgs(); int index[nbimgs];
  for (uint i = 0; i < nbimgs; i ++)
    index[i] = i;

  LINFO("Randomizing images...");
  randShuffle(index, nbimgs);

  try {
    // main loop:
    for (uint i = 0; i < nbimgs; i ++)
      {
        // load up the frame and show a fixation cross on a blank screen:
        d->clearScreen();
        LINFO("Loading '%s'...", manager.getExtraArg(index[i]).c_str());
        Image< PixRGB<byte> > image =
          Raster::ReadRGB(manager.getExtraArg(index[i]));

        SDL_Surface *surf = d->makeBlittableSurface(image, true);

        //create a fixation on top of the image
        if (keepfix.getVal())
          SDL_FillRect(surf, &rect, d->getUint32color(PixRGB<byte>(255, 0, 0)));

        LINFO("'%s' ready.", manager.getExtraArg(index[i]).c_str());

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

        d->waitNextRequestedVsync(false, true);
        d->pushEvent(std::string("===== Showing image: ") +
                     manager.getExtraArg(index[i]) + " =====");

        // start the eye tracker:
        et->track(true);

        // show the image:
        d->displaySurface(surf, -2);

        // wait for key:
        usleep(itsDur.getVal().usecs());

        // free the image:
        SDL_FreeSurface(surf);

        // make sure display if off before we stop the tracker:
        d->clearScreen();

        // stop the eye tracker:
        usleep(50000);
        et->track(false);

        //wait for requested time till next trial
        usleep(rutz::rand_range<int>(itsWait.getVal().min(), 
                                     itsWait.getVal().max()) * 1000);
      }

    d->clearScreen();
    d->displayText("Experiment complete. Thank you!");
    d->waitForKey();
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
