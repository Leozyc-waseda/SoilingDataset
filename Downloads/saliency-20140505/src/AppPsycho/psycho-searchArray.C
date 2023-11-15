/*!@file AppPsycho/psycho-searchArray.C Psychophysics display of still search arrays */

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
// Primary maintainer for this file: David J. Berg <dberg@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-searchArray.C $
//

#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Image/Image.H"
#include "Image/Range.H"
#include "Image/DrawOps.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Psycho/ArrayCreator.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Raster/Raster.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "Util/SimTime.H"
#include "Util/Timer.H"
#include "rutz/rand.h"

#include <vector>

//option for number of trials
static const ModelOptionDef OPT_ACTrials =
  { MODOPT_ARG(int), "ACTrials", &MOC_PSYCHOARRAYCREATOR,
    OPTEXP_CORE, "number of times to repeat the sequnce",
    "trials", '\0', "<int>", "1" };

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Search Array");
  OModelParam<std::string> itsConfigFile(&OPT_ACFileName, &manager);
  OModelParam<uint> itsItemRadius(&OPT_ACItemRadius, &manager);
  OModelParam<float> itsJitter(&OPT_ACJitterLevel, &manager);
  OModelParam<float> itsPpdx(&OPT_ACPpdX, &manager);
  OModelParam<float> itsPpdy(&OPT_ACPpdY, &manager);
  OModelParam<PixRGB<byte> > itsBgCol(&OPT_ACBackgroundColor, &manager);
  OModelParam<SimTime> itsDur(&OPT_DisplayTime, &manager);
  OModelParam<float> itsFixSize(&OPT_FixSize, &manager);
  OModelParam<bool> itsKeepFix(&OPT_KeepFix, &manager);
  OModelParam<int> itsTrials(&OPT_ACTrials, &manager);
  OModelParam<Range<int> > itsWait(&OPT_TrialWait, &manager);
  OModelParam<bool> itsPermTargs(&OPT_ACPermuteTargs, &manager);
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
  manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,"", 0, 0)==false)
    return(1);

  //create search array
  ArrayCreator ac(itsConfigFile.getVal(),
                  itsItemRadius.getVal(), itsJitter.getVal(),
                  d->getDims(), itsPpdx.getVal(),itsPpdy.getVal(),
                  itsBgCol.getVal(),
                  int(itsKeepFix.getVal())*itsFixSize.getVal(), 
                  itsPermTargs.getVal());

  LINFO("Array list created with %d trials", ac.size());

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  initRandomNumbers();

  // let's get all our ModelComponent instances started:
  manager.start();

  d->SDLdisplay::clearScreen(itsBgCol.getVal());

  //determine size of fixation point, set sdl fixation point
  const float ppd = sqrt(itsPpdx.getVal()*itsPpdx.getVal() + 
                         itsPpdy.getVal()*itsPpdy.getVal());
  const int size = int(ppd * itsFixSize.getVal());
  d->setFixationSize(size*2);

  // setup array of search array  indices:
  uint nbimgs = ac.size()*itsTrials.getVal(); int index[nbimgs];
  for (uint i = 0; i < nbimgs; ++i) index[i] = i % ac.size();
  randShuffle(index, nbimgs);
  std::vector<int> vindex(index, index + nbimgs);

  try {
    // main loop:
    for (uint i = 0; i < vindex.size(); i ++)
      {
        // load up the frame and show a fixation cross on a blank screen:
        d->SDLdisplay::clearScreen(itsBgCol.getVal());
        Image< PixRGB<byte> > image = ac.draw(vindex[i]);
        
        SDL_Surface *surf = d->makeBlittableSurface(image, true);
        LINFO("Array ready. %d trial", i);
        
        // give a chance to other processes (useful on single-CPU machines):
        sleep(1); if (system("/bin/sync")) LERROR("error in sync");
        LINFO("sync command complete");
        
        // eat up any extra fixations, saccades, or keys before we start
        // the next fixation round:
        et->clearEyeStatus();
        while (d->checkForKey() != -1) ;
        
        // display fixation to indicate that we are ready:
        d->displayWhiteDotFixation();
        
        // give us a chance to abort
        d->checkForKey();
        
        // ready to go whenever the monkey is ready (pulse on parallel port):
        if (!testrun.getVal())
          while(true)
            {
              if (et->isFixating()) break;
              if (d->checkForKey() != -1) break; // allow force start by key
            }
        
        //eat up any eye fixations or saccades, in case both were triggered
        et->clearEyeStatus();
        
        //store stimulus data to log file
        d->pushEvent(std::string("===== Showing Array: ") +
                     ac.toString(index[i]) + " =====");
        
        //wait for the next refresh
        d->waitNextRequestedVsync(false, true);
        
        // start the eye tracker:
        et->track(true);
        
        // show the image:
        d->displaySurface(surf, -2);
        
        //time starts when display is visible
        Timer itsTimer;
        
        //wait for the required time, or break if we recieve a signal. 
        while ( itsTimer.getSimTime() < itsDur.getVal() )
          {
            if (itsKeepFix.getVal())
              if (et->isSaccade())
                {
                  LINFO("Trial Aborted: Broke fixation");
                  d->pushEvent("Trial Aborted: Broke fixation");
                  
                  //put the stimulus back into the queue
                  int temp = vindex[i];
                  vindex.push_back(temp);
                  break;
                }
          }
        
        // free the image:
        SDL_FreeSurface(surf);
        
        // make sure display if off before we stop the tracker:
        d->SDLdisplay::clearScreen(itsBgCol.getVal());
        
        // stop the eye tracker:
        usleep(50000);
        et->track(false);
        
        //wait for requested time till next trial
        usleep(rutz::rand_range<int>(itsWait.getVal().min(), 
                                     itsWait.getVal().max()) * 1000);
      }
    
    d->SDLdisplay::clearScreen(itsBgCol.getVal());;
    LINFO("Experiment complete");
  }
  catch(...)
    {
      REPORT_CURRENT_EXCEPTION;
    }

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
