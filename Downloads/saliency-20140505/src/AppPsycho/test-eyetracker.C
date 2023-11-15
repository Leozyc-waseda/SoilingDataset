/*!@file AppPsycho/test-eyetracker.C test eye tracker functions */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/test-eyetracker.C $


#include "Component/ModelManager.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/PsychoOpts.H"
#include "Psycho/EyeTracker.H"
#include "Devices/KeyBoard.H"
#include "Component/EventLog.H"

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("test eye tracker");

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  manager.setOptionValString(&OPT_EventLogFileName, "test-et.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "UDP");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv," ", 0, 0)==false)
    return(1);

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  et->setEventLog(el);


  // let's get all our ModelComponent instances started:
  manager.start();
  
  try 
    {
      // main loop:
      KeyBoard keyBoard;
      LINFO("Listening for commands from host:");
      while(1)
        {
          if (et->isFixating()) LINFO("Fixation code recieved");
          if (et->isSaccade())  LINFO("Saccade code recieved");
          int key = keyBoard.getKeyAsChar(false);
          if (key == 32)
            {
              LINFO("starting eye tracker");
              et->track(true);
              sleep(1);
              LINFO("stopping eye tracker");
              et->track(false);
            }
          else if (key == 27) break;
        }
      LINFO("Keyboard spacebar pressed, exiting"); 
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
