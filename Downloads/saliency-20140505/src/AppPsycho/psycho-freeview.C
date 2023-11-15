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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-freeview.C $
// $Id: psycho-freeview.C 8521 2007-06-28 17:45:49Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Raster/Raster.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "Image/DrawOps.H"
#include "Image/ImageSet.H"
#include "Image/ShapeOps.H"
#include "GUI/GUIOpts.H"
#include "Util/StringUtil.H"
#include "Util/MathFunctions.H"

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Freeview");

  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  manager.exportOptions(MC_RECURSE);
  manager.setOptionValString(&OPT_SDLdisplayDims, "1920x1080");
  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<img1.ppm> ... <imgN.ppm>", 1, -1)==false)
    return(1);

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  // let's get all our ModelComponent instances started:
  manager.start();

  // let's display an ISCAN calibration grid:
  d->clearScreen();
  d->displayISCANcalib();
  d->waitForKey();

  // let's do an eye tracker calibration:
  d->displayText("<SPACE> to calibrate; other key to skip");
  int c = d->waitForKey();
  if (c == ' ') d->displayEyeTrackerCalibration();

  d->clearScreen();
  d->displayText("<SPACE> To Show Pictures");
  c = d->waitForKey();

  // setup array of movie indices:
  uint nbimgs = manager.numExtraArgs(); int index[nbimgs];
  for (uint i = 0; i < nbimgs; i ++) index[i] = i;
  if (c == ' ') { LINFO("Randomizing images..."); randShuffle(index, nbimgs); }

  // main loop:
  for (uint i = 0; i < nbimgs; i ++)
    {
      // load up the frame and show a fixation cross on a blank screen:
      d->clearScreen();
      LINFO("Loading '%s'...", manager.getExtraArg(index[i]).c_str());
      Image< PixRGB<byte> > image =
        Raster::ReadRGB(manager.getExtraArg(index[i]));

      SDL_Surface *surf = d->makeBlittableSurface(image, true);

      LINFO("'%s' ready.", manager.getExtraArg(index[i]).c_str());
      d->displayFixation();

      // ready to go whenever the user is ready:
      d->waitForKey();
      d->waitNextRequestedVsync(false, true);
      d->pushEvent(std::string("===== Showing image: ") +
                   manager.getExtraArg(index[i]) + " =====");

      // start the eye tracker:
      et->track(true);

      // blink the fixation:
      d->displayFixationBlink();

      // show the image:
      d->displaySurface(surf, -2);

      // fixed presentation time:
      usleep(2000000);

      // free the image:
      SDL_FreeSurface(surf);

      // make sure display if off before we stop the tracker:
      d->clearScreen();

      // stop the eye tracker:
      usleep(50000);
      et->track(false);

      if ( ((int)(i+1)%46) == 0)

        {

          // let's display an ISCAN calibration grid:
          d->clearScreen();
          d->displayISCANcalib();
          d->waitForKey();
          d->waitForKey();

          // let's do an eye tracker calibration:
          d->displayText("<SPACE> to calibrate; other key to skip");
          int c = d->waitForKey();
          if (c == ' ') d->displayEyeTrackerCalibration();

          d->clearScreen();
          d->displayText("<SPACE> To Cotinue Experiment");
          c = d->waitForKey();
        }
    }

  d->clearScreen();
  d->displayText("Experiment complete. Thank you!");
  d->waitForKey();

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
