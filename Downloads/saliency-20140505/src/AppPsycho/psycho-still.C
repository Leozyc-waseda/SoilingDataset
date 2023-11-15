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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-still.C $
// $Id: psycho-still.C 15232 2012-03-20 22:22:04Z pohetsn $
//

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/ShapeOps.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Raster/Raster.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Still");

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

  // let's do an eye tracker calibration:
  et->calibrate(d);

  d->clearScreen();
  d->displayText("<SPACE> for random play; other key for ordered");
  int c = d->waitForKey();

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

      // Prepare to display the image, and a random mask:      
      Image< PixRGB<byte> > mask(d->getWidth()/8, d->getHeight()/8, NO_INIT);
      for (int j = 0; j < mask.getSize(); j ++)
        mask.setVal(j, PixRGB<byte>(byte(randomUpToNotIncluding(256)),
                                    byte(randomUpToNotIncluding(256)),
                                    byte(randomUpToNotIncluding(256))));
      mask = zoomXY(mask, 8, 8);
      SDL_Surface *msurf = d->makeBlittableSurface(mask, true);

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

      // wait for key:
      c = d->waitForKey();

      // free the image:
      SDL_FreeSurface(surf);

      // Display mask:
      d->displaySurface(msurf, -2);
      usleep(150000);

      // make sure display if off before we stop the tracker:
      d->clearScreen();

      // stop the eye tracker:
      usleep(50000);
      et->track(false);
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
