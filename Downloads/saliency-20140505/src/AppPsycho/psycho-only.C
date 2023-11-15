/*!@file AppPsycho/psycho-only.C Pure Psychophysics display of still images */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-only.C $
// $Id: psycho-only.C 8426 2007-05-24 06:57:57Z itti $
//

#include "Component/ModelManager.H"
#include "Component/ComponentOpts.H"
#include "Component/EventLog.H"
#include "Image/Image.H"
#include "Psycho/PsychoDisplay.H"
#include "Raster/Raster.H"
#include "Util/Types.H"
#include <fstream>

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Pure Psychophysics Still");

  // Instantiate our various ModelComponents:
  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<fileList>", 1, -1)==false)
    return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  // get the list of images
  std::ifstream file(manager.getExtraArg(0).c_str());
  if (file == 0) LFATAL("Couldn't open object file: '%s'",
                        manager.getExtraArg(0).c_str());

  d->setEventLog(el);

  std::string imageName;
  while(getline(file,imageName))
    {
      // load up the frame and show a fixation cross on a blank screen:
      d->clearScreen();
      LINFO("Loading '%s'...", imageName.c_str());
      Image< PixRGB<byte> > image =
        Raster::ReadRGB(imageName);

      SDL_Surface *surf = d->makeBlittableSurface(image, true);

      LINFO("'%s' ready.", imageName.c_str());
      d->displayText("Press key when ready!");

      // ready to go whenever the user is ready:
      d->waitForKey();
      d->pushEvent(std::string("===== Showing image: ") +
                   imageName + " =====");

      // blink the fixation:
      d->displayFixationBlink();

      // show the image:
      d->displaySurface(surf, -2);

      // wait for key:
      d->waitForKey();

      // free the image:
      SDL_FreeSurface(surf);

      // make sure display if off before we stop the tracker:
      d->clearScreen();

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
