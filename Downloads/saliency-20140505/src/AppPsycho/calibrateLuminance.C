/*!@file AppPsycho/calibrateLuminance.C Match the luminance of 2 stimuli using
  standard flicker photometry */

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
// Primary maintainer for this file: Vidhya Navalpakkam <navalpak@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/calibrateLuminance.C $
// $Id: calibrateLuminance.C 8426 2007-05-24 06:57:57Z itti $
//

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Psycho/PsychoDisplay.H"
#include "GUI/GUIOpts.H"
#include "Raster/Raster.H"
#include "Util/Types.H"
#include <cstdlib>

// ######################################################################
// regenerates the test stimuli by changing the luminance
void updateLuminance(Image< PixRGB<byte> >& s2, int lumin, const float sat);
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Calibrate luminance");

  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  // set some display params
  manager.setOptionValString(&OPT_SDLdisplayDims, "1280x1024");
  d->setModelParamVal("PsychoDisplayBackgroundColor", PixRGB<byte>(0));
  d->setModelParamVal("PsychoDisplayTextColor", PixRGB<byte>(255));
  d->setModelParamVal("PsychoDisplayBlack", PixRGB<byte>(255));
  d->setModelParamVal("PsychoDisplayWhite", PixRGB<byte>(128));
  d->setModelParamVal("PsychoDisplayFixSiz", 5);
  d->setModelParamVal("PsychoDisplayFixThick", 5);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<img1.ppm> <img2.ppm>"
                               " <luminance2> <saturation2> ", 1, -1)==false)
    return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  // load up the frame and show a fixation cross on a blank screen:
  d->clearScreen();
  LINFO("Loading '%s'...", manager.getExtraArg(0).c_str());
  Image< PixRGB<byte> > s1 =
    Raster::ReadRGB(manager.getExtraArg(0));
  LINFO("Loading '%s'...", manager.getExtraArg(1).c_str());
  Image< PixRGB<byte> > s2 =
    Raster::ReadRGB(manager.getExtraArg(1));

  // read the luminance and saturation value of S2
  int lumin = manager.getExtraArgAs<int>(2);
  float sat = manager.getExtraArgAs<float>(3);

  SDL_Surface *surf1 = d->makeBlittableSurface(s1, false);
  SDL_Surface *surf2 = d->makeBlittableSurface(s2, false);

  LINFO("stimuli ready.");

  // ready to go whenever the user is ready:
  d->displayText("hit any key when ready");
  d->waitForKey();
  d->waitNextRequestedVsync(false, true);
  while (true) {
    d->clearScreen();
    d->displayFixation();
    // let's alternate between stimuli S1 and S2 for 4 sec
    for (int j = 0; j < 9; j ++) {
      d->displaySurface(surf1, -2, true);
      d->waitFrames(3);
      d->displaySurface(surf2, -2, true);
      d->waitFrames(3);
    }
    d->displaySurface(surf1, -2, true);

    // do the stimuli match in luminance
    d->displayText("Press <+><NUM> to increase luminance,"
                   "      <-><NUM> to decrease luminance"
                   "      and <ENTER> to accept");
    int c1 = d->waitForKey();
    int c2 = d->waitForKey();
    if (c1 == 43){
      // increase the luminance of S2
      SDL_FreeSurface(surf2);
      lumin += c2 - 48;
      updateLuminance(s2, lumin, sat);
      surf2 = d->makeBlittableSurface(s2, false);
    }
    else if (c1 == 45){
      // decrease the luminance of S2
      SDL_FreeSurface(surf2);
      lumin -= c2 - 48;
      updateLuminance(s2, lumin, sat);
      surf2 = d->makeBlittableSurface(s2, false);
    }
    else break; // accept values and proceed
  }

  d->clearScreen();

  // free the image:
  SDL_FreeSurface(surf1);
  SDL_FreeSurface(surf2);

  d->clearScreen();
  d->displayText("Experiment complete. Thank you!");
  d->waitForKey();

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}
// ######################################################################
void updateLuminance(Image< PixRGB<byte> >& s2, int lumin, const float sat){
  LINFO ("test luminance = %d", lumin);
  // change the r, g, b values used to draw the stimuli
  int red = (int)(lumin / (1.0f - 0.7875*sat));
  int green = (int) ((lumin - 0.2125*red)/0.7875);
  PixRGB<byte> zero(0, 0, 0);
  // since the stimuli is red saturated, blue = green
  LINFO ("r, g, b = %d, %d, %d", red, green, green);
  PixRGB<byte> rgb(red, green, green);
  int w = s2.getWidth(), h = s2.getHeight();
  // regenerate the image
  for (int x = 0; x < w; x++)
    for (int y = 0; y < h; y ++) {
      if (s2.getVal(x,y) == zero);  // do nothing
      else s2.setVal(x, y, rgb);
    }
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
