/*!@file AppPsycho/psycho-search.C Psychophysics display for a search for a
  target that is presented to the observer prior to the search */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-search.C $
// $Id: psycho-search.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "Component/ModelManager.H"
#include "Image/ColorOps.H" // for makeRGB()
#include "Image/CutPaste.H" // for inplacePaste()
#include "Image/Image.H"
#include "Image/MathOps.H"  // for inplaceSpeckleNoise()
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Raster/Raster.H"
#include "Util/MathFunctions.H"

#include <ctype.h>
#include <vector>

//! number of frames in the mask
#define NMASK 10

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Search");

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
  if (manager.parseCommandLine(argc, argv, "<imagelist.txt>", 1, 1) == false)
    return(1);

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  // let's get all our ModelComponent instances started:
  manager.start();

  // let's prepare the mask images:
  SDL_Surface *mask[NMASK]; int mindex[NMASK]; Dims ddims = d->getDims();

  for (int i = 0; i < NMASK; i ++) {
    Image<byte> r(ddims, ZEROS), g(ddims, ZEROS), b(ddims, ZEROS);
    inplaceSpeckleNoise(r, 1.0F, i+1, 255, true);
    inplaceSpeckleNoise(g, 1.0F, i+1, 255, true);
    inplaceSpeckleNoise(b, 1.0F, i+1, 255, true);
    mask[i] = d->makeBlittableSurface(makeRGB(r, g, b), false);

    // keep an array of indices that we will later randomize:
    mindex[i] = i;
  }

  // let's pre-load all the image names so that we can randomize them later:
  FILE *f = fopen(manager.getExtraArg(0).c_str(), "r");
  if (f == NULL) LFATAL("Cannot read stimulus file");
  char line[1024];
  std::vector<std::string> ilist, tlist;
  while(fgets(line, 1024, f))
    {
      // each line has two filenames: first the imagelet that contains
      // only the target, second the image that contains the target in
      // its environment:
      char *line2 = line; while(*line2 != '\0' && !isspace(*line2)) line2 ++;
      *line2++ = '\0'; line2[strlen(line2) - 1] = '\0';
      // now line is at the imagelet, line2 at the image
      tlist.push_back(std::string(line));
      ilist.push_back(std::string(line2));
    }
  fclose(f);

  // randomize stimulus presentation order:
  int nimg = ilist.size(); int imindex[nimg];
  for (int i = 0; i < nimg; i ++) imindex[i] = i;
  randShuffle(imindex, nimg);

  // let's display an ISCAN calibration grid:
  d->clearScreen();
  d->displayISCANcalib();
  d->waitForKey();

  // let's do an eye tracker calibration:
  d->displayText("<SPACE> to calibrate; other key to skip");
  int c = d->waitForKey();
  if (c == ' ') d->displayEyeTrackerCalibration(3, 3);

  // we are ready to start:
  d->clearScreen();
  d->displayText("<SPACE> to start experiment");
  d->waitForKey();

  // main loop:
  for (int im = 0; im < nimg; im ++) {
    int imnum = imindex[im];

    // load up the images and show a fixation cross on a blank screen:
    d->clearScreen();
    LINFO("Loading '%s' / '%s'...", ilist[imnum].c_str(),tlist[imnum].c_str());

    // get the imagelet and place it at a random position:
    Image< PixRGB<byte> > img = Raster::ReadRGB(tlist[imnum]);
    Image< PixRGB<byte> > rndimg(d->getDims(), NO_INIT);
    rndimg.clear(d->getGrey());
    int rndx = 40 + randomUpToNotIncluding(d->getDims().w() - img.getWidth() - 80);
    int rndy = 30 + randomUpToNotIncluding(d->getDims().h() - img.getHeight() - 60);
    inplacePaste(rndimg, img, Point2D<int>(rndx, rndy));
    SDL_Surface *surf1 = d->makeBlittableSurface(rndimg, false);
    char buf[256];
    sprintf(buf, "===== Showing imagelet: %s at (%d, %d) =====",
            tlist[imnum].c_str(), rndx, rndy);

    // compute coords of fixation cross centered on imagelet:
    int fixx = rndx + img.getWidth() / 2;
    int fixy = rndy + img.getHeight() / 2;

    // load up the full-scene image:
    img = Raster::ReadRGB(ilist[imnum]);
    SDL_Surface *surf2 = d->makeBlittableSurface(img, false);

    // randomize presentation order of mask frames:
    randShuffle(mindex, NMASK);

    // give a chance to other processes if single-CPU:
    usleep(200000);

    // ready to go whenever the user is ready:
    d->displayText("Ready");
    d->waitForKey();
    d->waitNextRequestedVsync(false, true);
    d->pushEvent(buf);

    // show the imagelet:
    d->displaySurface(surf1, 0, true);

    // wait for a bit (but for a reliable amount of time, so no usleep here):
    for (int i = 0; i < 80; i ++) d->waitNextRequestedVsync();

    // show the masks:
    d->pushEvent("===== Showing mask =====");
    for (int j = 0; j < NMASK; j ++)
      d->displaySurface(mask[mindex[j]], j+1, true);

    // show a fixation at center of where imagelet was shown:
    d->clearScreen();
    d->displayFixation(fixx, fixy);
    for (int i = 0; i < 40; i ++) d->waitNextRequestedVsync();

    // start the eye tracker:
    et->track(true);

    // show the fixation some more:
    for (int i = 0; i < 20; i ++) d->waitNextRequestedVsync();

    // show the image:
    d->pushEvent(std::string("===== Showing search image: ") + ilist[imnum] +
                 std::string(" ====="));
    d->displaySurface(surf2, 0, true);

    // wait for key; it will record reaction time in the logs:
    d->waitForKey();

    // free the imagelet and image:
    SDL_FreeSurface(surf1); SDL_FreeSurface(surf2);

    // stop the eye tracker:
    usleep(150000);
    et->track(false);

    // let's do a quickie eye tracker calibration once in a while:
    if (im > 0 && im % 20 == 0) {
      d->displayText("Ready for quick recalibration");
      d->waitForKey();
      d->displayEyeTrackerCalibration(3, 3);
      d->clearScreen();
      d->displayText("Ready to continue with the images");
      d->waitForKey();
    }
  }

  d->clearScreen();
  d->displayText("Experiment complete. Thank you!");
  d->waitForKey();

  // stop all our ModelComponents
  manager.stop();

  // free mask images:
  for (int i = 0; i < NMASK; i++) SDL_FreeSurface(mask[i]);

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
