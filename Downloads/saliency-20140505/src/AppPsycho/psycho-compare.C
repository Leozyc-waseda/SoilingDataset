/*!@file AppPsycho/psycho-compare.C Compare two still images */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-compare.C $
// $Id: psycho-compare.C 15310 2012-06-01 02:29:24Z itti $
//

#include "Component/ModelManager.H"
#include "Component/ComponentOpts.H"
#include "Component/EventLog.H"
#include "Image/Image.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "GUI/GUIOpts.H"
#include "Raster/Raster.H"
#include "Util/Types.H"
#include "Util/StringUtil.H"
#include "Util/MathFunctions.H"
#include <fstream>

//! Compare two images along a number of criteria
/*! This shows two images and asks the user to emit a judgement along
  a number of criteria. The two images are randomly presented on the
  left/right of fixation. After 5 seconds, criteria questions are
  displayed at the bottom of the screen. Input file format is:

  <num criteria>
  <textual description of criterion 1>
  ...
  <textual description of criterion N>
  <image filename 1> <image filename 2>
  ...
  <image filename 1> <image filename 2>


  EXAMPLE:

  3
  Which image is more beautiful?
  Which image contains more animals?
  Which image is more geeky?
  image001.png image002.png
  image003.png image004.png
  image005.png image006.png

*/

// ######################################################################
static int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psychophysics Comparison");

  // Instantiate our various ModelComponents:
  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);


  // set a default display size:
  manager.exportOptions(MC_RECURSE);
  manager.setOptionValString(&OPT_SDLdisplayDims, "1920x1080");
  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
 manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<fileList>", 1, -1) == false)
    return(1);

  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  // let's get all our ModelComponent instances started:
  manager.start();

  // get the list of images
  std::ifstream file(manager.getExtraArg(0).c_str());
  if (file == 0) LFATAL("Couldn't open file: '%s'",
                        manager.getExtraArg(0).c_str());

  initRandomNumbers();

  std::string line;
  if (!getline(file, line)) LFATAL("Bogus file format: missing num criteria");
  int numcrit; convertFromString(line, numcrit);
  std::vector<std::string> message;

  for (int i = 0; i < numcrit; ++ i)
    {
      if (!getline(file, line))
        LFATAL("Bogus file format: missing text for criterion %d", i+1);
      message.push_back(line);
    }

  // load up all the image pairs:
  std::vector<std::string> lines;
  while(getline(file, line)) lines.push_back(line);

  // create a randomized index:
  uint *idx = new uint[lines.size()];
  for (uint i = 0; i < lines.size(); i ++) idx[i] = i;
  randShuffle(idx, lines.size());
  LINFO("Randomized %" ZU " pairs of images.", lines.size());

  const uint dw = d->getWidth(), dh = d->getHeight();

  // let's do an eye tracker calibration:
  // let's display an ISCAN calibration grid:
  d->clearScreen();
  d->displayISCANcalib();
  d->waitForKey();

  // let's do an eye tracker calibration:
  d->displayText("<SPACE> to calibrate; other key to skip");
  int c = d->waitForKey();
  if (c == ' ') d->displayEyeTrackerCalibration();

  d->clearScreen();
  d->displayText("<SPACE> To Cotinue Experiment");
  c = d->waitForKey();

  for (uint ii = 0; ii < lines.size(); ++ii)
    {
      std::vector<std::string> tokens;
      split(lines[idx[ii]], " \t", std::back_inserter(tokens));
      if (tokens.size() != 2)
        LFATAL("Need two filenames per line: %s", lines[idx[ii]].c_str());

      // pick a random display order:
      uint idx1 = 0, idx2 = 1;
      if (randomUpToNotIncluding(2) > 0) { idx1 = 1; idx2 = 0; }

      // load up the two images and show a fixation cross on a blank screen:
      d->clearScreen();
      LINFO("Loading '%s'...", tokens[idx1].c_str());
      Image< PixRGB<byte> > image1 = Raster::ReadRGB(tokens[idx1]);
      LINFO("Loading '%s'...", tokens[idx2].c_str());
      Image< PixRGB<byte> > image2 = Raster::ReadRGB(tokens[idx2]);

      // Create a composite side-by-side image:
      Image< PixRGB<byte> > image(d->getDims(), NO_INIT);
      image.clear(d->getGrey());

      const uint m = 20;  // margin around the images
      const uint mm = 20; // half spacing between the images
      inplaceEmbed(image, image1,
                   Rectangle(Point2D<int>(m, m), Dims(dw/2-mm-m*2, dh-m*2)),
                   d->getGrey(), true);
      inplaceEmbed(image, image2,
                   Rectangle(Point2D<int>(dw/2+mm+m, m), Dims(dw/2-mm-m*2, dh-m*2)),
                   d->getGrey(), true);

      SDL_Surface *surf = d->makeBlittableSurface(image, true);

      LINFO("%s / %s ready.", tokens[idx1].c_str(), tokens[idx2].c_str());
      d->displayFixation();

      // ready to go whenever the user is ready:
      d->waitForKey();
      d->pushEvent(std::string("===== Showing images: ") +
                   tokens[idx1] + " / " + tokens[idx2] + " =====");

      // start the eye tracker:
      et->track(true);

      // blink the fixation:
      d->displayFixationBlink();

      // show the image:
      d->displaySurface(surf, -2);

      // sleep a bit and stop the tracker:
      usleep(5000000);
      et->track(false);

      const int fh = 20; // font height in pixels

      // display the questions and wait for a key each time:
      for (int i = 0; i < numcrit; ++i)
        {
          d->pushEvent(std::string("===== Question: ") +
                       message[i] + " =====");

          drawFilledRect(image, Rectangle(Point2D<int>(0, d->getHeight() - fh-10),
                                          Dims(d->getWidth(), fh+10)),
                         d->getGrey());

          writeText(image,
                    Point2D<int>((d->getWidth() - 10 * message[i].size()) / 2,
                            d->getHeight() - fh - 5),
                    message[i].c_str(), PixRGB<byte>(0, 0, 64),
                    d->getGrey(), SimpleFont::FIXED(10));

          SDL_Surface *s = d->makeBlittableSurface(image, true);
          d->displaySurface(s, -2);

          // wait for key (will log which key was pressed):
          d->waitForKey();

          SDL_FreeSurface(s);
        }

      // get ready for next one or end:
      d->clearScreen();
      SDL_FreeSurface(surf);
    }

  d->clearScreen();
  d->displayText("Experiment complete. Thank you!");
  d->waitForKey();

  // stop all our ModelComponents
  manager.stop();

  delete [] idx;

  // all done!
  return 0;
}

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  // simple wrapper around submain() to catch exceptions (because we
  // want to allow PsychoDisplay to shut down cleanly; otherwise if we
  // abort while SDL is in fullscreen mode, the X server won't return
  // to its original resolution)
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

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
