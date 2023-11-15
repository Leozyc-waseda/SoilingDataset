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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-searchGabor.C $
// $Id: psycho-searchGabor.C 10794 2009-02-08 06:21:09Z itti $
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
#include "Util/StringUtil.H"

#include <ctype.h>
#include <vector>
#include <string>
#include <fstream>

using namespace std;

//! number of frames in the mask
#define NMASK 10

// ######################################################################
static int submain(const int argc, char** argv)
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


  // let's pre-load all the image names so that we can randomize them later:
  FILE *f = fopen(manager.getExtraArg(0).c_str(), "r");
  if (f == NULL) LFATAL("Cannot read stimulus file");
  char line[1024];
  std::vector<std::string> ilist, tlist, rlist, slist;

 int correct = 0, accuracy = 100, total = 0;

  while(fgets(line, 1024, f))
    {
   std::vector<std::string> tokens;
     // each line has four filenames: first the imagelet that contains
      // only the target, second the image that contains the target in
      // its environment, third the image to report position of target
      //fourth the name of the spec file for the search array
      LINFO("line reads %s",line);
      split(line," ", std::back_inserter(tokens));

      // now line is at the imagelet, line2 at the image
      tlist.push_back(std::string(tokens[0]));
      ilist.push_back(std::string(tokens[1]));
      rlist.push_back(std::string(tokens[2]));
      slist.push_back(std::string(tokens[3]));
      LINFO("\nNew pair \nline1 reads: %s, \nline2 reads:%s, \nline 3 reads %s,\nline 4 reads %s,",tokens[0].c_str(), tokens[1].c_str(),tokens[2].c_str(), tokens[3].c_str());

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

  //******************* main loop:***********************//
  for (int im = 0; im < nimg; im++) {
    int imnum = imindex[im];

    // load up the images and show a fixation cross on a blank screen:
    d->clearScreen();
    LINFO("Loading '%s' / '%s'...", ilist[imnum].c_str(),tlist[imnum].c_str());

    // get the imagelet and place it at a random position:

    if(!Raster::fileExists(tlist[imnum]))
    {
      // stop all our ModelComponents
      manager.stop();
      LFATAL("i couldnt find image file %s", tlist[imnum].c_str());
    }

    Image< PixRGB<byte> > img = Raster::ReadRGB(tlist[imnum]);
    Image< PixRGB<byte> > rndimg(d->getDims(), NO_INIT);
    rndimg.clear(d->getGrey());
    int rndx = 0,rndy = 0;

    SDL_Surface *surf1 = d->makeBlittableSurface(img, true);
    char buf[256];

    if(!Raster::fileExists(ilist[imnum]))
    {
      manager.stop();
      LFATAL("i couldnt find image file %s", ilist[imnum].c_str());
    }
    img = Raster::ReadRGB(ilist[imnum]);
    SDL_Surface *surf2 = d->makeBlittableSurface(img, true);

    //randShuffle(mindex, NMASK);

   // load up the reporting number image:
       if(!Raster::fileExists(rlist[imnum]))
    {
      // stop all our ModelComponents
      manager.stop();
      LFATAL(" i couldnt find image file %s", tlist[imnum].c_str());
    }

    img = Raster::ReadRGB(rlist[imnum]);
    SDL_Surface *surf3 = d->makeBlittableSurface(img, true);

    // give a chance to other processes if single-CPU:
    usleep(200000);

    // ready to go whenever the user is ready:
    d->displayFixationBlink();
    //d->waitForKey();
    d->waitNextRequestedVsync(false, true);

    //************************ Display target************************//
    sprintf(buf, "===== Showing imagelet: %s at (%d, %d) =====",
            tlist[imnum].c_str(), rndx, rndy);

    d->pushEvent(buf);
    d->displaySurface(surf1, 0, true);
    usleep(2000000);

    d->clearScreen();

    //************************ Display array************************//

     // start the eye tracker:
    et->track(true);

    // show the image:
    d->pushEvent(std::string("===== Showing search image: ") + ilist[imnum] +
                 std::string(" ====="));

    d->displaySurface(surf2, 0, true);

    // wait for key; it will record reaction time in the logs:
    d->waitForKey();
    // stop the eye tracker:
    et->track(false); //we just want to record eye movements while subjects view the array


    //*********************** Display reporting image**************//
    // show the reporting image:
    d->pushEvent(std::string("===== Showing reporting image: ") + rlist[imnum] +
                 std::string(" ====="));
    d->displaySurface(surf3, 0, true);

    usleep(200000);
    d->displayText("Input the target number:");

    string inputString = d->getString('\n');

    //check user response
    char tmp[40];

    //lets open the spec file and extract the target number
    ifstream specFile(slist[imnum].c_str(), ifstream::in);

    bool found =false;
    string testLine;
    std::vector<std::string> specTokens;

   if(specFile.is_open())
      {
        while(!specFile.eof() && !found)
          {
            getline(specFile, testLine);
            string::size_type loc = testLine.find("target", 0);

            if(loc != string::npos)
               found = true;
          }
        if(!found)
          {
            manager.stop();
            LFATAL("couldnt find the target number from spec file");
          }

        split(testLine," ", std::back_inserter(specTokens));

        std::string responseString;
        responseString = inputString;
        int intResponse = atoi(responseString.c_str()), intActual = atoi(specTokens[1].c_str());

        total++;

        if (intResponse == intActual)
            {
              correct ++;
              accuracy = correct * 100 / total;
              sprintf(tmp, "Correct! Accuracy is %d%%", accuracy);
              d->displayText(tmp);
              d->pushEvent(std::string("===== Correct ====="));
              usleep(500000);
            }
          else
            {
              accuracy = correct * 100 / total;
              sprintf(tmp, "Wrong! Accuracy is %d%%", accuracy);
              d->displayText(tmp);
              d->pushEvent(std::string("===== Wrong ====="));
              usleep(500000);
            }

        specFile.close();
      }
    else
      {
        d->displayText("no target file found!");
        LFATAL("couldnt open the file -%s-",slist[imnum].c_str());
      }


    // free the imagelet and image:
    SDL_FreeSurface(surf1); SDL_FreeSurface(surf2); SDL_FreeSurface(surf3);


    // let's do a quiinckie eye tracker calibration once in a while:
    /*if (im > 0 && im % 20 == 0) {
      d->displayText("Ready for quick recalibration");
      d->waitForKey();
      d->displayEyeTrackerCalibration(3, 3);
      d->clearScreen();
      d->displayText("Ready to continue with the images");
      d->waitForKey();
      }*/

    //allow for a break after 50 trials then recalibrate

    if (im==50)
      {
        d->displayText("You may take a break press space to continue when ready");
        d->waitForKey();
        // let's display an ISCAN calibration grid:
        d->clearScreen();
        d->displayISCANcalib();
        d->waitForKey();
        // let's do an eye tracker calibration:
        d ->displayText("<SPACE> to calibrate; other key to skip");
        int c = d->waitForKey();
        if (c == ' ') d->displayEyeTrackerCalibration(3, 3);
     }



    et->recalibrate(d,15);

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
