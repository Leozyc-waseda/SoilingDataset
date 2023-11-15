/*!@file AppPsycho/psycho-stealth.C Psychophysics display of primary task's
  search arrays followed by stealthy insertion of secondary task's
  search arrays */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-stealth.C $
// $Id: psycho-stealth.C 8426 2007-05-24 06:57:57Z itti $
//

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "GUI/GUIOpts.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Raster/Raster.H"
#include "Util/Types.H"
#include <fstream>
#include <string>

// ######################################################################
extern "C" int main(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho stealth");

  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  // set some display params
  //manager.setOptionValString(&OPT_SDLdisplayDims, "1280x1024");
  manager.setOptionValString(&OPT_SDLdisplayDims, "1024x768");
  d->setModelParamVal("PsychoDisplayBackgroundColor", PixRGB<byte>(0));
  d->setModelParamVal("PsychoDisplayTextColor", PixRGB<byte>(255));
  d->setModelParamVal("PsychoDisplayBlack", PixRGB<byte>(255));
  d->setModelParamVal("PsychoDisplayWhite", PixRGB<byte>(128));
  d->setModelParamVal("PsychoDisplayFixSiz", 5);
  d->setModelParamVal("PsychoDisplayFixThick", 5);
  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");


  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<fileList> <stealthPrefix>"
                               "<numStealth>",
                               3,-1)==false)
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
  if (c == ' ') d->displayEyeTrackerCalibration(3, 3);

  d->clearScreen();

  // setup array of movie indices:
  std::ifstream file(manager.getExtraArg(0).c_str());
  if (file == 0) LFATAL("Couldn't open object file: '%s'",
                        manager.getExtraArg(0).c_str());
  // prefix or path to stealth trials
  std::string stealthPrefix =  manager.getExtraArg(1);
  // number of items per stealth trial
  int numStealth = atoi (manager.getExtraArg(2).c_str());

  std::string line;
  // for online feedback on primary task T1
  int correct = 0, accuracy = 100, total = 0;
  // find the number of reports on M,T,D for the secondary task T2
  int N[numStealth];
  for (int i = 0; i < numStealth; i++) N[i] = 0;
  while(getline(file, line))
    {
      // parse the line to get the file names
      // format: <image> <noCheat> <checkResponse> <reco>
      std::string::size_type end = line.find_first_of(" ");
      std::string::size_type start = 0;
      int i = 0; std::string imageName, noCheatName, checkName, recoName;
      while (end != std::string::npos) {
        std::string token = line.substr(start, end-start);
        switch(i)
          {
          case 0:
            imageName = token;
            break;
          case 1:
            noCheatName = token;
            break;
          case 2:
            checkName = token;
            break;
          }
        start = end + 1;
        end = line.find_first_of(" ", start);
        i ++;
      }
      // last token
      recoName = line.substr(start);

      // load up the frame and show a fixation cross on a blank screen:
      d->clearScreen();

      LINFO("Loading '%s'...", imageName.c_str());
      Image< PixRGB<byte> > image =
        Raster::ReadRGB(imageName);

      SDL_Surface *surf = d->makeBlittableSurface(image, true);

      LINFO("'%s' ready.", imageName.c_str());

      // ready to go whenever the user is ready:
      d->displayText("hit any key when ready");
      d->waitForKey();
      d->waitNextRequestedVsync(false, true);

      // start the eye tracker:
      et->track(true);
      d->pushEvent(std::string("===== Showing image: ") +imageName + " =====");

      // blink the fixation:
      d->displayFixationBlink();

      // show the image:
      d->displaySurface(surf, -2, true);

      if (imageName.find(stealthPrefix, 0) != std::string::npos) {
        // flash T2 trial
        for (int j = 0; j < 7; j ++) d->waitNextRequestedVsync();
      }
      else {
        // T1 trial
        c = d->waitForKey();
      }

      // free the image:
      SDL_FreeSurface(surf);

      // stop the eye tracker:
      usleep(50000);
      et->track(false);

      // flash a random grid of numbers and check response to prevent
      // cheating
      LINFO("Loading '%s'...", noCheatName.c_str());

      Image< PixRGB<byte> > noCheat =
        Raster::ReadRGB(noCheatName);

      surf = d->makeBlittableSurface(noCheat, true);


      LINFO("'%s' ready.", noCheatName.c_str());
      d->pushEvent(std::string("===== Showing noCheat: ") + noCheatName
                   + " =====");

      // flash the image for 99ms:
      d->displaySurface(surf, -2, true);
      for (int j = 0; j < 6; j ++) d->waitNextRequestedVsync();

       // free the image:
      SDL_FreeSurface(surf);

      // wait for response:
      d->displayText("Enter the number at the target location");
      c = d->waitForKey();
      int c2 = d->waitForKey();

      if (imageName.find(stealthPrefix, 0) == std::string::npos) {
        // check response for the T1 trial
        std::ifstream check (checkName.c_str());
        if (check == 0) LFATAL("Couldn't open check file: '%s'",
                               checkName.c_str());
        std::string response;
        total ++;
        while (getline(check, response))
          {
            LINFO (" reading from %s", checkName.c_str());
            LINFO (" subject entered %d%d", c, c2);
            LINFO (" reading %c%c from %s", response[0], response[1],
                   checkName.c_str());
            // check the response
            char tmp[40];
            if (((int)response[0] == c) && ((int)response[1] == c2))
              {
                correct ++;
                accuracy = correct * 100 / total;
                sprintf(tmp, "Correct! Accuracy is %d%%", accuracy);
                d->displayText(tmp);
                d->pushEvent(std::string("===== Correct ====="));
              }
            else
              {
                accuracy = correct * 100 / total;
                sprintf(tmp, "Wrong! Accuracy is %d%%", accuracy);
                d->displayText(tmp);
                d->pushEvent(std::string("===== Wrong ====="));
              }
            // maintain display
            for (int j = 0; j < 30; j ++) d->waitNextRequestedVsync();
          }
      }
      else {
        // check response for the T2 trial
        std::ifstream reco (recoName.c_str()); std::string sline;
        if (reco == 0) LFATAL("Couldn't open reco file: '%s'",
                               recoName.c_str());
        std::string response; int lineNum = 0;
        LINFO (" reading from %s", recoName.c_str());
        LINFO (" subject entered %c%c", c, c2);
        // compare subject's response to that stored in the reco file
        bool match = false;
        while(getline(reco, sline))
          {
            // parse the line to get the correct response
            // format: <row> <col> <imageName> <response>
            std::string::size_type end = sline.find_first_of(" "), start = 0;
            std::string response;
            while (end != std::string::npos) {
              std::string token = sline.substr(start, end-start);
              start = end + 1;
              end = sline.find_first_of(" ", start);
            }
            response = sline.substr(start);
            LINFO (" reading %c%c from %s", response[0], response[1],
                       recoName.c_str());
            // does the report match the stored response?
            if (((int)response[0] == c) && ((int)response[1] == c2))
              {
                // found match, update the number of reports
                N[lineNum] = N[lineNum] + 1;
                match = true;
                char s[40]; sprintf(s, "N = ( ");
                for (int i = 0; i < numStealth; i ++)
                  sprintf(s, "%s %d ", s, N[i]);
                LINFO ("%s )", s);
                break;
              }
            lineNum ++; // parse the next line
            i = 0; // reset token index for parsing the next line
          }
        if (match == false)
          d->displayText("Oops, You entered an invalid number...");

        // maintain display
        for (int j = 0; j < 30; j ++) d->waitNextRequestedVsync();

      }

    }

  d->clearScreen();
  d->displayText("Experiment complete. Thank you!");
  d->waitForKey();

  // output the number of reports
  char s[40]; sprintf(s, "N = ( ");
  for (int i = 0; i < numStealth; i ++)
    sprintf(s, "%s %d ", s, N[i]);
  LINFO ("%s )", s);
  LINFO ("accuracy = %d%%", accuracy);

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
