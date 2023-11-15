/*!@file AppPsycho/psycho-noCheat.C Psychophysics display of still images and
  check response to prevent cheating*/

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-spy3.C $
// $Id: psycho-spy3.C 10794 2009-02-08 06:21:09Z itti $
//

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Psycho/PsychoDisplay.H"
#include "GUI/GUIOpts.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Raster/Raster.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include <fstream>

// ######################################################################
extern "C" int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho no Cheating");

  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  // set some display params
  manager.setOptionValString(&OPT_SDLdisplayDims, "1920x1080");
  d->setModelParamVal("PsychoDisplayBackgroundColor", PixRGB<byte>(0));
  d->setModelParamVal("PsychoDisplayTextColor", PixRGB<byte>(255));
  d->setModelParamVal("PsychoDisplayBlack", PixRGB<byte>(255));
  d->setModelParamVal("PsychoDisplayWhite", PixRGB<byte>(128));
  d->setModelParamVal("PsychoDisplayFixSiz", 5);
  d->setModelParamVal("PsychoDisplayFixThick", 5);
  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<fileList>", 1, -1)==false)
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

  // get true random numbers:
  initRandomNumbers();

  // let's do an eye tracker calibration:
  d->displayText("<SPACE> to calibrate; other key to skip");
  int c = d->waitForKey();
  if (c == ' ') d->displayEyeTrackerCalibration(3, 3, 1);

  d->clearScreen();

  // setup array of stimuli:
  std::ifstream file(manager.getExtraArg(0).c_str());
  if (file == 0) LFATAL("Couldn't open object file: '%s'",
                        manager.getExtraArg(0).c_str());


  std::string line;
  getline(file, line);
  //parse the line to get the filepath
  //format: <abs filepath>
  std::string fpath = line.substr(0);

  getline(file, line);
  // parse the line to get the total number of images
  // format: <num>
  int numImgs = atoi(line.substr(0).c_str());
  std::string imgArr[numImgs][6];

  for(int a=0; a<numImgs; a++){
      getline(file, line);

      // parse the line to get the file names
      // format: <image> <target> <bg> <noCheat> <checkResponse> <reco>
      std::string::size_type end = line.find_first_of(" ");
      std::string::size_type start = 0;
      int i = 0; std::string imageName, targetName, bgName, noCheatName, checkName, recoName;
      while (end != std::string::npos) {
        std::string token = line.substr(start, end-start);
        switch(i)
          {
          case 0:
            imageName = token;
            break;
          case 1:
            targetName = token;
            break;
          case 2:
            bgName = token;
          case 3:
            noCheatName = token;
            break;
          case 4:
            checkName = token;
            break;
          }
        start = end + 1;
        end = line.find_first_of(" ", start);
        i ++;
      }
      // last token
      recoName = line.substr(start);

      // store file names in imgArr
      imgArr[a][0] = imageName;
      imgArr[a][1] = targetName;
      imgArr[a][2] = bgName;
      imgArr[a][3] = noCheatName;
      imgArr[a][4] = checkName;
      imgArr[a][5] = recoName;
    }

  //set up indices for randomizing images
  int index[numImgs];
  for (int i = 0; i < numImgs; i++) index[i] = i;
  LINFO("Randomizing images..."); randShuffle(index, numImgs);

  //begin experiment
  // int correct = 0, accuracy = 100, total = 0;
  int totalScore = 0;

  Timer timer, rt_timer;

  for(int a=0; a<numImgs; a++){

      std::string imageName, targetName, bgName, noCheatName, checkName, recoName;
      imageName = fpath +  imgArr[index[a]][0];
      targetName = fpath + imgArr[index[a]][1];
      bgName = fpath + imgArr[index[a]][2];
      noCheatName = fpath + imgArr[index[a]][3];
      checkName = fpath + imgArr[index[a]][4];
      recoName = fpath + imgArr[index[a]][5];

      // load up the frame and show a fixation cross on a blank screen:
      d->clearScreen();
      LINFO("Loading '%s'...", targetName.c_str());
      Image< PixRGB<byte> > target = Raster::ReadRGB(targetName);
      LINFO("Loading '%s'...", bgName.c_str());
      Image< PixRGB<byte> > background = Raster::ReadRGB(bgName);
      LINFO("Loading '%s'...", imageName.c_str());
      Image< PixRGB<byte> >image = Raster::ReadRGB(imageName);
      LINFO("Loading '%s'...", noCheatName.c_str());
      Image< PixRGB<byte> > noCheat = Raster::ReadRGB(noCheatName);

      SDL_Surface *surftarget = d->makeBlittableSurface(target, true);
      SDL_Surface *surfbackground = d->makeBlittableSurface(background, true);
      SDL_Surface *surfimage = d->makeBlittableSurface(image, true);
      SDL_Surface *surfnocheat = d->makeBlittableSurface(noCheat, true);

      LINFO("All images ready.");

      // ready to go whenever the user is ready:
      d->displayText("hit any key when ready");
      d->waitForKey();
      d->waitNextRequestedVsync(false, true);

      //fixation and target prime display
      // start the eye tracker:
      et->track(true);
      d->pushEvent(std::string("===== Showing target: ") + targetName +
                   " =====");

      // blink the fixation:
      d->displayFixationBlink();

      // show target image for 10*99ms:
      d->displaySurface(surftarget, -2, true);
      for (int j = 0; j < 30; j ++) d->waitNextRequestedVsync();
      SDL_FreeSurface(surftarget);

      // show bg image for 10*99ms:
      d->displaySurface(surfbackground, -2, true);
      for (int j = 0; j < 30; j ++) d->waitNextRequestedVsync();
      SDL_FreeSurface(surfbackground);

      // stop the eye tracker:
      usleep(50000);
      et->track(false);

      // load up the frame and show a fixation cross on a blank screen:
      d->clearScreen();

      //fixation and image display
      // start the eye tracker:
      et->track(true);
      d->pushEvent(std::string("===== Showing image: ") +imageName + " =====");

      // blink the fixation for a randomized amount of time:
      const int iter = 3 + randomUpToIncluding(5);
      d->displayFixationBlink(-1, -1, iter);

      // gobble up any keystrokes:
      while(d->checkForKey() != -1) ;

      // show the image until user responds, for up to 5s:
      d->displaySurface(surfimage, -2, true);
      double startTime = timer.getSecs();
      double reactTime = 0.0;
      rt_timer.reset();

      // wait for key; it will record reaction time in the logs:
      while(d->checkForKey() == -1 &&  (timer.getSecs() < startTime + 5.0))
        usleep(1000);
      reactTime =  rt_timer.getSecs();
      SDL_FreeSurface(surfimage);

      // stop the eye tracker:
      usleep(50000);
      et->track(false);

      // flash a random grid of numbers and check response to prevent
      // cheating
      d->pushEvent(std::string("===== Showing noCheat: ") + noCheatName
                   + " =====");

      // flash the image for 99ms:
      d->displaySurface(surfnocheat, -2, true);
      for (int j = 0; j < 3; j ++) d->waitNextRequestedVsync();
      SDL_FreeSurface(surfnocheat);

      // wait for response:
      d->displayText("Enter the number at the target location");
      c = d->waitForKey();
      int c2 = d->waitForKey();
      std::ifstream check (checkName.c_str());
      if (check == 0) LFATAL("Couldn't open check file: '%s'",
                             checkName.c_str());
      std::string response;
      //total ++;
      while (getline(check, response))
        {
          LINFO (" reading from %s", checkName.c_str());
          LINFO (" subject entered %d%d", c, c2);
          LINFO (" reading %c%c from %s", response[0], response[1],
                 checkName.c_str());

          // calculate score for reaction time
          int score = 0;
          if (reactTime <= 1.0) score = 10;
          else if (reactTime <= 2.0) score = 5;
          else if (reactTime <= 3.0) score = 4;
          else if (reactTime <= 4.0) score = 3;
          else if (reactTime <= 5.0) score = 2;
          else score = 1;

          // check the response
          char tmp[40];
          if (((int)response[0] == c) && ((int)response[1] == c2))
            {
              // correct ++;
              //accuracy = (correct * 100) / total;
              //sprintf(tmp, "Correct! Accuracy is %d%%", accuracy);

              totalScore += score;
              sprintf(tmp, "+%d...Total Score = %d", score, totalScore);
              d->displayText(tmp);
              d->pushEvent(std::string("===== Correct ====="));
            }
          else
            {
              //accuracy = (correct * 100) / total;
              // sprintf(tmp, "Wrong! Accuracy is %d%%", accuracy);

              sprintf(tmp, "+0...Total Score = %d", totalScore);
              d->displayText(tmp);
              d->pushEvent(std::string("===== Wrong ====="));
            }

          // maintain display
          for (int j = 0; j < 30; j ++) d->waitNextRequestedVsync();
        }

    et->recalibrate(d,20);
  }

  char tmp[40];
  sprintf(tmp, "===== Total Score = %d", totalScore);
  d->pushEvent(tmp);

  d->clearScreen();
  d->displayText("Experiment complete. Thank you!");
  d->waitForKey();

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

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
