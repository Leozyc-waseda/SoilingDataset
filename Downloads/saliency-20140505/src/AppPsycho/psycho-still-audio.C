/*!@file AppPsycho/psycho-still-audio.C Psychophysics display of still images */

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
// Primary maintainer for this file: Bella Rozenkrants <rozenkra@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-still-audio.C $
// $Id: psycho-still-audio.C 13712 2010-07-28 21:00:40Z itti $
//

#include "Component/ModelManager.H"
#include "Audio/AudioWavFile.H"
#include "Devices/AudioGrabber.H"
#include "Devices/AudioMixer.H"
#include "Devices/DeviceOpts.H"
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
#include <vector>
#include <pthread.h>
#include <string>
#include <math.h>
using std::string;

volatile bool recordaudio = false;
volatile bool keepgoing = true;
volatile int recnb = 0;
std::string subName = "empty";
std::string stimName = "empty";
int repNum = 0;
// thread mutex to control file naming
static pthread_mutex_t fileName= PTHREAD_MUTEX_INITIALIZER;
// ######################################################################

static void *audiorecorder(void *agbv)
{
  bool recording = false;
  std::vector<AudioBuffer<byte> > rec;


  while(keepgoing)
    {
      // start recording?
      if (recording == false && recordaudio == true)
        { rec.clear(); recording = true; }

      // stop recording?
      if (recording == true && recordaudio == false)
        {
          if(subName != "empty" && stimName != "empty")
            {
              pthread_mutex_lock(&fileName);

              // save our current records
              char fname[100];

              sprintf(fname, "/lab/ilab19/bella/audiofiles/r%d_sub%s_%s_rn%02d.wav",repNum,subName.c_str(), stimName.c_str(),recnb);

              writeAudioWavFile(fname, rec);

              // ready for next recording:
              recnb ++; recording = false;

                          pthread_mutex_unlock(&fileName);
            }
        }

      // we grab the audio data all the time, so that it does not
      // start piling up into the internal buffers of the audio
      // grabber; but we will save it only if we are in recording
      // mode. Note: the call to grab() is blocking and hence sets
      // the pace of our loop here. With 256 samples at 11.025kHz
      // that's about 23ms:
      AudioGrabber *agb = (AudioGrabber *)agbv;
      AudioBuffer<byte> data;

      agb->grab(data);

      if (data.nsamples() != 256U)
        LERROR("Recorded len %u is not 256!", data.nsamples());

      // continue recording?
      if (recording == true && recordaudio == true && data.nsamples() == 256U)
        // store in our queue of records:
        rec.push_back(data);
    }

  // terminate thread:
  pthread_exit(0);
  return NULL;
}

// ######################################################################
extern int submain(const int argc, char** argv)
{
  LINFO("starting program");
  MYLOGVERB = LOG_FULLTRACE;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Still Audio");

  nub::soft_ref<EyeTrackerConfigurator> etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  //added from psycho-movie2
  nub::soft_ref<AudioGrabber> agb(new AudioGrabber(manager));
  manager.addSubComponent(agb);

  // Instantiate our various ModelComponents:
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");

  LINFO("audio grabber initialized");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               " subjnum <img1.ppm> ... <imgN.ppm>", 2, -2)==false)
    return(1);

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  // let's get all our ModelComponent instances started:
  manager.start();

  // setup array of image indices:

  int nbimgs = manager.numExtraArgs();
  nbimgs = nbimgs - 1;
  int index[nbimgs];

  for (int i = 0; i < nbimgs; i ++)
    {
      index[i] = i+1;
      LINFO("%d=%s", i,manager.getExtraArg(i).c_str());
    }

   LINFO("Randomizing %d images...",nbimgs); randShuffle(index, nbimgs);

  for (int i=0;i< nbimgs;i++)
    LINFO("after randShuffle index[%d]=%d",i,index[i]);


  // let's display an ISCAN calibration grid:
  d->clearScreen();
  d->displayISCANcalib();
  d->waitForKey();

  // let's do an eye tracker calibration:
  d->displayText("<SPACE> to calibrate; other key to skip");
  int c = d->waitForKey();
  if (c == ' ') d->displayEyeTrackerCalibration(3, 3);
  d->displayText("<SPACE> to start experiment ");
  d->waitForKey();


 // get the audio going: (from psycho-movie2)
  pthread_t runner;
  pthread_create(&runner, NULL, &audiorecorder, (void *)(agb.get()));
  LINFO("after pthread created");
  char txt[100];

  // main loop:

    for (int rept = 1; rept < 3; rept ++)
    {
        if (rept == 2)
        {

          LINFO("Randomizing %d images...",nbimgs);
          randShuffle(index, nbimgs);

          //allow for a break after 100 trials and then recalibrate.
          d->displayText("You may take a break press space to continue when ready");
          d->waitForKey();

          // let's display an ISCAN calibration grid:
          d->clearScreen();
          d->displayISCANcalib();
          d->waitForKey();


          // let's do an eye tracker calibration:
          d ->displayText("<SPACE> to calibrate");
          int c = d->waitForKey();
          if (c == ' ') d->displayEyeTrackerCalibration(3, 3);
          d ->displayText("<SPACE> to continue with experiment");
          d->waitForKey();
           }

        for (int i = 0; i < nbimgs; i ++)
          {
            // recalibration every 20 trials
            if (i == 20 || i == 40 || i == 60 || i == 80)
              {
                d->displayText("Ready for quick recalibration?(press space)");
                int c = d->waitForKey();
                if (c == ' ') d->displayEyeTrackerCalibration(3, 3);
                d->clearScreen();
                d ->displayText("<SPACE> to continue with experiment");
                d->waitForKey();
              }


          // load up the frame and show a fixation cross on a blank screen:

          d->clearScreen();
          LINFO("Loading '%s'...", manager.getExtraArg(index[i]).c_str());
          Image< PixRGB<byte> > image =
            Raster::ReadRGB(manager.getExtraArg(index[i]));

          SDL_Surface *surf = d->makeBlittableSurface(image, true);

          LINFO("'%s' ready.", manager.getExtraArg(index[i]).c_str());
          d->displayFixation();

           pthread_mutex_lock(&fileName);
          repNum = rept;
          subName = std::string(manager.getExtraArg(0));
          stimName = std::string(manager.getExtraArg(index[i]));
          int len = stimName.length();
          len = len-26;
          stimName=stimName.substr(26,len);
          LINFO("stim name given = %s",stimName.c_str());
          pthread_mutex_unlock(&fileName);

          // ready to go whenever the user is ready:
          d->waitForKey();
          d->waitNextRequestedVsync(false, true);
          d->pushEvent(std::string("===== Showing image: ") +
                       manager.getExtraArg(index[i]) + " =====");

          // start the eye tracker:
          et->track(true);

          // blink the fixation:
          d->displayFixationBlink();

          // start audio recording, unless we were flagged to record later:
          sprintf(txt, "Start audio recording:  rep%d_%s_%s_%d_audio.wav",repNum, manager.getExtraArg(0).c_str(), manager.getExtraArg(index[i]).c_str(), recnb);

          d->pushEvent(txt);
          //recordaudio = true;

          // show the image:
          d->displaySurface(surf, -2);
          usleep(3000000);

          // free the image:
          SDL_FreeSurface(surf);

          // make sure display is off before we stop the tracker:
          d->clearScreen();

          // stop audio recording:
          sprintf(txt, "Stop audio recording: rep%d_%s_%s_%d_audio.wav",repNum, manager.getExtraArg(0).c_str(), manager.getExtraArg(index[i]).c_str(), recnb);

          d->pushEvent(txt);
          //recordaudio = false;

          // stop the eye tracker:
          usleep(30000);
          et->track(false);
          LINFO("i=%d index[0]=%d %s",i,index[0],manager.getExtraArg(index[0]).c_str());
          }

         }
  // kill the audio recording thread:
  keepgoing = false;
  LINFO("made it to keepgoing=false %d",(int)keepgoing);
  sleep(1);
  LINFO("made it to sleep");
  if (system("sync")) LERROR("Error sync()'ing");;
  LINFO("made it to sync");

  d->clearScreen();
  d->displayText("Experiment complete. Thank you!");
  LINFO("made it to display text");
  d->waitForKey();
  LINFO("made it to wait for key");



  // stop all our ModelComponents
  manager.stop();
  LINFO("made it to manager.stop");
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
