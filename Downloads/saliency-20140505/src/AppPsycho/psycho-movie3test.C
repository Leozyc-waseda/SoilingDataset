/*!@file AppPsycho/psycho-movie2.C Psychophysics interactive display of movies */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-movie3test.C $
// $Id: psycho-movie3test.C 10794 2009-02-08 06:21:09Z itti $
//

#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Audio/AudioWavFile.H"
#include "Devices/AudioGrabber.H"
#include "Devices/AudioMixer.H"
#include "Devices/DeviceOpts.H"
#include "Image/Image.H"
#include "Media/MPEGStream.H"
#include "Media/MediaOpts.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "GUI/SDLdisplay.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "Video/VideoFrame.H"
#include "Neuro/NeuroOpts.H"

#include <vector>
#include <pthread.h>

#define CACHELEN 150

volatile bool recordaudio = false;
volatile bool keepgoing = true;
volatile int recnb = 0;

static const ModelOptionDef OPT_AudioAfter =
  { MODOPT_FLAG, "AudioAfter", &MOC_DISPLAY, OPTEXP_SAVE,
    "Record audio after the movie presentation",
    "audio-after", '\0', "", "false" };

// ######################################################################
static void *audiorecorder(void *agbv)
{
  bool recording = false;
  std::vector<AudioBuffer<byte> > rec;
  AudioGrabber *agb = (AudioGrabber *)agbv;

  while(keepgoing)
    {
      // start recording?
      if (recording == false && recordaudio == true)
        { rec.clear(); recording = true; }

      // stop recording?
      if (recording == true && recordaudio == false)
        {
          // save our current records
          char fname[100]; sprintf(fname, "audio%04d.wav", recnb);
          writeAudioWavFile(fname, rec);

          // ready for next recording:
          recnb ++; recording = false;
        }

      // we grab the audio data all the time, so that it does not
      // start piling up into the internal buffers of the audio
      // grabber; but we will save it only if we are in recording
      // mode. Note: the call to grab() is blocking and hence sets
      // the pace of our loop here. With 256 samples at 11.025kHz
      // that's about 23ms:
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

// // ######################################################################
// static bool cacheFrame(nub::soft_ref<InputMPEGStream>& mp,
//                        std::deque<VideoFrame>& cache)
// {
//   const VideoFrame frame = mp->readVideoFrame();
//   if (!frame.initialized()) return false; // end of stream

//   cache.push_front(frame);
//   return true;
// }

// ######################################################################
static int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho Movie");

  OModelParam<bool> audioAfter(&OPT_AudioAfter, &manager);

  // Instantiate our various ModelComponents:
//   nub::soft_ref<InputMPEGStream> mp
//     (new InputMPEGStream(manager, "Input MPEG Stream", "InputMPEGStream"));
//   manager.addSubComponent(mp);

//   nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
//   manager.addSubComponent(d);

  nub::soft_ref<AudioMixer> mix(new AudioMixer(manager));
  manager.addSubComponent(mix);

  nub::soft_ref<AudioGrabber> agb(new AudioGrabber(manager));
  manager.addSubComponent(agb);

 //  nub::soft_ref<EyeTrackerConfigurator>
//     etc(new EyeTrackerConfigurator(manager));
//   manager.addSubComponent(etc);

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  // set a few defaults:
//   manager.setOptionValString(&OPT_InputMPEGStreamPreload, "true");
//   manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
//   manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");
  manager.setOptionValString(&OPT_AudioMixerLineIn, "false");
  manager.setOptionValString(&OPT_AudioMixerCdIn, "false");
  manager.setOptionValString(&OPT_AudioMixerMicIn, "true");
  manager.setOptionValString(&OPT_AudioGrabberBits, "8");
  manager.setOptionValString(&OPT_AudioGrabberFreq, "11025");
  manager.setOptionValString(&OPT_AudioGrabberBufSamples, "256");
  manager.setOptionValString(&OPT_AudioGrabberStereo, "false");

  // Parse command-line:
 //  if (manager.parseCommandLine(argc, argv,
//                                "<movie1.mpg> ... <movieN.mpg>", 1, -1)==false)
//     return(1);

//   if(manager.numExtraArgs() > 0)
//     {
//       infilename = manager.getExtraArgAs<std::string>(0);
//     }

  // hook our various babies up and do post-command-line configs:
  // nub::soft_ref<EyeTracker> et = etc->getET();
//   d->setEyeTracker(et);
//   d->setEventLog(el);
//   et->setEventLog(el);

  // let's get all our ModelComponent instances started:
  manager.start();

  // let's display a static low-level ISCAN calibration grid:
 //  d->clearScreen();
//   d->displayISCANcalib();
//   d->waitForKey();

//   // setup array of movie indices:
//   uint nbmovies = manager.numExtraArgs(); int index[nbmovies];
//   for (uint i = 0; i < nbmovies; i ++) index[i] = i;
//   LINFO("Randomizing movies..."); randShuffle(index,nbmovies);

  // get the audio going:
  pthread_t runner;
  pthread_create(&runner, NULL, &audiorecorder, (void *)(agb.get()));
  char txt[100];

  // main loop:
  std::deque<VideoFrame> cache;

  int movieCount = 1;

  for (uint i = 0; i < nbmovies; i ++)
    {
   //    // let's do an eye-tracker calibration once in a while:
//       if (i % 10 == 0)
//         {
//           d->displayText("<SPACE> for eye-tracker calibration");
//           d->waitForKey();
//           //    d->displayEyeTrackerCalibration(3, 3);
//           d->clearScreen();
//           if (i == 0) d->displayText("<SPACE> to start with the movies");
//           else d->displayText("<SPACE> to continue with the movies");
//           d->waitForKey();
//         }

//       // cache initial movie frames:
//       d->clearScreen();
//       if (cache.size()) LFATAL("ooops, cache not empty?");
//       bool streaming = true;
//       LINFO("Buffering '%s'...", manager.getExtraArg(index[i]).c_str());
//       // NOTE: we now specify --preload-mpeg=true with a
//       // setOptionValString() a few lines up from here
//       mp->setFileName(manager.getExtraArg(index[i]));
//       for (uint j = 0; j < CACHELEN; j ++)
//         {
//           streaming = cacheFrame(mp, cache);
//           if (streaming == false) break;  // all movie frames got cached!
//         }
//       LINFO("'%s' ready.", manager.getExtraArg(index[i]).c_str());

//       // give a chance to other processes (useful on single-CPU machines):
//       sleep(1); system("sync");

//       // display fixation to indicate that we are ready:
//       d->displayFixation();

      // ready to go whenever the user is ready:
   //    d->waitForKey(); int frame = 0;
//       d->waitNextRequestedVsync(false, true);
//       d->pushEvent(std::string("===== Playing movie: ") +
//                    manager.getExtraArg(index[i]) + " =====");

//       // start the eye tracker:
//       et->track(true);

//       // blink the fixation:
//       d->displayFixationBlink();

//       // create an overlay:
//       d->createVideoOverlay(VIDFMT_YUV420P); // mpeg stream returns YUV420P

      // start audio recording, unless we were flagged to record later:
      if (audioAfter.getVal() == false)
        {
          sprintf(txt, "Start audio recording: audio%04d.wav", recnb);
          d->pushEvent(txt);
          recordaudio = true;
        }

      // play the movie:
     //  while(cache.size())
//         {
//           // let's first cache one more frame:
//           if (streaming) streaming = cacheFrame(mp, cache);

//           // get next frame to display and put it into our overlay:
//           VideoFrame vidframe = cache.back();
//           d->displayVideoOverlay(vidframe, frame,
//                                  SDLdisplay::NEXT_VSYNC);
//           cache.pop_back();

//           ++frame;
//         }

      // destroy the overlay. Somehow, mixing overlay displays and
      // normal displays does not work. With a single overlay created
      // before this loop and never destroyed, the first movie plays
      // ok but the other ones don't show up:
   //    d->destroyYUVoverlay();
//       d->clearScreen();  // sometimes 2 clearScreen() are necessary



//       // wait for a while to allow people to finish their current sentence:
//       sleep(3);

      if(movieCount % 5 == 0)
        {
          // stop the eye tracker:
          et->track(false);
          d->displayText("Please describe what you saw in the last five videos.");
          sleep(5);

          // do we want to record some audio after the movie is done?
          if (audioAfter.getVal())
            {
              sprintf(txt, "Start audio recording after 5 movies: audio%04d.wav", recnb);
              d->pushEvent(txt);
              recordaudio = true;
              d->clearScreen();
              d->displayFixation();

              usleep(25000000); // 25 secs of recording

              // stop audio recording:
              sprintf(txt, "Stop audio recording after 5 movies: audio%04d.wav", recnb);
              d->pushEvent(txt);
              recordaudio = false;

              d->clearScreen();
            }
        }

      movieCount++;


   //    // stop the eye tracker:
//       et->track(false);

      // stop audio recording:
      sprintf(txt, "Stop audio recording: audio%04d.wav", recnb);
      d->pushEvent(txt);
      recordaudio = false;
    }

  // stop audio recording:
  sprintf(txt, "Stop audio recording: audio%04d.wav", recnb);
  d->pushEvent(txt);
  recordaudio = false;

  d->clearScreen();
  d->displayText("Experiment complete. Thank you!");
  d->waitForKey();

  // kill the audio recording thread:
  keepgoing = false;
  usleep(500000);

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
