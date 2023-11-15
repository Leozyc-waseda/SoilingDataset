/*!@file AppPsycho/psycho-EyeDetect.C Psychophysics display to cause
 * stereotyped eye movements */

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
// Primary maintainer for this file: David Berg <dberg@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-EyeDetect.C $

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "rutz/time.h"

#define TRIAL_LENGTH 15
#define MIN_COUNT 10
#define MAX_COUNT 10
#define MIN_X 100
#define MIN_Y 100
#define MAX_X 540
#define MAX_Y 380
#define PPD 23
#define RECALIB 8

//maybe add some spatial bluring so the pixilation isn't as noticible.
//maybe add a 1/f background noise

// ######################################################################
static int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho EyeDetect");

  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);

  nub::soft_ref<EyeTrackerConfigurator>
    etc(new EyeTrackerConfigurator(manager));
  manager.addSubComponent(etc);

  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  manager.setOptionValString(&OPT_EyeTrackerType, "ISCAN");


  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,"", 0, -1)==false)
    return(1);

  // hook our various babies up and do post-command-line configs:
  nub::soft_ref<EyeTracker> et = etc->getET();
  d->setEyeTracker(et);
  d->setEventLog(el);
  et->setEventLog(el);

  // EyeLink opens the screen for us, so make sure SDLdisplay is slave:
  if (etc->getModelParamString("EyeTrackerType").compare("EL") == 0)
    d->setModelParamVal("SDLslaveMode", true);

  //set our background color and text color to match our stim
  d->setModelParamVal("PsychoDisplayBackgroundColor", PixRGB<byte>(0,0,0));
  d->setModelParamVal("PsychoDisplayTextColor", PixRGB<byte>(255,255,255));
  d->setModelParamVal("PsychoDisplayFixSiz", 14);

  // let's get all our ModelComponent instances started:
  manager.start();

  // let's do a standard eye tracker calibration:
  et->calibrate(d);

  initRandomNumbers();
  //create some stims before display
  int spdln = 12;
  float speeds[12] = {.01, 0.05, 0.1, 0.3, 0.7, 1.0, 1.0, 5.0, 5.0, 10.0, 10.0, 10.0};
  int styln = 10;
  int stays[10] = {0,1,10,20,40,60,80,100,110,120};


   LINFO("Load Our 1/f Background Noise Pre-Generated...");
   FILE* f = fopen ("fnoise.bin", "r");
   fseek (f , 0 , SEEK_END);
   long lSize = ftell (f);
   rewind (f);
   Uint8 * buf;
   buf = (Uint8*) malloc (sizeof(Uint8)*lSize);
   if (fread(buf,2,lSize,f) != (unsigned int)(lSize)) LFATAL("fread error");
   fclose(f);
   lSize = lSize/(640*480);
   LINFO("LENGTH: %d\n",(int)lSize);

  for (int trials = 0; trials < TRIAL_LENGTH; ++trials)
     {
       //give other processes else a chance
       sleep(1); if (system("/bin/sync")) LERROR("error in sync()");

       LINFO("Creating Random Background for Trial");
       //create a new background image each time
       Image< PixRGB<byte> > img(640,480,ZEROS);

       int count = 0;
       int index = (int)floor(randomDouble()*lSize);
       for (int xx = 0; xx < 640; ++xx)
         for (int yy = 0; yy < 480; ++yy)
           {
             // LINFO("X:%d Y:%d",xx,yy);
             img.setVal(xx,yy,PixRGB<byte>((byte)buf[(640*480)*index + count],0,0));
             ++count;
           }

    //randomize the number of moves for each trial
       int num_loc = (int)floor(MIN_COUNT + randomDouble()*(MAX_COUNT - MIN_COUNT));

       //generate some random locations before we get started
       int location[num_loc][2];
       float speed[num_loc];
       int stay[num_loc-1];

       for (int ii = 0; ii < num_loc; ++ii)
         {
           int x =  (int)floor(MIN_X + randomDouble() * (MAX_X - MIN_X));
           int y =  (int)floor(MIN_Y + randomDouble() * (MAX_Y - MIN_Y));
           location[ii][0] = x;
           location[ii][1] = y;
           float sp =  speeds[(int)floor(randomDouble() * spdln)]*PPD;
           int st = stays[(int)floor(randomDouble() * (float)styln)];
           speed[ii] = sp;
           if (ii < num_loc-1)
             stay[ii] = st;
           //  LINFO(" Location X: %d, location Y: %d, Speed: %f, Fixation: %d\n",x,y,sp,st);
         }



    char str[50];
    sprintf(str,"Trial %d of %d: Follow the Red Dot <Space to Start>",trials+1,TRIAL_LENGTH);
    d->clearScreen();
    d->clearScreen();
    d->displayText(str);
    d->waitForKey();
    d->clearScreen();

    d->displayImage(img);
    // display fixation to indicate that we are ready:
    d->displayRedDotFixation();
    // ready to go whenever the user is ready:
    d->waitForKey();
    d->waitNextRequestedVsync(false, true);
    sprintf(str,"==== Trial: %d ====",trials+1);
    d->pushEvent(str);
    // start the eye tracker:
    et->track(true);
    // blink the fixation: this is 1 second I think
    d->displayRedDotFixationBlink(img);
    //display our randomly created stimulus
    LINFO("Displaying Train\n");
    d->waitNextRequestedVsync(false, true);
    d->displayMovingDotTrain(img,location, num_loc, speed, stay,PixRGB<byte>(255,0,0));
    //now we are done with our one trial clear,sleep and recalibrate
    d->waitNextRequestedVsync(false, true);
    d->clearScreen();
    d->clearScreen();
    // stop the eye tracker:
    usleep(50000);
    et->track(false);
    // let's do a quickie eye tracker recalibration:
    if (((trials+1) % RECALIB) == 0)
      {
        LINFO("Calibrating\n");
        et->recalibrate(d);
      }
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
