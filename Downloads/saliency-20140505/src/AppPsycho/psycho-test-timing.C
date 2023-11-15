/*!@file AppPsycho/psycho-EyeDetect.C pulse a white square on a black
   background, useful for testing timing of display. Duration will be
   50% of period. We use a YUVOverlay to match our video experiments
   
These are valid command options:
bin/psycho-test-timing 100 100 25 0 [01] [01] --sdl-dims=640x480 (no vblank wait)
bin/psycho-test-timing 100 100 25 1 [01] [01] --sdl-dims=640x480 --sdl-vblank-kludge=1
*/

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-test-timing.C $

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/DrawOps.H"
#include "Devices/ParPort.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/PsychoOpts.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Video/RgbConversion.H"
#include "Video/VideoFrame.H"
#include "Util/MathFunctions.H"

// ######################################################################
static int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Psycho test timing");

  OModelParam<std::string> itsPort(&OPT_EyeTrackerParDev, &manager);
  
  nub::soft_ref<EventLog> el(new EventLog(manager));
  manager.addSubComponent(el);
  
  nub::soft_ref<PsychoDisplay> d(new PsychoDisplay(manager));
  manager.addSubComponent(d);

  nub::soft_ref<ParPort> itsParPort(new ParPort(manager, "Parallel Port", "ParPort"));

  itsParPort->setModelParamVal("ParPortDevName", itsPort.getVal());  
  
  manager.addSubComponent(itsParPort);
  
  manager.setOptionValString(&OPT_EventLogFileName, "psychodata.psy");
  
  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,"<xpos> <ypos> <use par> <period in frames> <1=vsync> <1=hard sleep> <1=blink>",7,7)==false)
    return(1);
  
  // hook our various babies up and do post-command-line configs:
  d->setEventLog(el);
  d->setModelParamVal("PsychoDisplayBackgroundColor", PixRGB<byte>(0,0,0));
  
  // let's get all our ModelComponent instances started:
  manager.start();
  
  //stimulus position and size
  const int w = d->getDims().w(), h = d->getDims().h();
  const int fixrad = w / 100;
  const int siz2 = (fixrad - 1) / 2; // half cross size
  const int i = fromStr<int>(manager.getExtraArg(0));
  const int j = fromStr<int>(manager.getExtraArg(1));

  //make an image of our blank and target stimulus, convert our images
  //to yuv format and make a VideoFrame
  Rectangle rect = Rectangle::tlbrO(j - siz2, i - siz2, j + fixrad, i + fixrad);
  Image<PixRGB<byte> > itsStim(w, h, ZEROS);
  byte data[w*h + w*h/2];
  byte *y = &data[0], *u = &data[w*h], *v = &data[w*h + w*h/4];
  toVideoYUV422(itsStim, y, u, v);  
  VideoFrame blank = VideoFrame::deepCopyOf(VideoFrame(data, 
                                                       size_t(w*h + w*h/2), 
                                                       Dims(w,h), 
                                                       VIDFMT_YUV420P, 
                                                       false, true));
  
  drawFilledRect(itsStim, rect, PixRGB<byte>(255,255,255));  
  toVideoYUV422(itsStim, y, u, v);
  VideoFrame vid(data, size_t(w*h + w*h/2), Dims(w,h), 
                 VIDFMT_YUV420P, false, true);

  //create YUV overlay
  d->createVideoOverlay(VIDFMT_YUV420P, w, h);

  //grab some command line data  
  const bool usepar = fromStr<bool>(manager.getExtraArg(2));
  const int time = fromStr<int>(manager.getExtraArg(3));
  const bool vsync = fromStr<bool>(manager.getExtraArg(4));
  const bool hardwait = fromStr<bool>(manager.getExtraArg(5));
  const bool blink = fromStr<bool>(manager.getExtraArg(6));
  int frame = 0;
  int ltime = time;
  if (blink)
    ltime = time/2;

  //loop until key is hit
  if (usepar)
    itsParPort->WriteData(255, 0); //  strobe off

  while ((int)d->checkForKey() != 32)
    {
      //clear screen and sleep
      for (int ii = 0; ii < time; ++ii)
        {
          if (vsync)
            d->displayVideoOverlay(blank, frame, SDLdisplay::NEXT_VSYNC);
          else
            d->displayVideoOverlay(blank, frame, SDLdisplay::NO_WAIT);
          ++frame;
        }

      //hard wait here if if requested
      if (hardwait)
        d->waitForKey();

      //write log and pulse the parallel port
      if (usepar)
        {
          d->pushEventBegin("Parallel Port pulse");
          itsParPort->WriteData(255, 255); // strobe on
          itsParPort->WriteData(255, 0); //  strobe off
          d->pushEventEnd("Parallel Port pulse");
        }
      
      //display stimulus and leave on for 1/2 period
      for (int ii = 0; ii < ltime; ++ii)
        {
          if (vsync)
            d->displayVideoOverlay(vid, frame, SDLdisplay::NEXT_VSYNC);
          else
            d->displayVideoOverlay(vid, frame, SDLdisplay::NO_WAIT);
          ++frame;
          
          if (blink)
            {
              if (vsync)
                d->displayVideoOverlay(blank, frame, SDLdisplay::NEXT_VSYNC);
              else
                d->displayVideoOverlay(blank, frame, SDLdisplay::NO_WAIT);
              ++frame;
            }
        }
    }
  d->destroyYUVoverlay();
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
