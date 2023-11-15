/*!@file AppDevices/app-cam-saccades.C Use a pan-tilt camera for executing saccades */

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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/app-cam-saccades.C $
// $Id: app-cam-saccades.C 10982 2009-03-05 05:11:22Z itti $
//

#include "Channels/ChannelBase.H"
#include "Channels/ChannelOpts.H"
#include "Component/GlobalOpts.H"
#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/VCC4.H"
#include "GUI/XWinManaged.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/ShapeOps.H"
#include "Image/fancynorm.H"
#include "Media/MediaSimEvents.H"
#include "Neuro/NeuroOpts.H"
#include "Neuro/NeuroSimEvents.H"
#include "Neuro/SaccadeControllers.H"
#include "Neuro/ShapeEstimator.H"
#include "Neuro/SimulationViewerStd.H"
#include "Neuro/SpatialMetrics.H"
#include "Neuro/StdBrain.H"
#include "Neuro/VisualCortex.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "Transport/FrameIstream.H"
#include "Util/log.H"

#include <algorithm> // for std::max() and std::min()
#include <cmath>
#include <iostream>
#include <stdio.h>

//#define CAM_DISPLAY NULL
#define CAM_WINDOW_NAME "Saliency Calculated"
#define CAM_PREV_NAME "Captured Image"
#define ANG_FACTOR (180.0/Pi)
#define MAXPAN ((float)100.0)
#define MINPAN ((float)-100.0)
#define MAXTILT ((float)30.0)
#define MINTILT ((float)-30.0)

// make random saccade if the salmap evolves for more than TOO_MUCH_TIME s
#define TOO_MUCH_TIME SimTime::SECS(0.7)


float sqr (float x);
float randang();

// give back a random angle increment/decrement
inline float randang ()
{
  const float max_randang = 30.0;
  return ((float)(2.0 * rand() / RAND_MAX) - 1.0) * max_randang;
}

// sqr(x) = x*x;
inline float sqr (float x)
{
  return (x * x);
}

// ######################################################################
// ##### Main Program:
// ######################################################################
/*! This program takes input from a camera, calculates
  the most salient spot(s) and moves the camera there
  according to the following rules:<p>
  1) take image from camera
  2) calculate next most salient spot
  3) If (spot is already in list) goto 2
  4) store spot in list
  5) move to the spot
  6) goto 1<p>
  The list is finite (small) sized and "forgets" the oldest entries.
  When the search for the next most salient spot in a given image
  takes too much time, the camera is moved by a random amount.*/
int main(const int argc, const char **argv)
{
  LOG_FLAGS &= (~LOG_FULLTRACE);

  // instantiate a model manager:
  ModelManager manager("Saccade Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<SimEventQueueConfigurator>
    seqc(new SimEventQueueConfigurator(manager));
  manager.addSubComponent(seqc);

  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  nub::soft_ref<VCC4> pantilt(new VCC4(manager));
  manager.addSubComponent(pantilt);

  nub::soft_ref<StdBrain> brain(new StdBrain(manager));
  manager.addSubComponent(brain);

  nub::ref<SpatialMetrics> metrics(new SpatialMetrics(manager));
  manager.addSubComponent(metrics);

  // choose a V4Lgrabber by default, and a few custom grabbing
  // defaults, for backward compatibility with an older version of
  // this program:
  manager.setOptionValString(&OPT_FrameGrabberType, "V4L");
  manager.setOptionValString(&OPT_FrameGrabberDims, "320x240");
  manager.setOptionValString(&OPT_FrameGrabberChannel, "1");

  // set some more parameters as defaults
  manager.setOptionValString(&OPT_OriInteraction,"SubtractMean");
  manager.setOptionValString(&OPT_OrientComputeType,"Steerable");
  manager.setOptionValString(&OPT_RawVisualCortexChans,"OIC");
  manager.setOptionValString(&OPT_UseRandom,"false");
  manager.setOptionValString(&OPT_ShapeEstimatorMode,"FeatureMap");
  manager.setOptionValString(&OPT_ShapeEstimatorSmoothMethod,"Chamfer");
  manager.setOptionValString(&OPT_IORtype,"ShapeEst");
  manager.setOptionValString(&OPT_SVdisplayFOA,"true");
  manager.setOptionValString(&OPT_SVdisplayPatch,"false");
  manager.setOptionValString(&OPT_SVdisplayFOALinks,"false");
  manager.setOptionValString(&OPT_SVdisplayAdditive,"false");
  manager.setOptionValString(&OPT_SVdisplayTime,"false");
  manager.setOptionValString(&OPT_SVdisplayBoring,"false");
  metrics->setFOAradius(20);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful");
  const Dims dims = gb->peekDims();
  nub::soft_ref<SimEventQueue> seq = seqc->getQ();

  // let's get all our ModelComponent instances started:
  manager.start();

  float hdeg = 80, wdeg = 100;;
  float min_dist = std::min(hdeg, wdeg) / 12;
  const int num_mem = 10;
  float mem_pan[num_mem], mem_tilt[num_mem];
  int mem_ptr = 0;
  bool mem_filled = false;
  bool is_in_mem;
  int mem_top;

  pantilt->CameraInitialize(true);
  pantilt->PlainCommand(VCC4_SetZoomingWIDE);

  XWinManaged salWin(dims*2, -1, -1, CAM_WINDOW_NAME);
  XWinManaged capWin(dims*2, -1, -1, CAM_PREV_NAME);

  //StdBrain brain(4,2,4,3,4,4,SMDfoa,VCXNORM_FANCY);
  Image<PixRGB <byte> > capImg, salImg, capImg2;

  float campos_pan = 0.0, campos_tilt = 0.0;
  float oldpos_pan = 0.0, oldpos_tilt = 0.0;
  float newpos_pan, newpos_tilt;

  // main loop
  while (!salWin.pressedCloseButton() && !capWin.pressedCloseButton())
    {
      // wait until camera stops moving
      //while (pantilt->IsMoving()) usleep (10000);
      //usleep (100000);

      brain->reset(MC_RECURSE);

      // capture image and show in preview window
      for (int i = 0; i < 20; i++) capImg = gb->readRGB();
      capImg2 = capImg;
      drawDisk(capImg2, Point2D<int>(dims/2),
               3, PixRGB<byte>(255,0,0));
      capWin.drawImage(rescale(capImg2,dims*2));
      rutz::shared_ptr<SimEventInputFrame>
        e(new SimEventInputFrame(brain.get(), GenericFrame(capImg), 0));
      seq->post(e); // post the image to the brain

      // evolve
      seq->resetTime();
      while(true)
        {
          Point2D<int> winner(-1, -1);
          while(seq->now() < TOO_MUCH_TIME)
            {
              if (SeC<SimEventWTAwinner> e =
                  seq->check<SimEventWTAwinner>(0))
                {
                  winner = e->winner().p;
                  break;
                }
              if (seq->evolve() == SIM_BREAK) LFATAL("BREAK");
            }

          // too much time -> random saccade
          if (seq->now() >= TOO_MUCH_TIME || winner.isValid() == false)
            {
              newpos_pan = oldpos_pan + randang();
              newpos_pan = std::max(newpos_pan, MINPAN);
              newpos_pan = std::min(newpos_pan, MAXPAN);

              newpos_tilt = oldpos_tilt + randang();
              newpos_tilt = std::max(newpos_tilt, MINTILT);
              newpos_tilt = std::min(newpos_tilt, MAXTILT);

              LINFO("Using random position.");
              break;
            }

          // calculate new camera position and make sure they are in
          // the right range
          newpos_pan = oldpos_pan + (((float)winner.i -
                                      0.5 * (float)dims.w()) /
                                     (float)dims.w() * 0.5 * wdeg);
          //newpos_pan = oldpos_pan + ANG_FACTOR * tan(X_FACTOR *
          //((float)winner.i - 0.5*dims.w()));
          newpos_pan = std::max(newpos_pan, MINPAN);
          newpos_pan = std::min(newpos_pan, MAXPAN);

          newpos_tilt = oldpos_tilt + ((0.5 * (float)dims.h() -
                                        (float)winner.j) /
                                       (float)dims.h() * 0.5 * hdeg);
          //newpos_tile = oldpos_tilt + ANG_FACTOR * tan(Y_FACTOR *
          //((float)winner.j - 0.5*dims.h()));
          newpos_tilt = std::max(newpos_tilt, MINTILT);
          newpos_tilt = std::min(newpos_tilt, MAXTILT);

          is_in_mem = false;
          mem_top = mem_filled ? num_mem : mem_ptr;
          for (int i = 0; i < mem_top; i++)
            is_in_mem |= (sqr(newpos_pan - mem_pan[i]) +
                          sqr(newpos_tilt - mem_tilt[i]) < sqr(min_dist));

          if (!is_in_mem) break;

          LDEBUG("Using next cycle.");
          LDEBUG("newpos = %f, %f", newpos_pan, newpos_tilt);
#ifdef DEBUG
          std::cout << "Mark 15 \n";
          for (int i = 0; i < mem_top; i++)
            LDEBUG("mempos[%i] = %f, %f",i, mem_pan[i], mem_tilt[i]);
#endif
        }

      campos_pan  = newpos_pan;
      campos_tilt = newpos_tilt;

      mem_pan  [mem_ptr] = newpos_pan;
      mem_tilt [mem_ptr] = newpos_tilt;
      mem_ptr++;

      if (mem_ptr >= num_mem)
        {
          mem_ptr = 0;
          mem_filled = true;
        }

      LDEBUG("mem_ptr = %i",mem_ptr);
      LDEBUG("newpos = %i, %i",int(newpos_pan),int(newpos_tilt));

      oldpos_pan  = newpos_pan;
      oldpos_tilt = newpos_tilt;
      pantilt->gotoPosition (campos_pan, campos_tilt);

      // set the window's title bar to the ShapeEstimatorLabel
      //salWin.setTitle(brain->getSE()->getWinningLabel().c_str());

      // retrieve the image with marked salient spot and display it
      ////////////salImg = brain->getSV()->getTraj(seq->now());
      salWin.drawImage(rescale(salImg,dims*2));

      // end of the main loop
    }
  pantilt->PlainCommand(VCC4_GoHome);
  manager.stop();
  return 0;
}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
