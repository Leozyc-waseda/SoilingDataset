/*!@file Demo/app-chimpanzee.C Saliency-driven chimpanzee head */

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
// Primary maintainer for this file: Zack Gossman <gossman@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Demo/app-chimpanzee.C $
// $Id: app-chimpanzee.C 11630 2009-08-28 01:35:32Z dberg $
//

#include "Component/ModelManager.H"
#include "Demo/SaliencyMT.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "GUI/XWindow.H"
#include "Image/CutPaste.H"     // for inplacePaste()
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Neuro/NeuroOpts.H"
#include "Neuro/NeuroSimEvents.H"
#include "Neuro/EyeHeadControllerConfigurator.H"
#include "Neuro/SaccadeController.H"
#include "Raster/Raster.H"
#include "Simulation/SimEventQueue.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Devices/BeoMonkey.H"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#include "Image/Kernels.H"
#include "Image/Convolver.H"

//! Number of frames over which average framerate is computed
#define NAVG 20

#define EVEL 10  //should depend on distance
#define HVEL 10   //should depend on distance
//! Factor to display the sm values as greyscale:
#define SMFAC 0.05F

static bool goforever = true;  //!< Will turn false on interrupt signal

//! Signal handler (e.g., for control-C)
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; }

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager (for camera input):
  ModelManager manager("SaliencyMT Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<SimEventQueueConfigurator>
    seqc(new SimEventQueueConfigurator(manager));
  manager.addSubComponent(seqc);

  nub::ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  nub::ref<SaliencyMT> smt(new SaliencyMT(manager));
  manager.addSubComponent(smt);

  nub::ref<EyeHeadControllerConfigurator>
    ehcc(new EyeHeadControllerConfigurator(manager));
  manager.addSubComponent(ehcc);

  nub::ref<BeoMonkey> bc(new BeoMonkey(manager));
  manager.addSubComponent(bc);

  // Set the appropriate defaults for our machine that is connected to
  // the chimpanzee robot head:
  manager.exportOptions(MC_RECURSE);
  manager.setOptionValString(&OPT_FrameGrabberType, "V4L");
  manager.setOptionValString(&OPT_FrameGrabberChannel, "1");
  manager.setOptionValString(&OPT_FrameGrabberHue, "0");
  manager.setOptionValString(&OPT_FrameGrabberContrast, "16384");
  manager.setOptionValString(&OPT_FrameGrabberDims, "320x240");
  //  manager.setOptionValString(&OPT_SaccadeControllerType, "Threshfric");
  manager.setOptionValString(&OPT_EyeHeadControllerType, "Monkey");
  manager.setOptionValString(&OPT_SaccadeControllerEyeType, "Monkey");
  manager.setOptionValString(&OPT_SaccadeControllerHeadType, "Monkey");
  manager.setOptionValString(&OPT_SCeyeMaxIdleSecs, "500.0");
  manager.setOptionValString(&OPT_SCeyeThreshMinOvert, "1.0");
  manager.setOptionValString(&OPT_SCeyeThreshMaxCovert, "3.0");
  manager.setOptionValString(&OPT_SCeyeThreshMinNum, "3");
  //  manager.setOptionValString(&OPT_SCeyeSpringK, "1000000.0");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // do post-command-line configs:
  nub::soft_ref<SimEventQueue> seq = seqc->getQ();

  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful");
  const int w = gb->getWidth(), h = gb->getHeight();

  nub::ref<EyeHeadController> ehc = ehcc->getEHC();

  const int foa_size = std::min(w, h) / 12;
  manager.setModelParamVal("InputFrameDims", Dims(w, h),
                           MC_RECURSE | MC_IGNORE_MISSING);
  manager.setModelParamVal("SCeyeStartAtIP", true,
                           MC_RECURSE | MC_IGNORE_MISSING);
  manager.setModelParamVal("SCeyeInitialPosition", Point2D<int>(w/2,h/2),
                           MC_RECURSE | MC_IGNORE_MISSING);
  manager.setModelParamVal("FOAradius", foa_size,
                           MC_RECURSE | MC_IGNORE_MISSING);
  manager.setModelParamVal("FoveaRadius", foa_size,
                           MC_RECURSE | MC_IGNORE_MISSING);

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // get prepared to grab, move, display, etc:
  uint frame = 0U;                  // count the frames
  uint lastframe = 0U;              // last frame sent for processing
  Point2D<int> lastpoint(w/2, h/2);      // last point of fixation

  uint64 avgtime = 0; int avgn = 0; // for average framerate
  float fps = 0.0F;                 // to display framerate
  Timer tim;                        // for computation of framerate
  Timer masterclock;                // master clock for simulations

  int sml = 3;                      // pyramid level of saliency map
  Image<float> sm(w >> sml, h >> sml, ZEROS); // saliency map
  Point2D<int> fixation(-1, -1);         // coordinates of eye fixation

  // image buffer for display:
  Image<PixRGB<byte> > disp(w * 2, h + 20, ZEROS);
  disp += PixRGB<byte>(128);
  XWindow xwin(disp.getDims(), -1, -1, "USC Chimpanzee Demo");

  char info[1000];  // general text buffer for various info messages


  // ######################################################################
  try {
    // let's do it!
    manager.start();

    // get the frame grabber to start streaming:
    gb->startStream();

    // initialize the timers:
    tim.reset(); masterclock.reset();

    //set up a convoltion mask
    //  Image<float> gauss = gaussian<float>(0.0F,25.0F,0);
    //  Convolver cg(gauss,gauss.getDims());

    while(goforever)
      {
        // grab image:
        Image< PixRGB<byte> > ima = gb->readRGB();

        // display image:
        inplacePaste(disp, ima, Point2D<int>(0, 0));
        Image<float> dispsm = sm * SMFAC;
        inplacePaste(disp, Image<PixRGB<byte> >
                     (toRGB(quickInterpolate(dispsm, 1 << sml))),
                     Point2D<int>(w, 0));

        //        ehc->evolve(*seq);

        Point2D<int> ceye(-1, -1), chead(-1, -1);

        if (SeC<SimEventSaccadeStatusEye> e =
            seq->check<SimEventSaccadeStatusEye>(0)) ceye = e->position();
        if (SeC<SimEventSaccadeStatusHead> e =
            seq->check<SimEventSaccadeStatusHead>(0)) chead = e->position();

        if (ceye.isValid()) fixation = ceye;
        Point2D<int> fix2(fixation); fix2.i += w;
        if (fixation.i >= 0)
          {
            drawDisk(disp, fixation, foa_size/6+2, PixRGB<byte>(20, 50, 255));
            drawDisk(disp, fixation, foa_size/6, PixRGB<byte>(255, 255, 20));
            drawDisk(disp, fix2, foa_size/6+2, PixRGB<byte>(20, 50, 255));
            drawDisk(disp, fix2, foa_size/6, PixRGB<byte>(255, 255, 20));
          }

        xwin.drawImage(disp);

        // are we ready to process a new frame? if so, send our new one:
        if (smt->outputReady())
          {
            // let's get the previous results, if any:
            Image<float> out = smt->getOutput();
            if (out.initialized()) sm = out;

            //Lets blur the saliency map a little
            // sm = cg.fftConvolve(sm);

            // find most salient location and feed saccade controller:
            float maxval; Point2D<int> currwin; findMax(sm, currwin, maxval);
            WTAwinner newwin =
              WTAwinner::buildFromSMcoords(currwin, sml, true,
                                           masterclock.getSimTime(),
                                           maxval, false);
            if (newwin.isValid())
              {
                //      rutz::shared_ptr<SimEventWTAwinner>
                //   e(new SimEventWTAwinner(0, newwin));
                // seq->post(e);
              }

            // feed our current image as next one to process:
            smt->newInput(decXY(ima));
            lastframe = frame;
            lastpoint.i = newwin.p.i * 2; lastpoint.j = newwin.p.j * 2;
            //LINFO("Processing frame %u", frame);
          }

        // compute and show framerate and stats over the last NAVG frames:
        avgtime += tim.getReset(); avgn ++;
        if (avgn == NAVG)
          {
            fps = 1000.0F / float(avgtime) * float(avgn);
            avgtime = 0; avgn = 0;
          }


       // float x = ((float)lastpoint.i/640)*2 -1;
       // float y = ((float)lastpoint.j/480)*2 -1;
       float xeye = (float)ceye.i;
       float yeye = (float)ceye.j;
       float xhead = (float)chead.i;
       float yhead = (float)chead.j;

        // create an info string:
        sprintf(info, "USC Chimpanzee - %06u / %06u - [%03f %03f] - %.1ffps ",
                frame, lastframe, xeye, yeye, fps);

        writeText(disp, Point2D<int>(0, h), info,
                  PixRGB<byte>(255), PixRGB<byte>(127));

        //make a movement
        if (bc->isQueEmpty())
          {
            LINFO("%f %f/n",xhead,yhead);
            if ((xeye != -1) & (yeye != -1))
              {
                xeye = -1*((xeye/320)*2-1);
                yeye = -1*((yeye/240)*2-1);
                std::deque<Position> p = bc->blend(bc->getPathServo(BeoMonkey::H_EYE,xeye,EVEL),bc->getPathServo(BeoMonkey::V_EYE,yeye,EVEL));
                bc-> addSequence(p);
              }

            if ((xhead != -1) & (yhead != -1))
              {
                xhead = -1*((xhead/320)*2-1);
                yhead = -1*((yhead/240)*2-1);
                std::deque<Position> p = bc->blend(bc->getPathServo(BeoMonkey::H_HEAD,xhead,HVEL),bc->getPathServo(BeoMonkey::V_HEAD,yhead,HVEL));
                bc-> addSequence(p);
              }

          }
           bc->nextTimeStep();

        // ready for next frame:
        ++ frame;
        while(seq->now() < masterclock.getSimTime()) seq->evolve();
      }

    // get ready to terminate:
    manager.stop();

  } catch ( ... ) { };

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
