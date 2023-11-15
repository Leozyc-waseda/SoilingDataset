/*!@file Parallel/pvisionTCP4-master.C Grab & process over beowulf w/ pvisionTCP3. Like pvisionTCP3 but with input and output FrameSeries */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Parallel/pvisionTCP4-master.C $
// $Id: $
//

/*! This parallel vision processing master is for use with
  pvisionTCP3.  See the pvisionTCP3go script in bin/ for how to launch
  the slaves. The main difference between the TCP2 and TCP3 versions
  is that TCP3 uses load-balancing to send images to the slaves, and
  the master node is not only in charge of grabbing the video and
  displaying the results, but also of collecting the various
  conspicuity maps and assembling them into the saliency map. */

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"     // for inplacePaste()
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"     // for decX() etc.
#include "Image/Transforms.H"
#include "Neuro/NeuroOpts.H"
#include "Neuro/SaccadeControllerConfigurator.H"
#include "Neuro/SaccadeController.H"
#include "Parallel/pvisionTCP-defs.H"
#include "Simulation/SimEventQueue.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "Transport/FrameIstream.H"
#ifdef HAVE_SDL_SDL_H
#include "GUI/SDLdisplay.H"
#endif
#include "Util/Assert.H"
#include "Util/FpsTimer.H"
#include "Util/Timer.H"
#include "Util/sformat.H"
#include "Video/RgbConversion.H" // for toVideoYUV422()
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"

#include <signal.h>
#include <unistd.h>


//! prescale level by which we downsize images before sending them off
#define PRESCALE            2

//! averaging window
#define NAVG               20
#define TOTAL_FRAMES      100

static bool goforever = true;  //!< Will turn false on interrupt signal

//! Signal handler (e.g., for control-C)
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; }// exit(1); }

//! function to receive results from slaves
void receiveCMAPS(nub::soft_ref<Beowulf>& beo, Image<float> *cmap,
                  int32 *cmapframe);
uint nCMAPSreceived;  
FpsTimer tim;                     // for computation of framerate
double prevTime = 0.0;
std::vector<double> times;

// ######################################################################
int submain(int argc, char** argv)
{
#ifndef HAVE_SDL_SDL_H
  LFATAL("<SDL/SDL.h> must be installed to use this program");
#else
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("Parallel Vision TCP Version 3");

  // Instantiate our various ModelComponents:
  nub::soft_ref<SimEventQueueConfigurator>
    seqc(new SimEventQueueConfigurator(manager));
  manager.addSubComponent(seqc);

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Master", "BeowulfMaster", true));
  manager.addSubComponent(beo);

  nub::soft_ref<SaccadeControllerEyeConfigurator>
    scc(new SaccadeControllerEyeConfigurator(manager));
  manager.addSubComponent(scc);

  // let's set a bunch of defauls:
  manager.setOptionValString(&OPT_SaccadeControllerEyeType, "Threshfric");
  manager.setOptionValString(&OPT_FrameGrabberType, "V4L");
  manager.setOptionValString(&OPT_SCeyeMaxIdleSecs, "1000.0");
  manager.setOptionValString(&OPT_SCeyeThreshMinOvert, "4.0");
  manager.setOptionValString(&OPT_SCeyeThreshMaxCovert, "3.0");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // do post-command-line configs:
  nub::soft_ref<SimEventQueue> seq = seqc->getQ();

  int w = ifs->getWidth(),  h = ifs->getHeight();
  std::string dims = convertToString(Dims(w, h));
  LINFO("image size: [%dx%d]", w, h);
  //manager.setOptionValString(&OPT_InputFrameDims, dims);

  manager.setModelParamVal("InputFrameDims", Dims(w, h),
                           MC_RECURSE | MC_IGNORE_MISSING);

  nub::ref<SaccadeController> sc = scc->getSC();

  const int foa_size = 64;
  manager.setModelParamVal("InputFrameDims", Dims(w, h),
                           MC_RECURSE | MC_IGNORE_MISSING);
  manager.setModelParamVal("SCeyeStartAtIP", true,
                           MC_RECURSE | MC_IGNORE_MISSING);
  manager.setModelParamVal("SCeyeInitialPosition",Point2D<int>(w/2,h/2),
                           MC_RECURSE | MC_IGNORE_MISSING);
  manager.setModelParamVal("FOAradius", foa_size,
                           MC_RECURSE | MC_IGNORE_MISSING);
  manager.setModelParamVal("FoveaRadius", foa_size,
                           MC_RECURSE | MC_IGNORE_MISSING);
  //manager.setModelParamVal("SimulationTimeStep", SimTime::MSECS(1.0),
  //                         MC_RECURSE | MC_IGNORE_MISSING);

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // get prepared to grab, communicate, display, etc:
  PixRGB<byte> pix(255, 255, 0);    // yellow color for fixations
  TCPmessage smsg;                  // buffer to send messages to nodes
  Timer masterclock;                // master clock for simulations

  Image<float> cmap[NBCMAP4];       // array of conspicuity maps
  int32 cmapframe[NBCMAP4];         // array of cmap frame numbers
  for (int i = 0; i < NBCMAP4; i ++) cmapframe[i] = -1;
  int sml = 4;                      // pyramid level of saliency map
  Image<float> sm(w >> sml, h >> sml, ZEROS); // saliency map
  Point2D<int> fixation(-1, -1);         // coordinates of eye fixation

  nCMAPSreceived = 0;

  // image buffer for display:
  const PixRGB<byte> grey(127);
  std::string info;

  // let's get all our ModelComponent instances started:
  manager.start();

  // initialize the timers:
  masterclock.reset();
  
  Timer timer(1000000);
  // ##### MAIN LOOP:
  uint fNum = 0;
  ifs->updateNext(); Image< PixRGB<byte> > ima = ifs->readRGB();
  if(!ima.initialized()) { goforever = false; }
  uint nCMAPSsent = 0;
  //while(goforever)
  while(nCMAPSreceived < TOTAL_FRAMES*NBCMAP4)
    {
      timer.reset();
      // receive conspicuity maps:
      receiveCMAPS(beo, cmap, cmapframe);

      // grab an image:
      //ifs->updateNext(); Image< PixRGB<byte> > ima = ifs->readRGB();
      //if(!ima.initialized()) { goforever = false; }

      // display image in window:
      //Image<float> dispsm(sm); inplaceNormalize(dispsm, 0.0F, 255.0F);
      //Image<byte> dispsmb = dispsm; dispsmb = quickInterpolate(dispsmb, 2);
      //inplacePaste(ima, toRGB(dispsmb), Point2D<int>(w-dispsmb.getWidth(), 0));
      //drawRect(ima, Rectangle(Point2D<int>(w-dispsmb.getWidth()-1, 0),
      //                        dispsmb.getDims()), pix);

      sc->evolve(*seq);
      Point2D<int> eye = sc->getDecision(*seq);
      if (eye.isValid()) fixation = eye;
      uint64 t2 = timer.get();

      //if (fixation.isValid())
      //  {
      //    drawPatch(ima, fixation, 2, pix);
      //    drawCircle(ima, fixation, foa_size, pix, 2);
      //  }
      //writeText(ima, Point2D<int>(0, 0),
      //          sformat(" %.1ffps ", tim.getRecentFps()).c_str(),
      //          PixRGB<byte>(255), grey);
      //writeText(ima, Point2D<int>(0, h-20), info.c_str(),
      //          PixRGB<byte>(255), grey);

      //ofs->writeRGB(ima, "display", FrameInfo("display",SRC_POS));
      //LINFO(" %.1ffps ", tim.getRecentFps());

      // check for space bar pressed; note: will abort us violently if
      // ESC is pressed instead:
      //if (d->checkForKey() == ' ') goforever = false;

      // receive conspicuity maps:
      receiveCMAPS(beo, cmap, cmapframe);

      // send every other frame to processing:
      //if (tim.frameNumber() & 1)
      if (nCMAPSsent < TOTAL_FRAMES)
        {
          nCMAPSsent++;

          const int32 frame = int32(tim.frameNumber());

          // prescale image:
          //Image<PixRGB<byte> > ima2 =
          //  decY(lowPass5y(decX(lowPass5x(ima),1<<PRESCALE)),1<<PRESCALE);
          Image<PixRGB<byte> > ima2 =
            decY(decX(ima,1<<PRESCALE),1<<PRESCALE);

          // we do the job of BEO_RETINA on the master to reduce latency:
          // compute luminance and send it off:
          Image<byte> lum = luminance(ima2);

          // first, send off luminance to orientation slaves:
          smsg.reset(frame, BEO_ORI0); smsg.addImage(lum); beo->send(smsg);
          smsg.setAction(BEO_ORI15); beo->send(smsg);
          smsg.setAction(BEO_ORI30); beo->send(smsg);
          smsg.setAction(BEO_ORI45); beo->send(smsg);
          smsg.setAction(BEO_ORI60); beo->send(smsg);
          smsg.setAction(BEO_ORI75); beo->send(smsg);
          smsg.setAction(BEO_ORI90); beo->send(smsg);
          smsg.setAction(BEO_ORI105); beo->send(smsg);
          smsg.setAction(BEO_ORI120); beo->send(smsg);
          smsg.setAction(BEO_ORI135); beo->send(smsg);
          smsg.setAction(BEO_ORI150); beo->send(smsg);
          smsg.setAction(BEO_ORI165); beo->send(smsg);

          // and also send to flicker slave:
          smsg.setAction(BEO_FLICKER); beo->send(smsg);

          // finally, send to luminance slave:
          smsg.setAction(BEO_LUMINANCE); beo->send(smsg);

          // compute RG and BY and send them off:
          Image<byte> r, g, b, y; getRGBY(ima2, r, g, b, y, (byte)25);
          //Image<byte> r, g, b, y; getRGBY(ima, r, g, b, y, (byte)25);
          smsg.reset(frame, BEO_REDGREEN);
          smsg.addImage(r); smsg.addImage(g); beo->send(smsg);
          smsg.reset(frame, BEO_BLUEYELLOW);
          smsg.addImage(b); smsg.addImage(y); beo->send(smsg);

          uint64 t3 = timer.get();
          LINFO("time3[%d]: %f", fNum, (t3-t2)/1000.0F);
        }

      // receive conspicuity maps:
      receiveCMAPS(beo, cmap, cmapframe);

      // build our current saliency map:
      info = sformat("%06d / ", tim.frameNumber());
      Image<float> sminput;
      for (int i = 0; i < NBCMAP4; i ++)
        if (cmap[i].initialized())
          {
            if (sminput.initialized()) sminput += cmap[i];
            else sminput = cmap[i];
            info += sformat("%06d ", cmapframe[i]);
          }
        else
          info += "------ ";

      // inject saliency map input into saliency map:
      if (sminput.initialized()) sm = sm * 0.7F + sminput * 0.3F;

      // evolve our saccade controller up to now:
      while(seq->now() < masterclock.getSimTime()) seq->evolve();
      sc->evolve(*seq);

      // find most salient location and feed saccade controller:
      float maxval; Point2D<int> currwin; findMax(sm, currwin, maxval);
      WTAwinner newwin =
        WTAwinner::buildFromSMcoords(currwin, sml, true,
                                     masterclock.getSimTime(),
                                     maxval, false);
      if (newwin.isValid()) sc->setPercept(newwin, *seq);

      // receive conspicuity maps:
      receiveCMAPS(beo, cmap, cmapframe);

      // ready for next frame:
      tim.nextFrame();
      while(seq->now() < masterclock.getSimTime()) seq->evolve();

      fNum++;
    }

  // got interrupted; let's cleanup and exit:
  //d->destroyYUVoverlay();
  //double frate = tim.frameNumber() / (tim.getElapsedTime()).sec();
  //LINFO("Fr: %6.3f fps -> %8.3f ms/f\n", frate, 1000.0/frate);
  double time = (tim.getElapsedTime()).sec();
  double frate = TOTAL_FRAMES/time;
  LINFO("[%6d]: %15.6fsec:  %6.3f fps -> %8.3f ms/f\n", 
        fNum, time, frate, 1000/frate);

  Image<double> tvec(times.size(),1, NO_INIT);
  for(uint i = 0; i < times.size(); i++)
    tvec.setVal(i, times[i]);
  LINFO("standard deviation: %f", stdev(tvec));

  LINFO("Normal exit");
  manager.stop();
#endif
  return 0;
}

// ######################################################################
void receiveCMAPS(nub::soft_ref<Beowulf>& beo, Image<float> *cmap,
                  int32 *cmapframe)
{
  TCPmessage rmsg;      // buffer to receive messages from nodes
  int32 rframe, raction, rnode = -1, recnb=0; // receive from any node
  while(beo->receive(rnode, rmsg, rframe, raction)) // no wait
    {
      nCMAPSreceived++; 
      LINFO("--------------> nCMAPSreceived: %6d", nCMAPSreceived);


      if((nCMAPSreceived%16) == 0)
        {
          double time = (tim.getElapsedTime()).sec();
          times.push_back(time-prevTime);
          LINFO("time: %f", time - prevTime);
          prevTime = time;
        }

      //LINFO("received %d/%d from %d while at %d",
      //    rframe, raction, rnode, frame);
      switch(raction & 0xffff)
        {
        case BEO_CMAP: // ##############################
          {
            // get the map:
            Image<float> ima = rmsg.getElementFloatIma();

            // the map number is stored in the high 16 bits of the
            // raction field:
            int32 mapn = raction >> 16;
            if (mapn < 0 || mapn >= NBCMAP4) {
              LERROR("Bogus cmap number ignored");
              break;
            }

            // here is a totally asynchronous system example: we
            // just update our current value of a given cmap if
            // the one we just received is more recent than the
            // one we had so far:
            if (cmapframe[mapn] < rframe)
              { 
                cmap[mapn] = ima; cmapframe[mapn] = rframe; 
              }
          }

          break;
        default: // ##############################
          LERROR("Bogus action %d -- IGNORING.", raction);
          break;
        }
      // limit number of receives, so we don't hold CPU too long:
      recnb ++; if (recnb > NBCMAP4 * 2) break;
    }
}

// ######################################################################
extern "C" int main(int argc, char** argv)
{
  try
    {
      return submain(argc, argv);
    }
  catch (...)
    {
      REPORT_CURRENT_EXCEPTION;
      return 1;
    }
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
