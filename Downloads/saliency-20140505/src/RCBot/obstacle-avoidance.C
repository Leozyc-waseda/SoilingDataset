/*!@file RCBot/obstacle-avoidance.C avoide obstacles by determining
  the total right and left motion, and moving to the lowest motion.
  That is, move, where the space is free                                */
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
// Primary maintainer for this file: Lior Elazary <lelazary@yahoo.com>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/obstacle-avoidance.C $
// $Id: obstacle-avoidance.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "Component/ModelManager.H"
#include "Component/OptionManager.H"
#include "Controllers/PID.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/sc8000.H"
#include "GUI/XWindow.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "RCBot/Motion/MotionEnergy.H"
#include "Raster/Raster.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"
#include <math.h>

#define USE_V4L

XWindow *window;

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("Obstacle Avoidance");

#ifdef USE_V4L
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  nub::soft_ref<SC8000>
    sc8000(new SC8000(manager));
  manager.addSubComponent(sc8000);

  //calibrate the servos
  sc8000->calibrate(1, 13650, 10800, 16000);
  sc8000->calibrate(3, 14000, 12000, 16000);

#else
  // Instantiate our various ModelComponents:
  nub::soft_ref<InputFrameSeries>
    ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);
#endif

  // Parse command-line:
  if (manager.parseCommandLine((const int)argc, (const char**)argv,
                               "<image>", 0, 1) == false)
    return(1);

  SimTime stime = SimTime::ZERO(); Point2D<int> winner(-1, -1);

#ifdef USE_V4L
  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.get() == NULL)
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful -- ABORT");
  // get the frame grabber to start streaming:
  gb->startStream();
#else
  // do post-command-line configs:
  FrameState is = ifs->update(stime);
#endif

  manager.start();

  Timer timer(1000000);  timer.reset(); // reset the timer
  int frame = 0;

  int iw = gb->getWidth(), ih = gb->getHeight();
  // create the window
  window = new XWindow(Dims(iw*2+2, ih), -1, -1, "Output");

  // create the motion pyramid
  MotionEnergyPyrBuilder<byte> motionPyr(Gaussian5);

  int stop = 0;
  while(1)
  {
#ifdef USE_V4L
    Image< PixRGB<byte> > input = gb->readRGB();
#else
    const FrameState is = ifs->update(stime);
    if (is == FRAME_COMPLETE) break; // done receiving frames
    Image< PixRGB<byte> > input = ifs->readRGB();
    // empty image signifies end-of-stream
    if (!input.initialized()) break;
#endif

    Image<byte> lum = luminance(input);
    motionPyr.updateMotion(lum, 1);
    ImageSet<float> vMotionPyr = motionPyr.buildVerticalMotion();
    //ImageSet<float> hMotionPyr = motionPyr.buildHorizontalMotion();
    Image<float> motion = vMotionPyr[0];

    float motionLeft = 0, motionRight = 0;
    Image<float>::iterator motionPtr = motion.beginw();
    Image<float>::const_iterator motionPtrStop = motion.end();

    int inx = 0;
    while (motionPtr != motionPtrStop) {
      int y = inx / motion.getWidth();
      int x = inx - (y*motion.getWidth());

      if (y > 1){
        if (x < (motion.getWidth()/2)){
          motionLeft  += fabs(*motionPtr);
        } else {
          motionRight += fabs(*motionPtr);
        }
      }

      motionPtr++;
      inx++;
    }

    double val = motionRight + motionLeft;

    LINFO("Right %0.4f Left %0.4f Total %0.4f",        motionRight, motionLeft, val);

    if (val > 20) {  // Danger, Will Robinson! Danger!
      if (motionLeft > motionRight) {
        drawLine(lum, Point2D<int>(64,64), Point2D<int>(64+30,64-30), (byte)0, 2);
        sc8000->move(1, -1);
      } else {
        drawLine(lum, Point2D<int>(64,64), Point2D<int>(64-30,64-30), (byte)0, 2);
        sc8000->move(1, 1);
      }
    } else {
      sc8000->move(1, 0);
    }

    if (val > 4000 || stop) {
      LINFO("\n\nSTOP STOP STOP STOP \n");
      sc8000->move(3, 0);
      sleep(2);
    } else {
      sc8000->move(3, -0.270);
    }

    inplaceNormalize(motion, 0.0F, 255.0F);

    window->drawImage(lum);
    window->drawImage((Image<byte>)motion, lum.getWidth()+2, 0);

    frame++;
    stime += SimTime::SECS(0.1);
  }

  LINFO("Time taken %g\n", timer.getSecs());

  // stop all our ModelComponents
  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
