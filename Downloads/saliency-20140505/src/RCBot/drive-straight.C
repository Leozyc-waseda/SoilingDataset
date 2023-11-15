/*!@file RCBot/drive-straight.C atempt to drive the car straight
  by looking at the mean motion*/

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/drive-straight.C $
// $Id: drive-straight.C 9412 2008-03-10 23:10:15Z farhan $
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

#define UP_KEY 98
#define DOWN_KEY 104
#define LEFT_KEY 100
#define RIGHT_KEY 102

XWindow window1(Dims(256, 256), -1, -1, "Test Output 1");

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("Camera capture");

  // Instantiate our various ModelComponents:

  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  nub::soft_ref<SC8000> sc8000(new SC8000(manager));
  manager.addSubComponent(sc8000);

  PID<float> speed_pid(1.6, 0.3, 0.3, -0.6, 0.6);
  PID<float> steer_pid(4, 0, 0, -1, 1);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "P I D", 3, 3) == false)
    return(1);

  // Request a bunch of option aliases (shortcuts to lists of options):
  //REQUEST_OPTIONALIAS_NEURO(manager);
  //set the p rate

  //speed_pid.setPIDPgain(manager.getExtraArgAs<float>(0));
  //speed_pid.setPIDIgain(manager.getExtraArgAs<float>(1));
  //speed_pid.setPIDDgain(manager.getExtraArgAs<float>(2));

  steer_pid.setPIDPgain(manager.getExtraArgAs<float>(0));
  steer_pid.setPIDIgain(manager.getExtraArgAs<float>(1));
  steer_pid.setPIDDgain(manager.getExtraArgAs<float>(2));

  //calibrate the servos
  sc8000->calibrate(1, 13650, 10800, 16000);
  sc8000->calibrate(3, 14000, 12000, 16000);

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.get() == NULL)
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful -- ABORT");
  // int w = gb->getWidth(), h = gb->getHeight();

  //manager.setModelParamVal("InputFrameDims", Dims(w, h),
  //                         MC_RECURSE | MC_IGNORE_MISSING);

  // let's get all our ModelComponent instances started:
  manager.start();

  // get the frame grabber to start streaming:
  gb->startStream();

  // create the motion pyramid
  MotionEnergyPyrBuilder<byte> motionPyr(Gaussian5);

  // ########## MAIN LOOP: grab, process, display:
  int key = 0;

  //sc8000->move(3, -0.3); //move forward slowly

  int time = 0;
  while(key != 24)
  {
    // receive conspicuity maps:
    // grab an image:

    Image< PixRGB<byte> > ima = gb->readRGB();
    Image<byte> lum = luminance(ima);

    // build the motion pyramid
    motionPyr.updateMotion(lum, 1);
    ImageSet<float> hpyr = motionPyr.buildHorizontalMotion();
    ImageSet<float> vpyr = motionPyr.buildVerticalMotion();

    double speed = fabs(mean(vpyr[0]));
    double dir = (mean(hpyr[0]));

    /*
      int x = lum.getWidth()/2;
      int y = lum.getHeight()/2;

      drawLine(lum, Point2D<int>(x,y),
      Point2D<int>( (int)(x+75*cos(dir)), (int)(y-75*sin(dir)) ),
      (byte)0, 3);

      window1.drawImage(rescale(lum, 256, 256));
    */

    double speed_cor = speed_pid.update(0.175, speed);
    double steer_cor = steer_pid.update(0, dir);

    if (speed_cor > 0.6) speed_cor = 0.6;
    if (speed_cor < 0) speed_cor = 0;

    if (steer_cor > 1) steer_cor = 1;
    else if (steer_cor < -1) steer_cor = -1;

    LINFO("%i %f %f %f %f", time, 0.2 - speed , speed_cor, 0-dir, steer_cor);

    sc8000->move(3, speed_cor*-1);
    sc8000->move(1, steer_cor);

    /*
      key = window1.getLastKeyPress();

      if (key != last_key){
      switch (key){
      case UP_KEY: sc8000->move(3, -0.3); break;
      case DOWN_KEY: sc8000->move(3, 0.5); break;
      case LEFT_KEY: sc8000->move(1, 1); break;
      case RIGHT_KEY: sc8000->move(1, -1); break;
      case -1: sc8000->move(3,0); sc8000->move(1,0); break;
      }
      printf("Key press is %i\n", key);
      last_key = key;
      }
    */
    time++;
  }

  // got interrupted; let's cleanup and exit:
  LINFO("Normal exit");
  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
