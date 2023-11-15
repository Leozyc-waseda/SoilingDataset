/*!@file AppDevices/app-calibrate-VCC4.C Calibration for the VCC4 pan/tilt camera unit
 */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/app-calibrate-VCC4.C $
// $Id: app-calibrate-VCC4.C 6454 2006-04-11 00:47:40Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/VCC4.H"
#include "GUI/XWinManaged.H"
#include "Image/ColorOps.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/calibrateFunctions.H"
#include "Transport/FrameIstream.H"
#include "Util/log.H"
#include <iostream>
#include <stdio.h>

int main (const int argc, const char **argv)
{
  LOG_FLAGS &= (~LOG_FULLTRACE);

  // instantiate a model manager:
  ModelManager manager("Camera Calibrator");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  nub::soft_ref<VCC4> pantilt(new VCC4(manager));
  manager.addSubComponent(pantilt);

  // choose a V4Lgrabber by default, and a few custom grabbing
  // defaults, for backward compatibility with an older version of
  // this program:
  manager.setOptionValString(&OPT_FrameGrabberType, "V4L");
  manager.setOptionValString(&OPT_FrameGrabberDims, "640x480");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful");

  // let's get all our ModelComponent instances started:
  manager.start();

  int maxtilt = 30;  //, mintilt = -30;
  int maxpan = 100;  //, minpan = -100;
  const Dims dims = gb->peekDims();

  Image< PixRGB<byte> > cimg(dims,NO_INIT);
  Image<byte> bimg1(dims,NO_INIT), bimg2(dims,NO_INIT);

  pantilt->CameraInitialize(true);
  pantilt->gotoPosition(0,0,true);

  cimg = gb->readRGB();
  cimg = gb->readRGB();
  cimg = gb->readRGB();

  bimg1 = luminance(cimg);

  XWinManaged win1(dims,-1,-1,"Center"), win2(dims,-1,-1,"Periphery");
  win1.drawImage(bimg1);

  int dist = 0, angle = 0;
  int lbound = -10, ubound = 10;
  int est_pan = 100;
  int est_tilt = 80;
  float bounds = 0.1;
  int est;

  std::cout << "Tilt:\n";
  while ((dist < 0.9 * dims.h()) && (angle <= maxtilt))
    {
      cimg = gb->readRGB();
      angle++;
      pantilt->gotoPosition(0,angle,true);usleep(50000);

      bimg2 = luminance(cimg);
      win2.drawImage(bimg2);

      dist = findAlign(bimg1,bimg2,ALIGN_Y,lbound,ubound);

      est = dims.h() / est_tilt * angle;
      lbound = (int)(est * (1.0 - bounds));
      //lbound = dist;
      ubound = (int)(est * (1.0 + bounds));

      //lbound = dist;
      //ubound = dist * (angle + 2) / angle;
      if ((ubound - lbound) < 10)
        {
          ubound += 5;
          lbound -= 5;
        }
      if (ubound >= dims.h()) ubound = dims.h() - 1;

      std::cout << angle-1 << "\t" << dist << "\n";
      //std::cout << dist << " ";
    }
  pantilt->gotoPosition(0,0,true);

  std::cout << "\n";

  dist = 0; angle = 0;
  lbound = -10; ubound = 10;

  std::cout << "Pan:\n";
  while ((dist < 0.9 * dims.w()) && (angle <= maxpan))
    {
      cimg = gb->readRGB();
      angle++;
      pantilt->gotoPosition(angle,0,true);usleep(50000);

      bimg2 = luminance(cimg);
      win2.drawImage(bimg2);

      dist = findAlign(bimg1,bimg2,ALIGN_X,lbound,ubound);

      est = dims.w() / est_pan * angle;
      lbound = (int)(est * (1.0 - bounds));
      //lbound = dist;
      ubound = (int)(est * (1.0 + bounds));

      //lbound = dist;
      //ubound = dist * (angle + 2) / angle;
      if ((ubound - lbound) < 10)
        {
          ubound += 5;
          lbound -= 5;
        }
      if (ubound >= dims.w()) ubound = dims.w() - 1;


      std::cout << angle-1 << "\t" << dist << "\n";
      //std::cout << dist << " ";
    }

  std::cout << "\n";

  pantilt->gotoPosition(0,0,false);

  //while (!(win1.pressedCloseButton() || win2.pressedCloseButton())) sleep(1);

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
