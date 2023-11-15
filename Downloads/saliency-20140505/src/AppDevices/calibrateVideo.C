/*!@file AppDevices/calibrateVideo.C record from high speed camera XC and
Hokuyo Laser Range finder and save to disk*/

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
// Primary maintainer for this file: farhan
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/calibrateVideo.C $
// $Id: calibrateVideo.C 12962 2010-03-06 02:13:53Z irock $
//
#include "Component/ModelManager.H"
#include "Devices/RangeFinder.H"
#include "Component/OptionManager.H"
#include "Devices/Serial.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/DeviceOpts.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/ShapeOps.H"
#include "Image/CutPaste.H"
#include "Image/ImageCache.H"
#include "Image/Pixels.H"
#include "Image/MathOps.H"
#include "Image/IO.H"
#include "Image/Layout.H"
#include "GUI/SDLdisplay.H"
#include "GUI/GUIOpts.H"
#include "Raster/Raster.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Transport/FrameInfo.H"
#include "Util/Timer.H"
#include "Util/log.H"
#include "Util/sformat.H"
#include "Video/RgbConversion.H" // for toVideoYUV422()
#include "Raster/DeBayer.H" // for debayer()
#include "Raster/PlaintextWriter.H"
#include <algorithm>
#include <iterator>
#include <queue>
#include <fstream>
#include <iostream>


using namespace std;
Timer tim;

static int submain(const int argc, char** argv)
{
  // Image<PixRGB<byte> > dispImg(640,1024,ZEROS);
  Image<byte> ima(640,480,ZEROS);

  uint64 t;

  // instantiate a model manager:
  ModelManager manager("getting calibration image from recording camera");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::soft_ref<FrameGrabberConfigurator>
  gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  manager.setOptionValString(&OPT_FrameGrabberType, "XC");
  manager.setOptionValString(&OPT_FrameGrabberDims, "640x478");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "",0,0) == false)
    return(1);

 // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();

  if (gb.isInvalid())
    LFATAL("You need to have XC camera and XClibrary");


  manager.start();

  //start streaming camera
  gb->startStream();

   while(1)
  {
      //read from grabber
      ima = gb->readGray();
      Image<PixRGB<byte> > imgByte = ima;
      drawCircle(imgByte, Point2D<int>(320,240), 5,PixRGB<byte>(255,0,0));
      ofs->writeRGB(imgByte,"Output",FrameInfo("output",SRC_POS));

      LINFO("grab on");
      t = tim.get();
      if(t > 10000)
        {
          manager.stop();
          exit(1);
        }
  }

        manager.stop();
}



extern "C" int main(const int argc, char** argv)
{
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


