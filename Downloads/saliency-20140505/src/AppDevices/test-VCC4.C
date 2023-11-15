/*!@file AppDevices/test-VCC4.C Use a pan-tilt camera for executing saccades */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-VCC4.C $
// $Id: test-VCC4.C 4751 2005-07-01 16:50:30Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Devices/VCC4.H"
#include <cmath>
#include <iostream>


#define CAM_WIDTH 320
#define CAM_HEIGHT 240
#define CAM_DISPLAY NULL
#define CAM_WINDOW_NAME "Saliency Calculated"
#define CAM_PREV_NAME "Captured Image"
#define CAM_VIDEO_DEVICE "/dev/video0"
#define CAM_VIDEO_CHANNEL 2
#define CAM_SERIAL_DEVICE "/dev/ttyS0"
#define VCC4_HDEG 50
#define VCC4_WDEG 67
#define MAXPAN ((float)100.0)
#define MINPAN ((float)-100.0)
#define MAXTILT ((float)30.0)
#define MINTILT ((float)-30.0)

// make random saccade if the salmap evolves for more than TOO_MUCH_TIME s
#define TOO_MUCH_TIME 0.7


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
  MYLOGVERB = LOG_INFO;

  /*
  int min_dist = min(VCC4_HDEG, VCC4_WDEG) / 12;
  const int num_mem = 10;
  float mem_pan[num_mem], mem_tilt[num_mem];
  int mem_ptr = 0;
  bool mem_filled = false;
  bool is_in_mem;
  int mem_top;
  */

  std::cout << "Mark 1 \n";

  // instantiate a model manager:
  ModelManager manager("Test VCC4");

  // Instantiate our various ModelComponents:
  nub::soft_ref<VCC4> pantilt(new VCC4(manager));
  manager.addSubComponent(pantilt);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  std::cout << "Mark 2 \n";

  pantilt->CameraInitialize(true);
  std::cout << "Mark 3 \n";

  pantilt->PlainCommand(VCC4_SetZoomingWIDE);
  std::cout << "Mark 4 \n";

  pantilt->PlainCommand(VCC4_GoHome);


  manager.stop();
  return 0;
}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
