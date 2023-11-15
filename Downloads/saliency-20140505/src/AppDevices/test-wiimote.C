/*!@file AppDevices/test-wiimote.C Test the wiimote */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-wiimote.C $
// $Id: test-wiimote.C 14376 2011-01-11 02:44:34Z pez $
//

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/DrawOps.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "Raster/GenericFrame.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "GUI/ImageDisplayStream.H"
#include "GUI/XWinManaged.H"


#ifdef HAVE_LIBWIIMOTE
extern "C" {
#define _ENABLE_TILT
#define _ENABLE_FORCE
#undef HAVE_CONFIG_H
#include <libcwiimote/wiimote.h>
#include <libcwiimote/wiimote_api.h>
}
#endif

#define KEY_UP 98
#define KEY_DOWN 104
#define KEY_LEFT 100
#define KEY_RIGHT 102


int getKey(nub::ref<OutputFrameSeries> &ofs)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow("Output")
    : rutz::shared_ptr<XWinManaged>();
  return uiwin->getLastKeyPress();
}


int main(int argc, const char **argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Test wiimote");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  manager.start();

#ifdef HAVE_LIBWIIMOTE
        wiimote_t wiimote = WIIMOTE_INIT;
        //wiimote_report_t report = WIIMOTE_REPORT_INIT;

  LINFO("Press buttons 1 and 2 on the wiimote now to connect.");
  int nmotes = wiimote_discover(&wiimote, 1);
  if (nmotes == 0)
    LFATAL("no wiimotes were found");

  LINFO("found: %s\n", wiimote.link.r_addr);

  if (wiimote_connect(&wiimote, wiimote.link.r_addr) < 0)
    LFATAL("Unable to connect to wiimote");

  /* Activate the first led on the wiimote. It will take effect on the
     next call to wiimote_update. */

  wiimote.led.one  = 1;

  // let's get all our ModelComponent instances started:
  LINFO("Open ");

  LINFO("Status %i", wiimote_is_open(&wiimote));

  int speed = 100;
  int motor1_dir = 0;

  int motor2_dir = 0;

  Point2D<int> loc(128,128);
  while(wiimote_is_open(&wiimote))
  {
    /* The wiimote_update function is used to synchronize the wiimote
       object with the real wiimote. It should be called as often as
       possible in order to minimize latency. */

    if (wiimote_update(&wiimote) < 0) {
      wiimote_disconnect(&wiimote);
      break;
    }

                /* The wiimote object has member 'keys' which keep track of the
                   current key state. */

                if (wiimote.keys.home) { //press home to exit
                        wiimote_disconnect(&wiimote);
                }

                /* Activate the accelerometer when the 'A' key is pressed. */
                //if (wiimote.keys.a) {
                        wiimote.mode.acc = 1;
                //}
                //else {
                //        wiimote.mode.acc = 0;
                //}


    Image<PixRGB<byte> > img(255,255,ZEROS);

    drawLine(img, Point2D<int>(128, 128), Point2D<int>(128+(int)(wiimote.force.x*400), 128), PixRGB<byte>(255,0,0),3);
    drawLine(img, Point2D<int>(128, 128), Point2D<int>(128, 128+(int)(wiimote.force.y*400)), PixRGB<byte>(0,255,0),3);
    drawLine(img, Point2D<int>(128, 128), Point2D<int>(128+(int)(wiimote.force.z*400), 128), PixRGB<byte>(0,0,255),3);

    ofs->writeRGB(img, "Output", FrameInfo("output", SRC_POS));


    int key = getKey(ofs);
    if (key != -1)
    {
      switch(key)
      {
        case 10:  //l
          speed += 10;
          break;
        case 24:
          speed -= 10;
          break;
        case KEY_UP:
          motor1_dir = 2;
          motor2_dir = 2;
          break;
        case KEY_DOWN:
          motor1_dir = 1;
          motor2_dir = 1;
          break;
        case KEY_LEFT:
          motor1_dir = 2;
          motor2_dir = 1;
          break;
        case KEY_RIGHT:
          motor1_dir = 1;
          motor2_dir = 2;
          break;
        case 65: //space
          motor1_dir = 4;
          motor2_dir = 4;
          break;

      }
    }
    //send the data to the wiimote
    wiimote_write_byte(&wiimote, 0x04a40001, motor1_dir);
    wiimote_write_byte(&wiimote, 0x04a40002, speed);
    wiimote_write_byte(&wiimote, 0x04a40003, motor2_dir);
    wiimote_write_byte(&wiimote, 0x04a40004, speed);



                LINFO("KEYS %04x one=%d two=%d a=%d b=%d <=%d >=%d ^=%d v=%d h=%d +=%d -=%d\n",
                        wiimote.keys.bits,
                        wiimote.keys.one,
                        wiimote.keys.two,
                        wiimote.keys.a,
                        wiimote.keys.b,
                        wiimote.keys.left,
                        wiimote.keys.right,
                        wiimote.keys.up,
                        wiimote.keys.down,
                        wiimote.keys.home,
                        wiimote.keys.plus,
                        wiimote.keys.minus);

                LINFO("TILT x=%.3f y=%.3f z=%.3f\n",
                        wiimote.tilt.x,
                        wiimote.tilt.y,
                        wiimote.tilt.z);

                LINFO("FORCE x=%.3f y=%.3f z=%.3f (sum=%.3f)\n",
                        wiimote.force.x,
                        wiimote.force.y,
                        wiimote.force.z,
      sqrt(wiimote.force.x*wiimote.force.x+wiimote.force.y*wiimote.force.y+wiimote.force.z*wiimote.force.z));

  }

  // stop all our ModelComponents
  manager.stop();
#else
  LFATAL("Need the libwiimote");
#endif

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
