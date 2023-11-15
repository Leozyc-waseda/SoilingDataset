/*!@file Parallel/beograb-master.C Grab video & save PNM frames on beowulf
  local disks */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Parallel/beograb-master.C $
// $Id: beograb-master.C 12074 2009-11-24 07:51:51Z itti $
//

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "GUI/XWindow.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/Pixels.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/log.H"

#include <iostream>
#include <signal.h>
#include <unistd.h>
#include <cstdio>

static bool goforever = true;
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; exit(1); }

// ######################################################################
int main(const int argc, const char **argv)
{
  // suppress LDEBUG messages:
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("Beowulf-based FrameGrabber");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Master", "BeowulfMaster", true));
  manager.addSubComponent(beo);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<framepause>", 1, 1) == false)
    return(1);

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful");
  int framepause = manager.getExtraArgAs<int>(0);
  if (framepause < 5 || framepause > 2048)
    LFATAL("Invalid framepause %d [range 5..2048]", framepause);

  // let's get all our ModelComponent instances started:
  manager.start();

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // get prepared to grab, communicate, display, etc:
  XWindow xw(gb->peekDims(), -1, -1, "USC iLab Beograb");
  int32 frame = 0;              // count the frames
  Timer tim;                    // for computation of framerate & pause
  TCPmessage smsg;              // buffer to send messages to nodes
  Image< PixRGB<byte> > image;  // image grabbed and displayed
  const int nbnode = beo->getNbSlaves(); // number of slave nodes

  std::cerr << "***** READY. Press [RETURN] to start grabbing *****\n";
  getchar();

  // ########## MAIN LOOP: grab, process, display:
  while(goforever)
    {
      // initialize timer:
      tim.reset();

      // grab image:
      image = gb->readRGB();

      // display total time spent on this frame:
      if (tim.get() > 34)
        LINFO("*** Warning: grabbing %d took %llums", frame, tim.get());

      // display image:
      xw.drawImage(image);

      // send image to slave node, depending on frame number:
      smsg.reset(frame, 1);
      smsg.addImage(image);
      beo->send(frame % nbnode, smsg);

      // display total time spent on this frame:
      if (tim.get() > 34)
        LINFO("*** Warning: frame %d took %llums", frame, tim.get());

      // if necessary, sleep for a while until fixed amount of time elapsed:
      while(int(tim.get()) < framepause)
        {
          struct timespec ts, ts2;
          ts.tv_sec = 0; ts.tv_nsec = 1000000;  // sleep 1 ms
          nanosleep(&ts, &ts2);
        }

      // ready for next frame:
      frame++;
    }

  // stop all our ModelComponents
  manager.stop();

  exit(0);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
