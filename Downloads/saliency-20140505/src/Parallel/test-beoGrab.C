/*!@file Parallel/test-beoGrab.C Test frame grabbing and X display */

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
// Primary maintainer for this file: T. Nathan Mundhenk <mundhenk@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Parallel/test-beoGrab.C $
// $Id: test-beoGrab.C 6454 2006-04-11 00:47:40Z rjpeters $
//

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Transport/FrameIstream.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Util/sformat.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define NAVG 20

/*! This simple executable tests video frame grabbing through the
  video4linux driver (see V4Lgrabber.H) or the IEEE1394 (firewire)
  grabber (see IEEE1394grabber.H). Selection of the grabber type is
  made via the --fg-type=XX command-line option. */
int main(const int argc, const char **argv)
{
  // instantiate a model manager:
  ModelManager manager("Beo Frame Grabber Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Slave", "BeowulfSlave", false));
  manager.addSubComponent(beo);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful");

  TCPmessage rmsg;            // message being received and to process
  TCPmessage smsg;            // message being sent

  // let's get all our ModelComponent instances started:
  manager.start();

  // get ready for main loop:

  int c = 0;

  while(1) {
    int32 rframe, raction, rnode = -1;  // receive from any node
    while(beo->receive(rnode, rmsg, rframe, raction, 5))
    {
      Image< PixRGB<byte> > ima = gb->readRGB();
      smsg.reset(rframe, 0);
      smsg.addImage(ima);
      smsg.addString(sformat("%dx%d frame %d",
                             ima.getWidth(), ima.getHeight(),
                             c++).c_str());
      //com->send(node,smsg);
      beo->send(rnode, smsg);
      rmsg.reset(rframe, raction);
    }

  }

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
