/*!@file Parallel/test-beoGrab-master.C Test frame grabbing and X display */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Parallel/test-beoGrab-master.C $
// $Id: test-beoGrab-master.C 6431 2006-04-06 20:38:04Z rjpeters $
//

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Util/Types.H"
#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "GUI/XWinManaged.H"
#include "Util/log.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/*! This simple executable tests video frame grabbing through the
  video4linux driver (see V4Lgrabber.H) or the IEEE1394 (firewire)
  grabber (see IEEE1394grabber.H). Selection of the grabber type is
  made via the --fg-type=XX command-line option. */
int main(const int argc, const char **argv)
{
  // instantiate a model manager:
  ModelManager manager("Beo Frame Grabber Tester Master");

  // Instantiate our various ModelComponents:

  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Master", "BeowulfMaster", true));
  manager.addSubComponent(beo);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  TCPmessage rmsg;      // buffer to receive messages from nodes
  TCPmessage smsg;      // buffer to send messages to nodes

  // let's get all our ModelComponent instances started:
  manager.start();

  // get ready for main loop:
  XWinManaged* winA = 0;
  XWinManaged* winB = 0;
  XWinManaged* winC = 0;
  XWinManaged* winE = 0;
  CloseButtonListener listener;

  smsg.addInt32(1);

  // kick all of the slave nodes into action:
  for (int rnode = 0; rnode < beo->getNbSlaves(); ++rnode)
    beo->send(rnode,smsg);

  int32 rframe = 0, raction = 0;

  while(1)
  {
    int rnode = 0;
    if (rnode < beo->getNbSlaves()
        && beo->receive(rnode, rmsg, rframe, raction, 5))
      {
        Image< PixRGB<byte> > ima = rmsg.getElementColByteIma();

        const std::string info = rmsg.getElementString();
        LINFO("received info: '%s'", info.c_str());

        if (winA == 0)
          {
            winA = new XWinManaged(ima, beo->nodeName(rnode));
            listener.add(winA);
          }
        else
          winA->drawImage(ima);

        beo->send(rnode,smsg);
      }
    rmsg.reset(rframe, raction);

    rnode++;
    if (rnode < beo->getNbSlaves()
        && beo->receive(rnode, rmsg, rframe, raction, 5))
      {
        Image< PixRGB<byte> > ima = rmsg.getElementColByteIma();

        const std::string info = rmsg.getElementString();
        LINFO("received info: '%s'", info.c_str());

        if (winB == 0)
          {
            winB = new XWinManaged(ima, beo->nodeName(rnode));
            listener.add(winB);
          }
        else
          winB->drawImage(ima);

        beo->send(rnode,smsg);
      }
    rmsg.reset(rframe, raction);

    rnode++;
    if (rnode < beo->getNbSlaves()
        && beo->receive(rnode, rmsg, rframe, raction, 5))
      {
        Image< PixRGB<byte> > ima = rmsg.getElementColByteIma();

        const std::string info = rmsg.getElementString();
        LINFO("received info: '%s'", info.c_str());

        if (winC == 0)
          {
            winC = new XWinManaged(ima, beo->nodeName(rnode));
            listener.add(winC);
          }
        else
          winC->drawImage(ima);

        beo->send(rnode,smsg);
      }
    rmsg.reset(rframe, raction);

    rnode++;
    if (rnode < beo->getNbSlaves()
        && beo->receive(rnode, rmsg, rframe, raction, 5))
      {
        Image< PixRGB<byte> > ima = rmsg.getElementColByteIma();

        const std::string info = rmsg.getElementString();
        LINFO("received info: '%s'", info.c_str());

        if (winE == 0)
          {
            winE = new XWinManaged(ima, beo->nodeName(rnode));
            listener.add(winE);
          }
        else
          winE->drawImage(ima);

        beo->send(rnode,smsg);
      }
    rmsg.reset(rframe, raction);

    if (listener.pressedAnyCloseButton())
      break;
  }

  delete winA;
  delete winB;
  delete winC;
  delete winE;

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
