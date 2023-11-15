/*!@file AppDevices/test-multigrab.C Test multiple frame grabbing and X display */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-multigrab.C $
// $Id: test-multigrab.C 14321 2010-12-20 23:05:26Z itti $
//

#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/FrameGrabberFactory.H"
#include "GUI/XWindow.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Util/Timer.H"
#include "Util/log.H"
#include "Util/sformat.H"

#include <vector>
#include <cstdio>

#define NAVG 20

/*! This simple executable tests multiple video frame grabbing through
  either the video4linux2 driver (see V4L2grabber.H; when you specify
  --fg-type=V4L2 and then give the pathnames of V4L2 devices as
  arguments), or IEEE1394 (see IEEE1394grabber.H; when you specify
  --fg-type=1394 and pass the camera (subchannel) numbers as
  arguments). */
int main(const int argc, const char **argv)
{
  // instantiate a model manager:
  ModelManager manager("Multi Frame Grabber Tester");

  // Instantiate our various ModelComponents: We instantiate a
  // FrameGrabberConfigurator so that people can select the type and
  // set some options. Then, however, we are going to instantiate a
  // bunch of grabbers, as per the command-line argument. We will then
  // remove our FrameGrabberConfigurator and its baby, as it was just
  // a placeholder to allow command-line configuration to take place:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  // decide on which command-line options our model should export:
  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<dev1> ... <devN>", 1, -1) == false) return(1);

  // do post-command-line configs:
  int ngb = manager.numExtraArgs();

  // what grabber type did we end up with?
  std::string typ = manager.getOptionValString(&OPT_FrameGrabberType);
  bool fire = false; if (typ.compare("1394") == 0) fire = true;

  // instantiate a bunch of grabbers:
  std::vector< nub::soft_ref<FrameIstream> > gb;
  for (int i = 0; i < ngb; i ++)
    {
      const std::string name = sformat("grabber%03d", i);

      if (fire)
        {
          // instantiate a grabber. Note: We cancel the default USE_MY_VALS so that our grabber will inherit whichever
          // settings were specified at the command line:
          gb.push_back(nub::soft_ref<FrameIstream>(makeIEEE1394grabber(manager, name, name, 0)));
          manager.addSubComponent(gb[i]);
          gb[i]->exportOptions(MC_RECURSE);
          gb[i]->setModelParamString("FrameGrabberSubChan", manager.getExtraArg(i));
          gb[i]->setModelParamVal("FrameGrabberNbuf", 10);
        }
      else
        {
          // instantiate a grabber. Note: We cancel the default USE_MY_VALS so that our grabber will inherit whichever
          // settings were specified at the command line:
          gb.push_back(nub::soft_ref<FrameIstream>(makeV4L2grabber(manager, name, name, 0)));
          manager.addSubComponent(gb[i]);
          gb[i]->exportOptions(MC_RECURSE);
          gb[i]->setModelParamVal("FrameGrabberDevice", manager.getExtraArg(i));
        }
    }

  // we don't need the grabber that was in our configurator anymore:
  manager.removeSubComponent(*gbc);
  gbc.reset(NULL); // fully de-allocate the object and its children

  // get a window ready:
  XWindow win(Dims(ngb * gb[0]->getWidth(), gb[0]->getHeight()),
              -1, -1, "test-multigrab window");

  // let's get all our ModelComponent instances started:
  manager.start();

  // get the frame grabbers to start streaming:
  for (int i = 0; i < ngb; i ++) gb[i]->startStream();

  // get ready for main loop:
  Timer tim; uint64 t[NAVG]; int frame = 0;
  while(1) {
    tim.reset();

    int offx = 0;
    for (int i = 0; i < ngb; i ++)
      {
        Image< PixRGB<byte> > ima = gb[i]->readRGB();
        win.drawImage(ima, offx, 0);
        offx += gb[i]->getWidth();
      }

    // total time for multiple grabs and display:
    t[frame % NAVG] = tim.get();

    // compute and show framerate over the last NAVG frames:
    if (frame % NAVG == 0 && frame > 0)
      {
        uint64 avg = 0ULL; for (int i = 0; i < NAVG; i ++) avg += t[i];
        float avg2 = 1000.0F / float(avg) * float(NAVG);
        printf("Framerate for %d grab+display: %.1f fps\n", ngb, avg2);
      }
    frame ++;
  }

  // stop all our ModelComponents
  manager.stop();

  // all done!
  puts("All done!");
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
