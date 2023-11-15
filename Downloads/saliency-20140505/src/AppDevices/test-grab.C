/*!@file AppDevices/test-grab.C Test frame grabbing and X display */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-grab.C $
// $Id: test-grab.C 14290 2010-12-01 21:44:03Z itti $
//

#include "Component/ModelManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "GUI/XWindow.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Raster/GenericFrame.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"

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
  ModelManager manager("Frame Grabber Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

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

  // get ready for main loop:
  Timer tim; uint64 t[NAVG]; int frame = 0;
  GenericFrameSpec fspec = gb->peekFrameSpec();
  Dims windims = fspec.dims;
  if (fspec.nativeType == GenericFrame::RGBD) windims = Dims(windims.w() * 2, windims.h());
  XWindow win(windims, -1, -1, "test-grab window");
  int count = 0;

  // prepare a gamma table for RGBD displays (e.g., Kinect grabber):
  uint16 itsGamma[2048];
  for (int i = 0; i < 2048; ++i) {
    float v = i/2048.0;
    v = powf(v, 3)* 6;
    itsGamma[i] = v*6*256;
  }

  // get the frame grabber to start streaming:
  gb->startStream();

  while(1) {
    ++count; tim.reset();

    GenericFrame fr = gb->readFrame();

    Image< PixRGB<byte> > ima = fr.asRgbU8();

    if (fspec.nativeType == GenericFrame::RGBD) {
      Image<uint16> dimg = fr.asGrayU16(); // get the depth image

      Image<PixRGB<byte> > d(dimg.getDims(), NO_INIT);
      const int sz = dimg.size();
      for (int i = 0; i < sz; ++i) {
        uint v = dimg.getVal(i); if (v > 2047) v = 2047;
        int pval = itsGamma[v];
        int lb = pval & 0xff;
        switch (pval>>8) {
        case 0: d.setVal(i, PixRGB<byte>(255, 255-lb, 255-lb)); break;
        case 1: d.setVal(i, PixRGB<byte>(255, lb, 0)); break;
        case 2: d.setVal(i, PixRGB<byte>(255-lb, 255, 0)); break;
        case 3: d.setVal(i, PixRGB<byte>(0, 255, lb)); break;
        case 4: d.setVal(i, PixRGB<byte>(0, 255-lb, 255)); break;
        case 5: d.setVal(i, PixRGB<byte>(0, 0, 255-lb)); break;
        default: d.setVal(i, PixRGB<byte>(0, 0, 0)); break;
        }
      }
      ima = concatX(ima, d);
    }

    uint64 t0 = tim.get();  // to measure display time

    win.drawImage(ima);

    t[frame % NAVG] = tim.get();
    t0 = t[frame % NAVG] - t0;
    if (t0 > 20000ULL) LINFO("Display took %lluus", t0);


    // compute and show framerate over the last NAVG frames:
    if (frame % NAVG == 0 && frame > 0)
      {
        uint64 avg = 0ULL; for (int i = 0; i < NAVG; i ++) avg += t[i];
        float avg2 = 1000.0F / float(avg) * float(NAVG);
        printf("Framerate: %.1f fps\n", avg2);
      }
    frame ++;
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
