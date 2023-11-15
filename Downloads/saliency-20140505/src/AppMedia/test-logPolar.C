/*!@file AppMedia/test-logPolar.C */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/test-logPolar.C $
// $Id: test-logPolar.C 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef APPMEDIA_TEST_LOGPOLAR_C_DEFINED
#define APPMEDIA_TEST_LOGPOLAR_C_DEFINED

#include "Component/ModelManager.H"
#include "GUI/XWinManaged.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/LogPolarTransform.H"
#include "Image/Pixels.H"
#include "Media/FrameSeries.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Transport/FrameInfo.H"
#include "Util/CpuTimer.H"
#include "Util/Pause.H"
#include "Util/csignals.H"

// int main(int argc, const char** argv)
// {
//   Image<PixRGB<byte> > inp(512, 512, ZEROS);
//   inp.clear(PixRGB<byte>(255, 255, 255));

//   const int w = inp.getWidth();
//   const int h = inp.getHeight();

//   for (int j = 0; j < h; ++j)
//     for (int i = 0; i < w; ++i)
//       {
//         if (j < h/2)
//           {
//             if (i < w/2)
//               inp.setVal(i, j, PixRGB<byte>(255, 192, 192));
//             else
//               inp.setVal(i, j, PixRGB<byte>(192, 255, 192));
//           }
//         else
//           {
//             if (i < w/2)
//               inp.setVal(i, j, PixRGB<byte>(192, 192, 255));
//             else
//               inp.setVal(i, j, PixRGB<byte>(255, 255, 192));
//           }
//       }

// //   drawGrid(inp, 32, 32, 2, 2, PixRGB<byte>(255, 0, 0));

//   for (int ori = 0; ori < 360; ori += 30)
//     {
//       const double rad = ori * M_PI / 180.0;
//       drawLine(inp, Point2D<int>(inp.getWidth() / 2, inp.getHeight() / 2),
//                rad, inp.getWidth() * 2, PixRGB<byte>(255, 0, 0), 2);
//     }

//   for (int rad = 63; rad < inp.getWidth() / 2; rad += 64)
//     {
//       drawCircle(inp, Point2D<int>(inp.getWidth() / 2, inp.getHeight() / 2),
//                  rad, PixRGB<byte>(0, 0,
//                                    double(255.0 * rad / (inp.getWidth() / 2))),
//                  2);
//     }

//   CpuTimer tm;

//   LogPolarTransform t(inp.getDims(), Dims(375, 200), 0.3, 0.25);

//   tm.mark(); tm.report("setup"); tm.reset();

//   Image<PixRGB<byte> > btxf = t.transform(inp, PixRGB<byte>(255));

//   tm.mark(); tm.report("transform"); tm.reset();

//   XWinManaged win(inp);

//   XWinManaged win2(btxf);

//   while (!win.pressedCloseButton())
//     sleep(1);
// }

int submain(const int argc, const char **argv)
{
  volatile int signum = 0;
  catchsignals(&signum);

  ModelManager manager("Streamer");

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false)
    return(1);

  manager.start();

  ifs->startStream();

  const Dims indims = ifs->peekDims();

  LogPolarTransform t(indims, Dims(375, 200), 0.3, 0.25);

  int c = 0;

  PauseWaiter p;

  while (true)
    {
      if (signum != 0)
        {
          LINFO("quitting because %s was caught", signame(signum));
          return -1;
        }

      if (ofs->becameVoid())
        {
          LINFO("quitting because output stream was closed or became void");
          return 0;
        }

      if (p.checkPause())
        continue;

      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE)
        break;

      GenericFrame input = ifs->readFrame();
      if (!input.initialized())
        break;

      const FrameState os = ofs->updateNext();

      const Image<PixRGB<byte> > txf = logPolarTransform(t, input.asRgb(), PixRGB<byte>(255));

      ofs->writeFrame(input, "input",
                      FrameInfo("input frame", SRC_POS));

      ofs->writeRGB(txf, "logpolar",
                    FrameInfo("log-polar transform of input frame", SRC_POS));

      if (os == FRAME_FINAL)
        break;

      LDEBUG("frame %d", c++);

      if (ifs->shouldWait() || ofs->shouldWait())
        Raster::waitForKey();
    }

  return 0;
}

int main(const int argc, const char **argv)
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
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPMEDIA_TEST_LOGPOLAR_C_DEFINED
