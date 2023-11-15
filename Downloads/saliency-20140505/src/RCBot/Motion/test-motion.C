/*!@file RCBot/Motion/test-motion.C test the motion energy alg */

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
// Primary maintainer for this file: Lior Elazary <lelazary@yahoo.com>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/Motion/test-motion.C $
// $Id: test-motion.C 13756 2010-08-04 21:57:32Z siagian $
//

#include "Component/ModelManager.H"
#include "Component/OptionManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/Layout.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Image/SimpleFont.H"
#include "Media/FrameSeries.H"
#include "RCBot/Motion/MotionEnergy.H"
#include "Raster/Raster.H"
#include "Transport/FrameInfo.H"
#include "Util/Pause.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/csignals.H"
#include "Util/log.H"
#include <math.h>

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  volatile int signum = 0;
  catchsignals(&signum);

  ModelManager manager("Test Motion Energy");

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  if (manager.parseCommandLine((const int)argc, (const char**)argv,
                               "?pyrlevel=0? ?magthresh=1500?", 0, 2) == false)
    return(1);

  const int pyrlevel =
    manager.numExtraArgs() > 0
    ? manager.getExtraArgAs<int>(0)
    : 0;

  const float magthresh =
    manager.numExtraArgs() > 1
    ? manager.getExtraArgAs<float>(1)    
    : 1500.0f;

  manager.start();

  Timer timer(1000000);
  timer.reset();  // reset the timer
  int frame = 0;

  MotionEnergyPyrBuilder<byte> motionPyr(Gaussian5, 0.0f, 10.0f, 5,
                                         magthresh);

  PauseWaiter p;

  while (1)
    {
      if (signum != 0)
        {
          LINFO("quitting because %s was caught", signame(signum));
          break;
        }

      if (ofs->becameVoid())
        {
          LINFO("quitting because output stream was closed or became void");
          break;
        }

      if (p.checkPause())
        continue;

      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE) break; // done receiving frames

      Image< PixRGB<byte> > input = ifs->readRGB();
      // empty image signifies end-of-stream
      if (!input.initialized()) break;
      Image<byte> lum = luminance(input);

      motionPyr.updateMotion(lum, pyrlevel+1);

      Image<float> vMotion = motionPyr.buildVerticalMotionLevel(pyrlevel);
      Image<float> hMotion = motionPyr.buildHorizontalMotionLevel(pyrlevel);

      Image<float> quad = median3(quadEnergy(vMotion, hMotion));
      inplaceNormalize(quad, 0.0F, 255.0F);

      inplaceNormalize(vMotion, 0.0F, 255.0F);
      inplaceNormalize(hMotion, 0.0F, 255.0F);

      Image<byte> vimg = vMotion;
      writeText(vimg, Point2D<int>(1,1), "Vertical Motion", byte(0), byte(255),
                SimpleFont::fixedMaxWidth(8));

      Image<byte> himg = hMotion;
      writeText(himg, Point2D<int>(1,1), "Horizontal Motion", byte(0), byte(255),
                SimpleFont::fixedMaxWidth(8));

      const FrameState os = ofs->updateNext();

      const Layout<byte> disp = vcat(hcat(lum, Image<byte>(quad)), hcat(himg, vimg));

      ofs->writeGrayLayout(disp, "motion-energy",
                           FrameInfo("motion energy output images", SRC_POS));

      if (os == FRAME_FINAL)
        break;

      frame++;
    }

  LINFO("%d frames in %gs (%.2ffps)\n", frame, timer.getSecs(), frame / timer.getSecs());

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
