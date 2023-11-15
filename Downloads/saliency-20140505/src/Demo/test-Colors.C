/*!@file Demo/test-Colors.C Import images from a video device and display
the old and new ways to compute RG and BY color opponencies */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Demo/test-Colors.C $
// $Id: test-Colors.C 7754 2007-01-20 23:50:01Z itti $
//

#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "GUI/XWinManaged.H"
#include "Image/ColorOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/PyramidOps.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"
#include "Image/colorDefs.H"
#include "Transport/FrameIstream.H"
#include "Util/log.H"

#define DOSTAIN


// ######################################################################
// ##### Main Program:
// ######################################################################
/*! This program grabs a frame from a framegrabber and computes and
  displays a number of low-level features. This was done as a demo to
  explain the dissection of an image into features for a public
  outreach project.*/
int main(const int argc, const char **argv)
{
  LOG_FLAGS &= (~LOG_FULLTRACE);
  // a few constants for filters
  const int numGrabs = 3;
  const byte colMin = 0;
  const byte colMax = 255;
  const float RGBYthresh = 25.0F;

  // instantiate a model manager:
  ModelManager manager("Show Filters");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);


  // choose a V4Lgrabber by default, and a few custom grabbing defaults:
  manager.setOptionValString(&OPT_FrameGrabberType, "V4L");
  manager.setOptionValString(&OPT_FrameGrabberDevice,"/dev/v4l/video0");
  manager.setOptionValString(&OPT_FrameGrabberDims, "320x240");
  manager.setOptionValString(&OPT_FrameGrabberChannel, "0");
  manager.setOptionValString(&OPT_FrameGrabberMode,"YUV420P");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful");
  const Dims dims = gb->peekDims();

  // let's get all our ModelComponent instances started:
  manager.start();

  // create all the windows that we will need later on
  // and register them with a CloseButtonListener
  CloseButtonListener clist;
  XWinManaged xCap(dims,-1,-1,"Captured"); clist.add(xCap);

  XWinManaged xRL(dims,-1,-1,"pos(RG) - Standard, clamped");  clist.add(xRL);
  XWinManaged xGL(dims,-1,-1,"neg(RG) - Standard, clamped");  clist.add(xGL);
  XWinManaged xBL(dims,-1,-1,"pos(BY) - Standard, clamped");  clist.add(xBL);
  XWinManaged xYL(dims,-1,-1,"neg(BY) - Standard, clamped");  clist.add(xYL);

  XWinManaged xRD1(dims,-1,-1,"pos(RG) - Simple");  clist.add(xRD1);
  XWinManaged xGD1(dims,-1,-1,"neg(RG) - Simple");  clist.add(xGD1);
  XWinManaged xBD1(dims,-1,-1,"pos(BY) - Simple");  clist.add(xBD1);
  XWinManaged xYD1(dims,-1,-1,"neg(BY) - Simple");  clist.add(xYD1);

#ifdef DOSTAIN
  XWinManaged xCL(dims,-1,-1,"Colors, Standard"); clist.add(xCL);
  XWinManaged xCD1(dims,-1,-1,"Colors, Simple"); clist.add(xCD1);
#endif

  // main loop
  while (!clist.pressedAnyCloseButton())
    {
      // capture the image
      Image<PixRGB <byte> > iCap;
      for (int i = 0; i < numGrabs; ++i)
        iCap = gb->readRGB();
      xCap.drawImage(iCap);

      Image<float> rg,by,rgp,rgn,byp,byn;
      Image<PixRGB <byte> > cR,cG,cY,cB,cCol;
      Image<float> zero(iCap.getDims(),ZEROS);

      // old "Standard" method
      getRGBY(iCap,rg,by,RGBYthresh);

      rgp = rg; rgn = zero - rg;
      byp = by; byn = zero - by;

      inplaceClamp(rgp, 0.0f,1.0f);
      inplaceClamp(rgn, 0.0f,1.0f);
      inplaceClamp(byp, 0.0f,1.0f);
      inplaceClamp(byn, 0.0f,1.0f);

#ifdef DOSTAIN
      cR = colorStain(Image<byte>(rgp*255.0f),colMin,colMax,COL_RED);
      cG = colorStain(Image<byte>(rgn*255.0f),colMin,colMax,COL_GREEN);
      cB = colorStain(Image<byte>(byp*255.0f),colMin,colMax,COL_BLUE);
      cY = colorStain(Image<byte>(byn*255.0f),colMin,colMax,COL_YELLOW);
      cCol = takeMax(takeMax(cR,cG),takeMax(cB,cY));
      xRL.drawImage(cR); xGL.drawImage(cG);
      xBL.drawImage(cB); xYL.drawImage(cY);
      xCL.drawImage(cCol);
#else
      drawImage(xRL, rgp); drawImage(xGL, rgn);
      drawImage(xBL, byp); drawImage(xYL, byn);
#endif


      // new "Simple" method
      getRGBYsimple(iCap,rg,by,RGBYthresh);

      rgp = rg; rgn = zero - rg;
      byp = by; byn = zero - by;

      inplaceClamp(rgp, 0.0f,1.0f);
      inplaceClamp(rgn, 0.0f,1.0f);
      inplaceClamp(byp, 0.0f,1.0f);
      inplaceClamp(byn, 0.0f,1.0f);

#ifdef DOSTAIN
      cR = colorStain(Image<byte>(rgp*255.0f),colMin,colMax,COL_RED);
      cG = colorStain(Image<byte>(rgn*255.0f),colMin,colMax,COL_GREEN);
      cB = colorStain(Image<byte>(byp*255.0f),colMin,colMax,COL_BLUE);
      cY = colorStain(Image<byte>(byn*255.0f),colMin,colMax,COL_YELLOW);
      cCol = takeMax(takeMax(cR,cG),takeMax(cB,cY));
      xRD1.drawImage(cR); xGD1.drawImage(cG);
      xBD1.drawImage(cB); xYD1.drawImage(cY);
      xCD1.drawImage(cCol);
#else
      xRD1.drawImage(rgp); xGD1.drawImage(rgn);
      xBD1.drawImage(byp); xYD1.drawImage(byn);
#endif

      // end of the main loop
    }
  manager.stop();
}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
