/*!@file AppNeuro/app-show-filters.C Import images from a video device and display
a number of filters */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppNeuro/app-show-filters.C $
// $Id: app-show-filters.C 7293 2006-10-20 18:49:55Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "GUI/XWinManaged.H"
#include "Image/ColorOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
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
  const int numGrabs = 2;
  const int filterSize = 3;
  const int gaussLevel = 3;
  const byte colThresh = 100;
  const byte colMin = 0;
  const byte colMax = 255;
  const byte motThresh = 10;

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
  XWinManaged xCap(dims, -1, -1, "Captured");                clist.add(xCap);
  XWinManaged xHigh(dims, -1, -1, "High Frequencies");       clist.add(xHigh);
  XWinManaged xLow(dims, -1, -1, "Low Frequencies");         clist.add(xLow);
  XWinManaged xMot(dims, -1, -1, "Motion");            clist.add(xMot);
  XWinManaged xRed(dims, -1, -1, "Red");                     clist.add(xRed);
  XWinManaged xGrn(dims, -1, -1, "Green");                   clist.add(xGrn);
  XWinManaged xBlu(dims, -1, -1, "Blue");                    clist.add(xBlu);
  XWinManaged xYel(dims, -1, -1, "Yellow");                  clist.add(xYel);
  XWinManaged xCol(dims, -1, -1, "Color");            clist.add(xCol);

  // intialize all necessary images
  Image<PixRGB <byte> > iCap, cLow, cCol, cRed, cGrn, cBlu, cYel;
  Image<byte> iLum,iHigh,iLow,lowInp,iRed,iGrn,iBlu,iYel,oLow,diff,bin,iMot;
  Image<float> diffX, diffY, dUp, dDown, dLeft, dRight;
  bool firstTime = true;

  // main loop
  while (!clist.pressedAnyCloseButton())
    {
      // capture the image
      for (int i = 0; i < numGrabs; ++i)
        iCap = gb->readRGB();
      xCap.drawImage(iCap);

      // luminance
      iLum = luminance(iCap);

      // high and low frequencies
      cLow = buildPyrGaussian(iCap,0,gaussLevel+1,filterSize)[gaussLevel];
      cLow = rescale(cLow, iCap.getDims());
      iLow = luminance(cLow);
      xLow.drawImage(iLow);
      iHigh = iLum - iLow;
      inplaceNormalize(iHigh, byte(0), byte(255));
      xHigh.drawImage(iHigh);

      // the colors
      getRGBY(cLow, iRed, iGrn, iBlu, iYel, colThresh);
      inplaceNormalize(iRed, byte(0), byte(255));
      inplaceNormalize(iGrn, byte(0), byte(255));
      inplaceNormalize(iBlu, byte(0), byte(255));
      inplaceNormalize(iYel, byte(0), byte(255));
      cRed = colorStain(iRed, colMin, colMax, COL_RED);
      xRed.drawImage(cRed);
      cGrn = colorStain(iGrn, colMin, colMax, COL_GREEN);
      xGrn.drawImage(cGrn);
      cBlu = colorStain(iBlu, colMin, colMax, COL_BLUE);
      xBlu.drawImage(cBlu);
      cYel = colorStain(iYel, colMin, colMax, COL_YELLOW);
      xYel.drawImage(cYel);

      cCol = takeMax(takeMax(cRed,cGrn),takeMax(cBlu,cYel));
      xCol.drawImage(cCol);

      // motion
      if(!firstTime)
        {
          diff = absDiff(iLow, oLow);
          bin = makeBinary(diff, motThresh,0,1);
          iMot = (Image<byte>)(bin * iLum);
          xMot.drawImage(iMot);
        }

      firstTime = false;
      oLow = iLow;

      // end of the main loop
    }
  manager.stop();
}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
