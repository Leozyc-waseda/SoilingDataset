/*!@file AppMedia/app-imbed-image.C

   imbed a stimulus in an image
*/

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
// Primary maintainer for this file: David J. Berg <dberg@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-imbed-image.C $

#ifndef APPMEDIA_APP_IMBED_IMAGE_C_DEFINED
#define APPMEDIA_APP_IMBED_IMAGE_C_DEFINED

#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Image/Image.H"
#include "Image/CutPaste.H"
#include "Media/FrameSeries.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Transport/FrameInfo.H"
#include "Util/Pause.H"
#include "Util/csignals.H"
#include "Util/Timer.H"

static const ModelOptionCateg MOC_IMBEDOPT = {
  MOC_SORTPRI_2, "imbed image options" };

static const ModelOptionDef OPT_BackgroundPixel =
{ MODOPT_ARG(PixRGB<byte>), "BackgroundPixel", &MOC_IMBEDOPT, OPTEXP_CORE,
  "the color of the background to imbed our input in",
  "background-color", '\0', "< PixRGB<byte> >", "128,128,128"};

static const ModelOptionDef OPT_OutputDims =
{ MODOPT_ARG(Dims), "OutputDims", &MOC_IMBEDOPT, OPTEXP_CORE,
  "the Dimensions of the background image in which the input will be paisted. ",
  "background-dims", '\0', "<Dims>", "1920x1080"};

int submain(int argc, const char** argv)
{
  volatile int signum = 0;
  catchsignals(&signum);
  
  ModelManager manager("imbed image");
  OModelParam<PixRGB<byte> > itsBackgroundPix(&OPT_BackgroundPixel, &manager);
  OModelParam<Dims> itsBackgroundDims(&OPT_OutputDims, &manager);
  
  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);
  
  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);
  
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false)
    return(1);
  
  manager.start();
  
  ifs->startStream();
  
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
    
    const Image<PixRGB<byte> > rgbin = input.asRgb();

    if ((rgbin.getWidth() > itsBackgroundDims.getVal().w()) || (rgbin.getHeight() > itsBackgroundDims.getVal().h()))
      LFATAL("Input image must be smaller than the background dims (change --background-dims or rescale the input)");

    const FrameState os = ofs->updateNext();

    Image< PixRGB<byte> > out(itsBackgroundDims.getVal(), NO_INIT);
    out.clear(itsBackgroundPix.getVal());

    Point2D<int> pos( (out.getWidth() - rgbin.getWidth()) / 2,
                      (out.getHeight() - rgbin.getHeight()) / 2 );

    inplacePaste(out, rgbin, pos);

    LINFO("Imbedding image at: %d %d", pos.i, pos.j);

    ofs->writeFrame(GenericFrame(out), "imbedded output");
    
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
/* indent-tabs-mode: nil */
/* End: */

#endif // APPMEDIA_APP_IMBED_IMAGE_C_DEFINED
