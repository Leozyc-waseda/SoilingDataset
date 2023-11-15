/*!@file SeaBee/test-BinRecognizer.C test pipe recognizer   */
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/test-BinRecognizer.C $
// $Id: test-BinRecognizer.C 10794 2009-02-08 06:21:09Z itti $

#include "Component/ModelManager.H"

#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Media/MediaOpts.H"

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Image/CutPaste.H"
#include "Image/OpenCVUtil.H"

#include "BeoSub/IsolateColor.H"
#include "Image/DrawOps.H"
#include "Image/ColorOps.H"

#include "GUI/XWinManaged.H"

#include "SeaBee/BinRecognizer.H"


int main(int argc, char* argv[])
{

  MYLOGVERB = LOG_INFO;

  ModelManager manager("BinRecognizer Tester");

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "[image {*.ppm}]",
                               0, 1)
      == false) return(1);

  int w = ifs->getWidth(),  h = ifs->getHeight();
  std::string dims = convertToString(Dims(w, h));
  LINFO("image size: [%dx%d]", w, h);
  manager.setOptionValString(&OPT_InputFrameDims, dims);

  manager.setModelParamVal("InputFrameDims", Dims(w, h),
                           MC_RECURSE | MC_IGNORE_MISSING);

  manager.start();

  bool goforever = true;

  rutz::shared_ptr<XWinManaged> dispWin;
  dispWin.reset(new XWinManaged(Dims(w*2,h*2), 0, 0, "Bin Recognizer Display"));

  // input and output image
  rutz::shared_ptr<Image< PixRGB<byte> > > img;//
  img.reset(new Image< PixRGB<byte> >(w,h, ZEROS));

  //  rutz::shared_ptr<Image< PixRGB<byte> > > outputImg(new Image<PixRGB<byte> >(w,h, ZEROS));

  rutz::shared_ptr<BinRecognizer> binRecognizer(new BinRecognizer());
  rutz::shared_ptr<Point2D<int> > binCenter(new Point2D<int>(0,0));

  uint fNum = 0;
  while(goforever)
    {
      Image< PixRGB<byte> > dispImg(w*2,h*2, ZEROS);
      rutz::shared_ptr<Image< PixRGB<byte> > > outputImg(new Image<PixRGB<byte> >(w,h, ZEROS));

      ifs->updateNext(); *img = ifs->readRGB();
      if(!img->initialized()) {Raster::waitForKey(); break; }

      inplacePaste(dispImg, *img, Point2D<int>(0,0));

      uint staleCount = 0;

      binRecognizer->getBinLocation(img,
                                    outputImg,
                                    BinRecognizer::CONTOUR,
                                    binCenter,
                                    staleCount);

      drawLine(*outputImg, *binCenter,  *binCenter+2, PixRGB <byte> (255, 255,0), 3);

      inplacePaste(dispImg, *outputImg, Point2D<int>(0,h));

      dispWin->drawImage(dispImg, 0, 0);
      LINFO("%d",fNum); fNum++;

      //wait a little
      //Raster::waitForKey();
    }

  // get ready to terminate:
  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
