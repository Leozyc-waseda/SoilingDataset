/*!@file BeoSub/test-BeoSubCross.C find cross     */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/test-CrossRecognizer.C $
// $Id: test-CrossRecognizer.C 10794 2009-02-08 06:21:09Z itti $

#include "Component/ModelManager.H"

#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Media/MediaOpts.H"
#include "Raster/GenericFrame.H"

#include "Image/CutPaste.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"

#include "Image/DrawOps.H"
#include "Image/ColorOps.H"
#include "BeoSub/IsolateColor.H"

#include "GUI/XWinManaged.H"

#include "SeaBee/CrossRecognizer.H"


//END CAMERA STUFF
//canny
#define BOOSTBLURFACTOR 90.0
#define FREE_ARG char*
//#define PI 3.14159
#define FILLBLEED 4
#define INITIAL_TEMPERATURE 30.0
#define FINAL_TEMPERATURE 0.5
#define ALPHA 0.98
#define STEPS_PER_CHANGE 1000

#define BIN_ANGLE 0.588001425

int main(int argc, char* argv[])
{
 MYLOGVERB = LOG_INFO;

  ModelManager manager("CrossRecognizer Tester");

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
  dispWin.reset(new XWinManaged(Dims(w*2,h*2), 0, 0, "Cross Recognizer Display"));

  rutz::shared_ptr<CrossRecognizer> crossRecognizer(new CrossRecognizer());

  Image< PixRGB<byte> > img(w,h, ZEROS);

  // count of the number of points that are outside
  // std dev to determine if current avg center point is stale
  rutz::shared_ptr<uint> stalePointCount(new uint());
  *stalePointCount = 0;
  rutz::shared_ptr<float> crossAngle(new float());
  rutz::shared_ptr<Point2D<int> > crossCenterPoint(new Point2D<int>(0,0));
  uint fNum = 0;

  while(goforever)
    {
      Image< PixRGB<byte> > dispImg(w*2,h*2, ZEROS);
      rutz::shared_ptr<Image< PixRGB<byte> > > outputImg(new Image<PixRGB<byte> >(w,h, ZEROS));
      rutz::shared_ptr<Image<byte> > orangeIsoImage(new Image<byte>(w,h, ZEROS));

      ifs->updateNext(); img = ifs->readRGB();
      if(!img.initialized()) {Raster::waitForKey(); break; }

      inplacePaste(dispImg, img, Point2D<int>(0,0));

      //orangeIsoImage->resize(w,h);
      isolateOrange(img, *orangeIsoImage);

      inplacePaste(dispImg, toRGB(*orangeIsoImage), Point2D<int>(w,0));
      //Timer tim(1000000);

      crossRecognizer->getCrossLocation(orangeIsoImage,
                                        outputImg,
                                        CrossRecognizer::HOUGH,
                                        crossCenterPoint,
                                        crossAngle,
                                        stalePointCount);

      inplacePaste(dispImg, *outputImg, Point2D<int>(0,h));

      //wtf does this do?
      //Point2D<int> p(colorSegmentedImage.getWidth()/2, colorSegmentedImage.getHeight()/2);
      //Point2D<int> p2((int)(colorSegmentedImage.getWidth()/2+cos(crossAngle)*LINESCALE),
      //           (int)(colorSegmentedImage.getHeight()/2+sin(crossAngle)*LINESCALE));

      Image<PixRGB<byte> > crossOverlayImage = img;

      PixRGB <byte> crossColor;

      if(*stalePointCount <= 30)
        {
          crossColor =  PixRGB <byte> (0, 255,0);
        }
      else
        {
          crossColor =  PixRGB <byte> (255, 0 ,0);
        }
      float tmpAngle = *crossAngle;
      Point2D<int> tmpCenterPoint = *crossCenterPoint;
      drawCrossOR(crossOverlayImage,
                  tmpCenterPoint,
                  crossColor,
                  20,5, fabs(tmpAngle));


      inplacePaste(dispImg, crossOverlayImage, Point2D<int>(w,h));

      writeText( dispImg, Point2D<int>(0,0), sformat("Frame: %6d", fNum).c_str(),
                 PixRGB<byte>(255,0,0));
      writeText( dispImg, Point2D<int>(w,0), sformat("Edge Detect and Hough").c_str(),
                 PixRGB<byte>(255,0,0));
      writeText( dispImg, Point2D<int>(0,h), sformat("Identify Cross").c_str(),
                 PixRGB<byte>(255,0,0));

      dispWin->drawImage(dispImg, 0, 0);
      fNum++;

      //wait a little
      Raster::waitForKey();
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
