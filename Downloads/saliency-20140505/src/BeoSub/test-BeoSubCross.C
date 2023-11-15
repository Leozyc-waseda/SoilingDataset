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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoSubCross.C $
// $Id: test-BeoSubCross.C 15310 2012-06-01 02:29:24Z itti $

#include "Component/ModelManager.H"

#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Media/MediaOpts.H"
#include "Raster/GenericFrame.H"

#include "Image/CutPaste.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"

#include "GUI/XWinManaged.H"

#include "BeoSub/BeoSubCross.H"


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

  ModelManager manager("ComplexObject Tester");

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "[image {*.ppm}]",
                               0, 1)
      == false) return(1);

  // do post-command-line configs:

  std::string infilename;  //Name of the input image
  bool hasCompIma = false;
  Image<PixRGB<byte> > compIma;
  if(manager.numExtraArgs() > 0)
    {
      infilename = manager.getExtraArgAs<std::string>(0);
      compIma = Raster::ReadRGB(infilename);
      hasCompIma = true;
    }

  uint w = ifs->getWidth(),  h = ifs->getHeight();
  //uint w = 320, h = 240;
  std::string dims = convertToString(Dims(w, h));
  LDEBUG("image size: [%dx%d]", w, h);
  //manager.setOptionValString(&OPT_InputFrameDims, dims);
  //manager.setOptionValString(&OPT_InputFrameSource, "V4L2");
  //manager.setOptionValString(&OPT_FrameGrabberMode, "YUYV");
  //manager.setOptionValString(&OPT_FrameGrabberDims, "1024x576");
 // mgr->setOptionValString(&OPT_FrameGrabberBswap, "no");
  //manager.setOptionValString(&OPT_FrameGrabberFPS, "30");
  manager.setModelParamVal("InputFrameDims", Dims(w, h),
                           MC_RECURSE | MC_IGNORE_MISSING);

  manager.start();
  bool goforever = true;

  // input and output image
  Image< PixRGB<byte> > cameraImage;
  Image< PixRGB<byte> > outputImage;

  rutz::shared_ptr<BeoSubCross> crossRecognizer(new BeoSubCross());
  //(infilename, showMatches, objectname);

  // need an initial image if no comparison image is found
   if(!hasCompIma)
     {
//       ifs->updateNext(); ima = ifs->readRGB();
//       if(!ima.initialized()) goforever = false;
//       else compIma = ima;
     compIma.resize(w,h,NO_INIT);
     }
  int fNum = 0;
  int linescale = 80;
  // int sumCrossCenterX = 0, sumCrossCenterY = 0;
  // uint numPoints = 0;

  // count of the number of points that are outside
  // std dev to determine if current avg center point is stale
  uint stalePointCount = 0;

  //! window to display results
  rutz::shared_ptr<XWinManaged> itsWin;
  itsWin.reset(new XWinManaged(Dims(w*2,h*2),w+10, 30, "Cross Detection Output"));


  while(goforever)
    {
      //      LINFO("CYCLE %d\n", ++cycle);
      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE)
        break;

      //grab the images
      GenericFrame input = ifs->readFrame();
      if (!input.initialized())
        break;
      cameraImage = rescale(input.asRgb(), w, h);

      Timer tim(1000000);
      Image<PixRGB<byte> > dispImage(w*2,h*2,NO_INIT);

      inplacePaste(dispImage, cameraImage, Point2D<int>(0,0));

      Image<PixRGB<byte> > houghImage(w,h,NO_INIT);

      std::vector<LineSegment2D> lines = crossRecognizer->getHoughLines(cameraImage, houghImage);

      inplacePaste(dispImage, houghImage, Point2D<int>(w,0));

      std::vector<LineSegment2D> centerPointLines;
      Point2D<int> crossCenterPoint = crossRecognizer->getCrossCenter(lines, centerPointLines, stalePointCount);

      LDEBUG("Center Point X: %d, Y: %d", crossCenterPoint.i, crossCenterPoint.j);


//       bool isWithinStdDev = false;

      float crossAngle = crossRecognizer->getCrossDir(centerPointLines);
      Point2D<int> p(cameraImage.getWidth()/2, cameraImage.getHeight()/2);
      Point2D<int> p2((int)(cameraImage.getWidth()/2+cos(crossAngle)*linescale),
                 (int)(cameraImage.getHeight()/2+sin(crossAngle)*linescale));

      Image<PixRGB<byte> > crossOverlayImage = cameraImage;

      PixRGB <byte> crossColor;

      if(stalePointCount <= 30)
        {
          crossColor =  PixRGB <byte> (0, 255,0);
        }
      else
        {
          crossColor =  PixRGB <byte> (255, 0 ,0);
        }

      drawCrossOR(crossOverlayImage,
                  crossCenterPoint,
                  crossColor,
                  20,5, fabs(crossAngle));


      inplacePaste(dispImage, crossOverlayImage, Point2D<int>(0,h));


//   else foundCount = 0;


      //NEED TO FIX THIS
      //   if(!itsSetupOrangeTracker)
      //     {
      //       setupOrangeTracker(w,h);
      //       itsSetupOrangeTracker = true;
      //     }

      //   int mass = getOrangeMass(cameraImage, display);
      //   Image<byte> temp = quickInterpolate(segmenter->SITreturnCandidateImage(),4);
      // printf("mass:%d", mass);

      //  inplacePaste(dispImage, toRGB(temp), Point2D<int>(0,h*2));

      //   IplImage* dst = cvCreateImage(cvGetSize(img2ipl(cameraImage)), 32, 1 );
      //   IplImage* harris_dst = cvCreateImage(cvGetSize(img2ipl(edgeImage)), 32, 1 );

      //   cvCornerHarris(dst, harris_dst, 3);
      //   Image< PixRGB<byte> > harris = ipl2rgb(dst);

      //   inplacePaste(dispImage, harris, Point2D<int>(0,h*2));

      //   inplacePaste(dispImage, outputImage, Point2D<int>(w, h));
      //   drawLine(dispImage, Point2D<int>(0,h),Point2D<int>(w*2-1,h),
      //            PixRGB<byte>(255,255,255),1);
      //   drawLine(dispImage, Point2D<int>(w,0),Point2D<int>(w,h*2-1),
      //            PixRGB<byte>(255,255,255),1);
      writeText( dispImage, Point2D<int>(0,0), sformat("Frame: %6d", fNum).c_str(),
                 PixRGB<byte>(255,0,0));
      //   writeText( dispImage, Point2D<int>(w,0), sformat("Segment Color").c_str(),
      //              PixRGB<byte>(255,0,0));
      writeText( dispImage, Point2D<int>(w,0), sformat("Edge Detect and Hough").c_str(),
                 PixRGB<byte>(255,0,0));
      writeText( dispImage, Point2D<int>(0,h), sformat("Identify Cross").c_str(),
                 PixRGB<byte>(255,0,0));
      //   writeText( dispImage, Point2D<int>(0,h*2), sformat("Harris Corner").c_str(),
      //              PixRGB<byte>(255,0,0));

      //   std::string saveFName =  sformat("data/cross_%07d.ppm", fNum);
      //   LINFO("saving: %s",saveFName.c_str());
      //   itsWin->drawImage(dispImage, 0, 0);
      itsWin->drawImage(dispImage, 0, 0);
      fNum++;

      // decrease when:
      // it didn't find many lines, but there is some orange in the image

      // increase when:
      // it finds too many lines

      // increase and decrease the hough threshold
      // in an attempt to get some lines, but not too many.
  //   if (orange > 0 &&
  //       totlines < 20 &&
  //       houghThreshold > minThreshold) {
  //     houghThreshold--;  //make it more relaxed
  //     LINFO("Decreasing Hough Threshold");
  //   }
  //   else if (totlines > 20 && houghThreshold < maxThreshold) {
  //     houghThreshold++;  //make it stricter
  //     LINFO("Increasing Hough Threshold");
  //   }

//      LDEBUG("Line Count: %" ZU , lines.size());
//      LDEBUG("Adj Line Count: %d", totlines);
//      LDEBUG("Orange: %d", orange);
//      LDEBUG("Hough Threshold %d", houghThreshold);
//      LDEBUG("Found Streak: %d", foundCount);

     itsWin->drawImage(dispImage,0,0);
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
