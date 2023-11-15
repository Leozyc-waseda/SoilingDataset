/*!@file BeoSub/test-BeoSubBin.C find bin     */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoSubBin.C $
// $Id: test-BeoSubBin.C 12782 2010-02-05 22:14:30Z irock $

#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64
#include "BeoSub/BeoSubBin.H"

#include "Component/ModelManager.H"

#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Media/MediaOpts.H"

#include "Image/CutPaste.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ColorOps.H"
#include "Raster/Raster.H"
#include "Raster/GenericFrame.H"
#include "Devices/DeviceOpts.H"

#include "GUI/XWinManaged.H"

#include "Util/Timer.H"
#include "Image/DrawOps.H"
#include "Image/ShapeOps.H"

int main(int argc, char* argv[])
{

  MYLOGVERB = LOG_INFO;

  ModelManager manager("ComplexObject Tester");

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  manager.exportOptions(MC_RECURSE);

  //manager.setOptionValString(&OPT_InputFrameSource, "V4L2");
  manager.setOptionValString(&OPT_FrameGrabberMode, "YUYV");
  //manager.setOptionValString(&OPT_FrameGrabberDims, "1024x576");
  manager.setOptionValString(&OPT_FrameGrabberByteSwap, "no");
  manager.setOptionValString(&OPT_FrameGrabberFPS, "30");

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

  uint w = 320, h = 240;
  std::string dims = convertToString(Dims(w, h));
  LDEBUG("image size: [%dx%d]", w, h);
  manager.setOptionValString(&OPT_InputFrameDims, dims);

  manager.setModelParamVal("InputFrameDims", Dims(w, h),
                           MC_RECURSE | MC_IGNORE_MISSING);

  manager.start();
  bool goforever = true;

  // input and output image
  Image< PixRGB<byte> > cameraImage;
  Image< PixRGB<byte> > outputImage;

  rutz::shared_ptr<BeoSubBin> binRecognizer(new BeoSubBin());

  // need an initial image if no comparison image is found
   if(!hasCompIma)
     compIma.resize(w,h,NO_INIT);

  int fNum = 0;


  //! window to display results
  rutz::shared_ptr<XWinManaged> itsWin;
  itsWin.reset(new XWinManaged(Dims(w*2,h*2),0, 0, "Bin Detection Output"));

  Point2D<int> centerOfMass;
  std::vector<Point2D<int> > avgCenter;

  while(goforever)
    {
      //      LINFO("CYCLE %d\n", ++cycle);

      const FrameState is = ifs->updateNext();

      if(is == FRAME_COMPLETE)
        break;

      GenericFrame input = ifs->readFrame();
      if(!input.initialized()) { Raster::waitForKey(); break; }

      cameraImage = rescale(input.asRgb(), w, h);

      Timer tim(1000000);
      Image<PixRGB<byte> > dispImage(w*2,h*2,NO_INIT);

      inplacePaste(dispImage, cameraImage, Point2D<int>(0,0));




      Image<PixRGB<byte> > preHough(w, h, NO_INIT);
      Image<PixRGB<byte> > houghImage(w,h,NO_INIT);
      Image<PixRGB<byte> > blobImage(w, h, NO_INIT);
      Image< PixRGB<byte> > intersectImage(w, h, ZEROS);

      std::vector<LineSegment2D> bottomLines = binRecognizer->getHoughLines(cameraImage, intersectImage, houghImage);

      //Point2D<int> centerOfMass;
      float mass = binRecognizer->getBinSceneMass(cameraImage, blobImage, centerOfMass);
      drawCircle(blobImage, centerOfMass, 5, PixRGB<byte>(255, 0, 0));

      //drawCircle(blobImage, centerOfMass, 5, PixRGB<byte>(0, 255, 0));

      std::vector<LineSegment2D> test;

      //binRecognizer->removeOrangePipe(cameraImage);

      //binRecognizer->pruneLines(lines, test, &preHough);

      // NOTE THERE IS AN ISSUE WITH THE WHITENESS OF THE IMAGE ////

      // for the bottom camera
      //std::vector<LineSegment2D> bottomLines;
      binRecognizer->pruneLines(bottomLines, bottomLines, &preHough);
      //binRecognizer->getParallelIntersections(test, frontLines, preHough);
      //binRecognizer->binAngles.clear();
      //binRecognizer->pruneAngles(frontLines, binRecognizer->binAngles, &preHough);

      for(uint i = 0; i < binRecognizer->binAngles.size(); i++) {
        //printf("x: %d, y: %d, angle: %f\n", binAngles[i].pos.i, binAngles[i].pos.j, binAngles[i].angle);
        drawCircle(intersectImage, binRecognizer->binAngles[i].pos, 5, PixRGB<byte>(0, 255, 0));
      }


      // if we have both mass and angles, use weighted mass
      if(binRecognizer->binAngles.size() > 0 && mass > 0) {
        binRecognizer->getWeightedBinMass(binRecognizer->binAngles, centerOfMass, false, &intersectImage);
        drawCircle(intersectImage, centerOfMass, 5, PixRGB<byte>(255, 0, 0));
        avgCenter.push_back(centerOfMass);
      } // if we have mass, but no angles, use mass
      else if(binRecognizer->binAngles.size() <= 0 && mass > 0) {
        drawCircle(intersectImage, centerOfMass, 5, PixRGB<byte>(255, 0, 0));
        avgCenter.push_back(centerOfMass);
      }// if we have angles, but no mass, use center of angles
      else if(binRecognizer->binAngles.size() > 0 && mass <= 0) {
        binRecognizer->getBinCenter(binRecognizer->binAngles, centerOfMass);
        drawCircle(intersectImage, centerOfMass, 5, PixRGB<byte>(255, 0, 0));
      }


      /*// for the top camera
      std::vector<LineSegment2D> frontLines;
      // prune off repeated lines
      binRecognizer->pruneLines(test, frontLines, &preHough);
      // prune off non-horizontal lines
      binRecognizer->getParallelIntersections(frontLines, frontLines, preHough);
      //binRecognizer->binAngles.clear();
      // this is a redundant function
      //binRecognizer->pruneAngles(frontLines, binRecognizer->binAngles, &preHough);

      for(uint i = 0; i < binRecognizer->binAngles.size(); i++) {
        //printf("x: %d, y: %d, angle: %f\n", binAngles[i].pos.i, binAngles[i].pos.j, binAngles[i].angle);
        drawCircle(preHough, binRecognizer->binAngles[i].pos, 5, PixRGB<byte>(0, 255, 0));
        } */
      /*if(frontLines.size() > 0 && mass > 0) {
        binRecognizer->getWeightedBinMass(binRecognizer->binAngles, centerOfMass, false, &preHough);
        drawCircle(blobImage, centerOfMass, 5, PixRGB<byte>(255, 0, 0));
        avgCenter.push_back(centerOfMass);
      }
      else if(frontLines.size() > 0 && binRecognizer->binAngles.size() <= 0) {
        drawCircle(blobImage, centerOfMass, 5, PixRGB<byte>(255, 0, 0));
        avgCenter.push_back(centerOfMass);
        } */

      inplacePaste(dispImage, preHough, Point2D<int>(w,0));
      inplacePaste(dispImage, blobImage, Point2D<int>(0, h));
      inplacePaste(dispImage, intersectImage, Point2D<int>(w,h));


      writeText( dispImage, Point2D<int>(0,0), sformat("Frame: %6d", fNum).c_str(),
                 PixRGB<byte>(255,0,0));
      writeText( dispImage, Point2D<int>(0,h), sformat("Center of Mass: %f", mass).c_str(),
                 PixRGB<byte>(255,0,0));
      writeText( dispImage, Point2D<int>(w,0), sformat("Hough Lines").c_str(),
                 PixRGB<byte>(255,0,0));
      writeText( dispImage, Point2D<int>(w,h), sformat("Center of Bin").c_str(),
                 PixRGB<byte>(255,0,0));

      itsWin->drawImage(dispImage, 0, 0);
      fNum++;


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
