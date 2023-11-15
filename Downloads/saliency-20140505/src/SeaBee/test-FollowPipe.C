/*!@file SeaBee/test-FollowPipe.C test submarine pipe following  */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/test-FollowPipe.C $
// $Id: test-FollowPipe.C 10794 2009-02-08 06:21:09Z itti $

#include "Component/ModelManager.H"

#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Media/MediaOpts.H"

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Image/CutPaste.H"

#include "BeoSub/IsolateColor.H"
#include "Image/DrawOps.H"
#include "Image/ColorOps.H"

#include "GUI/XWinManaged.H"

#include "SeaBee/PipeRecognizer.H"
#include "BeoSub/BeoSubBin.H"
#include "SeaBee/BinRecognizer.H"


#include "SeaBee/SubController.H"

#include "SeaBee/SubGUI.H"

//whether or not to run the simulator
#define SIM_MODE true

#define CREATE_MOVIE true

int main(int argc, char* argv[])
{

  MYLOGVERB = LOG_INFO;

  ModelManager manager("PipeRecognizer Tester");


  nub::soft_ref<InputFrameSeries> ifs;
#if SIM_MODE == false
    ifs.reset(new InputFrameSeries(manager));
    manager.addSubComponent(ifs);
#endif

    nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
    manager.addSubComponent(ofs);


  nub::soft_ref<SubGUI> subGUI(new SubGUI(manager));
  manager.addSubComponent(subGUI);

  nub::soft_ref<SubController> subController(new SubController(manager, "SubController", "SubController", SIM_MODE));
  manager.addSubComponent(subController);

  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "[image {*.ppm}]",
                               0, 1)
      == false) return(1);

  int w, h;

#if SIM_MODE == false
    w = ifs->getWidth();
    h = ifs->getHeight();
#else
    w = 320;
    h = 240;
#endif

  std::string dims = convertToString(Dims(w, h));
  LINFO("image size: [%dx%d]", w, h);
  manager.setOptionValString(&OPT_InputFrameDims, dims);

  manager.setModelParamVal("InputFrameDims", Dims(w, h),
                           MC_RECURSE | MC_IGNORE_MISSING);

  manager.start();

  bool goforever = true;

  rutz::shared_ptr<XWinManaged> dispWin;
  dispWin.reset(new XWinManaged(Dims(w*2,h*2), 0, 0, "Pipe Recognizer Display"));

  rutz::shared_ptr<PipeRecognizer> pipeRecognizer(new PipeRecognizer());
  //rutz::shared_ptr<BeoSubBin> binRecognizer(new BeoSubBin());
  rutz::shared_ptr<Point2D<int> > pipeCenter(new Point2D<int>(0,0));

  rutz::shared_ptr<BinRecognizer> binRecognizer(new BinRecognizer());
  rutz::shared_ptr<Point2D<int> > binCenter(new Point2D<int>(0,0));


  Point2D<int> projPoint(Point2D<int>(0,0));
//   Point2D<int> projPoint2(Point2D<int>(0,0));

  rutz::shared_ptr<double> pipeAngle(new double);
//   rutz::shared_ptr<double> pipeAngleSmooth(new double);


  subGUI->startThread(ofs);
  subGUI->setupGUI(subController.get(), true);
  subGUI->addMeter(subController->getIntPressurePtr(),
                   "Int Pressure", 500, PixRGB<byte>(255, 0, 0));
  subGUI->addMeter(subController->getHeadingPtr(),
                   "Heading", 360, PixRGB<byte>(192, 255, 0));
  subGUI->addMeter(subController->getPitchPtr(),
                   "Pitch", 256, PixRGB<byte>(192, 255, 0));
  subGUI->addMeter(subController->getRollPtr(),
                   "Roll", 256, PixRGB<byte>(192, 255, 0));
  subGUI->addMeter(subController->getDepthPtr(),
                   "Depth", 300, PixRGB<byte>(192, 255, 0));

  subGUI->addMeter(subController->getThruster_Up_Left_Ptr(),
                   "Motor_Up_Left", -100, PixRGB<byte>(0, 255, 0));
  subGUI->addMeter(subController->getThruster_Up_Right_Ptr(),
                   "Motor_Up_Right", -100, PixRGB<byte>(0, 255, 0));
  subGUI->addMeter(subController->getThruster_Up_Back_Ptr(),
                   "Motor_Up_Back", -100, PixRGB<byte>(0, 255, 0));
  subGUI->addMeter(subController->getThruster_Fwd_Left_Ptr(),
                   "Motor_Fwd_Left", -100, PixRGB<byte>(0, 255, 0));
  subGUI->addMeter(subController->getThruster_Fwd_Right_Ptr(),
                   "Motor_Fwd_Right", -100, PixRGB<byte>(0, 255, 0));


  subGUI->addImage(subController->getSubImagePtr());
  subGUI->addImage(subController->getPIDImagePtr());

  // input and output images

  rutz::shared_ptr<Image< PixRGB<byte> > > img;
  img.reset(new Image< PixRGB<byte> >(w,h, ZEROS));

  // Image< PixRGB<byte> > img(w,h, ZEROS);
  rutz::shared_ptr<Image<byte> > orangeIsoImage(new Image<byte>(w,h, ZEROS));
  Image< PixRGB<byte> > dispImg(w*2,h*2, ZEROS);

  rutz::shared_ptr<Image< PixRGB<byte> > > pipelineOutputImg(new Image<PixRGB<byte> >(w,h, ZEROS));
  rutz::shared_ptr<Image< PixRGB<byte> > > outputImg(new Image<PixRGB<byte> >(w,h, ZEROS));



  Point2D<int> centerOfMass;
  std::vector<Point2D<int> > avgCenter;


#if SIM_MODE == true
  uint staleCount = 0;
#endif

  while(goforever)
    {
      // subController->runSimLoop();
      orangeIsoImage.reset(new Image<byte>(w,h, ZEROS));
      outputImg.reset(new Image<PixRGB<byte> >(w,h, ZEROS));

#if SIM_MODE == false
          ifs->updateNext(); *img = ifs->readRGB();
          if(!img->initialized()) {Raster::waitForKey(); break; }
#else
          //get sim image from bottom camera
          *img = subController->getImage(2);
#endif

      inplacePaste(dispImg, *img, Point2D<int>(0,0));

      orangeIsoImage->resize(w,h);
      isolateOrange(*img, *orangeIsoImage);

      inplacePaste(dispImg, toRGB(*orangeIsoImage), Point2D<int>(w,0));

      //Perform PipeLine Recognition
      //get all the orange lines in the image
      std::vector<LineSegment2D> pipelines = pipeRecognizer->getPipeLocation(orangeIsoImage,
                                                                             pipelineOutputImg,
                                                                             PipeRecognizer::HOUGH);


#if SIM_MODE == true

      binRecognizer->getBinLocation(img,
                                    outputImg,
                                    BinRecognizer::CONTOUR,
                                    binCenter,
                                    staleCount);

      drawCircle(*outputImg, *binCenter,  2, PixRGB <byte> (255, 0,0), 3);
#endif
      inplacePaste(dispImg, *outputImg, Point2D<int>(w,h));


      dispWin->drawImage(dispImg, 0, 0);


//       if(binCenter->i > 0 && binCenter->j > 0 && staleCount < 25)// && binCenter->i < 320 && binCenter->j < 240)
//         {
//           int xErr = (binCenter->i-(160));
//           int yErr = (binCenter->j-(120));
//           int desiredHeading = (int)(xErr*-1);

//           LINFO("xErr:%d,yErr:%d",binCenter->i,binCenter->j);

//           if(xErr > .15 * 320)
//             {
//               subController->setHeading(subController->getHeading()
//                                         + desiredHeading);
//             }
//           else if(xErr < .15 * 320)
//             {
//               subController->setHeading(subController->getHeading());
//               subController->setSpeed(50);
//             }
//           else if(yErr < .15 * 240)
//             {
//               subController->setSpeed(0);
//             }
//         }
//       else

      int minY = -1; //minimum midpoint y coordinate found
      int followLineIndex = -1; //index of pipeline with minimum y coordinate

      //iterates through pipelines and finds the topmost one in the image
      for(uint i = 0; i < pipelines.size(); i++)
        {
          LineSegment2D pipeline = pipelines[i];
          Point2D<int> midpoint = (pipeline.point1() + pipeline.point2())/2;

          if(midpoint.j < minY || minY == -1)
            {
              minY = midpoint.j;
              followLineIndex = i;
            }
        }

      //if we found a pipeline
      if(followLineIndex != -1)
        {

          LineSegment2D followLine = pipelines[followLineIndex];
          Point2D<int> midpoint = (followLine.point1() + followLine.point2())/2;

          projPoint.i = (int)(midpoint.i+30*cos(followLine.angle()));
          projPoint.j = (int)(midpoint.j+30*sin(followLine.angle()));

          drawLine(*pipelineOutputImg, midpoint, projPoint, PixRGB <byte> (255, 255,0), 3);


          inplacePaste(dispImg, *pipelineOutputImg, Point2D<int>(0,h));


          int desiredHeading = ((int)((followLine.angle()) * 180 / M_PI));

          if(desiredHeading > 360) desiredHeading = desiredHeading - 360;
          if(desiredHeading < 0) desiredHeading = desiredHeading + 360;

          int absHeading = (subController->getHeading() + desiredHeading) % 360;
          subController->setHeading(absHeading);

          //Wait for new heading to get to heading
          if(!((desiredHeading >= 270 && desiredHeading <= 275) || (desiredHeading <= 90 && desiredHeading >= 85)))
            {
              usleep(100000);
            }
          else
            {
              subController->setHeading(subController->getHeading());
              subController->setSpeed(75);
            }

        }
      else
        {
          subController->setHeading(subController->getHeading());
        }
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
