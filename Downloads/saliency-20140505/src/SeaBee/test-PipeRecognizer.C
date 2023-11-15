/*!@file SeaBee/test-PipeRecognizer.C test pipe recognizer   */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/test-PipeRecognizer.C $
// $Id: test-PipeRecognizer.C 12962 2010-03-06 02:13:53Z irock $

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

#include "SeaBee/PipeRecognizer.H"

#include "BeoSub/ColorSegmenter.H"

int main(int argc, char* argv[])
{

  MYLOGVERB = LOG_INFO;

  ModelManager manager("PipeRecognizer Tester");

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
  dispWin.reset(new XWinManaged(Dims(w*2,h*2), 0, 0, "Pipe Recognizer Display"));

  // input and output image
  Image< PixRGB<byte> > img(w,h, ZEROS);

  rutz::shared_ptr<PipeRecognizer> pipeRecognizer(new PipeRecognizer());

  uint fNum = 0;

  while(goforever)
    {
      Image< PixRGB<byte> > dispImg(w*2,h*2, ZEROS);
      rutz::shared_ptr<Image< PixRGB<byte> > >
        outputImg(new Image<PixRGB<byte> >(w,h, ZEROS));

      rutz::shared_ptr<Image<byte> > orangeIsoImage;
      orangeIsoImage.reset(new Image<byte>(w,h, ZEROS));

      //read an input image
      ifs->updateNext(); img = ifs->readRGB();
      if(!img.initialized()) {Raster::waitForKey(); break; }

      inplacePaste(dispImg, img, Point2D<int>(0,0));

      orangeIsoImage->resize(w,h);
      //get all of the orange pixels in the image
      isolateOrange(img, *orangeIsoImage);

      inplacePaste(dispImg, toRGB(*orangeIsoImage), Point2D<int>(w,0));

      //get all the orange lines in the image
      std::vector<LineSegment2D> pipelines =
        pipeRecognizer->getPipeLocation
        (orangeIsoImage, outputImg, PipeRecognizer::HOUGH);


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

          Point2D<int> projPoint;
          projPoint.i = (int)(midpoint.i+30*cos(followLine.angle()));
          projPoint.j = (int)(midpoint.j+30*sin(followLine.angle()));

          drawLine(*outputImg, midpoint, projPoint, PixRGB <byte> (255, 255,0), 3);

          inplacePaste(dispImg, *outputImg, Point2D<int>(0,h));

          dispWin->drawImage(dispImg, 0, 0);
          LINFO("%d",fNum); fNum++;
        }

      //wait for a key
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
