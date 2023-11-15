/*!@file SeaBee/PipeRecognizer.C finds pipelines in an image     */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/PipeRecognizer.C $
// $Id: PipeRecognizer.C 12962 2010-03-06 02:13:53Z irock $

#ifdef HAVE_OPENCV

#include "GUI/XWinManaged.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Image/DrawOps.H"

#include "MBARI/Geometry2D.H"
#include "Image/OpenCVUtil.H"
#include "Image/ColorOps.H"

#include "SeaBee/VisionRecognizer.H"
#include "SeaBee/PipeRecognizer.H"

// ######################################################################
PipeRecognizer::PipeRecognizer()
{
}

// ######################################################################
PipeRecognizer::~PipeRecognizer()
{ }

// ######################################################################
std::vector<LineSegment2D> PipeRecognizer::getPipeLocation
(rutz::shared_ptr<Image<PixRGB <byte> > > colorSegmentedImage,
 rutz::shared_ptr<Image<PixRGB <byte> > > outputImage,
 PipeRecognizeMethod method)
{
  if(!colorSegmentedImage->initialized())
       return std::vector<LineSegment2D>();

  Image<byte> lum = luminance(*colorSegmentedImage);

   switch(method)
     {
     case HOUGH:
       return calculateHoughTransform(lum,
                                      outputImage);
       break;

//      case LINE_BEST_FIT:
//         return calculateLineBestFit (colorSegmentedImage,
//                               outputImage,
//                               pipeCenter,
//                               pipeAngle
//                               );
//        break;

//      case CONTOUR:
//        return calculateContours (colorSegmentedImage,
//                           outputImage,
//                           pipeCenter,
//                           pipeAngle
//                           );
//        break;

     default:
       LERROR("Invalid pipe recognizer method specified");
       return std::vector<LineSegment2D>();
     }
}

// ######################################################################
std::vector<LineSegment2D> PipeRecognizer::calculateHoughTransform
(Image<byte>& colorSegmentedImage,
 rutz::shared_ptr<Image<PixRGB<byte> > > outputImage)
{
#ifndef HAVE_OPENCV
  LFATAL("OpenCV must be installed in order to use this function");
#else
  // Do edge detection (canny) on the image.
  IplImage cannyImage = getCannyImage( colorSegmentedImage );

  // Clear output image and set it equal to canny image.
  //  outputImage->clear();
  //rutz::shared_ptr<Image<PixRGB<byte> > > temp(new Image<PixRGB<byte> > ( toRGB( ipl2gray( &cannyImage ) ) );  // Cannot convert directly to RGB
  //since cannyImage has only 1 channel (black and white).
  //  temp.resize(outputImage->getDims());
  //  *outputImage += temp;

  // Do Hough transform.
  std::vector <LineSegment2D> lineSegments = getHoughLines( cannyImage );

  // Loop through hough lines and draw them to the screen.
  for(uint i = 0; i < lineSegments.size(); i++ )
    {
      Point2D<int> pt1 = lineSegments[i].point1();
      Point2D<int> pt2 = lineSegments[i].point2();
      //draw line segment in output image
      drawLine(*outputImage, pt1, pt2, PixRGB<byte>(255,0,0));
    }

  std::vector <LineSegment2D> prunedHoughLines = pruneHoughLines( lineSegments );

  return prunedHoughLines;

#endif // HAVE_OPENCV
}

// ######################################################################

uint PipeRecognizer::calculateLineBestFit
(Image<byte>  &colorSegmentedImage,
 Image<PixRGB <byte> >  &outputImage,
 Point2D<int> &pipeCenter,
 double &pipeAngle)
{return 0;}

uint PipeRecognizer::calculateContours
(Image<byte>  &colorSegmentedImage,
 Image<PixRGB <byte> >  &outputImage,
 Point2D<int> &pipeCenter,
 double &pipeAngle)
{return 0;}

// double PipeRecognizer::getOrangePixels(Image<byte> &cameraImage,
//                                                   double &avgX,
//                                                   double &avgY,
//                                                   double &sumX,
//                                                   double &sumY)
// {
//   Timer tim(1000000);

//   std::vector <Point2D<int> > edgePoints;
//   uint w = cameraImage.getWidth();
//   uint h = cameraImage.getHeight();

//   Image<byte> (*colorSegmentedImage)(w,h, ZEROS);

//   (*colorSegmentedImage) = cameraImage;

//   avgX = 0.0;
//   avgY = 0.0;
//   sumX = 0.0;
//   sumY = 0.0;

//   //Isolate the orange pixels in the image
//   tim.reset();

//   // isolateOrange(cameraImage, orangeIsoImage); //, fnb=0;


//   //find all the white edge pixels in the image and store them
//   for(int y = 0; y < orangeIsoImage.getHeight(); y++)
//     {
//       for(int x = 0; x < orangeIsoImage.getWidth(); x++)
//         {
//           if(orangeIsoImage.getVal(x,y) == 255)
//             {
//             // convert the x,y position of the pixel to an x,y position where
//             // the center of the image is the origin as opposed to the top left corner
//             // and store the pixel
//               edgePoints.push_back(Point2D<int>(x, y));

//               sumX += x;
//               sumY += y;
//             }
//         }
//     }

//   avgX = sumX/edgePoints.size();
//   avgY = sumY/edgePoints.size();

//   return getSlope(orangeIsoImage, edgePoints, avgX, avgY, sumX, sumY);
// }

// double PipeRecognizer::getSlope(Image<byte> &cameraImage,
//                std::vector <Point2D<int> > &points,
//                double avgX,
//                double avgY,
//                double sumX,
//                doubley sumY)
// {
//   double top = 0.0;
//   double bottom = 0.0;
//   double top2 = 0.0;
//   double bottom2 = 0.0;
//   double return_value = 0.0;
//   double return_value2 = 0.0;

//   int x = 0;
//   int y = 0;

//   /* loop through all the points in the picture and generate a slope
//      by finding the line of best fit*/
//   for(uint i = 0; i < points.size(); i++)
//     {
//       x = points[i].i;
//       y = points[i].j;

//           top += (x - avgX) * (y - avgY);
//           bottom += (x - avgX) * (x - avgX);

//       int tx =  x- cameraImage.getWidth()/2;
//       int ty =  y- cameraImage.getHeight()/2;
//       x = ty +cameraImage.getHeight()/2;
//       y = -tx + cameraImage.getWidth()/2;

//           top2 += (x - avgX) * (y - avgY);
//           bottom2 += (x - avgX) * (x - avgX);

//     }

//   if( bottom != 0.0 )
//       return_value =  atan2(top,bottom);
//   else
//       return_value = 1.62;  //if the bottom is zero, we have a vertical line,
//                            //so we want to return pi/2

//   if( bottom2 != 0.0 )
//       return_value2 =  (atan2(top2,bottom2)+3.14159/2);
//   else
//       return_value2 = (1.62+3.14159/2);


//   double e1 = 0.0;
//   double e2 = 0.0;
//   for(uint i = 0; i < points.size(); i++)
//     {

//       x = points[i].i;
//       y = points[i].j;

//       e1 =pow(x/bottom*top+avgY-y,2);

//       int tx =  x- cameraImage.getWidth()/2;
//       int ty =  y- cameraImage.getHeight()/2;
//       x = ty +cameraImage.getHeight()/2;
//       y = -tx + cameraImage.getWidth()/2;


//       e2 =pow(x/bottom2*top2+avgY-y,2);
//     }


//   if(e1<e2)
//     return return_value;
//   return return_value2;
// }

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
