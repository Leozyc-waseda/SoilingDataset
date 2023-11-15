/*!@file BeoSub/BeoSubPipe.C find pipe     */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubPipe.C $
// $Id: BeoSubPipe.C 15310 2012-06-01 02:29:24Z itti $

#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "GUI/XWinManaged.H"

#include "Image/Pixels.H"
#include "Raster/Raster.H"

#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"

#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/MathOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Transforms.H"

#include "BeoSub/hysteresis.H"
#include "VFAT/segmentImageTrackMC.H"
#include "BeoSub/HoughTransform.H"
#include "BeoSub/ColorTracker.H"
//#include "BeoSub/getPipeDir.H"
#include "BeoSub/CannyEdge.H"
#include "BeoSub/IsolateColor.H"
#include "rutz/compat_cmath.h" // for isnan()

#include "BeoSub/BeoSubPipe.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream> //needed for segmentImageTrackMC!
#include <math.h>
#include <vector>
#include <cmath>

// ######################################################################
BeoSubPipe::BeoSubPipe()
{
  houghThreshold = 30;
  minThreshold = 10;
  maxThreshold = 50;
  sigma = .7;
  tlow = 0.2;
  thigh = .97;
  linescale =80;

  // number of consecutive times that a line was found (going up to 8)
  foundCount = 0;

  itsWinInitialized = false;
  fNum = 0;
}

// ######################################################################
BeoSubPipe::~BeoSubPipe()
{ }

// ######################################################################
std::vector<LineSegment2D> BeoSubPipe::getHoughLines
(Image<byte> &cameraImage,
 Image< PixRGB <byte> > &outputImage)
{
  std::vector <LineSegment2D> lines;

#ifndef HAVE_OPENCV
  LFATAL("OpenCV must be installed in order to use this function");
#else

  IplImage *edge = cvCreateImage( cvGetSize(img2ipl(cameraImage)), 8, 1 );
  cvCanny( img2ipl(luminance(cameraImage)), edge, 100, 150, 3 );//150,200,3
  //  inplacePaste(dispImage, toRGB(edgeImage), Point2D<int>(0,h));

  Image< PixRGB<byte> >houghImage = ipl2gray(edge);
  //  tim.reset();
  //  rutz::shared_ptr<CvMemStorage*> storage;
  //  *storage = cvCreateMemStorage(0);
  CvMemStorage* storage = cvCreateMemStorage(0);

  outputImage.clear();

  outputImage = toRGB(houghImage);

  //  CvSeq* cvlines = cvHoughLines2(edge, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180
  CvSeq* cvlines = cvHoughLines2(edge, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180, 80 ,30, 10);
  //  t = tim.get();
  //  LDEBUG("houghTransform takes: %f ms", (float)t/1000.0);

  for(int i = 0; i < cvlines->total; i++ )
    {
      CvPoint* line = (CvPoint*)cvGetSeqElem(cvlines,i);
      Point2D<int> pt1 = Point2D<int>(line[0].x,line[0].y);
      Point2D<int> pt2 = Point2D<int>(line[1].x,line[1].y);
      //      cvLine( color_dst, line[0], line[1], CV_RGB(255,0,0), 3, 8 );

      lines.push_back(LineSegment2D(pt1,pt2));
      drawLine(outputImage, pt1, pt2, PixRGB<byte>(255,0,0));
    }

//     for(int i = 0; i < MIN(cvlines->total,10); i++ )
//     {
//       float* line = (float*)cvGetSeqElem(cvlines,i);
//       float rho = line[0];
//       float theta = line[1];
//       CvPoint pt1, pt2;
//       double a = cos(theta), b = sin(theta);
//       double x0 = a*rho, y0 = b*rho;
//       pt1.x = cvRound(x0 + 1000*(-b));
//       pt1.y = cvRound(y0 + 1000*(a));
//       pt2.x = cvRound(x0 - 1000*(-b));
//       pt2.y = cvRound(y0 - 1000*(a));
//       lines.push_back(LineSegment2D(Point2D<int>(pt1.x,pt1.y),Point2D<int>(pt2.x,pt2.y)));
//       drawLine(outputImage, Point2D<int>(pt1.x,pt1.y), Point2D<int>(pt2.x, pt2.y), PixRGB<byte>(255,0,0));
//     }

    //    delete edge;
    //    delete cvlines;

#endif // HAVE_OPENCV

    return lines;
}

// ######################################################################

float BeoSubPipe::pipeOrientation
(Image< PixRGB <byte> > &cameraImage, Image<PixRGB <byte> > &outputImage)
{

  // ------------ STEP 1: SEGMENT ORANGE  --------------------------

  Timer tim(1000000);
  uint w = cameraImage.getWidth();
  uint h = cameraImage.getHeight();
  Image<PixRGB<byte> > dispImage(w*2,h*2,NO_INIT);
  // Image<PixRGB<byte> > houghImage(w,h,NO_INIT);


  if(!itsWinInitialized)
    {
      itsWin.reset
        (new XWinManaged(Dims(w*2,h*2),
                         w+10, 0, "Pipe Direction Output"));
      itsWinInitialized = true;
    }

  Image<byte> orangeIsoImage(w,h, ZEROS);

  inplacePaste(dispImage, cameraImage, Point2D<int>(0,0));

  //Isolate the orange pixels in the image
  tim.reset();
  float orange = isolateOrange(cameraImage, orangeIsoImage); //, fnb=0;
  uint64 t = tim.get();
  LINFO("isoOrange[%f] takes: %f ms", orange, (float)t/1000.0);
  inplacePaste(dispImage, toRGB(orangeIsoImage), Point2D<int>(w,0));

  // ------------ STEP 2: GET EDGE PIXELS  --------------------------


  std::vector <LineSegment2D> lines;
  unsigned char *edge;
  char *dirfilename = NULL;
  canny(orangeIsoImage.getArrayPtr(), h, w,
        sigma, tlow, thigh, &edge, dirfilename);

  Image<byte> edgeImage(edge, w, h);
  inplacePaste(dispImage, toRGB(edgeImage), Point2D<int>(0,h));

  Image< PixRGB<byte> > houghImage = edgeImage;  //houghImage = orangeIsoImage;
  tim.reset();
  lines = houghTransform(edgeImage, M_PI/180, 1, houghThreshold, houghImage);
  t = tim.get();
  LINFO("houghTransform takes: %f ms", (float)t/1000.0);

  // ******* ---- ********

//   std::vector <LineSegment2D> lines;

//   tim.reset();
//   lines = getHoughLines(orangeIsoImage, outputImage);
//   //lines = houghTransform(edgeImage, M_PI/180, 1, houghThreshold, houghImage);
//   t = tim.get();
//   LINFO("houghTransform takes: %f ms", (float)t/1000.0);

  inplacePaste(dispImage, outputImage, Point2D<int>(0,h));

  // ------------ STEP 3: GET PIPE  --------------------------

    uint totlines=0;
  //std::vector<LineSegment2D> centerPipeLines = getPipes(lines);
  float pipeDir = getPipeDir(lines, 20, totlines);

  Point2D<int> p(cameraImage.getWidth()/2, cameraImage.getHeight()/2);
  Point2D<int> p2((int)(cameraImage.getWidth()/2+cos(pipeDir)*linescale),
             (int)(cameraImage.getHeight()/2+sin(pipeDir)*linescale));
  outputImage = houghImage;

//  for(uint i = 0; i < centerPipeLines.size(); i++)
//    {
//       drawLine(outputImage, centerPipeLines[i].point1(), centerPipeLines[i].point2(), PixRGB <byte> (255, 255,0), 3);
  drawLine(outputImage, p, p2, PixRGB <byte> (255, 255,0), 3);
      //itsWin->setTitle(sformat("p_dir: %10.6f", pipeDir).c_str());
      //    }

  // ------------ STEP 4: REPORT (DISPLAY RESULTS)  ---------------

  inplacePaste(dispImage, outputImage, Point2D<int>(w, h));
  drawLine(dispImage, Point2D<int>(0,h),Point2D<int>(w*2-1,h),
           PixRGB<byte>(255,255,255),1);
  drawLine(dispImage, Point2D<int>(w,0),Point2D<int>(w,h*2-1),
           PixRGB<byte>(255,255,255),1);
  writeText( dispImage, Point2D<int>(0,0), sformat("Frame: %6d", fNum).c_str(),
             PixRGB<byte>(255,0,0));
  writeText( dispImage, Point2D<int>(w,0), sformat("Segment Color").c_str(),
             PixRGB<byte>(255,0,0));
  writeText( dispImage, Point2D<int>(0,h), sformat("Detect Edge").c_str(),
             PixRGB<byte>(255,0,0));
  writeText( dispImage, Point2D<int>(w,h), sformat("Identify Pipe").c_str(),
             PixRGB<byte>(255,0,0));

//   std::string saveFName =  sformat("data/pipeDetect_%07d.ppm", fNum);
//   LINFO("saving: %s",saveFName.c_str());
//   Raster::WriteRGB(dispImage,saveFName);
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

//   LINFO("Line Count: %" ZU , lines.size());
//   LINFO("Adj Line Count: %d", totlines);
//   LINFO("Orange: %d", orange);
//   LINFO("Hough Threshold %d", houghThreshold);
//   LINFO("Found Streak: %d", foundCount);

  return 0.0;//pipeDir;
}

// ######################################################################
std::vector<LineSegment2D> BeoSubPipe::getPipes
(const std::vector<LineSegment2D> lines)
{
  uint nLines = lines.size();
  if(nLines == 0) { LDEBUG("*** NO LINES ***"); }

  std::vector< std::vector<LineSegment2D> > pipeLines;

  //Go through all the lines
  for(uint r = 0; r < nLines; r++)
    {
      int lnIndex = -1;

      //check to see if the current lines fits into a bucket
      for(uint c = 0; c < pipeLines.size(); c++)
        {
          if(lines[r].angleBetween(pipeLines[c][0]) < 5*(M_PI/180))//convert to radians
          {
            lnIndex = c;
            break;
          }

        }

      //if the line fits into a pre-existing bucket, add it to the bucket
      if(lnIndex > 0)
        {
          pipeLines[lnIndex].push_back(lines[r]);
          //average the old bucket's value with the new line added
          //so as to create a moving bucket
          Point2D<int> newPt1 = Point2D<int>(((lines[r].point1().i + pipeLines[lnIndex][0].point1().i)/2),
                            ((lines[r].point1().j + pipeLines[lnIndex][0].point1().j)/2));

          Point2D<int> newPt2 = Point2D<int>(((lines[r].point2().i + pipeLines[lnIndex][0].point2().i)/2),
                            ((lines[r].point2().j + pipeLines[lnIndex][0].point2().j)/2));

          pipeLines[lnIndex][0] = LineSegment2D(newPt1,newPt2);

        }
      //otherwise, create a new bucket
      else
        {
          std::vector<LineSegment2D> newCntrLines;
          newCntrLines.push_back(lines[r]);
          pipeLines.push_back(newCntrLines);
        }
    }

  std::vector<LineSegment2D> centerPipeLines;

  Point2D<int> two = Point2D<int>(2,2);

  for(uint c = 0; c < pipeLines.size(); c++)
    {
      if(pipeLines[c].size() == 2)
        {
          Point2D<int> endPoint1 = Point2D<int>((pipeLines[c][0].point1()+pipeLines[c][1].point1())/two);
          Point2D<int> endPoint2 = Point2D<int>((pipeLines[c][0].point2()+pipeLines[c][1].point2())/two);

          centerPipeLines.push_back(LineSegment2D(endPoint1,endPoint2));
        }
    }

  return centerPipeLines;

//   // get the mean
//   float sumLength = 0.0;  float meanLength = 0.0;
//   for (uint i = 0; i < nLines; i++) sumLength += lines[i].length();
//   meanLength = sumLength / nLines;

//   // get the standard deviation
//   float sumsqr = 0.0;
//   for (uint i = 0; i < nLines; i++)
//     sumsqr += pow((float)(lines[i].length()) - meanLength, 2.0);
//   float stdevLength = sqrt(sumsqr / (nLines-1));

//   // kick out lines that aren't within the stddev of the mean length
//   LINFO ("Mean: %f   StdDev: %f", meanLength, stdevLength);
//   std::vector<LineSegment2D> nlines;
//   for(uint i = 0; i < nLines; i++)
//     {
//       if (lines[i].length() > (meanLength - stdevLength) &&
//           lines[i].length() < (meanLength + stdevLength)   )
//         nlines.push_back(lines[i]);
//     }

//   if (nlines.size() == 0)
//     { LINFO("*** NO LINE LENGTH WITHIN STD DEV ***"); return avgPipeAngle; }

//   std::vector<LineSegment2D> flines = nlines;
//   LINFO("\n\n\n\nSum the Average...");
//   double sumAngle = 0;
//   std::vector<LineSegment2D>::iterator itr  = flines.begin();
//   while(itr < flines.end())
//     {
//       float angle = (*itr).angle();
//       LINFO("Line Angle[%4d,%4d][%4d,%4d]: %f",
//             (*itr).point1().i, (*itr).point1().j,
//             (*itr).point2().i, (*itr).point2().j,
//             angle * 180/M_PI);

//       // eliminate lines that are close to vertical
//       if(!isnan(angle))
//         {
//           sumAngle += angle; itr++;
//         }
//       else
//         { sumAngle += 90.0;
//           //itr = flines.erase(itr); LEBUG("Drop a line");
//         }
//     }
//   if (flines.size() == 0) return avgPipeAngle;

//   // also drop off angles that do not fall
//   // within a standard deviation of the average angle
//   float meanAngle = sumAngle / flines.size();
//   float stdAngleSum = 0.0;
//   float stdAngleSqr = 0.0;
//   for(uint i = 0; i < flines.size(); i++)
//     {
//       float angle = flines[i].angle();
//       stdAngleSum = angle - meanAngle;
//       stdAngleSqr += (stdAngleSum * stdAngleSum);
//     }
//   float stdevAngle = stdAngleSqr / flines.size();
//   double stdtemp = sqrt((double)stdevAngle);
//   stdevAngle = (float)stdtemp;

//   // throw out angles that do not fit within stdev of the angle
//   sumAngle = 0.0;  itr  = flines.begin();
//   while(itr < flines.end())
//     {
//       float angle = (*itr).angle();
//       if(angle >= (meanAngle - stdevAngle) &&
//          angle <= (meanAngle + stdevAngle))
//         { sumAngle += angle; itr++; }
//       else itr = flines.erase(itr);
//     }

//   if (flines.size() != 0) sumAngle /= flines.size();  else return avgPipeAngle;
//   if (sumAngle > 0) sumAngle -= M_PI;

//   float avgAngle = sumAngle;

//   // Smooth out the angles being displayed by averaging the position of the last 30 angles
//   uint angleBuffSize = angleBuff.size();

//   bool isWithinStdDev = false;
//   // Check if angle is within std dev
//   if(angleBuff.size() < 2
//      || (fabs(avgAngle - avgPipeAngle) <= stdDevPipeAngle ))
//     {
//       stdDevAngleCount = 0;
//       isWithinStdDev = true;
//     }
//   else
//     {
//       stdDevAngleCount++;
//     }

//   if(stdDevAngleCount > 5)
//     {
//       angleBuff.clear();
//       isWithinStdDev = true;
//     }

//   if(isWithinStdDev)
//     {
//       if(angleBuffSize >= 30)
//         {
//           angleBuff.pop_front();
//         }

//       angleBuff.push_back(avgAngle);

//       angleBuffSize = angleBuff.size();
//     }

//   if(angleBuffSize > 0)
//     {
//       float sumBuffAngle = 0.0;
//       std::list<float>::iterator it;

//       for(it = angleBuff.begin(); it != angleBuff.end(); ++it)
//         {
//           sumBuffAngle += (*it);
//         }

//       avgPipeAngle = sumBuffAngle / angleBuffSize;

//       //      LINFO("AVG Pipe Angle: %f",avgPipeAngle);


//       float sqrStdDevAngle = 0.0;
//       for(it = angleBuff.begin(); it != angleBuff.end(); ++it)
//       {
//         sqrStdDevAngle = ((*it - avgPipeAngle) * (*it - avgPipeAngle));
//       }

//       float stdevAngle = sqrStdDevAngle / angleBuffSize;
//       double stdTempAngle = sqrt((double)stdevAngle);
//       stdDevPipeAngle = (float)stdTempAngle;
//     }
//   else
//     {
//       avgPipeAngle = avgAngle;
//     }

// //   LDEBUG("StdDev: %f", stdevAngle * 180/M_PI);
// //   LDEBUG("Mean: %f", meanAngle * 180/M_PI);
// //   LDEBUG("Pipe Direction: %f", ((float)sumAngle) * 180.0/M_PI);
// //   LDEBUG("Line Count: %" ZU , lines.size());
// //   LDEBUG("Adjusted Line Count: %" ZU , flines.size());
//   return avgPipeAngle;
}

// ######################################################################
float BeoSubPipe::getPipeDir
(const std::vector<LineSegment2D> lines, const float thresh, uint &tot)
{
  uint nLines = lines.size();
  if(nLines == 0) { LINFO("*** NO LINES ***"); return avgPipeAngle; }

  // get the mean
  float sumLength = 0.0;  float meanLength = 0.0;
  for (uint i = 0; i < nLines; i++) sumLength += lines[i].length();
  meanLength = sumLength / nLines;

  // get the standard deviation
  float sumsqr = 0.0;
  for (uint i = 0; i < nLines; i++)
    sumsqr += pow((float)(lines[i].length()) - meanLength, 2.0);
  float stdevLength = sqrt(sumsqr / (nLines-1));

  // kick out lines that aren't within the stddev of the mean length
  LINFO ("Mean: %f   StdDev: %f", meanLength, stdevLength);
  std::vector<LineSegment2D> nlines;
  for(uint i = 0; i < nLines; i++)
    {
      if (lines[i].length() > (meanLength - stdevLength) &&
          lines[i].length() < (meanLength + stdevLength)   )
        nlines.push_back(lines[i]);
    }

  if (nlines.size() == 0)
    { LINFO("*** NO LINE LENGTH WITHIN STD DEV ***"); return avgPipeAngle; }

  std::vector<LineSegment2D> flines = nlines;
  LINFO("\n\n\n\nSum the Average...");
  double sumAngle = 0;
  std::vector<LineSegment2D>::iterator itr  = flines.begin();
  while(itr < flines.end())
    {
      float angle = (*itr).angle();
      LINFO("Line Angle[%4d,%4d][%4d,%4d]: %f",
            (*itr).point1().i, (*itr).point1().j,
            (*itr).point2().i, (*itr).point2().j,
            angle * 180/M_PI);

      // eliminate lines that are close to vertical
      if(!isnan(angle))
        {
          sumAngle += angle; itr++;
        }
      else
        { sumAngle += 90.0;
          //itr = flines.erase(itr); LEBUG("Drop a line");
        }
    }
  if (flines.size() == 0) return avgPipeAngle;

  // also drop off angles that do not fall
  // within a standard deviation of the average angle
  float meanAngle = sumAngle / flines.size();
  float stdAngleSum = 0.0;
  float stdAngleSqr = 0.0;
  for(uint i = 0; i < flines.size(); i++)
    {
     float angle = flines[i].angle();
      stdAngleSum = angle - meanAngle;
      stdAngleSqr += (stdAngleSum * stdAngleSum);
    }
  float stdevAngle = stdAngleSqr / flines.size();
  double stdtemp = sqrt((double)stdevAngle);
  stdevAngle = (float)stdtemp;

  // throw out angles that do not fit within stdev of the angle
  sumAngle = 0.0;  itr  = flines.begin();
  while(itr < flines.end())
    {
      float angle = (*itr).angle();
      if(angle >= (meanAngle - stdevAngle) &&
         angle <= (meanAngle + stdevAngle))
        { sumAngle += angle; itr++; }
      else itr = flines.erase(itr);
    }

  if (flines.size() != 0) sumAngle /= flines.size();  else return avgPipeAngle;
  if (sumAngle > 0) sumAngle -= M_PI;

  float avgAngle = sumAngle;

  // Smooth out the angles being displayed by averaging the position of the last 30 angles
  uint angleBuffSize = angleBuff.size();

  bool isWithinStdDev = false;
  // Check if angle is within std dev
  if(angleBuff.size() < 2
     || (fabs(avgAngle - avgPipeAngle) <= stdDevPipeAngle ))
    {
      stdDevAngleCount = 0;
      isWithinStdDev = true;
    }
  else
    {
      stdDevAngleCount++;
    }

  if(stdDevAngleCount > 5)
    {
      angleBuff.clear();
      isWithinStdDev = true;
    }

  if(isWithinStdDev)
    {
      if(angleBuffSize >= 30)
        {
          angleBuff.pop_front();
        }

      angleBuff.push_back(avgAngle);

      angleBuffSize = angleBuff.size();
    }

  if(angleBuffSize > 0)
    {
      float sumBuffAngle = 0.0;
      std::list<float>::iterator it;

      for(it = angleBuff.begin(); it != angleBuff.end(); ++it)
        {
          sumBuffAngle += (*it);
        }

      avgPipeAngle = sumBuffAngle / angleBuffSize;

      //      LINFO("AVG Pipe Angle: %f",avgPipeAngle);


      float sqrStdDevAngle = 0.0;
      for(it = angleBuff.begin(); it != angleBuff.end(); ++it)
      {
        sqrStdDevAngle = ((*it - avgPipeAngle) * (*it - avgPipeAngle));
      }

      float stdevAngle = sqrStdDevAngle / angleBuffSize;
      double stdTempAngle = sqrt((double)stdevAngle);
      stdDevPipeAngle = (float)stdTempAngle;
    }
  else
    {
      avgPipeAngle = avgAngle;
    }

//   LDEBUG("StdDev: %f", stdevAngle * 180/M_PI);
//   LDEBUG("Mean: %f", meanAngle * 180/M_PI);
//   LDEBUG("Pipe Direction: %f", ((float)sumAngle) * 180.0/M_PI);
//   LDEBUG("Line Count: %" ZU , lines.size());
//   LDEBUG("Adjusted Line Count: %" ZU , flines.size());
  return avgPipeAngle;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
