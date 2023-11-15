/*!@file BeoSub/BeoSubCross.C find pipe                                 */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubCross.C $
// $Id: BeoSubCross.C 14376 2011-01-11 02:44:34Z pez $

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
#include "BeoSub/CannyEdge.H"
#include "BeoSub/IsolateColor.H"
#include "rutz/compat_cmath.h" // for isnan()

#include "BeoSub/BeoSubCross.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream> //needed for segmentImageTrackMC!
#include <math.h>
#include <vector>
#include <cmath>

// ######################################################################
BeoSubCross::BeoSubCross()
{
  houghThreshold = 40;
  minThreshold = 10;
  maxThreshold = 50;
  sigma = 1.1;
  tlow = 0.3;
  thigh = 1.1;
  linescale = 80;
  avgCrossCenterX = 0;
  avgCrossCenterY = 0;
  avgCrossAngle = 0.0;
  stdDevCrossX = 0.0;
  stdDevCrossY = 0.0;
  stdDevCrossAngle = 0.0;

  stdDevAngleCount = 0;
  stdDevCenterCount = 0;

  color.resize(4,0.0F);
  std.resize(4,0.0F);
  norm.resize(4,0.0F);
  adapt.resize(4,0.0F);
  lowerBound.resize(4,0.0F);
  upperBound.resize(4,0.0F);

  // number of consecutive times that a line was found (going up to 8)
  foundCount = 0;

  itsSetupOrangeTracker = false;

  //  setupOrangeTracker();
}

// ######################################################################
BeoSubCross::~BeoSubCross()
{ }

// ######################################################################
void BeoSubCross::setupOrangeTracker()
{
  //  int width = 180;
  //int height = 120;
  int width = 360;
  int height = 240;

  segmenter = new segmentImageTrackMC<float,unsigned int, 4> (width*height);

  int wi = width/4;
  int hi = height/4;

  segmenter->SITsetFrame(&wi,&hi);

  segmenter->SITsetCircleColor(0,255,0);
  segmenter->SITsetBoxColor(255,255,0,0,255,255);
  segmenter->SITsetUseSmoothing(false,10);


  segmenter->SITtoggleCandidateBandPass(false);
  segmenter->SITtoggleColorAdaptation(true);

  //  std::cout<<"BECORE OPEN FILE"<<std::endl;
  colorConf.openFile("colortrack.conf", true);
  //  std::cout<<"AFTER OPEN FILE"<<std::endl;

  PixRGB<float> P1(colorConf.getItemValueF("ORNG_R"),
                   colorConf.getItemValueF("ORNG_G"),
                   colorConf.getItemValueF("ORNG_B"));

  PixH2SV1<float> ORNG(P1);
  //  std::cout<<"BEFORE H1 THING"<<std::endl;
  color[0] = ORNG.H1(); color[1] = ORNG.H2(); color[2] = ORNG.S(); color[3] = ORNG.V();

  printf("h1: %f, h2: %f, sat: %f, value: %f", color[0], color[1], color[2], color[3]);

  //! +/- tollerance value on mean for track
  std[0] = colorConf.getItemValueF("ORNG_std0");
  std[1] = colorConf.getItemValueF("ORNG_std1");
  std[2] = colorConf.getItemValueF("ORNG_std2");
  std[3] = colorConf.getItemValueF("ORNG_std3");

  //! normalizer over color values (highest value possible)
  norm[0] = colorConf.getItemValueF("ORNG_norm0");
  norm[1] = colorConf.getItemValueF("ORNG_norm1");
  norm[2] = colorConf.getItemValueF("ORNG_norm2");
  norm[3] = colorConf.getItemValueF("ORNG_norm3");

  //! how many standard deviations out to adapt, higher means less bias
  adapt[0] = colorConf.getItemValueF("ORNG_adapt0");
  adapt[1] = colorConf.getItemValueF("ORNG_adapt1");
  adapt[2] = colorConf.getItemValueF("ORNG_adapt2");
  adapt[3] = colorConf.getItemValueF("ORNG_adapt3");

  //! highest value for color adaptation possible (hard boundry)
  upperBound[0] = color[0] + colorConf.getItemValueF("ORNG_up0");
  upperBound[1] = color[1] + colorConf.getItemValueF("ORNG_up1");
  upperBound[2] = color[2] + colorConf.getItemValueF("ORNG_up2");
  upperBound[3] = color[3] + colorConf.getItemValueF("ORNG_up3");

  //! lowest value for color adaptation possible (hard boundry)
  lowerBound[0] = color[0] - colorConf.getItemValueF("ORNG_lb0");
  lowerBound[1] = color[1] - colorConf.getItemValueF("ORNG_lb1");
  lowerBound[2] = color[2] - colorConf.getItemValueF("ORNG_lb2");
  lowerBound[3] = color[3] - colorConf.getItemValueF("ORNG_lb3");

  //  std::cout<<"AFTER ARRAY STUFF"<<std::endl;
  segmenter->SITsetTrackColor(&color,&std,&norm,&adapt,&upperBound,&lowerBound);
  //  std::cout<<"AFTER SITsetTrackColor"<<std::endl;
}
// ######################################################################
int BeoSubCross::getOrangeMass(Image< PixRGB<byte> > image,
                               Image< PixRGB<byte> > &display)
{
  Image<PixRGB<byte> > Aux;
  Aux.resize(100,450,true);

  Image< PixH2SV2<float> > H2SVimage(image);

  segmenter->SITtrackImageAny(H2SVimage,&display,&Aux,true);

  int mass;
  segmenter->SITgetBlobWeight(mass);
  return mass;
}

// ######################################################################


std::vector<LineSegment2D> BeoSubCross::pruneLines (std::vector<LineSegment2D> lines)
{
  uint nLines = lines.size();
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
  LDEBUG ("Mean: %f   StdDev: %f", meanLength, stdevLength);
  std::vector<LineSegment2D> nlines;
  for(uint i = 0; i < nLines; i++)
    {
      if (lines[i].length() > (meanLength - stdevLength) &&
          lines[i].length() < (meanLength + stdevLength)   )
        nlines.push_back(lines[i]);
    }

  if (nlines.size() == 0)
    { LDEBUG("*** NO LINE LENGTH WITHIN STD DEV ***"); return lines; }

  std::vector<LineSegment2D> flines = nlines;
  LDEBUG("\n\n\n\nSum the Average...");
  double sumAngle = 0;
  std::vector<LineSegment2D>::iterator itr  = flines.begin();
  while(itr < flines.end())
    {
      float angle = (*itr).angle();
      LDEBUG("Line Angle[%4d,%4d][%4d,%4d]: %f",
            (*itr).point1().i, (*itr).point1().j,
            (*itr).point2().i, (*itr).point2().j,
            angle * 180/M_PI);

      // eliminate lines that are close to vertical
      if(!isnan(angle)) { sumAngle += angle; itr++; }
      else
        {
          sumAngle += 90.0;
          //          itr = flines.erase(itr);
          LDEBUG("Drop a line");
        }
    }
  if (flines.size() == 0) return lines;

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

  return flines;
}

// ######################################################################

std::vector<LineSegment2D> BeoSubCross::getHoughLines
(Image< PixRGB <byte> > &cameraImage,
 Image< PixRGB <byte> > &outputImage)
{
  std::vector <LineSegment2D> lines;

#ifndef HAVE_OPENCV
  LFATAL("OpenCV must be installed in order to use this function");
#else

  IplImage *edge = cvCreateImage( cvGetSize(img2ipl(cameraImage)), 8, 1 );
  cvCanny( img2ipl(luminance(cameraImage)), edge, 100, 150, 3 );//150,200,3

  //  inplacePaste(dispImage, toRGB(edgeImage), Point2D<int>(0,h));

  //  tim.reset();
//   rutz::shared_ptr<CvMemStorage*> storage;
//   *storage = cvCreateMemStorage(0);

  CvMemStorage* storage = cvCreateMemStorage(0);

  //outputImage.clear();

  outputImage = ipl2gray(edge);

  CvSeq* cvlines = cvHoughLines2(edge, storage, CV_HOUGH_STANDARD,
                                 1, CV_PI/180, 40 , 0, 0);
  //  t = tim.get();
  //  LDEBUG("houghTransform takes: %f ms", (float)t/1000.0);

    for(int i = 0; i < MIN(cvlines->total,10); i++ )
    {
      float* line = (float*)cvGetSeqElem(cvlines,i);
      float rho = line[0];
      float theta = line[1];
      CvPoint pt1, pt2;
      double a = cos(theta), b = sin(theta);
      double x0 = a*rho, y0 = b*rho;
      pt1.x = cvRound(x0 + 1000*(-b));
      pt1.y = cvRound(y0 + 1000*(a));
      pt2.x = cvRound(x0 - 1000*(-b));
      pt2.y = cvRound(y0 - 1000*(a));
      lines.push_back(LineSegment2D(Point2D<int>(pt1.x,pt1.y),
                                    Point2D<int>(pt2.x,pt2.y)));
      drawLine(outputImage, Point2D<int>(pt1.x,pt1.y),
               Point2D<int>(pt2.x, pt2.y), PixRGB<byte>(255,0,0));
    }

     cvReleaseMemStorage(&storage);
     cvReleaseImage(&edge);
     //cvReleaseSeq(&cvlines);

    //    delete edge;
    //    delete cvlines;

#endif // HAVE_OPENCV

    return lines;
}

// ######################################################################

Point2D<int> BeoSubCross::getCrossCenter (const std::vector<LineSegment2D> lines,
                                     std::vector<LineSegment2D> &centerPointLines,
                                     uint& stalePointCount)
{

  Point2D<int> avgCrossCenterPoint = Point2D<int>(avgCrossCenterX,avgCrossCenterY);

  uint nLines = lines.size();
  if(nLines == 0)
    {
      LDEBUG("*** NO LINES ***");
      stalePointCount++;
      return avgCrossCenterPoint;
    }

  std::vector<LineSegment2D> posAngleLines, negAngleLines;

  for(uint i = 0; i < nLines; i++)
    {
      if(lines[i].angle() >= 0)
        {
          posAngleLines.push_back(lines[i]);
        }
      else
        {
          negAngleLines.push_back(lines[i]);
        }
    }

  std::vector<LineSegment2D> fPosAngleLines = posAngleLines;//pruneLines(posAngleLines);
  std::vector<LineSegment2D> fNegAngleLines = negAngleLines;//pruneLines(negAngleLines);

  std::vector<LineSegment2D> flines;

  for(uint i = 0; i < fPosAngleLines.size(); i++)
  {
    flines.push_back(fPosAngleLines[i]);
  }

  for(uint i = 0; i < fNegAngleLines.size(); i++)
  {
    flines.push_back(fNegAngleLines[i]);
  }


  std::vector<std::vector<Point2D<int> > > intersectPts;

  //Go through all the remaining lines
  for(uint r = 0; r < flines.size(); r++)
    {
      for(uint s = 0; s < flines.size(); s++)
        {
          if(r != s)
            {
              double xIntersect, yIntersect = 0.0;
              //if two lines intersect
              if(flines[r].intersects(flines[s],xIntersect,yIntersect)
                 && (fabs(flines[r].angleBetween(flines[s])) >= 0.95 * M_PI/2)//convert to radians
                 && (fabs(flines[r].angleBetween(flines[s])) <= 1.05 * M_PI/2))//convert to radians
                {
                  int ptIndex = -1;

                  centerPointLines.push_back(flines[r]);
                  centerPointLines.push_back(flines[s]);


                  //check to see if the point at which they intersect fits into a pre-existing bucket
                  for(uint c = 0; c < intersectPts.size(); c++)
                    {
                      if(xIntersect < intersectPts[c][0].i + 5 &&
                         xIntersect > intersectPts[c][0].i - 5 &&
                         yIntersect < intersectPts[c][0].j + 5 &&
                         yIntersect > intersectPts[c][0].j - 5 )
                        {
                          ptIndex = c;
                        }

                    }

                  //if the point fits into a pre-existing bucket, add it to the bucket
                  if(ptIndex > 0)
                    {
                      int x = (int)(xIntersect);
                      int y = (int)(yIntersect);

                      intersectPts[ptIndex].push_back(Point2D<int>(x,y));
                      //average the old bucket's value with the new point added
                      //so as to create a moving bucket
                       intersectPts[ptIndex][0].i += x;
                       intersectPts[ptIndex][0].i /= 2;
                       intersectPts[ptIndex][0].j += y;
                       intersectPts[ptIndex][0].j /= 2;

                    }
                  //otherwise, create a new bucket
                  else
                    {
                      int x = (int)(xIntersect);
                      int y = (int)(yIntersect);

                      std::vector<Point2D<int> > newIntPts;
                      newIntPts.push_back(Point2D<int>(x,y));
                      intersectPts.push_back(newIntPts);
                    }
                }
            }
        }
    }

  if(intersectPts.size() == 0)
    {
      LDEBUG("*** NO INTERSECT POINTS FOUND ***");
      stalePointCount++;
      return avgCrossCenterPoint;
    }

  uint maxPts = intersectPts[0].size();
  uint maxIndex = 0;

  for(uint i = 0; i < intersectPts.size(); i++)
    {
      if(intersectPts[i].size() > maxPts)
        {
          maxPts = intersectPts[i].size();
          maxIndex = i;
        }
    }

  Point2D<int> finalPoint = intersectPts[maxIndex][0];


  // Smooth out the center points being displayed by averaging the position of the
  // last 30 center points.

  bool isWithinStdDev = false;
  // Check if center point is within std dev
  if(centerPointBuff.size() < 2
     || (abs(finalPoint.i - avgCrossCenterX) <= stdDevCrossX
         && abs(finalPoint.j - avgCrossCenterY) <= stdDevCrossY)
     )
    {
      stdDevCenterCount = 0;
      isWithinStdDev = true;
    }
  else
    {
      stdDevCenterCount++;

      LINFO("Current X: %d Y: %d",finalPoint.i, finalPoint.j);
      LINFO("AVG X: %d Y: %d",avgCrossCenterX, avgCrossCenterY);
      LINFO("STDDEV CROSS (X,Y):(%f,%f)",stdDevCrossX, stdDevCrossY);

    }

  stalePointCount = 0;

  uint centerBuffSize = centerPointBuff.size();

  if(stdDevCenterCount > 5)
    {
      isWithinStdDev = true;
      centerPointBuff.clear();
    }

  if(isWithinStdDev)
    {
      if(centerBuffSize >= 30)
        {
          centerPointBuff.pop_front();
        }

      centerPointBuff.push_back(finalPoint);

      centerBuffSize = centerPointBuff.size();
    }

  if(centerBuffSize > 0)
    {
      int sumX = 0, sumY = 0;
      std::list<Point2D<int> >::iterator it;

      for(it = centerPointBuff.begin(); it != centerPointBuff.end(); ++it)
        {
          sumX += (*it).i;
          sumY += (*it).j;
        }

      avgCrossCenterX = sumX / centerBuffSize;
      avgCrossCenterY = sumY / centerBuffSize;

      avgCrossCenterPoint = Point2D<int>(avgCrossCenterX,avgCrossCenterY);

      float sqrStdCenterX = 0.0;
      float sqrStdCenterY = 0.0;


      for(it = centerPointBuff.begin(); it != centerPointBuff.end(); ++it)
      {
        sqrStdCenterX += (((*it).i - avgCrossCenterX) * ((*it).i - avgCrossCenterX));
        sqrStdCenterY += (((*it).j - avgCrossCenterY) * ((*it).j - avgCrossCenterY));
      }

      float stdevCenterX = sqrStdCenterX / centerBuffSize;
      float stdevCenterY = sqrStdCenterY / centerBuffSize;
      double stdTempCenterX = sqrt((double)stdevCenterX);
      double stdTempCenterY = sqrt((double)stdevCenterY);
      stdDevCrossX = (float)stdTempCenterX;
      stdDevCrossY = (float)stdTempCenterY;
    }
  else
    {
      avgCrossCenterPoint = finalPoint;
    }
  //         stalePointCount++;
  return avgCrossCenterPoint;
}


// ######################################################################
float BeoSubCross::getCrossDir
(const std::vector<LineSegment2D> lines)
{
  uint nLines = lines.size();
  if(nLines == 0) { LDEBUG("*** NO LINES ***"); return avgCrossAngle;}

  //separate lines into positive and negative angles
  std::vector<LineSegment2D> posAngleLines, negAngleLines;

  for(uint i = 0; i < nLines; i++)
    {
      if(lines[i].angle() >= 0)
        {
          posAngleLines.push_back(lines[i]);
        }
      else
        {
          negAngleLines.push_back(lines[i]);
        }
    }

  std::vector<LineSegment2D> fPosAngleLines = posAngleLines;//pruneLines(posAngleLines);
  std::vector<LineSegment2D> fNegAngleLines = negAngleLines;//pruneLines(negAngleLines);

  double avgPosAngle, avgNegAngle;

  // Find average of positive angles
  //  LINFO("\n\n\n\nSum the Average...");
  double sumPosAngle = 0;
  std::vector<LineSegment2D>::iterator itr  = fPosAngleLines.begin();
  while(itr < fPosAngleLines.end())
    {
      float angle = (*itr).angle();
      LDEBUG("Line Angle[%4d,%4d][%4d,%4d]: %f",
            (*itr).point1().i, (*itr).point1().j,
            (*itr).point2().i, (*itr).point2().j,
            angle * 180/M_PI);

      // eliminate lines that are close to vertical
      if(!isnan(angle)) { sumPosAngle += angle; itr++; }
      else
        { sumPosAngle += 90.0;
          //itr = fPosAngleLines.erase(itr);
          LDEBUG("Drop a line");
        }
    }

  if (fPosAngleLines.size() == 0)
    {
      return avgCrossAngle;
    }
  else
    {
      avgPosAngle = sumPosAngle / fPosAngleLines.size();
    }

  // Find average of negative angles
  double sumNegAngle = 0;
  itr  = fNegAngleLines.begin();
  while(itr < fNegAngleLines.end())
    {
      float angle = (*itr).angle();
      LDEBUG("Line Angle[%4d,%4d][%4d,%4d]: %f",
            (*itr).point1().i, (*itr).point1().j,
            (*itr).point2().i, (*itr).point2().j,
            angle * 180/M_PI);

      // eliminate lines that are close to vertical
      if(!isnan(angle)) { sumNegAngle += angle; itr++; }
      else
        { sumNegAngle += 90.0;

        //        itr = fNegAngleLines.erase(itr);
        LDEBUG("Drop a line");
      }
    }


  if (fNegAngleLines.size() == 0)
    {
      return avgCrossAngle;
    }
  else
    {
      avgNegAngle = sumNegAngle / fNegAngleLines.size();
    }

  //Get the distance that the avg pos and neg angles are away from 90
  int posAngleDistance = abs(90 - (int)avgPosAngle);
  int negAngleDistance = abs(90 - abs((int)avgNegAngle));

  float sumAngle = 0.0;
  std::vector<LineSegment2D> flines;

  if(posAngleDistance <= negAngleDistance)
    {
      flines = fPosAngleLines;
      sumAngle = sumPosAngle;
    }
  else
    {
      flines = fNegAngleLines;
      sumAngle = sumNegAngle;
    }

//   // also drop off angles that do not fall
//   // within a standard deviation of the average angle
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

  float avgAngle = 0.0;

  if (flines.size() != 0)
    {
      avgAngle = sumAngle / flines.size();
    }
  else
    {
      return avgCrossAngle;
    }

  if (avgAngle > 0) avgAngle -= M_PI;


  // Smooth out the angles being displayed by averaging the position of the last 30 angles
  uint angleBuffSize = angleBuff.size();

  bool isWithinStdDev = false;
  // Check if angle is within std dev
  if(angleBuff.size() < 2
     || (fabs(avgAngle - avgCrossAngle) <= stdDevCrossAngle ))
    {
      stdDevAngleCount = 0;
      isWithinStdDev = true;
    }
  else
    {
      stdDevAngleCount++;
    }

  if(stdDevAngleCount > 10)
    {
      angleBuff.clear();
      isWithinStdDev = true;
    }

  if(isWithinStdDev)
    {
      if(angleBuffSize >= 50)
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

      avgCrossAngle = sumBuffAngle / angleBuffSize;

      //      LINFO("AVG Cross Angle: %f",avgCrossAngle);


      float sqrStdDevAngle = 0.0;
      for(it = angleBuff.begin(); it != angleBuff.end(); ++it)
      {
        sqrStdDevAngle = ((*it - avgCrossAngle) * (*it - avgCrossAngle));
      }

      float stdevAngle = sqrStdDevAngle / angleBuffSize;
      double stdTempAngle = sqrt((double)stdevAngle);
      stdDevCrossAngle = (float)stdTempAngle;
    }
  else
    {
      avgCrossAngle = avgAngle;
    }

  return avgCrossAngle;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
