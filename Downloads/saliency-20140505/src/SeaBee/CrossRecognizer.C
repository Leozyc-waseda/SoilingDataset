/*!@file SeaBee/PipeRecognizer.C find pipe     */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/CrossRecognizer.C $
// $Id: CrossRecognizer.C 10794 2009-02-08 06:21:09Z itti $

#include "SeaBee/CrossRecognizer.H"


// ######################################################################
CrossRecognizer::CrossRecognizer()
{
  //default values for hough transform
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

  foundCount = 0;

  itsSetupOrangeTracker = false;
}

// ######################################################################
CrossRecognizer::~CrossRecognizer()
{ }

// ######################################################################
void CrossRecognizer::getCrossLocation(rutz::shared_ptr<Image<byte> > colorSegmentedImage,
                                       rutz::shared_ptr<Image<PixRGB <byte> > > outputImage,
                                       CrossRecognizeMethod method,
                                       rutz::shared_ptr<Point2D<int> > crossCenterPoint,
                                       rutz::shared_ptr<float> crossAngle,
                                       rutz::shared_ptr<uint> stalePointCount
                                       )
{
  switch(method)
    {
    case HOUGH:
      calculateHoughTransform(colorSegmentedImage, outputImage, crossCenterPoint, crossAngle, stalePointCount);
      break;

    default:
      LERROR("Invalid pipe recognizer method specified");
    }

}

// ######################################################################

void CrossRecognizer::calculateHoughTransform(rutz::shared_ptr<Image<byte> > colorSegmentedImage,
                                             rutz::shared_ptr<Image<PixRGB <byte> > > outputImage,
                                             rutz::shared_ptr<Point2D<int> > crossCenterPoint,
                                             rutz::shared_ptr<float> crossAngle,
                                             rutz::shared_ptr<uint> stalePointCount
                                             )
{
  std::vector<LineSegment2D> lines = getHoughLines(colorSegmentedImage, outputImage);
  std::vector<LineSegment2D> centerPointLines;
  *crossCenterPoint = getCrossCenter(lines, centerPointLines, stalePointCount);

  LDEBUG("Center Point X: %d, Y: %d", crossCenterPoint->i, crossCenterPoint->j);

  *crossAngle = getCrossDir(centerPointLines);
}

// ######################################################################

std::vector<LineSegment2D> CrossRecognizer::pruneLines (std::vector<LineSegment2D> lines)
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

 Point2D<int> CrossRecognizer::getCrossCenter (const std::vector<LineSegment2D> lines,
                                         std::vector<LineSegment2D> &centerPointLines,
                                         rutz::shared_ptr<uint> stalePointCount)
{

  Point2D<int> avgCrossCenterPoint = Point2D<int>(avgCrossCenterX,avgCrossCenterY);

  uint nLines = lines.size();
  if(nLines == 0)
    {
      LDEBUG("*** NO LINES ***");
      ++(*stalePointCount);
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
      ++(*stalePointCount);
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

  *stalePointCount = 0;

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
float CrossRecognizer::getCrossDir(const std::vector<LineSegment2D> lines)
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

  float meanAngle = 0.0;
  float sumAngle = 0.0;
  std::vector<LineSegment2D> flines;

  if(posAngleDistance <= negAngleDistance)
    {
      flines = fPosAngleLines;
      meanAngle = avgPosAngle;
      sumAngle = sumPosAngle;
    }
  else
    {
      flines = fNegAngleLines;
      meanAngle = avgNegAngle;
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
std::vector<LineSegment2D> CrossRecognizer::getHoughLines
(rutz::shared_ptr< Image< byte > > colorSegmentedImage,
 rutz::shared_ptr< Image< PixRGB <byte> > > outputImage)
{

#ifndef HAVE_OPENCV
  LFATAL("OpenCV must be installed in order to use this function");
#else

  //find edges of segmented image using canny
  IplImage *edge = cvCreateImage( cvGetSize(img2ipl(*colorSegmentedImage)), 8, 1 );
  cvCanny( img2ipl(luminance(*colorSegmentedImage)), edge, 100, 150, 3 );//150,200,3

  //clear output image and set it equal to canny edge image
  outputImage->clear();
  *outputImage = toRGB(ipl2gray(edge));

  //storage for use in hough transform
  CvMemStorage* storage = cvCreateMemStorage(0);

  //perform hough transform and store hough lines
  CvSeq* cvLines = cvHoughLines2(edge, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180, 30, 20, 10);

  LINFO("cvLines %d",cvLines->total);

  //storage for hough line segments
  std::vector <LineSegment2D> lineSegments;

  //loop through hough lines, store them as line segments, and draw lines in output image
  for(int i = 0; i < cvLines->total; i++ )
    {
      //get a line
      CvPoint* line = (CvPoint*)cvGetSeqElem(cvLines,i);
      //get line end points
      Point2D<int> pt1 = Point2D<int>(line[0].x,line[0].y);
      Point2D<int> pt2 = Point2D<int>(line[1].x,line[1].y);


      //create line segment from endpoints and store
      lineSegments.push_back(LineSegment2D(pt1,pt2));

      //draw line segment in output image
      drawLine(*outputImage, pt1, pt2, PixRGB<byte>(255,0,0));
    }

  //clean up pointers
  //  delete edge;
  //  delete storage;
  //  delete cvLines;



#endif // HAVE_OPENCV










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
