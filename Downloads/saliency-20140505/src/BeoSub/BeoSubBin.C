/*!@file BeoSub/BeoSubBin.C find pipe     */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubBin.C $
// $Id: BeoSubBin.C 12782 2010-02-05 22:14:30Z irock $

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
//#include "BeoSub/IsolateColor.H"
#include "rutz/compat_cmath.h" // for isnan()

#include "BeoSub/BeoSubBin.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream> //needed for segmentImageTrackMC!
#include <math.h>
#include <vector>
#include <cmath>

// ######################################################################
BeoSubBin::BeoSubBin()
{
  imageAngles.resize(0);
  imageLines.resize(0);
  binAngles.resize(0);
  binLines.resize(0);
  binCenter = *(new Point2D<int>(-1, -1));

}

// ######################################################################
BeoSubBin::~BeoSubBin()
{ }



// ######################################################################
float BeoSubBin::getBinSceneMass(Image< PixRGB<byte> > &cameraImage, Image< PixRGB<byte> > &outputImage, Point2D<int>& center) {
#ifndef HAVE_OPENCV
  LFATAL("OpenCV must be installed in order to use this function");
  return 0;
#else
  IplImage *edge = cvCreateImage( cvGetSize(img2ipl(cameraImage)), 8, 1 );
  IplImage *out = cvCreateImage( cvGetSize(img2ipl(cameraImage)), 8, 1 );
  IplImage *edge2 = cvCreateImage( cvGetSize(img2ipl(cameraImage)), 8, 1);

  ////////// image manipulation ///////////////
  cvThreshold(img2ipl(luminance(cameraImage)), edge, 170, 255, CV_THRESH_TOZERO);

  IplConvKernel* dilation = cvCreateStructuringElementEx(8, 8, 0, 0, CV_SHAPE_RECT);
  cvErode(edge, edge2, dilation, 1);
  cvDilate(edge2, edge, dilation, 1);
  cvThreshold(edge, out, 200, 255, CV_THRESH_TOZERO);


  int mass=-1, x=0, y=0;
  Image<byte> isoWhite = luminance(ipl2gray(out));
  //////FIXME  mass = isolateWhite(isoWhite, isoWhite, x, y);

  outputImage = toRGB(isoWhite);

  center.i = x;
  center.j = y;

  cvReleaseImage(&edge);
  cvReleaseImage(&edge2);
  //  cvReleaseImage(&out);
  cvReleaseStructuringElement(&dilation);
  return mass;
#endif
}


std::vector<LineSegment2D> BeoSubBin::getHoughLines(Image< PixRGB <byte> > &cameraImage, Image< PixRGB <byte> > &preHough,
                                                    Image< PixRGB <byte> > &outputImage)
{
  std::vector <LineSegment2D> lines;

#ifndef HAVE_OPENCV
  LFATAL("OpenCV must be installed in order to use this function");
#else

  //  inplacePaste(dispImage, toRGB(edgeImage), Point2D<int>(0,h));
  IplImage *edge = cvCreateImage( cvGetSize(img2ipl(cameraImage)), 8, 1 );
  IplImage *out = cvCreateImage( cvGetSize(img2ipl(cameraImage)), 8, 1 );
  IplImage *edge2 = cvCreateImage( cvGetSize(img2ipl(cameraImage)), 8, 1 );
  //cvSobel(img2ipl(luminance(cameraImage)), edge, 0, 1, 3);





  /////////////////////// steps for bottom camera which only uses rectangulation ///////////////

  //////////////// image manipulation, may need tweaking /////////////////////
  // need to up contrast
  cvEqualizeHist(img2ipl(luminance(cameraImage)), edge);
  cvThreshold(img2ipl(luminance(cameraImage)), edge, 170, 255, CV_THRESH_TOZERO );

  cvThreshold(edge, edge2, 170, 255, CV_THRESH_TOZERO);
  IplConvKernel* dilation = cvCreateStructuringElementEx(8, 8, 0, 0, CV_SHAPE_RECT);
  out = edge;
  cvErode(edge, edge2, dilation, 1);
  cvDilate(edge2, edge, dilation, 1 );
  cvThreshold(edge, edge2, 200, 255, CV_THRESH_TOZERO );

  /////////////////////////////////////////////////////////


  cvCanny(edge, out, 100, 150, 3 );//150,200,3
  preHough =  ipl2gray(edge);


  CvMemStorage* storage = cvCreateMemStorage(0);

  outputImage.clear();
  outputImage = toRGB( ipl2gray( out ) );

  CvSeq* cvlines = cvHoughLines2(out, storage, CV_HOUGH_STANDARD, 1, CV_PI/180, 10, 10, 10);

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
      lines.push_back(LineSegment2D(Point2D<int>(pt1.x,pt1.y),Point2D<int>(pt2.x,pt2.y)));
      drawLine(outputImage, Point2D<int>(pt1.x,pt1.y), Point2D<int>(pt2.x, pt2.y), PixRGB<byte>(255,0,0));
    }
    ///////////////////////////////////////////////////////////////////////////////////




    ///////////// NOTE:::: will need to create overlap function that will remove orange, replace it with black ///////
    ////////////////// before any bin recognition is done /////////////////////////


    cvReleaseImage(&edge);
    cvReleaseImage(&edge2);
    //cvReleaseImage(&out);
    cvReleaseStructuringElement(&dilation);


#endif // HAVE_OPENCV
    return lines;
}

// ######################################################################
void BeoSubBin::pruneLines(std::vector<LineSegment2D>& lines, std::vector<LineSegment2D> &pruned, Image< PixRGB<byte> >* img ) {


  std::vector<LineSegment2D> nonSimilarLines;
  std::vector<LineSegment2D>::iterator iter;

  // add the first line to the nonSimilarLine vector
  // erase if from the input line vector we are checking
  if(lines.size() > 0) {
    nonSimilarLines.push_back(lines[0]);
    lines.erase(lines.begin());
    iter = lines.begin();
  }
  else
    LDEBUG("*** NO LINES ***");


  double x, y;



  //top border of image
  LineSegment2D border1 ( *(new Point2D<int>(0, 0)), *(new Point2D<int>(img->getWidth(), 0) ) );
  // left border of image
  LineSegment2D border2 ( *(new Point2D<int>(0, 0)), *(new Point2D<int>(0, img->getHeight()) ) );
  // right border of image
  LineSegment2D border3 ( *(new Point2D<int>(img->getWidth(), 0)), *(new Point2D<int>(img->getWidth(), img->getHeight()) ) );
  //bottom border of image
  LineSegment2D border4 ( *(new Point2D<int>(0, img->getHeight())), *(new Point2D<int>(img->getWidth(), img->getHeight()) ) );

  // check for similar lines
  for(uint i = 0; i < nonSimilarLines.size(); i++) {

    while(iter != lines.end()) {

      double angle = nonSimilarLines[i].angleBetween(*iter);
      angle *= 180/PI;
      double intersect = nonSimilarLines[i].intersects((*iter), x, y);

      // if the lines do not intersect in the image, possibly parallel
      if(!intersect) {

        double x2, y2;
        // if the parallel lines intersect the top and bottom images
        // at similar points, remove line from unpruned line list
        if((nonSimilarLines[i].intersects(border1, x, y) && (*iter).intersects(border1, x2, y2)) ||
           (nonSimilarLines[i].intersects(border4, x, y) && (*iter).intersects(border4, x2, y2))) {


          if(fabs(x - x2) < 5.0) {
            // drawLine(*img, (*iter).point1(), (*iter).point2(), PixRGB<byte>(255, 150, 0));
            lines.erase(iter);
          }

        }
        // if the parallel lines intersect the left and right images
        // at similar points, remove one of them
        else if((nonSimilarLines[i].intersects(border2, x, y) && (*iter).intersects(border2, x2, y2)) ||
                (nonSimilarLines[i].intersects(border3, x, y) && (*iter).intersects(border3, x2, y2))) {

          if(fabs(y - y2) < 5.0) {
            //drawLine(*img, (*iter).point1(), (*iter).point2(), PixRGB<byte>(255, 150, 0));
            lines.erase(iter);
          }

        }

      }
      // else if lines are not parallel,
      // if they intersect in the image at a very small angle
      // remove one of them
      else if(intersect && img->coordsOk((int)x, (int)y) && (fabs(angle) < 2.5 || fabs(angle) > 178.5)) {
        //drawLine(*img, (*iter).point1(), (*iter).point2(), PixRGB<byte>(255, 0, 150));
        lines.erase(iter);
      }

      if(iter != lines.end())
        iter++;
    }

    iter = lines.begin();
    // once all unpruned lines have been compared to current nonSimilarLine,
    // in which any similar lines have been removed.
    // add next unpruned line to nonSimilarLines vector
    // reiterate and see if any unpruned lines are similar to that one
    if(lines.size() > 0) {
      nonSimilarLines.push_back(*iter);
      lines.erase(iter);
    }


  }

  pruned = nonSimilarLines;
  // lines are now pruned
  binAngles.clear();

  // now remove similar angles at similar intersections
  pruneAngles(nonSimilarLines, binAngles, img);


  /// drawing ////////////////
  for(uint i = 0; i < nonSimilarLines.size(); i++)
    drawLine(*img, nonSimilarLines[i].point1(), nonSimilarLines[i].point2(), PixRGB<byte>(255,0,0));

//   for(uint i = 0; i < binAngles.size(); i++) {
//     //printf("x: %d, y: %d, angle: %f\n", binAngles[i].pos.i, binAngles[i].pos.j, binAngles[i].angle);
//     drawCircle(*img, binAngles[i].pos, 5, PixRGB<byte>(0, 255, 0));
//   }

  drawCircle(*img, *(new Point2D<int>(160, 120)), 5, PixRGB<byte>(200, 255, 0));

  /// we also want to return angles with their respective positions.
  //// which will require a struct that associates angle, with an x and a y;

}


void BeoSubBin::pruneAngles(std::vector<LineSegment2D>& lines, std::vector<BinAngles>& angles, Image< PixRGB< byte > >* img) {


  for(uint i = 0; i < lines.size(); i++) {

    for(uint j = i+1; j < lines.size(); j++) {

      double angle = lines[i].angleBetween(lines[j]);
      angle *= 180/PI;

      double x, y;
      lines[i].intersects(lines[j], x, y);

      // look through list of angles added so far,
      // if the angle we are currently looking at is
      // a similar angle, and it has a similar intersection
      // do not add it to angles vector
      bool add = true;
      for(uint k = 0; k < angles.size(); k++) {
        if(fabs(angles[k].angle - angle) < 5.0 &&
           (abs(angles[k].pos.i - (int)x) <= 2.0 && abs(angles[k].pos.j - (int)y) <= 2.0)) {
          add = false;
          break;
        }
      }

      // add angle if there no similar angles at intersections
      // in angles vector
      if(add && img->coordsOk((int)x, (int)y)) {
        BinAngles b;
        b.pos = *(new Point2D<int>((int)x, (int)y));
        b.angle = angle;
        angles.push_back(b);
      }

    }
  }

}

void BeoSubBin::removeOrangePipe(Image< PixRGB<byte> >& img) {

//   Image<byte> orange(img.getWidth(), img.getHeight(), ZEROS);

//   float mass = isolateOrange(img, orange);


//   if(mass > 0) {

//     for(int i = 0; i < img.getWidth(); i++) {
//       for(int j = 0; j < img.getHeight(); j++) {

//         if(orange.coordsOk(i, j) && orange.getVal(i, j) == 255)
//           img.setVal(i, j, 0);

//       }
//     }

//   }


}

void BeoSubBin::getWeightedBinMass(std::vector<BinAngles>& angles, Point2D<int> & center, bool cropWindow, Image<PixRGB<byte> > *img) {

  // find closest
  float DIST = 1;

  float avgY = 0;

  if(angles.size() > 0) {
    DIST = center.distance(angles[0].pos);
    avgY = angles[0].pos.j;
  }


  for(uint i = 1; i < angles.size(); i++) {
    float closer = center.distance(angles[i].pos);

    avgY += angles[i].pos.j;

    if(closer < DIST) //&& angles[i].pos.j > 100) //&& angles[i].angle < 80 && angles[i].angle > 40)
      DIST = closer;

  }

  avgY /= angles.size();
  avgY = 120 - avgY;

  float topCap = 0, bottomCap = 240;

  if(cropWindow && avgY < 0)
    bottomCap = 240 - avgY;
  else if(cropWindow)
    topCap = avgY;

  if(cropWindow) {
  LINFO("avgY: %f", avgY);
  drawLine(*img, Point2D<int>(0, (int)avgY), Point2D<int>(320, (int)avgY), PixRGB<byte>(255, 0, 0));
  drawLine(*img, Point2D<int>(0, (int)topCap), Point2D<int>(320, (int)topCap), PixRGB<byte>(255,0,0));
  drawLine(*img, Point2D<int>(0, (int)bottomCap), Point2D<int>(320, (int)bottomCap), PixRGB<byte>(255,0,0));
  }

  Point2D<int> weightedCenter = center;
  /// setup weighting system
  //// ORIGINAL DISTANCE / (WEIGHT)^2 from CLOSEST DISTANCE * WEIGHT = ORIGINAL DISTANCE

  printf("closest dist: %f\n", DIST);

  for(uint i = 0; i < angles.size(); i++) {

    if(angles[i].pos.j < bottomCap && angles[i].pos.j > topCap) {
      float currentDist = weightedCenter.distance(angles[i].pos);
      printf("current dist: %f, x: %d, y: %d\n", currentDist, angles[i].pos.i, angles[i].pos.j);
      float weight = 1 / (currentDist/DIST ); // * currentDist/DIST);
      printf("weight: %f\n", weight);
      float cx = (angles[i].pos.i - weightedCenter.i) * weight + weightedCenter.i;
      float cy = (angles[i].pos.j - weightedCenter.j) * weight + weightedCenter.j;

      Point2D<int> weightPoint((int)cx, (int)cy);
      printf("new point, x: %d, y: %d\n", weightPoint.i, weightPoint.j);
      center += weightPoint;
      center /= 2;
    }
  }

}

// THIS FUNCTION IS ONLY USED FOR FRONT CAMERA
void BeoSubBin::getParallelIntersections(std::vector<LineSegment2D>& lines, std::vector<LineSegment2D>& frontLines,
                                         Image< PixRGB<byte> >& img) {

  printf("size of lines: %d\n", (int)lines.size());

  bool moreThanOne;

  frontLines.clear();
  std::vector<LineSegment2D> tmp = lines;
  std::vector<LineSegment2D>::iterator iter2 = tmp.begin();

 // left border of image
  LineSegment2D border2 ( *(new Point2D<int>(0, 0)), *(new Point2D<int>(0, img.getHeight()) ) );
  // right border of image
  LineSegment2D border3 ( *(new Point2D<int>(img.getWidth(), 0)), *(new Point2D<int>(img.getWidth(), img.getHeight()) ) );

  // iterate through set of lines
  for(uint i = 0; i < tmp.size(); i++) {

    moreThanOne = false;
    // check current line with other lines in the set
    iter2 = tmp.begin() + i + 1;
    while(tmp.size() > i+1 && iter2 != tmp.end()) {

      double x, y;
      x = y = -1;
      // if both lines do not intersect each other in the image
      // and current lines are not in a horizontal direction, add to output vector
      if((!lines[i].intersects(*iter2, x, y) && !img.coordsOk((int)x, (int)y)) &&
         lines[i].intersects(border2, x, y) && lines[i].intersects(border3, x, y)) {

        drawLine(img, (*iter2).point1(), (*iter2).point2(), PixRGB<byte>(0,255,0));
        frontLines.push_back(*iter2);
        tmp.erase(iter2);
        moreThanOne = true;


      }
      else
        iter2++;
    }

    // if we found parallel lines,
    // make sure we add the line we were checking against
    if(moreThanOne && tmp.size() > i) {
      iter2 = tmp.begin() + i;
      drawLine(img, (*iter2).point1(), (*iter2).point2(), PixRGB<byte>(0,255,200));
      frontLines.push_back(*iter2);
      tmp.erase(iter2);
    }
  }

  // get intersections
  printf("size of frontLines before adding intersections: %d\n", (int)frontLines.size());
  uint size = frontLines.size();
  iter2 = tmp.begin();
  for(uint i = 0; i < size; i++) {
    iter2 = tmp.begin();
    while(tmp.size() > 0 && iter2 != tmp.end()) {

      double x, y;
      x = y = -1;
      if(frontLines[i].intersects(*iter2, x, y) && img.coordsOk((int)x, (int)y)) {
        frontLines.push_back(*iter2);
        tmp.erase(iter2);
      }
      else
        iter2++;
    }
  }


  printf("size of frontLines: %d\n", (int)frontLines.size());
}

void BeoSubBin::getBinCenter(std::vector<BinAngles>& angles, Point2D<int>& center) {

  for(uint i = 0; i < angles.size(); i++)
    center += angles[i].pos;

  center /= angles.size();
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
