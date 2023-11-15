/*!@file BeoSub/HoughTransform.C find hough transform for an image     */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/HoughTransform.C $
// $Id: HoughTransform.C 9412 2008-03-10 23:10:15Z farhan $


#ifndef HoughTransform_C
#define HoughTransform_C

#include "BeoSub/HoughTransform.H"

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Image/ColorOps.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Transforms.H"
#include "Raster/Raster.H"
#include "rutz/shared_ptr.h"
#include "BeoSub/hysteresis.H"
#include "VFAT/segmentImageTrackMC.H"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream> //needed for segmentImageTrackMC!
#include <math.h>
#include <list>
#include "Image/MatrixOps.H"
#include "BeoSub/CannyEdge.H"
#include "MBARI/Geometry2D.H"
using namespace std;

// finds all the lines in an image by using the hough transform
// thetaRes is the resolution of each theta
// dRes is the resolution of the D
// returns a vector with the line segments found
std::vector <LineSegment2D> houghTransform
(Image<byte> &inputImage, float thetaRes, float dRes, int threshold,
 Image< PixRGB<byte> > &output)
{
  //Timer tim(1000000);
  //uint t;

  uint w = inputImage.getWidth();
  uint h = inputImage.getHeight();

  // get the total number of angles and delta from the resolution of theta and D's
  int numAngle = (int) (M_PI / thetaRes);
  int numD = (int)(sqrt((w*w + h*h)/ dRes) / 2);
  int centerX = (int)(w/2);
  int centerY = (int)(h/2);

  std::vector <LineSegment2D> lines;
  std::vector <Point2D<int> > edgePoints;

  //find all the white edge pixels in the image and store them
  for(int y = 0; y < inputImage.getHeight(); y++)
    {
      for(int x = 0; x < inputImage.getWidth(); x++)
        {
          //if the pixel is an edge
          if(inputImage.getVal(x,y) == 255)
            // convert the x,y position of the pixel to an x,y position where
            // the center of the image is the origin as opposed to the top left corner
            // and store the pixel
            edgePoints.push_back(Point2D<int>(x - centerX, y - centerY));
        }
    }

  // store the edge points that belong to each line
  std::vector<std::vector<std::vector<Point2D<int> > > > ptList;
  ptList.resize(numD);
  for(int i = 0; i < numD; i++) ptList[i].resize(numAngle);

  // equation of the line is x*cos(theta) + y*sin(theta) = D

  // fill the accumulator
  for(uint c = 0; c < edgePoints.size(); c++)
    {
      int edgePointX = edgePoints[c].i;
      int edgePointY = edgePoints[c].j;

      //get all possible D and theta that can fit to that point
      for(int n = 0; n < numAngle; n++)
        {
          int r =  (int)(edgePointX * cos(n*thetaRes) + edgePointY * sin(n*thetaRes));

          // if the radius is negative, make it positive so as to avoid segfaults
          // (a point with radius r lies on the same line
          // as a point with radius -r if they share the same angle)
          r = abs(r);

          ptList[r][n].push_back(edgePoints[c]);

        }
    }

  // find the peaks, ie any number of points greater
  // than the threshold as well as a local maximum is considered a line
  for(int i = 1; i < numD - 1; i++)  {
    for(int j = 1; j < numAngle - 1; j++)
      {
        uint currentPointCount = ptList[i][j].size();

        if(currentPointCount > (unsigned int)(threshold) &&
           currentPointCount > ptList[i-1][j].size() &&
           currentPointCount > ptList[i+1][j].size() &&
           currentPointCount > ptList[i][j-1].size() &&
           currentPointCount > ptList[i][j+1].size()    ) // found a peak
          {
            //find the endpoints of the line segment
            int minx = ptList[i][j][1].i, maxx = ptList[i][j][1].i;
             int miny = ptList[i][j][1].j, maxy = ptList[i][j][1].j;
            //the indexes of the two endpoints in the vector
            uint mini = 0, maxi = 0;
            for(uint k = 1; k < currentPointCount; k++)
              {
                if(minx > ptList[i][j][k].i)
                  { minx =  ptList[i][j][k].i; mini = k; }
                if(maxx < ptList[i][j][k].i)
                  { maxx =  ptList[i][j][k].i; maxi = k; }
              }
//             LINFO("ksize: %d, min[%d]:%d,%d max[%d]:%d,%d", ptList[i][j].size(),
//                   mini, ptList[i][j][mini].i, ptList[i][j][mini].j,
//                   maxi, ptList[i][j][maxi].i, ptList[i][j][maxi].j);

            // if the line is vertical
            if(minx == maxx)
              {
                for(uint k = 1; k < currentPointCount; k++)
                  {
                    if(miny > ptList[i][j][k].j)
                      { miny =  ptList[i][j][k].j; mini = k; }
                    if(maxy > ptList[i][j][k].j)
                  { maxy =  ptList[i][j][k].j; maxi = k; }
                  }
              }

            // the two endpoints of the line segment
            Point2D<int> p1 = ptList[i][j][mini];
            Point2D<int> p2 = ptList[i][j][maxi];

            // convert from x,y where the center of the image is the origin
            // back to x,y where top left of image is the origin
            p1.i += centerX;
            p1.j += centerY;
            p2.i += centerX;
            p2.j += centerY;

            LineSegment2D thisLine(p1,p2);

            if(thisLine.isValid())
              {
                lines.push_back(thisLine);
                drawLine(output, p1, p2,  PixRGB<byte> (255,0,0), 1);
              }
          }
      }
  }

  return lines;

}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
