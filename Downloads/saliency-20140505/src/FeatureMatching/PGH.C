/*!@file FeatureMatching/PGH.C  Pairwise Geometric Histograms */



// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/FeatureMatching/PGH.C $
// $Id: PGH.C 12985 2010-03-09 00:18:49Z lior $
//

#ifndef PGH_C_DEFINED
#define PGH_C_DEFINED

#include "FeatureMatching/PGH.H"
#include "Image/DrawOps.H"
#include "GUI/DebugWin.H"
#include "Util/FastMathFunctions.H"
#include <fcntl.h>

// ######################################################################
PGH::PGH()
{
}

// ######################################################################
PGH::~PGH()
{

}

void PGH::addModel(const std::vector<Line>& lines, int id)
{
  ModelInfo model;
  model.id = id;
  model.lines = lines;


  //Create the histogram
  model.histogram = getGeometricHistogram(lines);
  SHOWIMG(model.histogram);

  itsModels.push_back(model);
}

int PGH::matchModel(const std::vector<Line>& lines)
{

  Image<float> histogram = getGeometricHistogram(lines);
  int id = 0;
  for(uint i=0; i<itsModels.size(); i++)
  {
    SHOWIMG(itsModels[i].histogram);
    SHOWIMG(histogram);
    double val = cmpHist(itsModels[i].histogram, histogram);
    LINFO("%i %f", itsModels[i].id, val);
  }

  return id;

}

double PGH::cmpHist(const Image<float>& hist1, const Image<float>& hist2)
{
  return sum(hist1-hist2);
}

Image<float>  PGH::getGeometricHistogram(const std::vector<Line>& lines)
{

  Image<float> histogram;

  for(uint i=0; i<lines.size(); i++)
    for(uint j=0; j<lines.size(); j++)
    {
      if (i==j) continue;
 
      float x = lines[j].pos.i - lines[i].pos.i;
      float y = lines[j].pos.j - lines[i].pos.j;

      float rot = lines[i].ori;

      Point2D<int> pos;
      pos.i = lines[i].pos.i + int(x * cos(rot) - y * sin(rot));
      pos.j = lines[i].pos.j + int(y * cos(rot) + x * sin(rot));
      float ori = lines[j].ori - rot;

      int x1 = int(cos(ori)*lines[j].length/2);
      int y1 = int(sin(ori)*lines[j].length/2);

      Point2D<int> pt1(pos.i-x1, pos.j+y1);
      Point2D<int> pt2(pos.i+x1, pos.j-y1);

      int d1 = lines[i].pos.j - pos.j+y1;
      int d2 = lines[i].pos.j - pos.j-y1;

      LINFO("d1 %i d2 %i", d1, d2);

      Image<PixRGB<byte> > img(320,240,ZEROS);
      drawLine(img,lines[i].pos, 0, lines[i].length,
          PixRGB<byte>(255,0,0));

      drawLine(img,pos, ori, lines[j].length,
          PixRGB<byte>(0,255,0));

      //drawLine(img, pt1, d1, PixRGB<byte>(0,0,255));
      //drawLine(img, pt2, d2, PixRGB<byte>(0,0,255));
      SHOWIMG(img);
    }

  return histogram;

}

//Image<float>  PGH::getGeometricHistogram(const std::vector<Line>& lines)
//{
//
//  #define DBL_EPSILON 2.2204460492503131e-16
//
//  int angleDim = 100;
//  int distDim = 100;
//
//
//  double angle_scale = (angleDim - 0.51) / fastacos(0);
//  double dist_scale = DBL_EPSILON;
//
//  Image<float> histogram(angleDim, distDim, ZEROS);
//  /* 
//     do 2 passes. 
//     First calculates maximal distance.
//     Second calculates histogram itself.
//     */
//  for(int pass = 1; pass <= 2; pass++ )
//  {
//    double dist_coeff = 0, angle_coeff = 0;
//
//    /* run external loop */
//    for(uint i = 0; i < lines.size(); i++ )
//    {
//      int dist = 0;
//
//      int x1 = int(cos(lines[i].ori)*lines[i].length/2);
//      int y1 = int(sin(lines[i].ori)*lines[i].length/2);
//
//      Point2D<int> pt1(lines[i].pos.i-x1, lines[i].pos.j+y1);
//      Point2D<int> pt2(lines[i].pos.i+x1, lines[i].pos.j-y1);
//
//      int dx = pt2.i - pt1.i;
//      int dy = pt2.j - pt1.j;
//
//      if( (dx | dy) != 0 )
//      {
//        if( pass == 2 )
//        {
//          dist_coeff = (1/lines[i].length) * dist_scale;
//          angle_coeff = (1/lines[i].length) * ( ACOS_TABLE_SIZE / 2);
//        }
//
//        /* run internal loop (for current edge) */
//        for(uint j = 0; j < lines.size(); j++ )
//        {
//          int x1 = int(cos(lines[j].ori)*lines[j].length/2);
//          int y1 = int(sin(lines[j].ori)*lines[j].length/2);
//
//          Point2D<int> pt3(lines[j].pos.i-x1, lines[j].pos.j+y1);
//          Point2D<int> pt4(lines[j].pos.i+x1, lines[j].pos.j-y1);
//
//
//          if( i != j )        /* process edge pair */
//          {
//
//
//            int d1 = (pt3.j - pt1.j) * dx - (pt3.i - pt1.i) * dy;
//            int d2 = (pt4.j - pt1.j) * dx - (pt2.i - pt1.i) * dy;
//            int cross_flag;
//            int hist_row = 0;
//
//            if( pass == 2 )
//            {
//
//              int dp = (pt4.i - pt3.i) * dx + (pt4.j - pt3.j) * dy;
//
//              dp = int( dp * angle_coeff * (1/lines[j].length) ) + (ACOS_TABLE_SIZE / 2);
//              dp = std::max( dp, 0 );
//              dp = std::min( dp, ACOS_TABLE_SIZE - 1 );
//
//              hist_row =  (int)round( fastacos(dp) * angle_scale );
//
//              Image<PixRGB<byte> > img(320,240,ZEROS);
//              drawLine(img, pt1, pt2, PixRGB<byte>(0,255,0));
//              drawLine(img, pt3, pt4, PixRGB<byte>(0,255,0));
//              LINFO("Ang %f", fastacos(dp)*180/M_PI);
//              SHOWIMG(img);
//
//              d1 = (int)round( d1 * dist_coeff );
//              d2 = (int)round( d2 * dist_coeff );
//            }
//
//            cross_flag = (d1 ^ d2) < 0;
//
//            d1 = abs( d1 );
//            d2 = abs( d2 );
//
//            if( pass == 2 )
//            {
//              if( d1 >= distDim )
//                d1 = distDim - 1;
//              if( d2 >= distDim )
//                d2 = distDim - 1;
//
//              if( !cross_flag )
//              {
//                if( d1 > d2 )   /* make d1 <= d2 */
//                {
//                  d1 ^= d2;
//                  d2 ^= d1;
//                  d1 ^= d2;
//                }
//
//                for( ; d1 <= d2; d1++ )
//                  histogram.setVal(hist_row, d1, histogram.getVal(hist_row, d1)+1);
//              } else {
//                for( ; d1 >= 0; d1-- )
//                  histogram.setVal(hist_row, d1, histogram.getVal(hist_row, d1)+1);
//                for( ; d2 >= 0; d2-- )
//                  histogram.setVal(hist_row, d2, histogram.getVal(hist_row, d2)+1);
//              }
//            }
//            else    /* 1st pass */
//            {
//              d1 = std::max( d1, d2 );
//              dist = std::max( dist, d1 );
//            }
//          }           /* end of processing of edge pair */
//
//        }               /* end of internal loop */
//
//        if( pass == 1 )
//        {
//          double scale = dist * (1/lines[i].length);
//
//          dist_scale = std::max( dist_scale, scale );
//        }
//      }
//    }                       /* end of external loop */
//
//    if( pass == 1 )
//    {
//      dist_scale = (distDim - 0.51) / dist_scale;
//    }
//
//  }                           /* end of pass on loops */
//
//  return histogram;
//}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

