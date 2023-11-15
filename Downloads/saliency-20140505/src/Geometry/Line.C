/*!@file FeatureMatching/Line.C Line segments algs */

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
// $HeadURL: $
// $Id: $
//

#ifndef LINE_C_DEFINED
#define LINE_C_DEFINED

#include "FeatureMatching/Line.H"
#include "Image/DrawOps.H"
#include "GUI/DebugWin.H"
#include "Util/FastMathFunctions.H"

// ######################################################################
Line::Line(float sx, float sy, float ex, float ey) :
  p1(sx,sy),
  p2(ex,ey),
  itsDirectionIdx(-1)
{}

// ######################################################################
Line::Line(Point2D<float> inP1, Point2D<float> inP2) :
  p1(inP1),
  p2(inP2),
  itsDirectionIdx(-1)
{}

// ######################################################################
Line::Line(Point2D<int> inP1, Point2D<int> inP2) :
  p1(inP1),
  p2(inP2),
  itsDirectionIdx(-1)
{}

// ######################################################################
double Line::getOri()
{
  double theta = atan2(p2.j-p1.j,p2.i-p1.i);
  if (theta<0)
    theta += M_PI;
  return theta;
}

// ######################################################################
double Line::getLength()
{
  double dx = p2.i - p1.i;
  double dy = p2.j - p1.j;
  return sqrt(dx*dx + dy*dy);
}

// ######################################################################
void Line::scale(double s)
{
  p1.i *= s;
  p1.j *= s;
  p2.i *= s;
  p2.j *= s;
}

// ######################################################################
void Line::scale(double sx, double sy)
{
  p1.i *= sx;
  p1.j *= sy;
  p2.i *= sx;
  p2.j *= sy;
}

// ######################################################################
Point2D<float> Line::getCenter()
{
  Point2D<float> c = (p1+p2)/2;

  return c;

}

// ######################################################################
void Line::rotate(double theta)
{

  double sinTheta;
  double cosTheta;
  double mat[2][2];

  sinTheta = sin(theta);
  cosTheta = cos(theta);
  mat[0][0] = cosTheta;
  mat[0][1] = -sinTheta;
  mat[1][0] = sinTheta;
  mat[1][1] = cosTheta;

  double x, y;
  x = p1.i, y = p1.j;
  p1.i = x*mat[0][0] + y*mat[0][1];
  p1.j = x*mat[1][0] + y*mat[1][1];

  x = p2.i, y = p2.j;
  p2.i = x*mat[0][0] + y*mat[0][1];
  p2.j = x*mat[1][0] + y*mat[1][1];
}

// ######################################################################
void Line::shear(double k1, double k2)
{
  float tmpP1i = p1.i + p1.j*k2;
  float tmpP1j = p1.i*k1 + p1.j;

  float tmpP2i = p2.i + p2.j*k2;
  float tmpP2j = p2.i*k1 + p2.j;

  p1.i = tmpP1i;
  p1.j = tmpP1j;
  p2.i = tmpP2i;
  p2.j = tmpP2j;
}

// ######################################################################
void Line::trans(Point2D<float> p)
{
  p1 += p;
  p2 += p;

}

// ######################################################################
void Line::quantize(int numDirections)
{
  Point2D<float> center = getCenter();
  trans(center*-1);

  double ori = getOri();
  itsDirectionIdx =  floor ((ori  *numDirections) / (M_PI+1e-5));
  double dOri = (((itsDirectionIdx)*M_PI)/numDirections + M_PI/(2*numDirections)) - ori;
  rotate(dOri);
  trans(center);
}

void Line::setQuantizedDir(int numDirections)
{
  double ori = getOri();
  itsDirectionIdx = (int) floor ((ori  *numDirections) / (M_PI+1e-5));
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

