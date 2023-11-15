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

#ifndef POLYGON_C_DEFINED
#define POLYGON_C_DEFINED

#include "FeatureMatching/Polygon.H"
#include "Image/DrawOps.H"
#include "GUI/DebugWin.H"
#include "Util/FastMathFunctions.H"

void Polygon::addLine(const Line& line)
{
  itsLines.push_back(line);
}

// ######################################################################
void Polygon::quantize(int nDirections)
{
  itsNumDirections = nDirections;
  for(uint i=0; i<itsLines.size(); i++)
    itsLines[i].quantize(nDirections);
}

// ######################################################################
void Polygon::rotate(double theta)
{
  for(uint i=0; i<itsLines.size(); i++)
    itsLines[i].rotate(theta);
}

// ######################################################################
void Polygon::shear(double k1, double k2)
{
  for(uint i=0; i<itsLines.size(); i++)
    itsLines[i].shear(k1,k2);
}

// ######################################################################
void Polygon::scale(float s)
{
  itsLength = 0;
  for(uint i=0; i<itsLines.size(); i++)
  {
    itsLines[i].scale(s);
    itsLines[i].setQuantizedDir(itsNumDirections);
    itsLength += itsLines[i].getLength();
  }

}

// ######################################################################
void Polygon::scale(float sx, float sy)
{
  itsLength = 0;
  for(uint i=0; i<itsLines.size(); i++)
  {
    itsLines[i].scale(sx, sy);
    itsLines[i].setQuantizedDir(itsNumDirections);
    itsLength += itsLines[i].getLength();
  }
}

// ######################################################################
void Polygon::setCOM()
{
  Point2D<float> center(0,0);
  for (uint i=0 ; i<itsLines.size(); i++)
    center += itsLines[i].getCenter();
  center /= itsLines.size();

  center *= -1; //Shift by center

  trans(center);
}

// ######################################################################
void Polygon::trans(Point2D<float> p)
{
  for(uint i=0; i<itsLines.size(); i++)
    itsLines[i].trans(p);
}

// ######################################################################
void Polygon::setWeights()
{
  for(uint i=0; i<itsLines.size(); i++)
    itsLines[i].setWeight(itsLines[i].getLength()/itsLength);
}

// ######################################################################
void Polygon::getBoundary(double &minx, double &miny, double &maxx, double &maxy)
{
  minx = miny = 1e+10;
  maxx = maxy = -1e+10;

  for (uint i=0 ; i<itsLines.size() ; i++)
  {
    Point2D<float> p1 = itsLines[i].getP1();
    Point2D<float> p2 = itsLines[i].getP2();

    if (minx > p1.i) minx = p1.i;
    if (minx > p2.i) minx = p2.i;

    if (maxx < p1.i) maxx = p1.i;
    if (maxx < p2.i) maxx = p2.i;

    if (miny > p1.j) miny = p1.j;
    if (miny > p2.j) miny = p2.j;

    if (maxy < p1.j) maxy = p1.j;
    if (maxy < p2.j) maxy = p2.j;
  }	
}


// ######################################################################
double Polygon::getRadius()
{
  double radius = 0;
  Point2D<float> center(0,0);
  for(uint i=0; i<itsLines.size(); i++)
  {
    double dist = center.distance(itsLines[i].getP1());
    if (dist > radius)
      radius = dist;

    dist = center.distance(itsLines[i].getP2());
    if (dist > radius)
      radius = dist;
  }
  return radius;
}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

