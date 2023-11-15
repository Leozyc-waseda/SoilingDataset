/*!@file SceneUnderstanding/Camera.C  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/Camera.C $
// $Id: Camera.C 13413 2010-05-15 21:00:11Z itti $
//

#ifndef Camera_C_DEFINED
#define Camera_C_DEFINED

#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "plugins/SceneUnderstanding/Camera.H"

// ######################################################################
Camera::Camera(Point3D<float> location,
    Point3D<float> orientation, float focalRatio,
    float width, float height) :
  itsLocation(location),
  itsOrientation(orientation),
  itsFocalRatio(focalRatio),
  itsWidth(width),
  itsHeight(height)
{
}

// ######################################################################
Camera::~Camera()
{

}

// ######################################################################
Point2D<float> Camera::project(Point3D<float> point)
{
  float dx = cos(itsOrientation.y) * (sin(itsOrientation.z)*(point.y-itsLocation.y) +
                                      cos(itsOrientation.z)*(point.x-itsLocation.x)) -
             sin(itsOrientation.y)*(point.z-itsLocation.z);
  float dy = sin(itsOrientation.x) * (cos(itsOrientation.y)*(point.z-itsLocation.z) +
                                      sin(itsOrientation.y)*(sin(itsOrientation.z)*(point.y-itsLocation.y) +
                                                             cos(itsOrientation.z)*(point.x-itsLocation.x))) +
             cos(itsOrientation.x)*(cos(itsOrientation.z)*(point.y-itsLocation.y) -
                                    sin(itsOrientation.z)*(point.x-itsLocation.x));
  float dz = cos(itsOrientation.x) * (cos(itsOrientation.y)*(point.z-itsLocation.z) +
                                      sin(itsOrientation.y)*(sin(itsOrientation.z)*(point.y-itsLocation.y) +
                                                             cos(itsOrientation.z)*(point.x-itsLocation.x))) -
             sin(itsOrientation.x)*(cos(itsOrientation.z)*(point.y-itsLocation.y) -
                                    sin(itsOrientation.z)*(point.x-itsLocation.x));

  //Project the point onto the 2d camera plane
  Point2D<float> projPoint;

  projPoint.i = (itsWidth/2.0F) + dx * itsFocalRatio/dz;
  projPoint.j = (itsHeight/2.0F) + dy * itsFocalRatio/dz;

  return projPoint;
}

Point3D<float> Camera::inverseProjection(const Point2D<float> point,
    const float dist)
{

  Point3D<float> ip(0,0,0);

  float dz = dist;
  float dx = ((float)point.i - (itsWidth/2.0F))/(itsFocalRatio/dz);
  float dy = ((float)point.j - (itsHeight/2.0F))/(itsFocalRatio/dz);

  ip.x = dx+itsLocation.x;
  ip.y = dy+itsLocation.y;
  ip.z = dz+itsLocation.z;

  return ip;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

