/*!@file BeoSub/BeeBrain/OceanObject.C information of objects to find   */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/OceanObject.C $
// $Id: OceanObject.C 9057 2007-11-28 04:29:48Z beobot $
//

#include "OceanObject.H"

// ######################################################################
// Constructors
OceanObject::OceanObject()
{
  oceanObjectMutex = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(oceanObjectMutex,NULL);

  itsPosition = Point3D(-1,-1,-1);
  itsOrientation = 0.0;
  itsFrequency = 0.0;
  itsDistance = 0.0;
  itsMass = 0.0;
}

// ######################################################################
OceanObject::OceanObject(OceanObjectType type)
{

  oceanObjectMutex = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(oceanObjectMutex,NULL);

  itsType = type;
  itsPosition = Point3D(-1,-1,-1);
  itsOrientation = 0.0;
  itsFrequency = 0.0;
  itsDistance = 0.0;
  itsMass = 0.0;
}


// ######################################################################
// Accessor methods
void OceanObject::setStatus(OceanObjectStatus s)
{
  pthread_mutex_lock(oceanObjectMutex);
  itsStatus = s;
  pthread_mutex_unlock(oceanObjectMutex);
}

// ######################################################################
void OceanObject::setType(OceanObjectType t)
{
  pthread_mutex_lock(oceanObjectMutex);
  itsType = t;
  pthread_mutex_unlock(oceanObjectMutex);
}

// ######################################################################
void OceanObject::setPosition(Point3D pos)
{
  pthread_mutex_lock(oceanObjectMutex);
  itsPosition = pos;
  pthread_mutex_unlock(oceanObjectMutex);
}

// ######################################################################
void OceanObject::setOrientation(Angle ori)
{
  pthread_mutex_lock(oceanObjectMutex);
  itsOrientation = ori;
  pthread_mutex_unlock(oceanObjectMutex);
}

// ######################################################################
void OceanObject::setFrequency(float freq)
{
  pthread_mutex_lock(oceanObjectMutex);
  itsFrequency = freq;
  pthread_mutex_unlock(oceanObjectMutex);
}

// ######################################################################
void OceanObject::setDistance(float dist)
{
  pthread_mutex_lock(oceanObjectMutex);
  itsDistance = dist;
  pthread_mutex_unlock(oceanObjectMutex);
}

// ######################################################################
void OceanObject::setMass(float mass)
{
  pthread_mutex_lock(oceanObjectMutex);
  itsMass = mass;
  pthread_mutex_unlock(oceanObjectMutex);
}

// ######################################################################
void OceanObject::setId(uint id)
{
  pthread_mutex_lock(oceanObjectMutex);
  itsId = id;
  pthread_mutex_unlock(oceanObjectMutex);
}

// ######################################################################
Point3D OceanObject::getPosition()
{
  pthread_mutex_lock(oceanObjectMutex);
  Point3D p = itsPosition;
  pthread_mutex_unlock(oceanObjectMutex);
  return p;
}

// ######################################################################
rutz::shared_ptr<Point3D> OceanObject::getPositionPtr()
{
  pthread_mutex_lock(oceanObjectMutex);
  rutz::shared_ptr<Point3D> p (new Point3D(itsPosition));
  pthread_mutex_unlock(oceanObjectMutex);
  return p;
}

// ######################################################################
Angle OceanObject::getOrientation()
{
  pthread_mutex_lock(oceanObjectMutex);
  Angle o = itsOrientation;
  pthread_mutex_unlock(oceanObjectMutex);
  return o;
}

// ######################################################################
rutz::shared_ptr<Angle> OceanObject::getOrientationPtr()
{
  pthread_mutex_lock(oceanObjectMutex);
  rutz::shared_ptr<Angle> o (new Angle(itsOrientation));
  pthread_mutex_unlock(oceanObjectMutex);
  return o;
}


// ######################################################################
float OceanObject::getFrequency()
{
  pthread_mutex_lock(oceanObjectMutex);
  float f = itsFrequency;
  pthread_mutex_unlock(oceanObjectMutex);
  return f;
}

// ######################################################################
float OceanObject::getDistance()
{
  pthread_mutex_lock(oceanObjectMutex);
  float d = itsDistance;
  pthread_mutex_unlock(oceanObjectMutex);
  return d;
}

// ######################################################################
float OceanObject::getMass()
{
  pthread_mutex_lock(oceanObjectMutex);
  float m = itsMass;
  pthread_mutex_unlock(oceanObjectMutex);
  return m;
}

// ######################################################################
uint OceanObject::getId()
{
  pthread_mutex_lock(oceanObjectMutex);
  uint id = itsId;
  pthread_mutex_unlock(oceanObjectMutex);
  return id;
}

// ######################################################################
OceanObject::OceanObjectStatus OceanObject::getStatus()
{
  pthread_mutex_lock(oceanObjectMutex);
  OceanObjectStatus currentStatus = itsStatus;
  pthread_mutex_unlock(oceanObjectMutex);
  return currentStatus;
}

// ######################################################################
OceanObject::OceanObjectType OceanObject::getType()
{
  pthread_mutex_lock(oceanObjectMutex);
  OceanObjectType currentType = itsType;
  pthread_mutex_unlock(oceanObjectMutex);
  return currentType;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
