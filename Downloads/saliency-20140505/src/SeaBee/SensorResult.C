/*!@file SeaBee/SensorResult.C information of objects to find   */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/SensorResult.C $
// $Id: SensorResult.C 10794 2009-02-08 06:21:09Z itti $
//

#include "SensorResult.H"

// ######################################################################
// Constructors
SensorResult::SensorResult()
{
  sensorResultMutex = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(sensorResultMutex,NULL);

  itsStatus = NOT_FOUND;
  itsPosition = Point3D(-1,-1,-1);
  itsOrientation = -1.0;
  itsFrequency = -1.0;
  itsDistance = -1.0;
  itsMass = -1.0;
  itsFrameNumber = 0;
  itsTimer.reset(new Timer(1000000));
}

// ######################################################################
SensorResult::SensorResult(SensorResultType type)
{

  sensorResultMutex = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(sensorResultMutex,NULL);

  itsType = type;
  itsStatus = NOT_FOUND;
  itsPosition = Point3D(-1,-1,-1);
  itsOrientation = -1.0;
  itsFrequency = -1.0;
  itsDistance = -1.0;
  itsMass = -1.0;
  itsFrameNumber = 0;
  itsTimer.reset(new Timer(1000000));
}


// ######################################################################
void SensorResult::copySensorResult(SensorResult sr)
{
  pthread_mutex_lock(sensorResultMutex);
  itsPosition = sr.getPosition();
  itsOrientation = sr.getOrientation();
  itsFrequency = sr.getFrequency();
  itsDistance = sr.getDistance();
  itsMass = sr.getMass();
  itsType = sr.getType();
  itsStatus = sr.getStatus();
  itsTimer->reset();
  pthread_mutex_unlock(sensorResultMutex);
}


// ######################################################################
void SensorResult::startTimer()
{
  itsTimer.reset(new Timer(1000000));
}

// ######################################################################
// Accessor methods
void SensorResult::setStatus(SensorResultStatus s)
{
  pthread_mutex_lock(sensorResultMutex);
  itsStatus = s;
  pthread_mutex_unlock(sensorResultMutex);
}

// ######################################################################
void SensorResult::setType(SensorResultType t)
{
  pthread_mutex_lock(sensorResultMutex);
  itsType = t;
  pthread_mutex_unlock(sensorResultMutex);
}

// ######################################################################
void SensorResult::setPosition(Point3D pos)
{
  pthread_mutex_lock(sensorResultMutex);
  itsPosition = pos;
  itsTimer->reset();
  pthread_mutex_unlock(sensorResultMutex);
}

// ######################################################################
void SensorResult::setOrientation(Angle ori)
{
  pthread_mutex_lock(sensorResultMutex);
  itsOrientation = ori;
  itsTimer->reset();
  pthread_mutex_unlock(sensorResultMutex);
}

// ######################################################################
void SensorResult::setFrequency(float freq)
{
  pthread_mutex_lock(sensorResultMutex);
  itsFrequency = freq;
  itsTimer->reset();
  pthread_mutex_unlock(sensorResultMutex);
}

// ######################################################################
void SensorResult::setDistance(float dist)
{
  pthread_mutex_lock(sensorResultMutex);
  itsDistance = dist;
  itsTimer->reset();
  pthread_mutex_unlock(sensorResultMutex);
}

// ######################################################################
void SensorResult::setMass(float mass)
{
  pthread_mutex_lock(sensorResultMutex);
  itsMass = mass;
  itsTimer->reset();
  pthread_mutex_unlock(sensorResultMutex);
}

// ######################################################################
void SensorResult::setId(uint id)
{
  pthread_mutex_lock(sensorResultMutex);
  itsId = id;
  itsTimer->reset();
  pthread_mutex_unlock(sensorResultMutex);
}

// ######################################################################
void SensorResult::setFrameNum(uint fnum)
{
  pthread_mutex_lock(sensorResultMutex);
  itsFrameNumber = fnum;
  itsTimer->reset();
  pthread_mutex_unlock(sensorResultMutex);
}

// ######################################################################
Point3D SensorResult::getPosition()
{
  pthread_mutex_lock(sensorResultMutex);
  Point3D p = itsPosition;
  pthread_mutex_unlock(sensorResultMutex);
  return p;
}

// ######################################################################
rutz::shared_ptr<Point3D> SensorResult::getPositionPtr()
{
  pthread_mutex_lock(sensorResultMutex);
  rutz::shared_ptr<Point3D> p (new Point3D(itsPosition));
  pthread_mutex_unlock(sensorResultMutex);
  return p;
}

// ######################################################################
Angle SensorResult::getOrientation()
{
  pthread_mutex_lock(sensorResultMutex);
  Angle o = itsOrientation;
  pthread_mutex_unlock(sensorResultMutex);
  return o;
}

// ######################################################################
rutz::shared_ptr<Angle> SensorResult::getOrientationPtr()
{
  pthread_mutex_lock(sensorResultMutex);
  rutz::shared_ptr<Angle> o (new Angle(itsOrientation));
  pthread_mutex_unlock(sensorResultMutex);
  return o;
}


// ######################################################################
float SensorResult::getFrequency()
{
  pthread_mutex_lock(sensorResultMutex);
  float f = itsFrequency;
  pthread_mutex_unlock(sensorResultMutex);
  return f;
}

// ######################################################################
float SensorResult::getDistance()
{
  pthread_mutex_lock(sensorResultMutex);
  float d = itsDistance;
  pthread_mutex_unlock(sensorResultMutex);
  return d;
}

// ######################################################################
float SensorResult::getMass()
{
  pthread_mutex_lock(sensorResultMutex);
  float m = itsMass;
  pthread_mutex_unlock(sensorResultMutex);
  return m;
}

// ######################################################################
uint SensorResult::getId()
{
  pthread_mutex_lock(sensorResultMutex);
  uint id = itsId;
  pthread_mutex_unlock(sensorResultMutex);
  return id;
}

// ######################################################################
uint SensorResult::getFrameNum()
{
  pthread_mutex_lock(sensorResultMutex);
  uint fnum = itsFrameNumber;
  pthread_mutex_unlock(sensorResultMutex);
  return fnum;
}

// ######################################################################
SensorResult::SensorResultStatus SensorResult::getStatus()
{
  pthread_mutex_lock(sensorResultMutex);
  SensorResultStatus currentStatus = itsStatus;
  int lastUpdate = itsTimer->get();

  if(lastUpdate > NOT_FOUND_TIME)
    {
      if(currentStatus == SensorResult::FOUND)
        currentStatus = SensorResult::LOST;
      else
        currentStatus = SensorResult::NOT_FOUND;
    }

  pthread_mutex_unlock(sensorResultMutex);
  return currentStatus;
}

// ######################################################################
SensorResult::SensorResultType SensorResult::getType()
{
  pthread_mutex_lock(sensorResultMutex);
  //  int lastAccess = itsTimer->get();
  //LINFO("LastAccess:%d",lastAccess);
  SensorResultType currentType = itsType;
  pthread_mutex_unlock(sensorResultMutex);
  return currentType;
}

// ######################################################################
bool SensorResult::downwardCoordsOk()
{
  pthread_mutex_lock(sensorResultMutex);
  bool isValid = (itsPosition.isValidZ() &&
                  itsPosition.isValidX());
  pthread_mutex_unlock(sensorResultMutex);

  return isValid;
}

// ######################################################################
bool SensorResult::forwardCoordsOk()
{
  pthread_mutex_lock(sensorResultMutex);
  bool isValid = (itsPosition.isValidY() &&
                  itsPosition.isValidX());
  pthread_mutex_unlock(sensorResultMutex);

  return isValid;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
