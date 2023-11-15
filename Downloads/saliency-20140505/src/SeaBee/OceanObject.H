/*!@file SeaBee/OceanObject.H  information of objects to find  */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/OceanObject.H $
// $Id: OceanObject.H 9272 2008-02-15 20:32:46Z beobot $
//

#ifndef __OCEAN_OBJECT_H__
#define __OCEAN_OBJECT_H__

#include "Point3D.H"
#include "Util/Angle.H"

#include "rutz/shared_ptr.h"

#include <pthread.h>
#include <string>

class OceanObject
{
 public:

  enum OceanObjectType { BUOY, PIPE, BIN, CROSS, PINGER };

  enum OceanObjectStatus { FOUND, NOT_FOUND, LOST};

  OceanObject();

  OceanObject(OceanObjectType type);

  void setStatus(OceanObjectStatus status);

  void setType(OceanObjectType type);

  void setPosition(Point3D pos);

  void setOrientation(Angle ori);

  void setFrequency(float freq);

  void setDistance(float dist);

  void setMass(float mass);

  void setId(uint id);

  Point3D getPosition();

  rutz::shared_ptr<Point3D> getPositionPtr();

  Angle getOrientation();

  rutz::shared_ptr<Angle> getOrientationPtr();

  float getFrequency();

  float getDistance();

  float getMass();

  uint getId();

  OceanObjectType getType();

  OceanObjectStatus getStatus();

 private:

  pthread_mutex_t* oceanObjectMutex;

  enum OceanObjectType itsType;

  enum OceanObjectStatus itsStatus;

  Point3D itsPosition;

  Angle itsOrientation;
  float itsFrequency;
  float itsDistance;
  float itsMass;

  uint itsId;

  uint itsFrameNumber;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
