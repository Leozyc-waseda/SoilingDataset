/*!@file SeaBee/SensorResult.H  information of objects to find  */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/SensorResult.H $
// $Id: SensorResult.H 10794 2009-02-08 06:21:09Z itti $
//

#ifndef __SENSOR_RESULT_H__
#define __SENSOR_RESULT_H__

#include "Point3D.H"
#include "Util/Angle.H"
#include "Util/Timer.H"

#include "rutz/shared_ptr.h"

#include <pthread.h>
#include <string>

#define NOT_FOUND_TIME 3000000 //3 seconds

class SensorResult
{
 public:

  enum SensorResultType { BUOY,
                          PIPE,
                          BIN,
                          CROSS,
                          PINGER,
                          SALIENCY,
                          STEREO };

  enum SensorResultStatus { FOUND, NOT_FOUND, LOST};

  SensorResult();

  SensorResult(SensorResultType type);

  void copySensorResult(SensorResult sr);

  void startTimer();

  void setStatus(SensorResultStatus status);

  void setType(SensorResultType type);

  void setPosition(Point3D pos);

  void setOrientation(Angle ori);

  void setFrequency(float freq);

  void setDistance(float dist);

  void setMass(float mass);

  void setId(uint id);

  void setFrameNum(uint fnum);

  Point3D getPosition();

  rutz::shared_ptr<Point3D> getPositionPtr();

  Angle getOrientation();

  rutz::shared_ptr<Angle> getOrientationPtr();

  float getFrequency();

  float getDistance();

  float getMass();

  uint getId();

  uint getFrameNum();

  SensorResultType getType();

  SensorResultStatus getStatus();

  //checks if position in downward (Z,X) plane is valid
  bool downwardCoordsOk();

  //checks if position in forward (Y,X) plane is valid
  bool forwardCoordsOk();

 private:

  pthread_mutex_t* sensorResultMutex;

  enum SensorResultType itsType;

  enum SensorResultStatus itsStatus;

  Point3D itsPosition;

  Angle itsOrientation;
  float itsFrequency;
  float itsDistance;
  float itsMass;

  uint itsId;

  uint itsFrameNumber;

  rutz::shared_ptr<Timer> itsTimer;

};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
