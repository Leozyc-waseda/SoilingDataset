/*!@file BeoSub/BeoSubSensor.H Basic sensor encapsulation for the BeoSub */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubSensor.H $
// $Id: BeoSubSensor.H 5709 2005-10-13 07:53:05Z ilab16 $
//

#ifndef BEOSUBSENSOR_H_DEFINED
#define BEOSUBSENSOR_H_DEFINED

#include "Util/Angle.H"
#include "Util/Types.H"
#include <deque>
#include <pthread.h>

//! Definition and access functions for a BeoSub sensor
/*! This class encapsulates a basic BeoSub sensor. Sensed data
  consists of streams of T (the template argument) values.
  Measurements received (e.g., through BeoChip events) are passed to
  the BeoSubSensor via the newMeasurement() function. Current filtered
  value is available via the getValue() function. The class is
  thread-safe. */
template <class T> class BeoSubSensor
{
public:
  //! Constructor
  /* Build a new sensor.
     @param qlen length of the internal queue used to store the received
     data and compute the filtered sensor value.
     @param decay exponential decay by which older values contribute to the
     returned average measurement. */
  BeoSubSensor(const uint qlen = 6, const double decay = 1.0);

  //! Destructor
  ~BeoSubSensor();

  //! Receive a new measurement value
  void newMeasurement(const T& val);

  //! Get current filtered value
  T getValue() const;

  //! Clear all past measurements
  void reset();

  //! Check whether received values are well behaved
  bool check() const;

private:
  std::deque<T> itsQ;
  uint itsQlen;
  bool itsCacheValid;
  T itsCachedValue;
  double itsDecay;
  pthread_mutex_t itsMutex;
};

//! Helper function: compute averaged sensor measurements
template <class T>
T averageBeoSubSensorValue(const std::deque<T> data,
                           const double factor = 1.0);

//! Helper function: compute averaged sensor measurements, Angle version
Angle averageBeoSubSensorValue(std::deque<Angle> data,
                               const double factor = 1.0);

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
