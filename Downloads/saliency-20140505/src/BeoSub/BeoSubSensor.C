/*!@file BeoSub/BeoSubSensor.C Encapsulation of a sensor for the BeoSub */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubSensor.C $
// $Id: BeoSubSensor.C 5709 2005-10-13 07:53:05Z ilab16 $
//

#include "BeoSub/BeoSubSensor.H"
#include "Util/log.H"
#include <algorithm>
#include <cmath>

// ######################################################################
template <class T>
BeoSubSensor<T>::BeoSubSensor(const uint qlen, const double decay) :
  itsQ(), itsQlen(qlen), itsCacheValid(false), itsDecay(decay)
{
  pthread_mutex_init(&itsMutex, NULL);
}

// ######################################################################
template <class T>
BeoSubSensor<T>::~BeoSubSensor()
{
  pthread_mutex_destroy(&itsMutex);
}

// ######################################################################
template <class T>
void BeoSubSensor<T>::newMeasurement(const T& val)
{
  pthread_mutex_lock(&itsMutex);
  itsCacheValid = false;
  itsQ.push_front(val);
  while (itsQ.size() > itsQlen) itsQ.pop_back();
  pthread_mutex_unlock(&itsMutex);
}

// ######################################################################
template <class T>
T BeoSubSensor<T>::getValue() const
{
  // compute the average of the data:
  T val;
  pthread_mutex_lock(const_cast<pthread_mutex_t *>(&itsMutex));
  if (itsCacheValid)
    val = itsCachedValue;
  else
    {
      val = averageBeoSubSensorValue(itsQ, itsDecay);
      (const_cast<BeoSubSensor *>(this))->itsCachedValue = val;
      (const_cast<BeoSubSensor *>(this))->itsCacheValid = true;
    }
  pthread_mutex_unlock(const_cast<pthread_mutex_t *>(&itsMutex));
  return val;
}

// ######################################################################
template <class T>
void BeoSubSensor<T>::reset()
{
  pthread_mutex_lock(&itsMutex);
  itsQ.clear();
  itsCacheValid = false;
  pthread_mutex_unlock(&itsMutex);
}

// ######################################################################
template <class T>
bool BeoSubSensor<T>::check() const
{

  // FIXME: will need to think about that now that the class is templatized

  /*
  // get min, max and average:
  float mi = 1.0e30F, ma = -1.0e30F, avg = 0.0F; int n = 0;
  pthread_mutex_lock(&itsMutex);
  for (std::deque<float>::const_iterator i = itsQ.begin(); i < itsQ.end(); i++)
    {
      float val = *i;
      if (val < mi) mi = val;
      if (val > ma) ma = val;
      avg += val;
      ++ n;
    }
  pthread_mutex_unlock(&itsMutex);

  // if we did not accumulate any data, we are not ok!
  if (n == 0) { LINFO("No data -- RETURNING FALSE"); return false; }

  // compare (max-min) to avg:
  float diff = fabs(ma - mi); avg = fabs(avg) / float(n);
  if (diff > 0.25 * avg)
    {
      LINFO("max-min=%f, avg=%f -- RETURNING FALSE", diff, avg);
      return false;
    }
  */
  // everything seems okay:
  return true;
}

// ######################################################################
template <class T>
T averageBeoSubSensorValue(const std::deque<T> data, const double factor)
{
  T val(0); double fac = 1.0;
  for (typename std::deque<T>::const_iterator
         i = data.begin(); i < data.end(); ++i)
    {
      val += (*i) * fac;
      fac *= factor;
    }
  if (data.size()) val /= data.size();
  return val;
}

// ######################################################################
Angle averageBeoSubSensorValue(std::deque<Angle> data, const double factor)
{ return averageVectorAngle(data.begin(), data.end(), factor); }


// Instantiations:
template class BeoSubSensor<float>;
template class BeoSubSensor<double>;
template class BeoSubSensor<Angle>;

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
