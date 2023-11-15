/*!@file BeoSub/BeoSubIMU.H class for interfacing with the IMU */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubIMU.H $
// $Id: BeoSubIMU.H 11659 2009-09-05 01:27:31Z siagian $
//

#ifndef BEOSUBIMU_H_DEFINED
#define BEOSUBIMU_H_DEFINED

#include "Component/OptionManager.H"
#include "Devices/Serial.H"
#include "Util/Angle.H"
#include <pthread.h>

//! A hook which will be called when a new IMU reading is received
class BeoSubIMUListener {
public:
  //! Destructor
  virtual ~BeoSubIMUListener();

  //! New data was received
  virtual void newData(const float xa, const float ya, const float za,
                       const Angle xv, const Angle yv, const Angle zv) = 0;
};

//! Class for interfacing with the IMU
/*! This is an IMU. It sends X,Y,Z acceleration and X,Y,Z angular velocity*/
class BeoSubIMU: public ModelComponent
{
public:
  //! Initialize
  BeoSubIMU(OptionManager& mgr,
            const std::string& descrName = "IMU",
            const std::string& tagName = "IMU",
            const char *dev = "/dev/ttyS0");

  //! Destructor
  ~BeoSubIMU();

  //! Setup a listener that will be called each time new data is received
  void setListener(rutz::shared_ptr<BeoSubIMUListener> listener);

  //! Get the current X acceleration
  float getXaccel();

  //! Get the current Y acceleration
  float getYaccel();

  //! Get the current Z acceleration
  float getZaccel();

  //! Get the current X velocity
  Angle getXvel();

  //! Get the current Y velocity
  Angle getYvel();

  //! Get the current Z velocity
  Angle getZvel();

  void run();

protected:
  void start2(); //!< get started
  void stop1(); //!< get stopped
  float accelConvert(int data);
  float rateConvert(int data);

private:
  nub::soft_ref<Serial> itsSerial;
  rutz::shared_ptr<BeoSubIMUListener> itsListener;
  Angle itsXvel, itsYvel, itsZvel;
  float itsXaccel, itsYaccel, itsZaccel;
  bool itsKeepgoing;
  pthread_t itsRunner;
  pthread_mutex_t itsLock;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
