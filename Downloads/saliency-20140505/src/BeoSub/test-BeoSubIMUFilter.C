/*!@file BeoSub/test-BeoSubIMUFilter.C test the IMU */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   //
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoSubIMUFilter.C $
// $Id: test-BeoSubIMUFilter.C 7880 2007-02-09 02:34:07Z itti $
//

#include "BeoSub/BeoSubIMU.H"
#include "Devices/HMR3300.H"
#include "Component/ModelManager.H"
#include <math.h>

//Global vars set by listeners
float xAccel = 0, yAccel = 0, zAccel = 0;
Angle xVel = 0, yVel = 0, zVel = 0, cHeading = 0, cPitch = 0, cRoll = 0;
Angle hError = 0, pError = 0, rError = 0;
Angle realPitch = 0, realRoll = 0;
float realXAccel = 0, realZAccel = 0;

bool firstTime = true;
pthread_mutex_t outLock;

class TestHMR3300Listener : public HMR3300Listener {
public:
  //! Destructor
  virtual ~TestHMR3300Listener() {};

  //! New data was received
  virtual void newData(const Angle heading, const Angle pitch,
                       const Angle roll)
  {
          pthread_mutex_lock(&outLock);
          printf("CH: %3.4f CP: %3.4f CR: %3.4f\n", heading.getVal(), pitch.getVal(), roll.getVal());
          pthread_mutex_unlock(&outLock);
  }
};


#include <cstdio>

//! A hook which will be called when a new IMU reading is received
class TestBeoSubIMUListener : public BeoSubIMUListener {
public:
  //! Destructor
  virtual ~TestBeoSubIMUListener() {}

  //! New data was received
  virtual void newData(const float xa, const float ya, const float za,
                       const Angle xv, const Angle yv, const Angle zv)
  {
          pthread_mutex_lock(&outLock);
          printf("XA: %3.4f YA:%3.4f ZA:%3.4f XV:%3.4f XY:%3.4f XZ: %3.4f\n", xa, xa, xa, xv.getVal(), yv.getVal(), zv.getVal());
          pthread_mutex_unlock(&outLock);
  }
};

int main(const int argc, const char **argv)
{
  // get a manager going:
  ModelManager manager("IMU Manager");

  // instantiate our model components:
  nub::soft_ref<BeoSubIMU> imu(new BeoSubIMU(manager));
  manager.addSubComponent(imu);

  nub::soft_ref<HMR3300> hmr(new HMR3300(manager) );
  manager.addSubComponent(hmr);


  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<serdev>", 2, 2) == false)
    return(1);

  // let's configure our serial devices:
  imu->setModelParamVal("IMUSerialPortDevName",
                        manager.getExtraArg(0), MC_RECURSE);

  hmr->setModelParamVal("HMR3300SerialPortDevName",
                        manager.getExtraArg(1), MC_RECURSE);

  // let's register our listeners:
  rutz::shared_ptr<TestBeoSubIMUListener> lis(new TestBeoSubIMUListener);
  rutz::shared_ptr<BeoSubIMUListener> lis2; lis2.dynCastFrom(lis); // cast down
  imu->setListener(lis2);

  rutz::shared_ptr<TestHMR3300Listener> lisn(new TestHMR3300Listener);
  rutz::shared_ptr<HMR3300Listener> lisn2; lisn2.dynCastFrom(lisn); // cast down
  hmr->setListener(lisn2);

  // initialize output lock
  pthread_mutex_init(&outLock, NULL);

  // get started:
  manager.start();

  // this is completely event driven, so here we just sleep. When data
  // is received, it will trigger our listener:
  while (1) sleep(1000);

  // stop everything and exit:
  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
