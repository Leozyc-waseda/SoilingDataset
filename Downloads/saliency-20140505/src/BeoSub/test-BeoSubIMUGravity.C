/*!@file BeoSub/test-BeoSubIMUGravity.C test the IMU */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoSubIMUGravity.C $
// $Id: test-BeoSubIMUGravity.C 7880 2007-02-09 02:34:07Z itti $
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

class TestHMR3300Listener : public HMR3300Listener {
public:
  //! Destructor
  virtual ~TestHMR3300Listener() {};

  //! New data was received
  virtual void newData(const Angle heading, const Angle pitch,
                       const Angle roll)
  {

    //The following code is meant to be run upon initialization. It assumes that the sub is on a level surfac and then calculates the error due to mounting of the compass.
    if(firstTime){
      hError = heading; pError = pitch; rError = roll;
      firstTime = false;
    }
    cHeading = heading; cPitch = pitch; cRoll = roll;

    realPitch = (cPitch - pError);
    realRoll = (cRoll - rError);

    //NOTE: The gravity constant may need to be more precise AND may need to be negative!
    realXAccel = (xAccel - (sin(realPitch.getVal())*-9.82) - (sin(realRoll.getVal())*-9.82));
    realZAccel = (zAccel - (cos(realPitch.getVal())*-9.82) - (cos(realRoll.getVal())*-9.82));

    printf("X-Accel = %f, Y-Accel = %f, Z-Accel = %f, X-Vel = %f, Y-Vel = %f, Z-Vel = %f, Heading = %f Pitch = %f Roll = %f\n\n",
         xAccel, yAccel, zAccel, xVel.getVal(), yVel.getVal(), zVel.getVal(),
          cHeading.getVal(), realPitch.getVal(), realRoll.getVal());

    //Re-calculate accelerations based on compass reading and gravity and output again
    printf("REAL xAccel = %f, REAL zAccel = %f\n\n", realXAccel, realZAccel);
    //printf("xAccel = %f, zAccel = %f\n", xAccel, zAccel);

    //printf("heading = %f, pitch = %f, roll = %f\n", cHeading.getVal(), realPitch.getVal(), realRoll.getVal());
  }
};


//! A hook which will be called when a new IMU reading is received
class TestBeoSubIMUListener : public BeoSubIMUListener {
public:
  //! Destructor
  virtual ~TestBeoSubIMUListener() {}

  //! New data was received
  virtual void newData(const float xa, const float ya, const float za,
                       const Angle xv, const Angle yv, const Angle zv)
  {
    xAccel = xa; yAccel = ya; zAccel = za;
    xVel = xv; yVel = yv; zVel = zv;
    //printf("XBOOGLY: %f\n", xAccel);
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
