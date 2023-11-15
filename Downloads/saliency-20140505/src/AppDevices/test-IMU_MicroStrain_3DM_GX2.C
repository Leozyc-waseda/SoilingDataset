/*!@file AppDevices/test-IMU_MicroStrain_3DM_GX2.C test the
  MicroStrain 3DM_GX2 IMU */

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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-IMU_MicroStrain_3DM_GX2.C $
// $Id: test-IMU_MicroStrain_3DM_GX2.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Devices/IMU_MicroStrain_3DM_GX2.H"
#include "Component/ModelManager.H"

int main(const int argc, const char **argv)
{
  // get a manager going:
  ModelManager manager("IMU Manager");

  // instantiate our model components:
  nub::soft_ref<IMU_MicroStrain_3DM_GX2>
    imu(new IMU_MicroStrain_3DM_GX2(manager));
  manager.addSubComponent(imu);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<serdev> [command]", 1, 1) == false)
    return(1);

  // let's configure our serial device:
  imu->configureSerial(manager.getExtraArg(0));
  LINFO("Using: %s", manager.getExtraArg(0).c_str());

  // get started:
  manager.start();

  // set data requested to be roll, pitch, and yaw
//   imu->setDataRequested(ACCEL_AND_ANG_RATE);
//   imu->setDataRequested(MAGNETOMETER);
  imu->setDataRequested(ROLL_PITCH_YAW);

  // this is completely event driven, so here we just sleep. When data
  // is received, it will trigger our listener:
  while (1)
    {
      if(imu->newData())
        {
//           // get Acceleration and Angular Rate
//           AccelAndAngRateRecord record;
//           imu->getAccelerationAndAngularRate(record);
//           LINFO("Acceleration x:%15.6f y:%15.6f z:%15.6f",
//                 record.accelX, record.accelY, record.accelZ);
//           LINFO("Angular Rate x:%15.6f y:%15.6f z:%15.6f",
//                 record.angRateX , record.angRateY, record.angRateZ);

//           // get magnetometer direction and magnitude
//           MagnetometerRecord mRecord;
//           imu->getMagnetometer(mRecord);
//           LINFO("Magnetometer x:%15.6f y:%15.6f z:%15.6f",
//                 mRecord.magX, mRecord.magY, mRecord.magZ);

          // get roll, pitch, and yaw
          RollPitchYawRecord rpyRecord;
          imu->getRollPitchYaw(rpyRecord);
          LINFO("Euler Angle  r:%15.6f p:%15.6f y:%15.6f",
                rpyRecord.roll, rpyRecord.pitch, rpyRecord.yaw);
        }
      else
        usleep(1000);
    }

  // stop everything and exit:
  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
