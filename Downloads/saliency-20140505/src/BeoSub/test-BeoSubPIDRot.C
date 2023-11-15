/*!@file BeoSub/test-BeoSubPIDRot.C test submarine ballasts */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoSubPIDRot.C $
// $Id: test-BeoSubPIDRot.C 6990 2006-08-11 18:13:51Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Devices/BeoChip.H"
#include "BeoSub/BeoSubIMU.H"
#include "BeoSub/BeoSubBallast.H"
#include "Controllers/PID.H"

#include <cstdlib>
#include <iostream>
#include <pthread.h>

nub::soft_ref<BeoChip> bc_ptr;
PID<float> *pid;

float cvel;
float cdelta;
bool start = true;

/* IMU is z-down */
void thrust(const float vel, const float delta) {
  /* thrust should be z-down as well */
  float left = vel+delta;
  if(left > 1) left = 1;
  else if(left < -1) left = -1;
  float right = vel-delta;
  if(right > 1) right = 1;
  else if(right < -1) right = -1;
  bc_ptr->setServo(3, left);
  bc_ptr->setServo(2, right);
  //bc_ptr->setServo(3, -1);
  //bc_ptr->setServo(2, 1);
}

// here a listener for the compass:
class IMUListener : public BeoSubIMUListener {
public:
  //! Destructor
  virtual ~IMUListener() {};

  //! New data was received
  virtual void newData(const float xa, const float ya, const float za,
                       const Angle xv, const Angle yv, const Angle zv)
  {
          float current = zv.getVal();
          // feed the PID controller:
          cdelta  = pid->update(0, current);
          LINFO("%f %f", current, cdelta);

          if(start) {
                  //thrust(0, 1);
                  thrust(cvel, cdelta);
          }
  }
};

//! This program provides basic testing of a BeoSubBallast
int main(const int argc, const char* argv[])
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("BeoSubPIDRot test program");

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeoChip> bc(new BeoChip(manager));
  manager.addSubComponent(bc);
  bc_ptr = bc;

  // instantiate our model components:
  nub::soft_ref<BeoSubIMU> imu(new BeoSubIMU(manager));
  manager.addSubComponent(imu);

  //pid = new PID<float>(0.1, 0.001, 0, -0.5, 0.5);
  pid = new PID<float>(0.5, 0.001, 0, -500, 500);
  //pid = new PID<float>(0.5, 0.1, 0, -0.5, 0.5);

  if (manager.parseCommandLine(argc, argv, "<BeoChip> <IMU>", 2, 2) == false)
    return(1);

  // let's configure our serial device:
  bc->setModelParamVal("BeoChipDeviceName", manager.getExtraArg(0));
  bc->setModelParamVal("BeoChipUseRTSCTS", false);
  imu->setModelParamVal("IMUSerialPortDevName", manager.getExtraArg(1), MC_RECURSE);

  rutz::shared_ptr<IMUListener> lis(new IMUListener);
  rutz::shared_ptr<BeoSubIMUListener> lis2; lis2.dynCastFrom(lis); // cast down
  imu->setListener(lis2);

  // let's get all our ModelComponent instances started:
  manager.start();

  LINFO("Reseting the BeoChip...");
  bc->reset(MC_RECURSE);
  sleep(1);

  LINFO("Waiting for a bit. Turn BeoChip on if not already on."); sleep(1);
  LINFO("Echo request (should bring an echo reply back)...");
  bc->echoRequest(); sleep(1);

  cvel = 0.8;
  char c;
  std::cin >> c;
  start = false;


  manager.stop();
  thrust(0, 0);
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
