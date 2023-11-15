/*!@file BeoSub/test-BeoSubIMU.C test the IMU */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoSubIMU.C $
// $Id: test-BeoSubIMU.C 6990 2006-08-11 18:13:51Z rjpeters $
//

#include "BeoSub/BeoSubIMU.H"
#include "Component/ModelManager.H"

//! A hook which will be called when a new IMU reading is received
class TestBeoSubIMUListener : public BeoSubIMUListener {
public:
  //! Destructor
  virtual ~TestBeoSubIMUListener() {}

  //! New data was received
  virtual void newData(const float xa, const float ya, const float za,
                       const Angle xv, const Angle yv, const Angle zv)
  {
    LINFO("<X-Accel=%8.4f, Y-Accel=%8.4f, Z-Accel=%8.4f, X-Vel=%8.4f, Y-Vel=%8.4f, Z-Vel=%8.4f>",
          xa, ya, za, xv.getVal(), yv.getVal(), zv.getVal());
  }
};

int main(const int argc, const char **argv)
{
  // get a manager going:
  ModelManager manager("IMU Manager");

  // instantiate our model components:
  nub::soft_ref<BeoSubIMU> imu(new BeoSubIMU(manager));
  manager.addSubComponent(imu);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<serdev>", 1, 1) == false)
    return(1);

  // let's configure our serial device:
  imu->setModelParamVal("IMUSerialPortDevName",
                        manager.getExtraArg(0), MC_RECURSE);

  // let's register our listener:
  rutz::shared_ptr<TestBeoSubIMUListener> lis(new TestBeoSubIMUListener);
  rutz::shared_ptr<BeoSubIMUListener> lis2; lis2.dynCastFrom(lis); // cast down
  imu->setListener(lis2);

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
