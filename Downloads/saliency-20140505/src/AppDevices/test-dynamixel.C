/*!@file AppDevices/test-dynamixel.C dynamixel controller  */

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
// Primary maintainer for this file: Chin-Kai Chang <chinkaic@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-dynamixel.C $
// $Id: test-dynamixel.C 8094 2012-11-12 17:11:57Z kai $
//

#include "Component/ModelManager.H"
#include "Devices/Dynamixel.H"
#include "Util/log.H"

#include <stdio.h>

int main(int argc, const char **argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Test Model for Dynamixel Class");

  // Instantiate our various ModelComponents:
  nub::soft_ref<Dynamixel> dynamixel(new Dynamixel(manager));
  manager.addSubComponent(dynamixel);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<servo> <position>[<rpm> <ttyUSB[X]> <baud(1,34)>]", 2, 5) == false) return(1);

  // Let's get some of the option values, to configure our window:
  int deviceID = 0;
  int baud = 34;
  int rpm = 30;

  int sernum = atoi(manager.getExtraArg(0).c_str());
  int serpos = atoi(manager.getExtraArg(1).c_str());
  if(argc >= 4)
    rpm = atoi(manager.getExtraArg(2).c_str());
  if(argc >= 5)
    deviceID = atoi(manager.getExtraArg(3).c_str());
  if(argc >= 6)
    baud   = atoi(manager.getExtraArg(4).c_str());

  // let's get all our ModelComponent instances started:
  manager.start();

  //set the window width/height

  dynamixel->init(baud,deviceID);//baud 200000/(1+34) =~ 57600,ttyUSB0

  int temp = dynamixel->getTemperature(sernum);
  LINFO("Servo Temperature %d C",temp);

  float volt = dynamixel->getVoltage(sernum);
  LINFO("Servo Voltage %4.2f V",volt);


  dynamixel->setLed(sernum,true);
  LINFO("Servo LED on");


  dynamixel->setTorque(sernum,true);
  LINFO("Servo Torque enable");

  // command the ssc
  dynamixel->setSpeed(sernum, rpm);

  dynamixel->move(sernum, serpos);
  LINFO("Moved servo %d to position %d", sernum, serpos);

  //wait until it's done
  while(dynamixel->isMoving(sernum)){
    float deg = dynamixel->getPosition(sernum);
    LINFO("Current Servo %d in %6.2f deg ",sernum,deg);
  }


  dynamixel->setLed(sernum,false);
  LINFO("Servo LED off");


  dynamixel->setTorque(sernum,false);
  LINFO("Servo Torque disable");
  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
