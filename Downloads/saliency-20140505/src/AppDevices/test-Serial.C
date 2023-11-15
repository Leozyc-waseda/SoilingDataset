/*!@file AppDevices/test-Serial.C Send raw data and recive data */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-Serial.C $
// $Id: test-Serial.C 15095 2011-11-23 06:39:29Z kai $
//

#include "Component/ModelManager.H"
#include "Devices/Serial.H"
#include "Util/Types.H"
#include "Util/log.H"
#include <unistd.h>
#include <stdio.h>
#include <signal.h>

int main(int argc, const char **argv)
{
        std::string USBDEV = "/dev/ttyUSB0";

  // Instantiate a ModelManager:
  ModelManager manager("Test Serial");

  nub::soft_ref<Serial> serial(new Serial(manager));
  manager.addSubComponent(serial);


  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "Serial Data", 0, 40) == false) return(1);

  //serial->configure (USBDEV.c_str(), 115200, "8N1", false, false, 0);
	serial->configureSearch ("motorboard", 115200,"ttyUSB","8N1",false,false,0);
//  serial->configure("/dev/ttyUSB0", 115200);

  // let's get all our ModelComponent instances started:
  manager.start();
  usleep(1000);

  int numChar = manager.numExtraArgs();
  unsigned char buff[1024];
  for(int i=0; i<numChar; i++)
    buff[i] = manager.getExtraArgAs<int>(i);

  printf("Sending: ");
  for(int i=0; i<numChar; i++)
    printf("%i ", buff[i]);
  printf("\n");


//  std::vector<unsigned char> frame = serial->readFrame(buff, numChar);
//
//  printf("Got Result: ");
//  for(unsigned int i=0; i<frame.size(); i++)
//    printf("%d ", frame.at(i));
//  printf("\n");

  serial->write(buff, numChar);
  sleep(1);

  int ret= serial->read(buff, 1024);
  printf("Got Result From %s port %s\n", serial->getDeviceDescriptor().c_str(),serial->getDevName().c_str());
  printf("Got %i: ", ret);
  for(int i=0; i<ret; i++)
    printf(" %3d  ", buff[i]);
  printf("\n");

  printf("Hex %i: ", ret);
  for(int i=0; i<ret; i++)
    printf(" %3x  ", buff[i]);
  printf("\n");

  printf("Str %i: ", ret);
  for(int i=0; i<ret; i++)
    printf("[%3c] ", buff[i]);
  printf("\n");
  sleep(1);

//	serial->configureSearch ("motorboard", 115200,"ttyUSB","8N1",false,false,0);
	serial->searchDevice("ttyUSB","motorboard");
  sleep(1);
  serial->write(buff, numChar);
  sleep(1);

  ret= serial->read(buff, 1024);
  printf("Got Result From %s port %s\n", serial->getDeviceDescriptor().c_str(),serial->getDevName().c_str());
  printf("Got %i: ", ret);
  for(int i=0; i<ret; i++)
    printf(" %3d  ", buff[i]);
  printf("\n");

  printf("Hex %i: ", ret);
  for(int i=0; i<ret; i++)
    printf(" %3x  ", buff[i]);
  printf("\n");

  printf("Str %i: ", ret);
  for(int i=0; i<ret; i++)
    printf("[%3c] ", buff[i]);
  printf("\n");
  sleep(1);
  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}
