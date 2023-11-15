/*!@file BeoSub/BeoSubDepthSensor.C class for interfacing with a depth sensor */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubDepthSensor.C $
// $Id: BeoSubDepthSensor.C 4679 2005-06-24 04:59:53Z rjpeters $
//

#include "BeoSub/BeoSubDepthSensor.H"
#include <cstdio>
#include <sstream>
#include <string>
#include <unistd.h>

void *Sensor_run(void *c);
// ######################################################################
void *Sensor_run(void *c)
{
  BeoSubDepthSensor *d = (BeoSubDepthSensor *)c;
  d->run();
  return NULL;
}

// ######################################################################
BeoSubDepthSensor::BeoSubDepthSensor(OptionManager& mgr,
                                     const std::string& descrName,
                                     const std::string& tagName,
                                     const char *dev) :
  ModelComponent(mgr, descrName, tagName),
  serial(new Serial(mgr)),
  keepgoing(true), depth(0.0f)
{
  serial->configure(dev, 9600, "8N1");
  addSubComponent(serial);
  pthread_mutex_init(&lock, NULL);
}

// ######################################################################
void BeoSubDepthSensor::start2()
{ pthread_create(&runner, NULL, &Sensor_run, (void *)this); }

// ######################################################################
void BeoSubDepthSensor::stop1()
{
  keepgoing = false;
  usleep(300000); // make sure thread has exited
}

// ######################################################################
BeoSubDepthSensor::~BeoSubDepthSensor()
{ pthread_mutex_destroy(&lock); }

// ######################################################################
void BeoSubDepthSensor::run()
{
  unsigned char c = 255;
  while(keepgoing)
    {
      // skip to next CR:
      while(c != '\r') c = serial->read();

      // get data until next CR:
      std::string str;
      while( (c = serial->read() ) != '\r')
        if (isdigit(c) || c == '.' || c == '-') str += c;
        else str += ' ';

      // convert to floats:
      std::stringstream strs(str);
      int b1, b2, b3; strs >> b1 >> b2 >> b3;
      float d = float(b1) + (float(b2) + float(b3) / 256.0F) / 256.0F;

      pthread_mutex_lock(&lock);
      depth = d;
      pthread_mutex_unlock(&lock);
    }

  pthread_exit(0);
}

// ######################################################################
float BeoSubDepthSensor::get()
{
  volatile float val;
  pthread_mutex_lock(&lock);
  val = depth;
  pthread_mutex_unlock(&lock);
  return val;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
