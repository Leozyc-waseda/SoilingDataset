/*!@file BeoSub/BeoSubIMU.C class for interfacing with the IMU */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubIMU.C $
// $Id: BeoSubIMU.C 6990 2006-08-11 18:13:51Z rjpeters $
//

#include "BeoSub/BeoSubIMU.H"
#include <string>

void *IMU_run(void *c);

// ######################################################################
void *IMU_run(void *c)
{
  BeoSubIMU *d = (BeoSubIMU *) c;
  d ->run();
  return NULL;
}

// ######################################################################
BeoSubIMUListener::~BeoSubIMUListener()
{ }

// ######################################################################
BeoSubIMU::BeoSubIMU(OptionManager& mgr, const std::string& descrName,
                     const std::string& tagName, const char *dev) :
  ModelComponent(mgr, descrName, tagName),
  itsSerial(new Serial(mgr, descrName+" Serial Port", tagName+"SerialPort")),
  itsKeepgoing(true)
{
  itsSerial->configure(dev, 115200, "8N1", false, true, 0);
  addSubComponent(itsSerial);
  pthread_mutex_init(&itsLock, NULL);
}

// ######################################################################
void BeoSubIMU::start2()
{ pthread_create(&itsRunner, NULL, &IMU_run, (void *) this); }

// ######################################################################
void BeoSubIMU::stop1()
{
  itsKeepgoing = false;
  usleep(300000); // make sure thread has exited
}

// ######################################################################
void BeoSubIMU::setListener(rutz::shared_ptr<BeoSubIMUListener> listener)
{ itsListener = listener; }

// ######################################################################
BeoSubIMU::~BeoSubIMU()
{
  pthread_mutex_destroy(&itsLock);
}

// ######################################################################
void BeoSubIMU::run()
{
  while(itsKeepgoing)
    {
      const int psize = 22;
      unsigned char packet[psize];
      packet[0] = 0x7F; packet[1] = 0xFF;
      int checksum;

      do
        {
          // initialize variables:
          int count = 2; checksum = 0x7FFF;
          bool tagFound = false, FFFound = false;

          while (count < psize)
            {
              // read one byte from serial port:
              unsigned char c; int size = itsSerial->read(&c, 1);
              if (size != 1) { PLERROR("Serial port read error"); continue; }

              // if we have already found the tag, just fill the packet in:
              if (tagFound)
                {
                  packet[count] = c;
                  if (count % 2 == 0) checksum += int(c);
                  else checksum += ((int(c)) << (sizeof(char)*8));
                  ++ count;
                }

              // finding the first byte of the tag:
              if (c == 0xFF) FFFound = true;

              // trigger tag found when a complete tag is detected:
              if (FFFound && c == 0x7F) tagFound = true;
            }
        } while ((checksum & 0x0000FFFF)); // exit iff checksum correct

      // get the data:
      pthread_mutex_lock(&itsLock);
      itsXaccel = accelConvert((packet[5] << 8) + packet[4]);
      itsYaccel = accelConvert((packet[9] << 8) + packet[8]);
      itsZaccel = accelConvert((packet[13] << 8) + packet[12]);
      itsXvel = rateConvert((packet[7] << 8) + packet[6]);
      itsYvel = rateConvert((packet[11] << 8) + packet[10]);
      itsZvel = rateConvert((packet[15] << 8) + packet[14]);

      // if we have a listener, let's notify it:
      if (itsListener.is_valid())
        itsListener->newData(itsXaccel, itsYaccel, itsZaccel,
                             itsXvel, itsYvel, itsZvel);
      pthread_mutex_unlock(&itsLock);
    }

  pthread_exit(0);
}

// ######################################################################
float BeoSubIMU::accelConvert(int data)
{
  if (data > 0x7FFF)
    {
      int temp = (((~data)+1) & 0x0000FFFF);
      return -temp / 326.3F;
    }
  else return data / 326.3F; // scale factor
}

// ######################################################################
float BeoSubIMU::rateConvert(int data)
{
  if (data > 0x7FFF)
    {
      int temp = (((~data)+1) & 0x0000FFFF); // 2's complement
      return -temp / 100.0F;
    }
  else return data / 100.0F; // scale factor
}

// ######################################################################
float BeoSubIMU::getXaccel()
{
  pthread_mutex_lock(&itsLock);
  const float ret = itsXaccel;
  pthread_mutex_unlock(&itsLock);
  return ret;
}

// ######################################################################
float BeoSubIMU::getYaccel()
{
  pthread_mutex_lock(&itsLock);
  const float ret = itsYaccel;
  pthread_mutex_unlock(&itsLock);
  return ret;
}

// ######################################################################
float BeoSubIMU::getZaccel()
{
  pthread_mutex_lock(&itsLock);
  const float ret = itsZaccel;
  pthread_mutex_unlock(&itsLock);
  return ret;
}

// ######################################################################
Angle BeoSubIMU::getXvel()
{
  pthread_mutex_lock(&itsLock);
  const Angle ret = itsXvel;
  pthread_mutex_unlock(&itsLock);
  return ret;
}

// ######################################################################
Angle BeoSubIMU::getYvel()
{
  pthread_mutex_lock(&itsLock);
  const Angle ret = itsYvel;
  pthread_mutex_unlock(&itsLock);
  return ret;
}

// ######################################################################
Angle BeoSubIMU::getZvel()
{
  pthread_mutex_lock(&itsLock);
  const Angle ret = itsZvel;
  pthread_mutex_unlock(&itsLock);
  return ret;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
