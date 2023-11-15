/*!@file BeoSub/BeoSubMotor.C Low-level driver for BeoSub motors */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubMotor.C $
// $Id: BeoSubMotor.C 7183 2006-09-20 00:02:57Z rjpeters $
//

// core of the code contributed by Harris Chiu, USC/ISI

#include "BeoSub/BeoSubMotor.H"

#include "Component/OptionManager.H"
#include "Util/Assert.H"
#ifdef HAVE_SYS_IO_H
#include <sys/io.h> /* for glibc */
#endif
#include <unistd.h> /* for libc5 */

#define BSM_NUMCHAN 9

// ######################################################################
BeoSubMotor::BeoSubMotor(OptionManager& mgr, const std::string& descrName,
                         const std::string& tagName,
                         const int def_parallel_port_addr) :
  ModelComponent(mgr, descrName, tagName),
  itsPort("BeoSubMotorParPortAddr", this, def_parallel_port_addr)
{
  strobelo = 205; strobehi = 204;

  chMin[0] = 59;  chMax[0] = 117;
  chMin[1] = 1;   chMax[1] = 255;
  chMin[2] = 25;  chMax[2] = 150;
  chMin[3] = 1;   chMax[3] = 255;
  chMin[4] = 25;  chMax[4] = 133;
  chMin[5] = 128; chMax[5] = 128;
  chMin[6] = 1;   chMax[6] = 255;
  chMin[7] = 128; chMax[7] = 128;
  chMin[8] = 128; chMax[8] = 128;

  chDefault[0] = 78;
  chDefault[1] = 83;
  chDefault[2] = 90;
  chDefault[3] = 83;
  chDefault[4] = 93;
  chDefault[5] = 128;
  chDefault[6] = 124;
  chDefault[7] = 128;
  chDefault[8] = 128;

  if (geteuid()) LFATAL("Need to run as root");
#ifdef HAVE_IOPERM
  ioperm(itsPort.getVal(), 3, 1);
#else
  LFATAL("libc must have ioperm() to use this function");
#endif
  reset();
}

// ######################################################################
BeoSubMotor::~BeoSubMotor()
{
  reset();
}

// ######################################################################
bool BeoSubMotor::WritePort()
{
#ifndef HAVE_SYS_IO_H
  LFATAL("Oops! I need <sys/io.h> for this operation");
  return false;
#else
  const int timeout = 100;

  for (int i = 0; i < timeout; i++)
    {
      char PortValue = inb(itsPort.getVal() + 1);
      if (PortValue < 0)
        {
          for (int j = 0; j < 9; j++) SendCh(ch[j]);
          return true;
        }
      usleep(10000);
    }
  LERROR("Parallel port busy -- GIVING UP");
  return false; // port was busy
#endif // HAVE_SYS_IO_H
}

// ######################################################################
void BeoSubMotor::SendCh(const int value)
{
#ifndef HAVE_SYS_IO_H
  LFATAL("Oops! I need <sys/io.h> for this operation");
#else
  outb(value, itsPort.getVal());
  outb(strobelo, (itsPort.getVal() + 2));
  for (int i = 0; i < 5000; i ++) ;
  outb(strobehi, (itsPort.getVal() + 2));
  for (int i = 0; i < 5000; i ++) ;
#endif // HAVE_SYS_IO_H
}

// ######################################################################
void BeoSubMotor::reset()
{
  for (int i = 0; i < BSM_NUMCHAN; i ++) ch[i] = chDefault[i];
  WritePort();
  usleep(100000);
}

// ######################################################################
bool BeoSubMotor::setValue(const byte value, const int channel,
                           const bool immed)
{
  ASSERT(channel >= 0 && channel < BSM_NUMCHAN);

  if (value < chMin[channel]) ch[channel] = chMin[channel];
  else if (value > chMax[channel]) ch[channel] = chMax[channel];
  else ch[channel] = value;

  if (immed) return WritePort();
  return true;
}

// ######################################################################
bool BeoSubMotor::pulseValue(const int channel, const bool positive)
{
  ASSERT(channel >= 0 && channel < BSM_NUMCHAN);
  int length = 600000; int delta = 80;
  if (channel == BSM_UPDOWN) { length = 1000000; delta = 100; }
  // go to rest:
  if (setValue(chDefault[channel], channel, true) == false) return false;
  usleep(100000);

  // turn on:
  if (positive)
    {
      if (setValue(chDefault[channel] - delta, channel, true) == false)
        return false;
    }
  else
    {
      if (setValue(chDefault[channel] + delta, channel, true) == false)
        return false;
    }
  usleep(length);

  // back to rest:
  if (setValue(chDefault[channel], channel, true) == false) return false;
  usleep(200000);

  // success!
  return true;
}

// ######################################################################
byte BeoSubMotor::getValue(const int channel) const
{
  ASSERT(channel >= 0 && channel < BSM_NUMCHAN);
  return ch[channel];
}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
