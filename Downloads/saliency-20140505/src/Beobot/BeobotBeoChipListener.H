/*!@file Beobot/BeobotBeoChipListener.H listener for the buttons of the beoChip */

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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotBeoChipListener.H $
// $Id: BeobotBeoChipListener.H 8267 2007-04-18 18:24:24Z rjpeters $
//

#include "Devices/BeoChip.H"
#include "Devices/DeviceOpts.H"

// ######################################################################
//! Beobot BeoChipListener
class BeobotBeoChipListener : public BeoChipListener
{
public:
  BeobotBeoChipListener(nub::soft_ref<BeoChip> bc) :
    itsBeoChip(bc), minp0(9999), maxp0(0), minp1(9999), maxp1(0),
    counter0(0), counter1(0), kbd(0x1f), moveServo(false) { }

  virtual ~BeobotBeoChipListener() { }

  virtual void event(const BeoChipEventType t, const int valint,
                     const float valfloat)
  {
    LDEBUG("Event: %d val = %d, fval = %f", int(t), valint, valfloat);
    switch(t)
      {
      case PWM0:
        if (valint < minp0) minp0 = valint;
        else if (valint > maxp0) maxp0 = valint;
        if(moveServo)
          itsBeoChip->setServo(0, valfloat);
        if (++counter0 >= 10)
          {
            itsBeoChip->lcdPrintf(5, 2, "%04d  %04d-%04d",
                                  valint, minp0, maxp0);
            itsBeoChip->lcdPrintf(6, 1, "%03d",
                                  itsBeoChip->getServoRaw(0));
            counter0 = 0;
          }
        break;
      case PWM1:
        if (valint < minp1) minp1 = valint;
        else if (valint > maxp1) maxp1 = valint;
        if(moveServo)
          itsBeoChip->setServo(1, valfloat);
        if (++counter1 >= 10)
          {
            itsBeoChip->lcdPrintf(5, 3, "%04d  %04d-%04d",
                                  valint, minp1, maxp1);
            itsBeoChip->lcdPrintf(17, 1, "%03d",
                                  itsBeoChip->getServoRaw(1));
            counter1 = 0;
          }
        break;
      case KBD: kbd = valint;  break;
      case RESET: LERROR("BeoChip RESET occurred!"); break;
      case ECHOREP: LINFO("BeoChip Echo reply received."); break;
      case INOVERFLOW: LERROR("BeoChip input overflow!"); break;
      case SERIALERROR: LERROR("BeoChip serial error!"); break;
      case OUTOVERFLOW: LERROR("BeoChip output overflow!"); break;
      default: LERROR("Unknown event %d received!", int(t)); break;
      }
  }

  nub::soft_ref<BeoChip> itsBeoChip;
  int minp0, maxp0, minp1, maxp1;
  int counter0, counter1;
  int kbd;
  bool moveServo;
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
