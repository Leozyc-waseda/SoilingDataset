/*!@file Beobot/BeobotEffectors.C controls the effectors of the Beobots
  (e.g., motors) */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotEffectors.C $
// $Id: BeobotEffectors.C 6795 2006-06-29 20:45:32Z rjpeters $
//

#include "Beobot/BeobotEffectors.H"

#include "Component/OptionManager.H"
#include "Devices/lcd.H"
#include "Devices/ssc.H"

// ######################################################################
BeobotEffectors::BeobotEffectors(OptionManager& mgr,
                                 const std::string& descrName,
                                 const std::string& tagName) :
  ModelComponent(mgr, descrName, tagName),
  myBeoChip(new BeoChip(mgr))
{
  // register our subcomponents:
  addSubComponent(myBeoChip);
}

// ##################################################################
BeobotEffectors::~BeobotEffectors()
{ }

// ##################################################################
void BeobotEffectors::performAction(BeobotAction& action)
{
  turn(action.getTurn());
  speed(action.getSpeed());
  changegear(action.getGear());
}

// ##################################################################
void BeobotEffectors::center()
{ myBeoChip->setServo(bbc.speedServoNum,0.0F); }

// ##################################################################
void BeobotEffectors::turn(const float angle)
{ myBeoChip->setServo(bbc.steerServoNum, angle); }

// ##################################################################
void BeobotEffectors::speed(const float sp)
{ myBeoChip->setServo(bbc.speedServoNum, sp); }

// ##################################################################
void BeobotEffectors::stop()
{ myBeoChip->setServo(bbc.speedServoNum, 0.0F); }

// ##################################################################
void BeobotEffectors::changegear(const int gear)
{
  if (gear == 0)
    myBeoChip->setServo(bbc.gearServoNum, -1.0);
  else if (gear == 1)
    myBeoChip->setServo(bbc.gearServoNum,  0.0);
  else
    myBeoChip->setServo(bbc.gearServoNum,  1.0);
}

// ##################################################################
void BeobotEffectors::display(const char* message)
{ myBeoChip->lcdPrintf(0,0, "%s", message); }

// ##################################################################
void BeobotEffectors::cleardisplay()
{ myBeoChip->lcdClear(); }

// ##################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
