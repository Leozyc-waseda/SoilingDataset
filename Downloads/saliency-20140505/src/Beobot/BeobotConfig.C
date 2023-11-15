/*!@file Beobot/BeobotConfig.C A set of Beobot configuration parameters */
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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotConfig.C $
// $Id: BeobotConfig.C 6795 2006-06-29 20:45:32Z rjpeters $

#include "Beobot/BeobotConfig.H"
#include "Component/ModelManager.H"

// ######################################################################
BeobotConfig::BeobotConfig()
{
  readconfig.openFile( "Beobot.conf" );
  init();
}

// ######################################################################
BeobotConfig::BeobotConfig( std::string fileName )
{
  readconfig.openFile( fileName.c_str() );
  init();
}

// ######################################################################
BeobotConfig::~BeobotConfig()
{ }

// ######################################################################
void BeobotConfig::init()
{
  // speed servo number on BeoChip
  speedServoNum =
    (unsigned int)readconfig.getItemValueF( "speedServoNum" );
  LDEBUG("speedServoNum: %d", speedServoNum);

  // steering servo number on BeoChip
  steerServoNum =
    (unsigned int)readconfig.getItemValueF( "steerServoNum" );
  LDEBUG("steerServoNum: %d", steerServoNum);

  // gear servo number on BeoChip
  gearServoNum =
    (unsigned int)readconfig.getItemValueF( "gearServoNum" );
  LDEBUG("gearServoNum: %d", gearServoNum);

  // speed servo neutral position
  speedNeutralVal =
    (unsigned int)readconfig.getItemValueF( "speedNeutralVal" );
  LDEBUG("speedNeutralVal: %d", speedNeutralVal);

  // speed servo minimum position
  speedMinVal =
    (unsigned int)readconfig.getItemValueF( "speedMinVal" );
  LDEBUG("speedMinVal: %d", speedMinVal);

  // speed servo maximum position
  speedMaxVal =
    (unsigned int)readconfig.getItemValueF( "speedMaxVal" );
  LDEBUG("speedMaxVal: %d", speedMaxVal);

  // steering servo neutral position
  steerNeutralVal =
    (unsigned int)readconfig.getItemValueF( "steerNeutralVal" );
  LDEBUG("steerNeutralVal: %d", steerNeutralVal);

  // steering servo minimum position
  steerMinVal =
    (unsigned int)readconfig.getItemValueF( "steerMinVal" );
  LDEBUG("steerMinVal: %d", steerMinVal);

  // steering servo maximum position
  steerMaxVal =
    (unsigned int)readconfig.getItemValueF( "steerMaxVal" );
  LDEBUG("steerMaxVal: %d", steerMaxVal);

  // gear servo neutral position
  gearNeutralVal =
    (unsigned int)readconfig.getItemValueF( "gearNeutralVal" );
  LDEBUG("gearNeutralVal: %d", gearNeutralVal);

  // gear servo minimum position
  gearMinVal =
    (unsigned int)readconfig.getItemValueF( "gearMinVal" );
  LDEBUG("gearMinVal: %d", gearMinVal);

  // gear servo maximum position
  gearMaxVal =
    (unsigned int)readconfig.getItemValueF( "gearMaxVal" );
  LDEBUG("gearMaxVal: %d", gearMaxVal);

  // PWM0 neutral values
  pwm0NeutralVal =
    (unsigned int)readconfig.getItemValueF( "pwm0NeutralVal" );
  LDEBUG("pwm0NeutralVal: %d", pwm0NeutralVal);

  // PWM0 minimum values
  pwm0MinVal =
    (unsigned int)readconfig.getItemValueF( "pwm0MinVal" );
  LDEBUG("pwm0MinVal: %d", pwm0MinVal);

  // PWM0 maximum values
  pwm0MaxVal =
    (unsigned int)readconfig.getItemValueF( "pwm0MaxVal" );
  LDEBUG("pwm0MaxVal: %d", pwm0MaxVal);

  // PWM1 neutral values
  pwm1NeutralVal =
    (unsigned int)readconfig.getItemValueF( "pwm1NeutralVal" );
  LDEBUG("pwm1NeutralVal: %d", pwm1NeutralVal);

  // PWM1 minimum values
  pwm1MinVal =
    (unsigned int)readconfig.getItemValueF( "pwm1MinVal" );
  LDEBUG("pwm1MinVal: %d", pwm1MinVal);

  // PWM1 maximum values
  pwm1MaxVal =
    (unsigned int)readconfig.getItemValueF( "pwm1MaxVal" );
  LDEBUG("pwm1MaxVal: %d", pwm1MaxVal);

  // BeoChip port location
  beoChipPort = readconfig.getItemValueS( "beoChipPort" );
  LDEBUG("beoChipPort: %s", beoChipPort.c_str());
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
