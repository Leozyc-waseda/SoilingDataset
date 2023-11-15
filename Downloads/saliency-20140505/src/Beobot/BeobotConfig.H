/*!@file Beobot/BeobotConfig.H A set of car configuration parameters    */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotConfig.H $
// $Id: BeobotConfig.H 6780 2006-06-22 01:29:58Z beobot $

#ifndef BEOBOT_BEOBOTCONFIG_H_DEFINED
#define BEOBOT_BEOBOTCONFIG_H_DEFINED

#include "Util/readConfig.H"

/*! This class uses readConfig to extract config parameters for BeobotControl
    It is either from "beobot.conf" or another specified file              */

class BeobotConfig
{
public:
  //!default constructor, open config fileName and read
  BeobotConfig(std::string fileName);
  //!default constructor, open config "Beobot.conf" and read
  BeobotConfig();
  //!default destructor
  ~BeobotConfig();

  //! speed servo neutral position
  unsigned int speedNeutralVal;
  //! speed servo minimum position
  unsigned int speedMinVal;
  //! speed servo maximum position
  unsigned int speedMaxVal;

  //! steer servo neutral position
  unsigned int steerNeutralVal;
  //! steer servo minimum position
  unsigned int steerMinVal;
  //! steer servo maximum position
  unsigned int steerMaxVal;

  //! gear servo neutral position
  unsigned int gearNeutralVal;
  //! gear servo minimum position
  unsigned int gearMinVal;
  //! gear servo maximum position
  unsigned int gearMaxVal;

  //! PWM0 pulse neutral value
  unsigned int pwm0NeutralVal;
  //! PWM0 pulse minimum value
  unsigned int pwm0MinVal;
  //! PWM0 pulse maximum value
  unsigned int pwm0MaxVal;

  //! PWM1 pulse neutral value
  unsigned int pwm1NeutralVal;
  //! PWM1 pulse minimum value
  unsigned int pwm1MinVal;
  //! PWM1 pulse maximum value
  unsigned int pwm1MaxVal;

  //! speed servo number on BeoChip
  unsigned int speedServoNum;
  //! steering servo number on BeoChip
  unsigned int steerServoNum;
  //! gear servo number on BeoChip
  unsigned int gearServoNum;
  //! BeoChip serial port device;
  std::string beoChipPort;

private:
  readConfig readconfig;
  void init();
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
