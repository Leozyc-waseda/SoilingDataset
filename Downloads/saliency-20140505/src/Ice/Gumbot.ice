/*!@file Gumbot.ice ice definition file for the Gumbot */

//////////////////////////////////////////////////////////////////// //
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
// Primary maintainer for this file: Lior Elazary <lelazary@yahoo.com>
// $HeadURL$
// $Id$
//

#include <Ice/ImageIce.ice>

#ifndef Gumbot_ICE
#define Gumbot_ICE


module Robots {

  enum GumbotModes {SafeMode, FullMode, SpotMode, CoverMode, CoverAndDockMode};

  interface Gumbot {


    //!set the speed -1.0 (reverse) ... 1.0 (forward)
    float getSpeed();
    //!get the speed -1.0 ... 1.0
    short setSpeed(float speed);

    //! gets steering angle; input from -1.0 (full left) to 1.0 (full right)
    float getSteering();
    //! sets steering angle; input from -1.0 (full left) to 1.0 (full right)
    short setSteering(float steeringPos);

    //! gets the image sensor i
    ImageIceMod::ImageIce getImageSensor(short i); 

    //! Get image sersor dims
    ImageIceMod::DimsIce getImageSensorDims(short i);

    //! get sensor values
    short getSensorValue(short i);

    //! turn the motors on or off
    void motorsOff(short i);

    //! set the motors speed individually
    void setMotor(short i, float val);

    //! Send raw comands to the robot
    short sendRawCmd(string data);

    //!Play the songs, like the imperial march song
    void playSong(short song);

    //!Send a start command
    void sendStart();

    //!Go into the various modes
    void setMode(GumbotModes mode);

    //!Set the various demos
    void setDemo(short demo);

    //!Set LEDs
    void setLED(short led, short color, short intensity);


    // Shuts down the server
    void shutdown();
  };
};

#endif
