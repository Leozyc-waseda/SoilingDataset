/*!@file RCBot/BotControl.H  abstract robot control (can use corba)  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/BotControl.H $
// $Id: BotControl.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef BOTCONTROL_H
#define BOTCONTROL_H

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/sc8000.H"
#include "Devices/ssc.H"
#include "GUI/XWinManaged.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Devices/BeoChip.H"

//! Class to abstract robot control for various robots
//! Can control the robot as well as get sensor values

enum RBEOCHIP {RBEOCHIP_RESET, RBEOCHIP_CAL, RBEOCHIP_SETSERVO};

class BotControl : public ModelComponent {
public:

  // various robot types
  enum RobotType {RCBOT_JEEP, RCBOT_TRUCK, WIREBOT, BEOBOT, SIMULATION};

  BotControl(OptionManager &mgr,
             const std::string& descrName = "Robot Controller",
             const std::string& tagName = "BotControl",
             RobotType inBotType = SIMULATION);

  //! Destructor
  ~BotControl();

  //! Init process
  void init();
  //!set the speed -1.0 (reverse) ... 1.0 (forward)
  float getSpeed();
  //!get the speed -1.0 ... 1.0
  bool setSpeed(const float speedPos);

  //! gets steering angle; input from -1.0 (full left) to 1.0 (full right)
  float getSteering();
  //! sets steering angle; input from -1.0 (full left) to 1.0 (full right)
  bool setSteering(const float steeringPos);

  //! gets the image sensor i
  Image<PixRGB<byte > > getImageSensor(int i = 0);

  //! Get image sersor dims
  void getImageSensorDims(short &w, short &h, int i = 0);

  //! Added extra info to the display
  void setInfo(const char *info, Point2D<int> trackLoc, Point2D<int> recLoc);

  //! Get user input from window
  int getUserInput(Point2D<int> &loc);

  //! Set Display
  void setDisplay(bool sd);

protected:
  nub::soft_ref<SC8000> sc8000;   //!< sc8000 servo controller
  nub::soft_ref<SSC> ssc;         //!< ssc servo controller
  nub::soft_ref<BeoChip> bc;      //!< BeoChip controller
//   nub::soft_ref<FrameGrabberConfigurator> gbc;
//   nub::soft_ref<FrameIstream> gb;
  nub::soft_ref<InputFrameSeries> ifs;
  XWinManaged *xwin;

private:
  //!The type of robot we are using
  RobotType botType;

  //!servo def
  int speedServo;
  int steerServo;
  int driveServoLeft;
  int driveServoRight;

  //! Speed position
  float speed;

  //! steer position
  float steering;

  //! Image dims
  int width; int height;

  // display windows
  void showImg(Image<PixRGB<byte> > &img);

  // setDisplay
  bool dispImg;

  //extra info
  char *extraInfo;
  Point2D<int> trackLoc;
  Point2D<int> recLoc;

  //to compute frame/sec
  unsigned long long int avgtime;
  int avgn;
  float fps;
  Timer timer;
  static const int NAVG = 20;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
