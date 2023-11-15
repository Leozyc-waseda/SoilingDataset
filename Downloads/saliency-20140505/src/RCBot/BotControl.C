/*!@file RCBot/BotControl.C  abstract robot control (can use corba)  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/BotControl.C $
// $Id: BotControl.C 13993 2010-09-20 04:54:23Z itti $
//
// ######################################################################

#include "RCBot/BotControl.H"
#include "Component/OptionManager.H"
#include "Beobot/BeobotConfig.H"

BeobotConfig bbc;

// ######################################################################
BotControl::BotControl(OptionManager& mgr, const std::string& descrName,
                       const std::string& tagName, RobotType inBotType) :
  ModelComponent(mgr, descrName, tagName),
  botType(inBotType)
{
  // switch for the the actuator system
  switch(botType)
    {
    case RCBOT_JEEP:
    case RCBOT_TRUCK:
      sc8000 = nub::ref<SC8000>(new SC8000(mgr));
      addSubComponent(sc8000);
      break;

    case WIREBOT:
      ssc = nub::soft_ref<SSC>(new SSC(mgr));
      addSubComponent(ssc);
      break;

    case BEOBOT:
      bc = nub::soft_ref<BeoChip>(new BeoChip(mgr));
      addSubComponent(bc);

      // configure the BeoChip
      bc->setModelParamVal("BeoChipDeviceName", std::string("/dev/ttyS0"));
      break;

    case SIMULATION:
      break;
    }

  // for frame grabber:
  // --in=ieee1394 --framegrabber-mode=YUV44 --framegrabber-dims=160x120
  //
  // for mpg input:
  // --in=fname.mpg --rescale-input=160x120 --crop-input=x1,y1,x2,y2
  //
  // for raster series input:
  // --in=prefix#.ppm --rescale-input=160x120 --crop-input=x1,y1,x2,y2
  ifs =  nub::ref<InputFrameSeries>(new InputFrameSeries(mgr));
  addSubComponent(ifs);

  // default is we do not display the image
  dispImg = true;
}

// ######################################################################
void BotControl::setDisplay(bool sd)
{
  if(dispImg != sd)
    {
      dispImg = sd;
      if(dispImg)
        xwin = new XWinManaged(Dims(width,height), -1, -1, "Bot Control");
      else
        delete(xwin);
    }
}

// ######################################################################
BotControl::~BotControl()
{  }

// ######################################################################
void BotControl::init()
{
  LINFO("initializing BotControl");
  switch(botType)
    {
    case RCBOT_JEEP:
      //set the servos
      speedServo = 3;
      steerServo = 1;

      //calibrate the servos
      sc8000->calibrate(steerServo, 13650, 10800, 16000); //for steering
      sc8000->calibrate(speedServo, 14000, 12000, 16000); //for speed
      break;

    case RCBOT_TRUCK:
      LINFO("RC TRUCK");
      //set the servos
      speedServo = 3;
      steerServo = 4;

      //calibrate the servos
      sc8000->calibrate(steerServo, 10000, 15000, 20000); //for steering
      sc8000->calibrate(speedServo, 10000, 15000, 19000); //for speed

      sc8000->calibrate(1, 13000, 16000, 20000); //for speed
      sc8000->calibrate(2, 10000, 17000, 19000); //for speed
      sc8000->move(1, 0);
      sc8000->move(2, 0);
      break;

    case WIREBOT:
      LINFO("Wire Bot");
      //set the servos
      driveServoRight = 0;
      driveServoLeft = 1;

      //calibrate the servos
      //FIXME ssc->calibrate(driveServoRight, 1157, 757, 1557); //for steering
      //FIXME ssc->calibrate(driveServoLeft, 1161, 761, 1561); //for speed

      ssc->move(driveServoRight, 0);
      ssc->move(driveServoLeft, 0);
      break;

    case BEOBOT:
      LINFO("BeoBot");

      // reset the beochip:
      LINFO("Resetting BeoChip...");
      bc->resetChip(); sleep(1);

      //set the servos
      speedServo = bbc.speedServoNum;
      steerServo = bbc.steerServoNum;

      // keep the gear at the lowest speed/highest torque
      bc->setServoRaw(bbc.gearServoNum, bbc.gearMinVal);

      // turn on the keyboard
      bc->debounceKeyboard(true);
      bc->captureKeyboard(true);

      // calibrate the PWMs:
      bc->calibratePulse(0,
                         bbc.pwm0NeutralVal,
                         bbc.pwm0MinVal,
                         bbc.pwm0MaxVal);
      bc->calibratePulse(1,
                         bbc.pwm1NeutralVal,
                         bbc.pwm1MinVal,
                         bbc.pwm1MaxVal);
      bc->capturePulse(0, true);
      bc->capturePulse(1, true);
      break;

    case SIMULATION:
      break;
    }

  // reset fps
  avgtime = 0; avgn = 0; fps = 0.0F;
  timer.reset();

  width = ifs->getWidth();
  height = ifs->getHeight();
  xwin = new XWinManaged(Dims(width,height), -1, -1, "Bot Control");
}

// ######################################################################
float BotControl::getSpeed()
{
  switch (botType)
    {
    case WIREBOT:
      return 0.0;
    case BEOBOT:
      return (bc->getServo(speedServo));

    case RCBOT_JEEP:
    case RCBOT_TRUCK:
      return sc8000->getPosition(speedServo);

    case SIMULATION:
      // nothing to do
      return 0.0;
    }
  return 0.0;
}

// ######################################################################
bool BotControl::setSpeed(const float speedPos)
{
  speed = speedPos;
  switch (botType)
    {
    case WIREBOT:
      return (ssc->move(driveServoRight, -1*speed)
              && ssc->move(driveServoLeft, speed) );
    case BEOBOT:
      return (bc->setServo(speedServo, speed));

    case RCBOT_JEEP:
    case RCBOT_TRUCK:
      return sc8000->move(speedServo, speed);

    case SIMULATION:
      // nothing to do
      return false;
    }

  return false;
}

// ######################################################################
float BotControl::getSteering()
{
  switch (botType)
    {
    case WIREBOT:
      return 0.0;
    case BEOBOT:
      return (bc->getServo(steerServo));

    case RCBOT_JEEP:
    case RCBOT_TRUCK:

      return sc8000->getPosition(steerServo);

    case SIMULATION:
      // nothing to do
      return 0.0;
    }
  return 0.0;
}

// ######################################################################
bool BotControl::setSteering(const float steeringPos)
{
  steering = steeringPos;

  switch (botType)
    {
    case WIREBOT:
      return (ssc->move(driveServoRight, steeringPos) &&
              ssc->move(driveServoLeft, steeringPos) );

    case BEOBOT:
      return (bc->setServo(steerServo, steering));

    case RCBOT_JEEP:
    case RCBOT_TRUCK:
      return sc8000->move(steerServo, steeringPos);

    case SIMULATION:
      // nothing to do
      return false;
    }
  return false;
}

// ######################################################################
Image<PixRGB<byte> > BotControl::getImageSensor(int i)
{
  ifs->updateNext();
  Image<PixRGB<byte> > img = ifs->readRGB();

  // compute frame rate
  avgtime += timer.getReset(); avgn ++;
  if (avgn == NAVG)
    {
      fps = 1000.0F / float(avgtime) * float(avgn);
      avgtime = 0; avgn = 0;
    }

  if (img.initialized() && dispImg) showImg(img);
  return img;
}

// ######################################################################
void BotControl::getImageSensorDims(short &w, short &h, int i)
{
  w = width;
  h = height;
}

// ######################################################################
void BotControl::setInfo(const char *info, Point2D<int> inTrackLoc,
                         Point2D<int> inRecLoc)
{
  extraInfo = strdup(info);
  trackLoc = inTrackLoc;
  recLoc = inRecLoc;
}

// ######################################################################
void BotControl::showImg(Image<PixRGB<byte> > &img)
{
  Image<PixRGB<byte> > disp = img;
  char info[255];

  if (recLoc.isValid())
    drawDisk(disp, Point2D<int>(recLoc.i, recLoc.j),
             10, PixRGB<byte>(255, 255, 20));
  if (trackLoc.isValid())
    drawDisk(disp, Point2D<int>(trackLoc.i, trackLoc.j),
             6, PixRGB<byte>(20, 50, 255));

  sprintf(info, " %.1ffps %s", fps, extraInfo);

  writeText(disp, Point2D<int>(0,220), info,
            PixRGB<byte>(255), PixRGB<byte>(127));

  xwin->drawImage(disp);
}

// ######################################################################
int BotControl::getUserInput(Point2D<int> &loc)
{
  loc = xwin->getLastMouseClick();
  return  xwin->getLastKeyPress();
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
