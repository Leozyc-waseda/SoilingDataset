/*!@file SeaBee/SubController.C  Control motors and pid */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/SubController.C $
// $Id: SubController.C 10794 2009-02-08 06:21:09Z itti $
//

#include "SeaBee/SubController.H"
#include "Component/ModelOptionDef.H"
#include "Devices/DeviceOpts.H"
#include "Util/Assert.H"
#include "Image/DrawOps.H"

#define CMD_DELAY 6000
#define INT_CLVL_PRESSURE 225

const ModelOptionCateg MOC_SeaBee_Controller = {
    MOC_SORTPRI_3, "SeaBee controller related options" };

const ModelOptionDef OPT_SeaBeeSimMode =
{ MODOPT_FLAG, "SeabeeSimMode", &MOC_SeaBee_Controller, OPTEXP_CORE,
    "Run in simulator mode",
    "seabee-sim-mode", '\0', "", "false" };

namespace
{
  class SubControllerPIDLoop : public JobWithSemaphore
  {
  public:
    SubControllerPIDLoop(SubController* subCtrl)
      :
      itsSubController(subCtrl),
      itsPriority(1),
      itsJobType("controllerLoop")
    {}

    virtual ~SubControllerPIDLoop() {}

    virtual void run()
    {
      ASSERT(itsSubController);
      while(1)
        {
          itsSubController->sendHeartBeat();
          itsSubController->updatePID();
        }
    }

    virtual const char* jobType() const
    { return itsJobType.c_str(); }

    virtual int priority() const
    { return itsPriority; }

  private:
    SubController* itsSubController;
    const int itsPriority;
    const std::string itsJobType;
  };
}


// ######################################################################
SubController::SubController(OptionManager& mgr,
                             const std::string& descrName,
                             const std::string& tagName,
                             const bool simulation):

  ModelComponent(mgr, descrName, tagName),
  itsSimulation(&OPT_SeaBeeSimMode, this, ALLOW_ONLINE_CHANGES),
  itsDesiredPitch(30),
  itsDesiredRoll(0),
  itsDesiredHeading(20),
  itsDesiredDepth(200),
  itsDesiredSpeed(0),
  itsDesiredTurningSpeed(0),
  itsCurrentPitch(0),
  itsCurrentRoll(0),
  itsCurrentHeading(0),
  itsCurrentDepth(-1),
  itsCurrentSpeed(0),
  speedScale("speedScale", this, 1, ALLOW_ONLINE_CHANGES),
  itsSpeedScale(speedScale.getVal()),
  depthRatio("depthRatio", this, 1.3, ALLOW_ONLINE_CHANGES),
  itsDepthRatio(depthRatio.getVal()),
  itsPitchPID(0.5f, 0.0, 0.0, -20, 20),
  itsRollPID(0.0f, 0, 0, -20, 20),
  headingP("headingP", this, 2.0, ALLOW_ONLINE_CHANGES),
  headingI("headingI", this, 0, ALLOW_ONLINE_CHANGES),
  headingD("headingD", this, 0, ALLOW_ONLINE_CHANGES),
  itsHeadingPID(headingP.getVal(), headingI.getVal(), headingD.getVal(), -20, 20),
  depthP("depthP", this, -13.0, ALLOW_ONLINE_CHANGES),
  depthI("depthI", this, -0.5, ALLOW_ONLINE_CHANGES),
  depthD("depthD", this, -8.0, ALLOW_ONLINE_CHANGES),
  itsDepthPID(depthP.getVal(), depthI.getVal(), depthD.getVal(), -20, 20),
  itsCurrentThruster_Up_Left(0),
  itsCurrentThruster_Up_Right(0),
  itsCurrentThruster_Up_Back(0),
  itsCurrentThruster_Fwd_Right(0),
  itsCurrentThruster_Fwd_Left(0),
  setCurrentThruster_Up_Left("ThrusterUpLeft", this, 0, ALLOW_ONLINE_CHANGES),
  setCurrentThruster_Up_Right("ThrusterUpRight", this, 0, ALLOW_ONLINE_CHANGES),
  setCurrentThruster_Up_Back("ThrusterUpBack", this, 0, ALLOW_ONLINE_CHANGES),
  setCurrentThruster_Fwd_Right("ThrusterFwdRight", this, 0, ALLOW_ONLINE_CHANGES),
  setCurrentThruster_Fwd_Left("ThrusterFwdLeft", this, 0, ALLOW_ONLINE_CHANGES),
  pitchP("pitchP", this, 0, ALLOW_ONLINE_CHANGES),
  pitchI("pitchI", this, 0, ALLOW_ONLINE_CHANGES),
  pitchD("pitchD", this, 0, ALLOW_ONLINE_CHANGES),
  rollP("rollP", this, 0, ALLOW_ONLINE_CHANGES),
  rollI("rollI", this, 0, ALLOW_ONLINE_CHANGES),
  rollD("rollD", this, 0, ALLOW_ONLINE_CHANGES),
  motorsOn("motorsOn", this, false, ALLOW_ONLINE_CHANGES),
  pidOn("pidOn", this, true, ALLOW_ONLINE_CHANGES),
  guiOn("guiOn", this, true, ALLOW_ONLINE_CHANGES),
  depthPIDDisplay("Depth PID Disp", this, false, ALLOW_ONLINE_CHANGES),
  pitchPIDDisplay("Pitch PID Disp", this, false, ALLOW_ONLINE_CHANGES),
  headingPIDDisplay("Heading PID Disp", this, false, ALLOW_ONLINE_CHANGES),
  rollPIDDisplay("Roll PID Disp", this, false, ALLOW_ONLINE_CHANGES),

  setDepthValue("DepthValue", this, 450, ALLOW_ONLINE_CHANGES),
  setPitchValue("PitchValue", this, itsDesiredPitch, ALLOW_ONLINE_CHANGES),
  setRollValue("RollValue", this, 0, ALLOW_ONLINE_CHANGES),
  setHeadingValue("HeadingValue", this, 20, ALLOW_ONLINE_CHANGES),
  itsPIDImage(256, 256, ZEROS),
  itsSubImage(256, 256, ZEROS),
  itsPrevDepth(0),
  itsPrevPrevDepth(0),
  itsDepthCount(0)
{
  if (itsSimulation.getVal())
    {
      itsBeeStemSim = nub::soft_ref<BeeStemSim>(new BeeStemSim(mgr));
      addSubComponent(itsBeeStemSim);

    }
  else
    {
      itsBeeStemTiny = nub::soft_ref<BeeStemTiny>(new BeeStemTiny(mgr,"BeeStemTiny", "BeeStemTiny", "/dev/ttyS1"));
      addSubComponent(itsBeeStemTiny);

      itsForwardCam = nub::ref<V4L2grabber>(new V4L2grabber(mgr));
      itsForwardCam->exportOptions(MC_RECURSE);
      addSubComponent(itsForwardCam);

      itsBottomCam = nub::ref<V4L2grabber>(new V4L2grabber(mgr));
      itsBottomCam->exportOptions(MC_RECURSE);
      addSubComponent(itsBottomCam);
    }

}

void SubController::start1()
{
  if (!(itsSimulation.getVal()))
    {
      itsForwardCam->setModelParamVal("FrameGrabberDevice",std::string("/dev/video0"));
      itsBottomCam->setModelParamVal("FrameGrabberDevice",std::string("/dev/video1"));
    }
}

void SubController::start2()
{

  killMotors();
  sleep(1);

  //setup pid loop thread
  itsThreadServer.reset(new WorkThreadServer("SubController",1)); //start a single worker thread
  itsThreadServer->setFlushBeforeStopping(false);
  rutz::shared_ptr<SubControllerPIDLoop> j(new SubControllerPIDLoop(this));
  itsThreadServer->enqueueJob(j);

  setHeading(getHeading());

  if (!(itsSimulation.getVal()))
    {
      itsForwardCam->startStream();
      itsBottomCam->startStream();
    }

}

// ######################################################################
SubController::~SubController()
{

  //killMotors();
}

void SubController::sendHeartBeat()
{
  //  itsBeeStem->setHeartBeat();
  usleep(10000);
}

// ######################################################################
void SubController::initSensorVals()
{
  itsBeeStemTiny->getSensors(itsCurrentHeading,
                             itsCurrentPitch,
                             itsCurrentRoll,
                             itsCurrentDepth,
                             itsCurrentIntPressure);
}

// ######################################################################
bool SubController::setHeading(int heading)
{
  //<TODO mmontalbo> check for heading greater than 360
  heading = heading % 360;

  itsDesiredHeading = heading;
  return true;
}

// ######################################################################
bool SubController::setPitch(int pitch)
{
  itsDesiredPitch = pitch;
  return true;
}

bool SubController::setDepth(int depth)
{
  itsDesiredDepth = depth;
  return true;
}


// ######################################################################
bool SubController::setRoll(int roll)
{
  itsDesiredRoll = roll;
  return true;
}

bool SubController::setSpeed(int speed)
{
  itsDesiredSpeed = speed;
  itsCurrentSpeed = itsDesiredSpeed;
  return true;
}

bool SubController::setTurningSpeed(int speed)
{
  itsDesiredTurningSpeed = speed;
  return true;
}


// ######################################################################

void SubController::updateHeading(unsigned int heading)
{
  itsCurrentHeading = heading;
}

void SubController::updatePitch(int pitch)
{
  itsCurrentPitch = pitch;
}

void SubController::updateRoll(int roll)
{
  itsCurrentRoll = roll;
}

void SubController::updateDepth(unsigned int depth)
{

  itsAvgDepth.push_back(depth);

  if (itsAvgDepth.size() > 40)
    {

      long avg = 0;
      for(std::list<int>::iterator itr=itsAvgDepth.begin(); itr != itsAvgDepth.end(); ++itr)
        avg += *itr;
      itsAvgDepth.pop_front();

      itsCurrentDepth = avg/itsAvgDepth.size();
    }
}

void SubController::updatePID()
{

  if (itsSimulation.getVal())
    itsBeeStemSim->getSensors(itsCurrentHeading, itsCurrentPitch, itsCurrentRoll, itsCurrentDepth, itsCurrentIntPressure);
  else
    itsBeeStemTiny->getSensors(itsCurrentHeading, itsCurrentPitch, itsCurrentRoll, itsCurrentDepth, itsCurrentIntPressure);

  if(guiOn.getVal())
    {
      genPIDImage();
      genSubImage();
    }


  if (pidOn.getVal())
    {

      int desiredHeading;

      if(itsCurrentHeading >= 180) {
        desiredHeading = itsDesiredHeading + 360 - itsCurrentHeading;
      }
      else {
        desiredHeading = itsDesiredHeading - itsCurrentHeading;
      }

      while(desiredHeading > 360)
        desiredHeading -= 360;
      while(desiredHeading < 0)
        desiredHeading += 360;

      if(desiredHeading >180)
        desiredHeading = -(360 - desiredHeading);

      float headingCorrection = (float)itsHeadingPID.update(desiredHeading, 0);
      float pitchCorrection = itsPitchPID.update((float)itsDesiredPitch, (float)itsCurrentPitch);
      //float rollCorrection = itsRollPID.update(itsDesiredRoll, (float)itsCurrentRoll);
      float depthCorrection = itsDepthPID.update((float)itsDesiredDepth, (float)itsCurrentDepth);

      if (depthCorrection > 75) depthCorrection = 75;
      if (depthCorrection < -75) depthCorrection = -75;

      // LINFO("Heading | Desired: %d, Current %d,  Correction %f", normDes, normHead, headingCorrection);

      int polCorr = 1;

      if (!itsSimulation.getVal())
        {
          polCorr = -1;
        }

      int thruster_Fwd_Left;
      int thruster_Fwd_Right;
      if(itsDesiredTurningSpeed != 0)
        {
          thruster_Fwd_Left  = polCorr*(THRUSTER_FWD_LEFT_THRESH  - (int)(itsDesiredTurningSpeed) - itsCurrentSpeed);
          thruster_Fwd_Right = THRUSTER_FWD_RIGHT_THRESH  + (int)(itsDesiredTurningSpeed) - itsCurrentSpeed;
        }
      else
        {
          thruster_Fwd_Left  = polCorr*(THRUSTER_FWD_LEFT_THRESH  - (int)(headingCorrection) - itsCurrentSpeed);
          thruster_Fwd_Right = THRUSTER_FWD_RIGHT_THRESH  + (int)(headingCorrection) - itsCurrentSpeed;
        }

      itsCurrentThruster_Fwd_Left = (int)(thruster_Fwd_Left * itsSpeedScale);
      itsCurrentThruster_Fwd_Right = (int)(thruster_Fwd_Right * itsSpeedScale);

      if (itsDesiredDepth != -1) //hack to set the depth manually
        {
          int thruster_Up_Left  =  -1*(THRUSTER_UP_LEFT_THRESH  + (int)((depthCorrection/itsDepthRatio) + pitchCorrection));
          int thruster_Up_Right =  THRUSTER_UP_RIGHT_THRESH  + (int)((depthCorrection/itsDepthRatio) + pitchCorrection);
          int thruster_Up_Back  =  THRUSTER_UP_BACK_THRESH  + (int)(depthCorrection - pitchCorrection);

          itsCurrentThruster_Up_Left =  (int)(thruster_Up_Left * itsSpeedScale);
          itsCurrentThruster_Up_Right =  (int)(thruster_Up_Right * itsSpeedScale);
          itsCurrentThruster_Up_Back =  (int)(thruster_Up_Back * itsSpeedScale);

//           itsCurrentThruster_Up_Back = 0;
//           itsCurrentThruster_Up_Left = 0;
//           itsCurrentThruster_Up_Right = 0;
        }
    }



  if (!motorsOn.getVal())
    {
      itsCurrentThruster_Up_Left = 0;
      itsCurrentThruster_Fwd_Left = 0;
      itsCurrentThruster_Up_Back = 0;
      itsCurrentThruster_Fwd_Right = 0;
      itsCurrentThruster_Up_Right = 0;
    }

  if (itsSimulation.getVal())
    itsBeeStemSim->setThrusters(
                                itsCurrentThruster_Up_Left,
                                itsCurrentThruster_Fwd_Left,
                                itsCurrentThruster_Up_Back,
                                itsCurrentThruster_Fwd_Right,
                                itsCurrentThruster_Up_Right);
  else
    itsBeeStemTiny->setThrusters(
                                 itsCurrentThruster_Up_Left,
                                 itsCurrentThruster_Fwd_Left,
                                 itsCurrentThruster_Up_Back,
                                 itsCurrentThruster_Fwd_Right,
                                 itsCurrentThruster_Up_Right);



}

void SubController::setThruster(int thruster, int val)
{
  //LINFO("Set Thruster %i %i", thruster, val);
  switch(thruster)
    {
    case THRUSTER_UP_LEFT:   itsCurrentThruster_Up_Left = val;  break;
    case THRUSTER_UP_RIGHT:  itsCurrentThruster_Up_Right = val;  break;
    case THRUSTER_UP_BACK:   itsCurrentThruster_Up_Back = val;  break;
    case THRUSTER_FWD_RIGHT: itsCurrentThruster_Fwd_Right = val;  break;
    case THRUSTER_FWD_LEFT:  itsCurrentThruster_Fwd_Left = val;  break;
    default: LINFO("Invalid motor %i", thruster); break;
    }

}

void SubController::setIntPressure(unsigned int pressure)
{
  itsCurrentIntPressure = pressure;
}


void SubController::killMotors()
{
  //itsBeeStem->setThrusters(0, 0, 0, 0, 0);
}


void SubController::paramChanged(ModelParamBase* const param, const bool valueChanged, ParamClient::ChangeStatus* status)
{

  //////// Pitch PID constants/gain change ////////
  if (param == &pitchP && valueChanged)
    itsPitchPID.setPIDPgain(pitchP.getVal());
  else if(param == &pitchI && valueChanged)
    itsPitchPID.setPIDIgain(pitchI.getVal());
  else if(param == &pitchD && valueChanged)
    itsPitchPID.setPIDDgain(pitchD.getVal());

  //////// Roll PID constants/gain change ///////
  else if(param == &rollP && valueChanged)
    itsRollPID.setPIDPgain(rollP.getVal());
  else if(param == &rollI && valueChanged)
    itsRollPID.setPIDIgain(rollI.getVal());
  else if(param == &rollD && valueChanged)
    itsRollPID.setPIDDgain(rollD.getVal());

  //////// Heading PID constants/gain change ////
  else if(param == &headingP && valueChanged)
    itsHeadingPID.setPIDPgain(headingP.getVal());
  else if(param == &headingI && valueChanged)
    itsHeadingPID.setPIDIgain(headingI.getVal());
  else if(param == &headingD && valueChanged)
    itsHeadingPID.setPIDDgain(headingD.getVal());

  /////// Depth PID constants/gain change ////
  else if(param == &depthP && valueChanged)
    itsDepthPID.setPIDPgain(depthP.getVal());
  else if(param == &depthI && valueChanged)
    itsDepthPID.setPIDIgain(depthI.getVal());
  else if(param == &depthD && valueChanged)
    itsDepthPID.setPIDDgain(depthD.getVal());
  else if(param == &setDepthValue && valueChanged)
    setDepth(setDepthValue.getVal());
  else if(param == &setPitchValue && valueChanged)
    setPitch(setPitchValue.getVal());
  else if(param == &setRollValue && valueChanged)
    setRoll(setRollValue.getVal());
  else if(param == &setHeadingValue && valueChanged)
    setHeading(setHeadingValue.getVal());

  //Thruster_Settings
  else if(param == &setCurrentThruster_Up_Left && valueChanged)
    setThruster(THRUSTER_UP_LEFT, setCurrentThruster_Up_Left.getVal());
  else if(param == &setCurrentThruster_Up_Right && valueChanged)
    setThruster(THRUSTER_UP_RIGHT, setCurrentThruster_Up_Right.getVal());
  else if(param == &setCurrentThruster_Up_Back && valueChanged)
    setThruster(THRUSTER_UP_BACK, setCurrentThruster_Up_Back.getVal());
  else if(param == &setCurrentThruster_Fwd_Left && valueChanged)
    setThruster(THRUSTER_FWD_LEFT, setCurrentThruster_Fwd_Left.getVal());
  else if(param == &setCurrentThruster_Fwd_Right && valueChanged)
    setThruster(THRUSTER_FWD_RIGHT, setCurrentThruster_Fwd_Right.getVal());

  else if(param == &speedScale && valueChanged)
    itsSpeedScale = speedScale.getVal();
  else if(param == &depthRatio && valueChanged)
    itsDepthRatio = depthRatio.getVal();


}

////////////////// GUI Related ////////////////////////////////////

void SubController::genPIDImage()
{
  static int x = 0;

  int depthErr, pitchErr, headingErr, rollErr;

  depthErr = (itsDesiredDepth - itsCurrentDepth);
  pitchErr = (itsDesiredPitch - itsCurrentPitch);
  headingErr = (itsDesiredHeading - itsCurrentHeading);
  rollErr = (itsDesiredRoll - itsCurrentRoll);

  while (headingErr <= -180) headingErr += 360;
  while (headingErr > 180) headingErr -= 360;



  int depth_y = (256/2) + (depthErr*2);
  if (depth_y > 255) depth_y = 255;
  if (depth_y < 0) depth_y = 0;

  int pitch_y = (256/2) + (pitchErr*2);
  if (pitch_y > 255) pitch_y = 255;
  if (pitch_y < 0) pitch_y = 0;

  int heading_y = (256/2) + (headingErr*2);
  if (heading_y > 255) heading_y = 255;
  if (heading_y < 0) heading_y = 0;

  int roll_y = (256/2) + (rollErr*2);
  if (roll_y > 255) roll_y = 255;
  if (roll_y < 0) roll_y = 0;


  if (!x)
    {
      itsPIDImage.clear();
      drawLine(itsPIDImage, Point2D<int>(0, 256/2), Point2D<int>(256, 256/2), PixRGB<byte>(255,0,0));
    }
  if(depthPIDDisplay.getVal()) itsPIDImage.setVal(x,depth_y,PixRGB<byte>(0,255,0));
  if(pitchPIDDisplay.getVal()) itsPIDImage.setVal(x,pitch_y,PixRGB<byte>(255,0,0));
  if(headingPIDDisplay.getVal()) itsPIDImage.setVal(x,heading_y,PixRGB<byte>(0,0,255));
  if(rollPIDDisplay.getVal()) itsPIDImage.setVal(x,roll_y,PixRGB<byte>(255,255,0));

  x = (x+1)%256;

}

void SubController::genSubImage()
{
  itsSubImage.clear();
  drawCircle(itsSubImage, Point2D<int>(128,128), 100, PixRGB<byte>(255,0,0));
  int x = (int)(100.0*cos((itsCurrentHeading-90)*(M_PI/180))); //shift by 90 so that 0 is up
  int y = (int)(100.0*sin((itsCurrentHeading-90)*(M_PI/180)));
  drawArrow(itsSubImage, Point2D<int>(128,128), Point2D<int>(128+x,128+y), PixRGB<byte>(0,255,0));

  int dx = (int)(100.0*cos((itsDesiredHeading-90)*(M_PI/180))); //shift by 90 so that 0 is up
  int dy = (int)(100.0*sin((itsDesiredHeading-90)*(M_PI/180)));
  drawArrow(itsSubImage, Point2D<int>(128,128), Point2D<int>(128+dx,128+dy), PixRGB<byte>(0,0,255));

}

const Image<PixRGB<byte> > SubController::getImage(int camera)
{

  if (itsSimulation.getVal())
    {
      return itsBeeStemSim->getImage(camera);
    }
  else
    {
      GenericFrame inputFrame;
      Image<PixRGB<byte> > img;

      switch(camera)
        {
        case 1:
          inputFrame = itsBottomCam->readFrame();
          if(!inputFrame.initialized()) return Image<PixRGB<byte> >();

          img = inputFrame.asRgb();
          break;

        case 2:
          inputFrame = itsForwardCam->readFrame();
          if(!inputFrame.initialized()) return Image<PixRGB<byte> >();

          img = inputFrame.asRgb();
          break;

        default:
          return Image<PixRGB<byte> >();
        }

      img = rescale(img, 320, 240);

      return img;
    }

  return Image<PixRGB<byte> >();

}

const Dims SubController::peekDims()
{
  return Dims(320,240);
}

bool SubController::isSimMode()
{
  return itsSimulation.getVal();
}

int SubController::getHeadingErr()
{
  return abs(itsDesiredHeading - itsCurrentHeading);
}

int SubController::getDepthErr()
{
  return abs(itsDesiredDepth - itsCurrentDepth);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
