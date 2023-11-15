/*!@file ArmControl/ArmController.C  Control motors and pid */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/ArmController.C $
// $Id: ArmController.C 10794 2009-02-08 06:21:09Z itti $
//

#include "ArmControl/ArmController.H"
#include "ArmControl/RobotArm.H"
#include "Component/ModelOptionDef.H"
#include "Devices/DeviceOpts.H"
#include "Util/Assert.H"
#include "Image/DrawOps.H"

namespace
{
  class ArmControllerPIDLoop : public JobWithSemaphore
  {
    public:
      ArmControllerPIDLoop(ArmController* armCtrl)
        :
          itsArmController(armCtrl),
          itsPriority(1),
          itsJobType("controllerLoop")
    {}

      virtual ~ArmControllerPIDLoop() {}

      virtual void run()
      {
        ASSERT(itsArmController);
        while(1)
        {
          if (itsArmController->isControllerOn())
            itsArmController->updatePID();
          else
            sleep(1);
        }
      }

      virtual const char* jobType() const
      { return itsJobType.c_str(); }

      virtual int priority() const
      { return itsPriority; }

    private:
      ArmController* itsArmController;
      const int itsPriority;
      const std::string itsJobType;
  };
}

// ######################################################################
ArmController::ArmController(OptionManager& mgr,
    const std::string& descrName,
    const std::string& tagName,
    nub::soft_ref<RobotArm> robotArm ):

  ModelComponent(mgr, descrName, tagName),
  itsDesiredBase(0),
  itsDesiredSholder(0),
  itsDesiredElbow(0),
  itsDesiredWrist1(0),
  itsDesiredWrist2(0),
  itsDesiredGripper(0),

  itsDesiredSpeed(0),

  itsCurrentBase(0),
  itsCurrentSholder(0),
  itsCurrentElbow(0),
  itsCurrentWrist1(0),
  itsCurrentWrist2(0),
  itsCurrentGripper(0),

  itsCurrentSpeed(0),

  itsBasePID(0.8f, 0.0, 0.2, -20, 20,
      ERR_THRESH,30,-30),
  itsSholderPID(1.0f, 0, 0.65, -20, 20,
      ERR_THRESH,20,-60),
  itsElbowPID(0.7f, 0, 0.6, -20, 20,
      ERR_THRESH,44,-20),
  itsWrist1PID(0.7f, 0, 0.2, -20, 20,
      ERR_THRESH,30,-30),
  itsWrist2PID(0.7f, 0, 0.2, -20, 20,
      ERR_THRESH,30,-30),

  itsCurrentMotor_Base(0),
  itsCurrentMotor_Sholder(0),
  itsCurrentMotor_Elbow(0),
  itsCurrentMotor_Wrist1(0),
  itsCurrentMotor_Wrist2(0),

  setCurrentMotor_Base("MotorBase", this, 0, ALLOW_ONLINE_CHANGES),
  setCurrentMotor_Sholder("MotorSholder", this, 0, ALLOW_ONLINE_CHANGES),
  setCurrentMotor_Elbow("MotorElbow", this, 0, ALLOW_ONLINE_CHANGES),
  setCurrentMotor_Wrist1("MotorWrist1", this, 0, ALLOW_ONLINE_CHANGES),
  setCurrentMotor_Wrist2("MotorWrist2", this, 0, ALLOW_ONLINE_CHANGES),


  baseP("baseP", this, 0.8, ALLOW_ONLINE_CHANGES),
  baseI("baseI", this, 0, ALLOW_ONLINE_CHANGES),
  baseD("baseD", this, 0.2, ALLOW_ONLINE_CHANGES),

  sholderP("sholderP", this, 1.0, ALLOW_ONLINE_CHANGES),
  sholderI("sholderI", this, 0, ALLOW_ONLINE_CHANGES),
  sholderD("sholderD", this, 0.65, ALLOW_ONLINE_CHANGES),

  elbowP("elbowP", this, 0.7, ALLOW_ONLINE_CHANGES),
  elbowI("elbowI", this, 0, ALLOW_ONLINE_CHANGES),
  elbowD("elbowD", this, 0.6, ALLOW_ONLINE_CHANGES),

  wrist1P("wrist1P", this, 0.7, ALLOW_ONLINE_CHANGES),
  wrist1I("wrist1I", this, 0, ALLOW_ONLINE_CHANGES),
  wrist1D("wrist1D", this, 0.2, ALLOW_ONLINE_CHANGES),

  wrist2P("wrist2P", this, 0.7, ALLOW_ONLINE_CHANGES),
  wrist2I("wrist2I", this, 0, ALLOW_ONLINE_CHANGES),
  wrist2D("wrist2D", this, 0.2, ALLOW_ONLINE_CHANGES),

  motorsOn("motorsOn", this, false, ALLOW_ONLINE_CHANGES),
  pidOn("pidOn", this, false, ALLOW_ONLINE_CHANGES),
  guiOn("guiOn", this, true, ALLOW_ONLINE_CHANGES),
  controllerOn("controllerOn", this, true, ALLOW_ONLINE_CHANGES),
  motorsSpeed("motorsSpeed", this, 100, ALLOW_ONLINE_CHANGES),

  basePIDDisplay("Base PID Disp", this, true, ALLOW_ONLINE_CHANGES),
  sholderPIDDisplay("Sholder PID Disp", this, false, ALLOW_ONLINE_CHANGES),
  elbowPIDDisplay("Elbow PID Disp", this, false, ALLOW_ONLINE_CHANGES),
  wrist1PIDDisplay("Wrist1 PID Disp", this, false, ALLOW_ONLINE_CHANGES),
  wrist2PIDDisplay("Wrist2 PID Disp", this, false, ALLOW_ONLINE_CHANGES),

  basePos("BasePos", this, itsDesiredBase, ALLOW_ONLINE_CHANGES),
  sholderPos("SholderPos", this, itsDesiredSholder, ALLOW_ONLINE_CHANGES),
  elbowPos("ElbowPos", this, itsDesiredElbow, ALLOW_ONLINE_CHANGES),
  wrist1Pos("Wrist1Pos", this, itsDesiredWrist1, ALLOW_ONLINE_CHANGES),
  wrist2Pos("Wrist2Pos", this, itsDesiredWrist2, ALLOW_ONLINE_CHANGES),

  itsPIDImage(256, 256, ZEROS),
  itsRobotArm(robotArm),

  itsAvgn(0),
  itsAvgtime(0),
  itsLps(0)

{
  //itsRobotArm = nub::soft_ref<Scorbot>(new Scorbot(mgr,"Scorbot", "Scorbot", "/dev/ttyUSB0"));
  //itsRobotArm = nub::soft_ref<ArmSim>(new ArmSim(mgr));
  addSubComponent(itsRobotArm);

  itsMaxJointPos.base = 0;          itsMinJointPos.base = 0;
  itsMaxJointPos.sholder = 0;       itsMinJointPos.sholder = 0;
  itsMaxJointPos.elbow = 0;         itsMinJointPos.elbow = 0;
  itsMaxJointPos.wrist1 = 0;        itsMinJointPos.wrist1 = 0;
  itsMaxJointPos.wrist2 = 0;        itsMinJointPos.wrist2 = 0;
  itsMaxJointPos.gripper = 0;       itsMinJointPos.gripper = 0;

  //For scrobt
  // Base -5647~7431
  // Sholder -1296~3264
  // Elbow -1500~4630
}

void ArmController::start2()
{

  killMotors();
  sleep(1);

  itsCurrentBase    = itsRobotArm->getEncoder(RobotArm::BASE);
  itsCurrentElbow   = itsRobotArm->getEncoder(RobotArm::ELBOW);
  itsCurrentSholder = itsRobotArm->getEncoder(RobotArm::SHOLDER);
  itsCurrentWrist1  = itsRobotArm->getEncoder(RobotArm::WRIST1);
  itsCurrentWrist2  = itsRobotArm->getEncoder(RobotArm::WRIST2);

  //setup pid loop thread
  itsThreadServer.reset(new WorkThreadServer("ArmController",1)); //start a single worker thread
  itsThreadServer->setFlushBeforeStopping(false);
  rutz::shared_ptr<ArmControllerPIDLoop> j(new ArmControllerPIDLoop(this));
  itsThreadServer->enqueueJob(j);


}

// ######################################################################
ArmController::~ArmController()
{
  killMotors();
}

// ######################################################################
bool ArmController::setBasePos(int base, bool rel)
{
  if (rel)
    itsDesiredBase += base;
  else
    itsDesiredBase = base;
  return true;
}

// ######################################################################
bool ArmController::setElbowPos(int elbow, bool rel)
{
  if (rel)
    itsDesiredElbow += elbow;
  else
    itsDesiredElbow = elbow;

  return true;
}

// ######################################################################
bool ArmController::setSholderPos(int sholder, bool rel)
{
  if (rel)
    itsDesiredSholder += sholder;
  else
    itsDesiredSholder = sholder;

  return true;
}

// ######################################################################
bool ArmController::setWrist1Pos(int wrist1, bool rel)
{
  if (rel)
  {
    itsDesiredWrist1 += wrist1;
  //  itsDesiredWrist2 -= wrist1;
  }
  else
  {
    itsDesiredWrist1 = wrist1;
    //itsDesiredWrist2 = -1*wrist1;
  }
  return true;
}

bool ArmController::setWrist2Pos(int wrist2, bool rel)
{
  if (rel)
  {
    //itsDesiredWrist1 += wrist2;
    itsDesiredWrist2 += wrist2;
  }
  else
  {
    //itsDesiredWrist1 = wrist2;
    itsDesiredWrist2 = wrist2;
  }
  return true;
}

// ######################################################################
bool ArmController::setSpeed(int speed)
{
  itsBasePID.setSpeed((float)speed/100.0);
  itsSholderPID.setSpeed((float)speed/100.0);
  itsElbowPID.setSpeed((float)speed/100.0);
  itsWrist1PID.setSpeed((float)speed/100.0);
  itsWrist2PID.setSpeed((float)speed/100.0);
  return true;
}


// ######################################################################

void ArmController::updateBase(int base)
{
  itsCurrentBase = base;
}

void ArmController::updateElbow(unsigned int elbow)
{
  itsCurrentElbow = elbow;
}

void ArmController::updateSholder(int sholder)
{
  itsCurrentSholder = sholder;
}

void ArmController::updateWrist1(unsigned int wrist1)
{
  itsCurrentWrist1 = wrist1 ;
}

void ArmController::updateWrist2(unsigned int wrist2)
{
  itsCurrentWrist1 = wrist2 ;
}

ArmController::JointPos ArmController::getJointPos()
{
  JointPos jointPos;
  jointPos.base = itsRobotArm->getEncoder(RobotArm::BASE);
  jointPos.sholder = itsRobotArm->getEncoder(RobotArm::SHOLDER);
  jointPos.elbow = itsRobotArm->getEncoder(RobotArm::ELBOW);
  jointPos.wrist1 = itsRobotArm->getEncoder(RobotArm::WRIST1);
  jointPos.wrist2 = itsRobotArm->getEncoder(RobotArm::WRIST2);
  jointPos.gripper = itsCurrentGripper;

  return jointPos;
}

bool ArmController::isFinishMove()
{
    if(abs(itsDesiredBase - itsCurrentBase) > 40 ||
      abs(itsDesiredSholder - itsCurrentSholder) > 40 ||
      abs(itsDesiredElbow - itsCurrentElbow) > 40 ||
      abs(itsDesiredWrist1 - itsCurrentWrist1) > 40 ||
      abs(itsDesiredWrist2 - itsCurrentWrist2) > 40 )
      return false;

    return true;
}

void ArmController::setMinJointPos(const ArmController::JointPos &jointPos)
{

  itsMinJointPos.base = jointPos.base;
  itsMinJointPos.sholder = jointPos.sholder;
  itsMinJointPos.elbow = jointPos.elbow;
  itsMinJointPos.wrist1 = jointPos.wrist1;
  itsMinJointPos.wrist2 = jointPos.wrist2;
  itsMinJointPos.gripper = jointPos.gripper;

}

void ArmController::setMaxJointPos(const ArmController::JointPos &jointPos)
{
  itsMaxJointPos.base = jointPos.base;
  itsMaxJointPos.sholder = jointPos.sholder;
  itsMaxJointPos.elbow = jointPos.elbow;
  itsMaxJointPos.wrist1 = jointPos.wrist1;
  itsMaxJointPos.wrist2 = jointPos.wrist2;
  itsMaxJointPos.gripper = jointPos.gripper;


}

bool ArmController::setJointPos(const ArmController::JointPos &jointPos, bool block)
{
  if (!jointPos.reachable)
    return true;

  itsDesiredBase = jointPos.base;
  itsDesiredSholder = jointPos.sholder;
  itsDesiredElbow = jointPos.elbow;
  itsDesiredWrist1 = jointPos.wrist2;
  itsDesiredWrist2 = jointPos.wrist1;
  //itsDesiredGripper = jointPos.gripper;

  if (block) //wait untill move is finished
  {
    while(!isFinishMove()){
      itsRobotArm->simLoop();
//      itsRobotArm->getFrame(-1);
    }
  }

  //After the move, set the gripper
  //if (itsDesiredGripper != itsCurrentGripper)
  //setGripper(itsDesiredGripper);

  return isFinishMove();

}

void ArmController::resetJointPos(JointPos& jointPos, int val)
{
  jointPos.base = val;
  jointPos.sholder = val;
  jointPos.elbow = val;
  jointPos.wrist1 = val;
  jointPos.wrist2 = val;
  jointPos.gripper = val;

}

void ArmController::setGripper(int pos)
{

  switch(pos)
  {
    case 0:
      itsRobotArm->setMotor(RobotArm::GRIPPER, -75);
      break;
    case 1:
      itsRobotArm->setMotor(RobotArm::GRIPPER, 75);
      break;
  }
  itsCurrentGripper = pos;
  sleep(1);
}

// ######################################################################
void ArmController::updatePID()
{
        //Showing the running speed
        //getLps();

  itsCurrentBase    = itsRobotArm->getEncoder(RobotArm::BASE);
  itsCurrentElbow   = itsRobotArm->getEncoder(RobotArm::ELBOW);
  itsCurrentSholder = itsRobotArm->getEncoder(RobotArm::SHOLDER);
  itsCurrentWrist1  = itsRobotArm->getEncoder(RobotArm::WRIST1);
  itsCurrentWrist2  = itsRobotArm->getEncoder(RobotArm::WRIST2);

  //limit the joint positions
  if (itsDesiredBase    >= itsMaxJointPos.base)
    itsDesiredBase   = itsMaxJointPos.base;
  if (itsDesiredSholder >= itsMaxJointPos.sholder)
    itsDesiredSholder= itsMaxJointPos.sholder;
  if (itsDesiredElbow   >= itsMaxJointPos.elbow)
    itsDesiredElbow  = itsMaxJointPos.elbow;
  if (itsDesiredWrist1  >= itsMaxJointPos.wrist1)
    itsDesiredWrist1 = itsMaxJointPos.wrist1;
  if (itsDesiredWrist2  >= itsMaxJointPos.wrist2)
    itsDesiredWrist2 = itsMaxJointPos.wrist2;
  if (itsDesiredGripper >= itsMaxJointPos.gripper)
    itsDesiredGripper= itsMaxJointPos.gripper;

  if (itsDesiredBase    < itsMinJointPos.base)
    itsDesiredBase   = itsMinJointPos.base;
  if (itsDesiredSholder < itsMinJointPos.sholder)
    itsDesiredSholder= itsMinJointPos.sholder;
  if (itsDesiredElbow   < itsMinJointPos.elbow)
    itsDesiredElbow  = itsMinJointPos.elbow;
  if (itsDesiredWrist1  < itsMinJointPos.wrist1)
    itsDesiredWrist1 = itsMinJointPos.wrist1;
  if (itsDesiredWrist2  < itsMinJointPos.wrist2)
    itsDesiredWrist2 = itsMinJointPos.wrist2;
  if (itsDesiredGripper < itsMinJointPos.base)
    itsDesiredGripper= itsMinJointPos.gripper;

  if(guiOn.getVal())
  {
    genPIDImage();
  }

  if (pidOn.getVal())
  {
    itsCurrentMotor_Base = (int)itsBasePID.update((float)itsDesiredBase, (float)itsCurrentBase);
    if (!itsBasePID.getRunPID())
    {
      LINFO("Base off");
      itsCurrentMotor_Base = 0;
      itsDesiredBase = itsCurrentBase;
      itsBasePID.setPIDOn(true);
    }
    itsCurrentMotor_Sholder = (int)itsSholderPID.update((float)itsDesiredSholder, (float)itsCurrentSholder);
    if (!itsSholderPID.getRunPID())
    {
      LINFO("Sholder off");
      itsDesiredSholder = itsCurrentSholder;
      itsCurrentMotor_Sholder = 0;
      itsSholderPID.setPIDOn(true);
    }
    itsCurrentMotor_Elbow = (int)itsElbowPID.update((float)itsDesiredElbow, (float)itsCurrentElbow);
    if (!itsElbowPID.getRunPID())
    {
      LINFO("Elbow off");
      itsDesiredElbow = itsCurrentElbow;
      itsCurrentMotor_Elbow = 0;
      itsElbowPID.setPIDOn(true);
    }
    itsCurrentMotor_Wrist1 = (int)itsWrist1PID.update((float)itsDesiredWrist1, (float)itsCurrentWrist1);
    if (!itsWrist1PID.getRunPID())
    {
      itsDesiredWrist1 = itsCurrentWrist1;
      itsCurrentMotor_Wrist1 = 0;
      itsWrist1PID.setPIDOn(true);
    }
    itsCurrentMotor_Wrist2 = (int)itsWrist2PID.update((float)itsDesiredWrist2, (float)itsCurrentWrist2);
    if (!itsWrist2PID.getRunPID())
    {
      itsDesiredWrist2 = itsCurrentWrist2;
      itsCurrentMotor_Wrist2 = 0;
      itsWrist2PID.setPIDOn(true);
    }
  }

  if (!motorsOn.getVal())
  {
    itsCurrentMotor_Base =0;
    itsCurrentMotor_Sholder =0;
    itsCurrentMotor_Elbow =0;
    itsCurrentMotor_Wrist1 =0;
    itsCurrentMotor_Wrist2 =0;

  }



//  LINFO("Set motor %i", itsCurrentMotor_Base);
  itsRobotArm->setMotor(RobotArm::BASE, itsCurrentMotor_Base);
  itsRobotArm->setMotor(RobotArm::SHOLDER, itsCurrentMotor_Sholder);
  itsRobotArm->setMotor(RobotArm::ELBOW, itsCurrentMotor_Elbow);
  itsRobotArm->setMotor(RobotArm::WRIST1, itsCurrentMotor_Wrist1);
  itsRobotArm->setMotor(RobotArm::WRIST2, itsCurrentMotor_Wrist2);

}

void ArmController::setMotor(int motor, int val)
{
  switch(motor)
  {
    case (RobotArm::BASE):
      itsCurrentMotor_Base = val;
      break;
    case (RobotArm::SHOLDER):
      itsCurrentMotor_Sholder = val;
      break;
    case (RobotArm::ELBOW):
      itsCurrentMotor_Elbow = val;
      break;
    case (RobotArm::WRIST1):
      itsCurrentMotor_Wrist1 = val;
      break;
    case (RobotArm::WRIST2):
      itsCurrentMotor_Wrist2 = val;
      break;
    default: LINFO("Invalid motor %i", motor); break;
  }

}


void ArmController::killMotors()
{
  itsRobotArm->stopAllMotors();
  itsCurrentMotor_Base =0;
  itsCurrentMotor_Sholder =0;
  itsCurrentMotor_Elbow =0;
  itsCurrentMotor_Wrist1 =0;
  itsCurrentMotor_Wrist2 =0;
  pidOn.setVal(false);
}


void ArmController::paramChanged(ModelParamBase* const param, const bool valueChanged, ParamClient::ChangeStatus* status)
{

  //////// Base PID constants/gain change ////////
  if (param == &baseP && valueChanged)
    itsBasePID.setPIDPgain(baseP.getVal());
  else if(param == &baseI && valueChanged)
    itsBasePID.setPIDIgain(baseI.getVal());
  else if(param == &baseD && valueChanged)
    itsBasePID.setPIDDgain(baseD.getVal());

  //////// Elbow PID constants/gain change ////
  else if(param == &elbowP && valueChanged)
    itsElbowPID.setPIDPgain(elbowP.getVal());
  else if(param == &elbowI && valueChanged)
    itsElbowPID.setPIDIgain(elbowI.getVal());
  else if(param == &elbowD && valueChanged)
    itsElbowPID.setPIDDgain(elbowD.getVal());

  //////// Sholder PID constants/gain change ///////
  else if(param == &sholderP && valueChanged)
    itsSholderPID.setPIDPgain(sholderP.getVal());
  else if(param == &sholderI && valueChanged)
    itsSholderPID.setPIDIgain(sholderI.getVal());
  else if(param == &sholderD && valueChanged)
    itsSholderPID.setPIDDgain(sholderD.getVal());

  /////// Wrist1 PID constants/gain change ////
  else if(param == &wrist1P && valueChanged)
    itsWrist1PID.setPIDPgain(wrist1P.getVal());
  else if(param == &wrist1I && valueChanged)
    itsWrist1PID.setPIDIgain(wrist1I.getVal());
  else if(param == &wrist1D && valueChanged)
    itsWrist1PID.setPIDDgain(wrist1D.getVal());

  /////// Wrist2 PID constants/gain change ////
  else if(param == &wrist2P && valueChanged)
    itsWrist2PID.setPIDPgain(wrist2P.getVal());
  else if(param == &wrist2I && valueChanged)
    itsWrist2PID.setPIDIgain(wrist2I.getVal());
  else if(param == &wrist2D && valueChanged)
    itsWrist2PID.setPIDDgain(wrist2D.getVal());


  //Motor_Settings
  else if(param == &basePos && valueChanged)
    setBasePos(basePos.getVal());
  else if(param == &sholderPos && valueChanged)
    setSholderPos(sholderPos.getVal());
  else if(param == &elbowPos && valueChanged)
    setElbowPos(elbowPos.getVal());
  else if(param == &wrist1Pos && valueChanged)
    setWrist1Pos(wrist1Pos.getVal());
  else if(param == &wrist2Pos && valueChanged)
    setWrist2Pos(wrist2Pos.getVal());

  //Position settings
  else if(param == &setCurrentMotor_Base && valueChanged)
    setMotor(RobotArm::BASE, setCurrentMotor_Base.getVal());
  else if(param == &setCurrentMotor_Sholder && valueChanged)
    setMotor(RobotArm::SHOLDER, setCurrentMotor_Sholder.getVal());
  else if(param == &setCurrentMotor_Elbow && valueChanged)
    setMotor(RobotArm::ELBOW, setCurrentMotor_Elbow.getVal());
  else if(param == &setCurrentMotor_Wrist2 && valueChanged)
    setMotor(RobotArm::WRIST2, setCurrentMotor_Wrist2.getVal());
  else if(param == &setCurrentMotor_Wrist1 && valueChanged)
    setMotor(RobotArm::WRIST1, setCurrentMotor_Wrist1.getVal());

  else if(param == &motorsSpeed && valueChanged)
    setSpeed(motorsSpeed.getVal());

}

////////////////// GUI Related ////////////////////////////////////

void ArmController::genPIDImage()
{
  static int x = 0;

  int wrist1Err, wrist2Err, baseErr, elbowErr, sholderErr;

  baseErr = (itsDesiredBase - itsCurrentBase);
  elbowErr = (itsDesiredElbow - itsCurrentElbow);
  sholderErr = (itsDesiredSholder - itsCurrentSholder);
  wrist1Err = (itsDesiredWrist1 - itsCurrentWrist1);
  wrist2Err = (itsDesiredWrist2 - itsCurrentWrist2);

  while (elbowErr <= -180) elbowErr += 360;
  while (elbowErr > 180) elbowErr -= 360;


  int scale = 2;

  int wrist1_y = (256/2) + (wrist1Err/scale);
  if (wrist1_y > 255) wrist1_y = 255;
  if (wrist1_y < 0) wrist1_y = 0;

  int wrist2_y = (256/2) + (wrist2Err/scale);
  if (wrist2_y > 255) wrist2_y = 255;
  if (wrist2_y < 0) wrist2_y = 0;

  int base_y = (256/2) + (baseErr/scale);
  if (base_y > 255) base_y = 255;
  if (base_y < 0) base_y = 0;

  int elbow_y = (256/2) + (elbowErr/scale);
  if (elbow_y > 255) elbow_y = 255;
  if (elbow_y < 0) elbow_y = 0;

  int sholder_y = (256/2) + (sholderErr/scale);
  if (sholder_y > 255) sholder_y = 255;
  if (sholder_y < 0) sholder_y = 0;


  if (!x)
  {
    itsPIDImage.clear();
    drawLine(itsPIDImage, Point2D<int>(0, 256/2), Point2D<int>(256, 256/2), PixRGB<byte>(255,0,0));
  }

  if(basePIDDisplay.getVal())
    drawCircle(itsPIDImage, Point2D<int>(x,base_y), 2, PixRGB<byte>(255,0,0));
  if(sholderPIDDisplay.getVal())
    itsPIDImage.setVal(x,sholder_y,PixRGB<byte>(255,255,0));
  if(elbowPIDDisplay.getVal())
    itsPIDImage.setVal(x,elbow_y,PixRGB<byte>(0,0,255));
  if(wrist1PIDDisplay.getVal())
    itsPIDImage.setVal(x,wrist1_y,PixRGB<byte>(0,255,0));

  x = (x+1)%256;

}

int ArmController::getElbowErr()
{
  return (itsDesiredElbow - itsCurrentElbow);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
