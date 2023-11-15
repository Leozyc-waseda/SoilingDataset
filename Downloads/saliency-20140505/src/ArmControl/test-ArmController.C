/*!@file ArmControl/test-ArmController.C test the arm controller  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/test-ArmController.C $
// $Id: test-ArmController.C 10794 2009-02-08 06:21:09Z itti $
//

//
#include "Image/Layout.H"
#include "Image/MatrixOps.H"
#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "GUI/GeneralGUI.H"
#include "ArmControl/ArmController.H"
#include "ArmControl/ArmSim.H"
#include "GUI/XWinManaged.H"
#include "GUI/ImageDisplayStream.H"
#include "Raster/GenericFrame.H"
#include "Media/FrameSeries.H"
#include "ArmControl/RobotArm.H"
#include "Transport/FrameInfo.H"


#include <unistd.h>
#include <stdio.h>
#include <signal.h>

#define KEY_UP 98
#define KEY_DOWN 104
#define KEY_LEFT 100
#define KEY_RIGHT 102

#define ARMSIM

//! Signal handler (e.g., for control-C)
void terminate(int s)
{
        LERROR("*** INTERRUPT ***");
        exit(0);
}

int getKey(nub::soft_ref<OutputFrameSeries> &ofs)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow("GUIDisplay")
    : rutz::shared_ptr<XWinManaged>();
  return uiwin->getLastKeyPress();
}

void recordCmd(nub::soft_ref<ArmController> &armCtrl, std::vector<ArmController::JointPos> &jointPositions)
{

  ArmController::JointPos jointPos = armCtrl->getJointPos();
  LINFO("Recording Pos: %i %i %i %i %i %i",
      jointPos.base,
      jointPos.sholder,
      jointPos.elbow,
      jointPos.wrist1,
      jointPos.wrist2,
      jointPos.gripper);
  jointPositions.push_back(jointPos);

}

void playCmd(nub::soft_ref<ArmController> &armCtrl, std::vector<ArmController::JointPos> &jointPositions)
{
  for(uint i=0; i<jointPositions.size(); i++)
  {
    ArmController::JointPos jointPos = jointPositions[i];
      LINFO("Playing Pos: %i %i %i %i %i %i",
        jointPos.base,
        jointPos.sholder,
        jointPos.elbow,
        jointPos.wrist1,
        jointPos.wrist2,
        jointPos.gripper);
      armCtrl->setJointPos(jointPos);
  }
}





int main(int argc, const char **argv)
{
  // Instantiate a ModelManager:
  ModelManager *mgr = new ModelManager("Test Arm Controller");

//  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(*mgr));
//  mgr->addSubComponent(ifs);

  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
  mgr->addSubComponent(ofs);

//  nub::soft_ref<ArmSim> robotArm(new ArmSim(*mgr));
  nub::soft_ref<Scorbot> robotArm(new Scorbot(*mgr,"Scorbot", "Scorbot", "/dev/ttyUSB0"));

  nub::soft_ref<ArmController> armController(new ArmController(*mgr,
        "ArmController", "ArmController", robotArm));
  mgr->addSubComponent(armController);

  nub::soft_ref<GeneralGUI> armGUI(new GeneralGUI(*mgr));
  mgr->addSubComponent(armGUI);

  mgr->exportOptions(MC_RECURSE);


  // Parse command-line:
  if (mgr->parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // let's get all our ModelComponent instances started:
  mgr->start();

  //start the gui thread
  armGUI->startThread(ofs);
  sleep(1);
  //setup gui for various objects
  armGUI->setupGUI(armController.get(), true);

  //Main GUI Window


  armGUI->addMeter(armController->getMotor_Base_Ptr(),
      "Motor_Base", -100, PixRGB<byte>(0, 255, 0));
  armGUI->addMeter(armController->getMotor_Sholder_Ptr(),
      "Motor_Sholder", -100, PixRGB<byte>(0, 255, 0));
  armGUI->addMeter(armController->getMotor_Elbow_Ptr(),
      "Motor_Elbow", -100, PixRGB<byte>(0, 255, 0));
  armGUI->addMeter(armController->getMotor_Wrist1_Ptr(),
      "Motor_Wrist1", -100, PixRGB<byte>(0, 255, 0));
  armGUI->addMeter(armController->getMotor_Wrist2_Ptr(),
      "Motor_Wrist2", -100, PixRGB<byte>(0, 255, 0));


  armGUI->addImage(armController->getPIDImagePtr());

  //set The min-max joint pos
  ArmController::JointPos jointPos;
  jointPos.base = 8000;
  jointPos.sholder = 5000;
  jointPos.elbow = 5000;
  jointPos.wrist1 = 2000;
  jointPos.wrist2 = 2000;
  //jointPos.base = 3000;
  //jointPos.sholder = 3000;
  //jointPos.elbow = 2000;
  //jointPos.wrist1 = 0;
  //jointPos.wrist2 = 0;
  armController->setMaxJointPos(jointPos);

  jointPos.base = -8000;
  jointPos.sholder = -1500;
  jointPos.elbow = -3000;
  jointPos.wrist1 = -2000;
  jointPos.wrist2 = -2000;
  armController->setMinJointPos(jointPos);



  int speed = 75;
  std::vector<ArmController::JointPos> jointPositions;
  armController->setMotorsOn(true);
  armController->setPidOn(true);
  while(1) {
          //armController->setBasePos(1000, false);
          //sleep(6);
          //armController->setBasePos(0, false);
          //sleep(6);
    if(0){//if it's sim
      robotArm->simLoop();
      Image<PixRGB<byte> > armCam = flipVertic(robotArm->getFrame(-1));
      ofs->writeRGB(armCam, "ArmSiminput", FrameInfo("ArmSiminput", SRC_POS));
    }
    int key = -1; //getKey(ofs);
    if (key != -1)
    {
      switch(key)
      {

        //Controll arm
        case 10: //1
          armController->setBasePos(10, true);
          break;
        case 24: //q
          armController->setBasePos(-10, true);
          break;
        case 11: //2
          armController->setSholderPos(-10, true);
          break;
        case 25: //w
          armController->setSholderPos(10, true);
          break;
        case 12: //3
          armController->setElbowPos(10, true);
          break;
        case 26: //e
          armController->setElbowPos(-10, true);
          break;
        case 13: //4
          armController->setWrist1Pos(10, true);
          armController->setWrist2Pos(10, true);
          break;
        case 27: //r
          armController->setWrist1Pos(-10, true);
          armController->setWrist2Pos(-10, true);
          break;
        case 14: //5
          armController->setWrist1Pos(10, true);
          armController->setWrist2Pos(-10, true);
          break;
        case 28: //t
          armController->setWrist1Pos(-10, true);
          armController->setWrist2Pos(10, true);
          break;
        case 15: //6
          armController->setGripper(0);
          break;
        case 29: //y
          armController->setGripper(1);
          break;
        case 65: //space
          armController->killMotors();
          break;

        //Record and play joint positions
        case 54: //c record position
          recordCmd(armController, jointPositions);
          break;
        case 46: //l play back pos
          playCmd(armController, jointPositions);
          break;
        case 56: //b clear joint pos
          LINFO("Clearing joint positions");
          jointPositions.clear();
          break;

        //Change speed
        case KEY_UP:
          speed += 1;
          break;
        case KEY_DOWN:
          speed -= 1;
          break;

      }

      if (speed < 0) speed = 0;
      if (speed > 100) speed = 100;

      LINFO("Key = %i", key);
    }

    usleep(1000);
  }//End While(1)

  // stop all our ModelComponents
  mgr->stop();

  // all done!
  return 0;
}

