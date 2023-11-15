/*! @file ArmControl/train-eye.C  grab slient objects */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// Primary maintainer for this file: Chin-Kai Chang <chinkaic@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/train-eye.C $
// $Id: train-eye.C 10794 2009-02-08 06:21:09Z itti $
//

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/Transforms.H"
#include "Image/DrawOps.H"
#include "Image/Rectangle.H"
#include "Image/MathOps.H"
#include "Image/MatrixOps.H"
#include "Image/Layout.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Neuro/BeoHeadBrain.H"
#include "Util/Timer.H"
#include "GUI/GeneralGUI.H"
#include "GUI/ImageDisplayStream.H"
#include "GUI/DebugWin.H"
#include "GUI/XWinManaged.H"
#include "RCBot/Motion/MotionEnergy.H"
#include "ArmControl/CtrlPolicy.H"
#include "ArmControl/ArmSim.H"
#include "ArmControl/RobotArm.H"
#include "ArmControl/ArmController.H"
#include "ArmControl/ArmPlanner.H"
#include "Devices/DeviceOpts.H"
#include "Util/MathFunctions.H"
#include <unistd.h>
#include <stdio.h>
#include <signal.h>


//#define BEOHEAD
#define SCORBOT
//#define GUIDISPLAY
//#define CPDISPLAY
//#define OUTPUT
ModelManager *mgr;
nub::soft_ref<Scorbot> scorbot;
nub::soft_ref<ArmSim> armSim;
nub::soft_ref<ArmController> armControllerScorbot;
nub::soft_ref<ArmController> armControllerArmSim;
nub::soft_ref<ArmPlanner> armPlanner;
nub::soft_ref<BeoHeadBrain> beoHeadBrain;

double desireCup[3] = {-1*1.125/4-0.15,0.15,0.25};//cup
double desireBlock[3] = {-1*1.125/4,0,0.05};//red spot

//! Signal handler (e.g., for control-C)
void terminate(int s)
{
#ifdef SCORBOT
      armControllerScorbot->setBasePos( 0 , false);
      armControllerScorbot->setSholderPos( 0 , false);
      armControllerScorbot->setElbowPos( 0 , false);
      armControllerScorbot->setWrist1Pos(0 , false);
      armControllerScorbot->setWrist2Pos(0 , false);
      while(!armControllerScorbot->isFinishMove())
      {
        usleep(1000);
      }
#endif
#ifdef BEOHEAD
//  beoHeadBrain->setRelaxNeck(true);
  beoHeadBrain->relaxHead();
#endif
        LERROR("*** INTERRUPT ***");
        exit(0);
}
Point2D<int> getClick(nub::soft_ref<OutputFrameSeries> &ofs)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow("Output")
    : rutz::shared_ptr<XWinManaged>();
  return uiwin->getLastMouseClick();
}
int getKey(nub::soft_ref<OutputFrameSeries> &ofs)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow("Output")
    : rutz::shared_ptr<XWinManaged>();
  return uiwin->getLastKeyPress();
}
void output()
{
  double *pos = armSim->getEndPos();
  printf("%f %f %f ",pos[0],pos[1],pos[2]);
  printf("%d %d %d\n",armControllerArmSim->getBase(),
      armControllerArmSim->getSholder(),
      armControllerArmSim->getElbow());
}
void sync()
{
#ifdef SCORBOT
      int basePos = armControllerArmSim->getBase();
      int sholderPos = armControllerArmSim->getSholder();
      int elbowPos = armControllerArmSim->getElbow();
      int wrist1Pos = armControllerArmSim->getWrist1();
      int wrist2Pos = armControllerArmSim->getWrist2();

      elbowPos +=sholderPos;

      armControllerScorbot->setBasePos( basePos , false);
      armControllerScorbot->setSholderPos( sholderPos , false);
      sleep(2);
      armControllerScorbot->setElbowPos( -1*elbowPos , false);
      sleep(2);

      wrist1Pos += (int)((0.2496*(float)-1*elbowPos)-39.746) ;//+ pitch + (roll + offset) ;
      wrist2Pos -= (int)((0.2496*(float)-1*elbowPos)-39.746) ;//- pitch + (roll + offset);
      armControllerScorbot->setWrist1Pos( wrist1Pos , false);
      armControllerScorbot->setWrist2Pos( wrist2Pos , false);
#endif
      output();

}
void train()
{
  double loc[3];
  for(int i = 0;i<3;i++)
    loc[i] = desireBlock[i];
  for(int j=0;j< 10;j++){
    for(int i = 0;i< 10;i++){
      LINFO("Try Loc %f,%f,%f", loc[0], loc[1], loc[2]);
      armSim->moveObject(loc);
      if(armPlanner->move(loc,0.05)){
        LINFO("Success move to Loc %f,%f,%f", loc[0], loc[1], loc[2]);
        loc[0]-=0.01;
      }else{
      }
    }
  loc[1]-=0.01;
  loc[0] = desireBlock[0];
  }
}
int main(const int argc, const char **argv)
{

  mgr = new ModelManager("Test ObjRec");

#ifdef SCORBOT
  scorbot = nub::soft_ref<Scorbot>(new Scorbot(*mgr,"Scorbot", "Scorbot", "/dev/ttyUSB0"));
  armControllerScorbot =nub::soft_ref<ArmController>(new ArmController(*mgr, "ArmControllerScorbot", "ArmControllerScorbot", scorbot));
  mgr->addSubComponent(armControllerScorbot);
#endif
  armSim = nub::soft_ref<ArmSim>(new ArmSim(*mgr));
  armControllerArmSim =nub::soft_ref<ArmController>( new ArmController(*mgr,
        "ArmControllerArmSim", "ArmControllerArmSim", armSim));
  armPlanner = nub::soft_ref<ArmPlanner>(new ArmPlanner(*mgr,"ArmPlanner","ArmPlanner",
        armControllerScorbot ,armControllerArmSim, armSim));

  MYLOGVERB = LOG_INFO;

  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
  mgr->addSubComponent(ofs);

  mgr->addSubComponent(armControllerArmSim);

#ifdef BEOHEAD
  beoHeadBrain = nub::soft_ref<BeoHeadBrain>(new BeoHeadBrain(*mgr));
  mgr->addSubComponent(beoHeadBrain);
#endif
  nub::soft_ref<GeneralGUI> armSimGUI(new GeneralGUI(*mgr));
  mgr->addSubComponent(armSimGUI);
#ifdef SCORBOT
  nub::soft_ref<GeneralGUI> scorbotGUI(new GeneralGUI(*mgr));
  mgr->addSubComponent(scorbotGUI);
#endif
  mgr->exportOptions(MC_RECURSE);

  if (mgr->parseCommandLine(
        (const int)argc, (const char**)argv, "", 0, 0) == false)
    return 1;
  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  mgr->start();

#ifdef GUIDISPLAY
  //start the gui thread
  armSimGUI->startThread(ofs);
#ifdef CPDISPLAY
  armSimGUI->addImage(armPlanner->getImagePtr());
#endif
  sleep(1);
  //setup gui for various objects

  //Main GUI Window

  armSimGUI->addMeter(armControllerArmSim->getBasePtr(),
      "Sim_Base", -3000, PixRGB<byte>(0, 255, 0));
  armSimGUI->addMeter(armControllerArmSim->getSholderPtr(),
      "Sim_Sholder", -3000, PixRGB<byte>(0, 255, 0));
  armSimGUI->addMeter(armControllerArmSim->getElbowPtr(),
      "Sim_Elbow", -3000, PixRGB<byte>(0, 255, 0));
  armSimGUI->addMeter(armControllerArmSim->getWrist1Ptr(),
      "Sim_Wrist1", -3000, PixRGB<byte>(0, 255, 0));
  armSimGUI->addMeter(armControllerArmSim->getWrist2Ptr(),
      "Sim_Wrist2", -3000, PixRGB<byte>(0, 255, 0));
#ifdef SCORBOT
  armSimGUI->addMeter(armControllerScorbot->getBasePtr(),
      "Motor_Base", -3000, PixRGB<byte>(0, 255, 0));
  armSimGUI->addMeter(armControllerScorbot->getSholderPtr(),
      "Motor_Sholder", -3000, PixRGB<byte>(0, 255, 0));
  armSimGUI->addMeter(armControllerScorbot->getElbowPtr(),
      "Motor_Elbow", -3000, PixRGB<byte>(0, 255, 0));
  armSimGUI->addMeter(armControllerScorbot->getWrist1Ptr(),
      "Motor_Wrist1", -3000, PixRGB<byte>(0, 255, 0));
  armSimGUI->addMeter(armControllerScorbot->getWrist2Ptr(),
      "Motor_Wrist2", -3000, PixRGB<byte>(0, 255, 0));
#endif

#endif
#ifdef BEOHEAD
  beoHeadBrain->initHead();
  beoHeadBrain->setRelaxNeck(false);
#endif
#ifdef SCORBOT
  armControllerScorbot->setMotorsOn(true);
  armControllerScorbot->setPidOn(true);
#endif
  armControllerArmSim->setPidOn(true);
  armControllerArmSim->setMotorsOn(true);

  //set The min-max joint pos
  ArmController::JointPos jointPos;
  jointPos.base = 3000;
  jointPos.sholder = 3000;
  jointPos.elbow = 2000;
  jointPos.wrist1 = 9999;
  jointPos.wrist2 = 9999;
  armControllerArmSim->setMaxJointPos(jointPos);

  jointPos.base = -8000;
  jointPos.sholder = -1500;
  jointPos.elbow = -3000;
  jointPos.wrist1 = -9999;
  jointPos.wrist2 = -9999;
  armControllerArmSim->setMinJointPos(jointPos);

  //Move the arm to 0 pos
  jointPos.base = 0;
  jointPos.sholder = 0;
  jointPos.elbow = 0;
  jointPos.wrist1 = 0;
  jointPos.wrist2 = 0;
  armControllerArmSim->setJointPos(jointPos);

#ifdef SCORBOT
  //set The min-max joint pos
  jointPos.base = 7431;
  jointPos.sholder = 3264;
  jointPos.elbow = 4630;
  jointPos.wrist1 = 9999;
  jointPos.wrist2 = 9999;
  armControllerScorbot->setMaxJointPos(jointPos);

  jointPos.base = -5647;
  jointPos.sholder = -1296;
  jointPos.elbow = -2900;
  jointPos.wrist1 = -9999;
  jointPos.wrist2 = -9999;
  armControllerScorbot->setMinJointPos(jointPos);
#endif

  double *desireObject = desireBlock;
  armSim->addObject(ArmSim::BOX,desireObject);

  while(1)
  {
    armSim->simLoop();
    Image<PixRGB<byte> > armCam = flipVertic(armSim->getFrame(-1));
#ifdef BEOHEAD
    Image< PixRGB<byte> > inputImg = beoHeadBrain->getLeftEyeImg();
#endif
    Image<PixRGB<byte> > simCam = flipVertic(armSim->getFrame(-1));


#ifdef BEOHEAD
    if (!inputImg.initialized())
      continue;

    Point2D<int> targetLoc = beoHeadBrain->getTargetLoc();
    if (targetLoc.isValid())
      drawCircle(inputImg, targetLoc, 3, PixRGB<byte>(255,0,0));

    simCam += inputImg/2;
#endif
    Layout<PixRGB<byte> > outDisp;
#ifdef BEOHEAD
    outDisp = vcat(outDisp, hcat(inputImg, simCam));
    outDisp = hcat(outDisp, armCam);
#else
    outDisp = vcat(outDisp, hcat(armCam, simCam));
#endif
    ofs->writeRgbLayout(outDisp, "Output", FrameInfo("Output", SRC_POS));

    Point2D<int> clickLoc = getClick(ofs);
    if (clickLoc.isValid())
    {
     LINFO("clickPos %ix%i", clickLoc.i, clickLoc.j);
     clickLoc.i = clickLoc.i%320;//Let all window can click
     double loc[3];
     armSim->getObjLoc(clickLoc.i, clickLoc.j, loc);
#ifdef BEOHEAD
      beoHeadBrain->setTarget(clickLoc);
#endif
     LINFO("Loc %f,%f,%f", loc[0], loc[1], loc[2]);
     desireObject[0] = loc[0];
     desireObject[1] = loc[1];
     desireObject[2] = 0.04+ 0.04;//loc[2];
     armSim->moveObject(desireObject);
     LINFO("MOve beefore");
     if(armPlanner->move(desireObject,0.05))
       LINFO("Success move to %f %f %f",desireObject[0],desireObject[1],desireObject[2]);

     LINFO("MOve AAfter");
    }
#ifdef BEOHEAD
    NeoBrain::Stats stats = beoHeadBrain->getStats();
    LINFO("Eye Pan %f Tilt %f",stats.rightEyePanPos,stats.rightEyeTiltPos);
    LINFO("Head Pan %f Tilt %f Yaw %f",stats.headPanPos,stats.headTiltPos,stats.headYawPos);
    targetLoc = beoHeadBrain->getTargetLoc();
    LINFO("TrackPoint %i Tilt %i",targetLoc.i,targetLoc.j);
#endif
    int key = getKey(ofs);
    if (key != -1)
    {
      switch(key)
      {

        //Controll arm
        case 10: //1
          break;
        case 24: //q
          break;
        case 11: //2
          break;
        case 25: //w
          break;
        case 12: //3
          break;
        case 26: //e
          break;
        case 13: //4
          break;
        case 27: //r
          break;
        case 14: //5
          break;
        case 28: //t
          train();
          break;
        case 15: //6
#ifdef SCORBOT
          armControllerScorbot->setGripper(0);//close
#endif
          break;
        case 29: //y
#ifdef SCORBOT
          armControllerScorbot->setGripper(1);//open
#endif
          break;
        case 65: //space
          armControllerArmSim->killMotors();
#ifdef SCORBOT
          armControllerScorbot->killMotors();
#endif
          break;
        case 41: //f
          break;
        case 55: //v
          break;
        case 42: //g
          //go to desire block
          if(armPlanner->move(desireBlock,0.05))
            LINFO("Success move to %f %f %f",desireBlock[0],desireBlock[1],desireBlock[2]);
          break;
        case 56: //b roll
          break;
        case 57: //n
          break;
        case 39: //s
          break;
        case 43:// h
#ifdef SCORBOT
          armControllerScorbot->setElbowPos( 0 , false);
          sleep(2);
          armControllerScorbot->setWrist1Pos( 0, false);
          armControllerScorbot->setWrist2Pos( 0 ,false);
          armControllerScorbot->setSholderPos( 0 , false);
          armControllerScorbot->setBasePos( 0 , false);
          //armControllerScorbot->setGripper(1);//open

#endif
          armControllerArmSim->setWrist1Pos( 0, false);
          armControllerArmSim->setWrist2Pos( 0 ,false);
          armControllerArmSim->setElbowPos( 0 , false);
          armControllerArmSim->setSholderPos( 0 , false);
          armControllerArmSim->setBasePos( 0 , false);
          //armControllerArmSim->setGripper(1);//open
          break;

      }//End Switch
      output();

      LINFO("Key = %i", key);
    }

  }//End While(1)
  mgr->stop();

  return 0;

}

