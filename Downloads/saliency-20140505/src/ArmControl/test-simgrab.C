/*! @file ArmControl/test-simgrab.C  grab slient objects */

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
// Primary maintainer for this file: Chin-Kai Chang<chinkaic@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/test-simgrab.C $
// $Id: test-simgrab.C 10794 2009-02-08 06:21:09Z itti $
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
#include "Util/Timer.H"
#include "GUI/GeneralGUI.H"
#include "GUI/ImageDisplayStream.H"
#include "GUI/DebugWin.H"
#include "GUI/XWinManaged.H"
#include "ArmControl/ArmSim.H"
#include "ArmControl/RobotArm.H"
#include "ArmControl/ArmController.H"
#include "Util/MathFunctions.H"
#include <unistd.h>
#include <stdio.h>
#include <signal.h>

double getDistance(double desire[3]);
  ModelManager *mgr = new ModelManager("Test ObjRec");
  nub::soft_ref<ArmSim> armSim(new ArmSim(*mgr));
  nub::soft_ref<ArmController> armControllerArmSim(new ArmController(*mgr,
        "ArmControllerArmSim", "ArmControllerArmSim", armSim));

//! Signal handler (e.g., for control-C)
void terminate(int s)
{

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
void sync()
{
//      int basePos = armControllerArmSim->getBase();
//      int sholderPos = armControllerArmSim->getSholder();
//      int elbowPos = armControllerArmSim->getElbow();
//      int wrist1Pos = armControllerArmSim->getWrist1();
//      int wrist2Pos = armControllerArmSim->getWrist2();
//
//      elbowPos +=sholderPos;
//

}
void output()
{
  double desire[3] = {-1.125/4,0,0};
  double dist = getDistance(desire);
  double *pos = armSim->getEndPos();
  printf("%f %f %f ",pos[0],pos[1],pos[2]);
  printf("%d %d %d %f\n",armControllerArmSim->getBase(),
      armControllerArmSim->getSholder(),
      armControllerArmSim->getElbow(),dist);

}
double getDistance(double desire[3])
{
  double *pos = armSim->getEndPos();
  double sum = 0;
  for(int i=0;i<3;i++)
    sum+= (pos[i]-desire[i])*(pos[i]-desire[i]);
  return sum;
}
void moveMotor(int motor,int move)
{
    switch(motor)
    {
      case 0:
        armControllerArmSim->setBasePos(move, true);
        break;
      case 1:
        armControllerArmSim->setSholderPos(move, true);
        break;
      case 2:
        armControllerArmSim->setElbowPos(move, true);
        break;
      default:
        break;
    }
    while(!armControllerArmSim->isFinishMove())
    {
      armSim->simLoop();
      usleep(1000);
    }



}

bool gibbsSampling()
{
  double desire[3] = {-1*1.125/4,0,0.05};//red spot
  double current_distance = getDistance(desire);
  double prev_distance = current_distance;

  if(current_distance < 0.01){//close than 1cm
    output();
    return true;
  }
  do{
    int motor =  randomUpToIncluding(2);//get 0,1,2 motor
    int move =  randomUpToIncluding(200)-100;//get -100~100 move

    LINFO("Move motor %d with %d dist %f",motor,move,current_distance);
    prev_distance = current_distance;
    moveMotor(motor,move);
    current_distance = getDistance(desire);
    LINFO("Motor moved %d with %d dist %f",motor,move,current_distance);

    //After random move
    if(current_distance > prev_distance)//if getting far
    {
      moveMotor(motor,-move);
    }
  }while(current_distance > prev_distance);

  return false;
}


int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
  mgr->addSubComponent(ofs);

  mgr->addSubComponent(armControllerArmSim);

  nub::soft_ref<GeneralGUI> armSimGUI(new GeneralGUI(*mgr));
  mgr->addSubComponent(armSimGUI);

  mgr->exportOptions(MC_RECURSE);

  if (mgr->parseCommandLine(
        (const int)argc, (const char**)argv, "", 0, 0) == false)
    return 1;
  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  mgr->start();

  initRandomNumbers();

  //start the gui thread
  armSimGUI->startThread(ofs);
  sleep(1);
  //setup gui for various objects
  //Main GUI Window
  armSimGUI->addMeter(armControllerArmSim->getBasePtr(),
      "Sim_Base", -1000, PixRGB<byte>(0, 255, 0));
  armSimGUI->addMeter(armControllerArmSim->getSholderPtr(),
      "Sim_Sholder", -1000, PixRGB<byte>(0, 255, 0));
  armSimGUI->addMeter(armControllerArmSim->getElbowPtr(),
      "Sim_Elbow", -1000, PixRGB<byte>(0, 255, 0));
  armSimGUI->addMeter(armControllerArmSim->getWrist1Ptr(),
      "Sim_Wrist1", -1000, PixRGB<byte>(0, 255, 0));
  armSimGUI->addMeter(armControllerArmSim->getWrist2Ptr(),
      "Sim_Wrist2", -1000, PixRGB<byte>(0, 255, 0));

  armSimGUI->addImage(armControllerArmSim->getPIDImagePtr());

  armControllerArmSim->setMotorsOn(true);
  armControllerArmSim->setPidOn(true);
  bool isGibbs = false;
  while(1)
  {
    armSim->simLoop();
    Image<PixRGB<byte> > armCam = flipVertic(armSim->getFrame(-1));
    ofs->writeRGB(armCam, "Output", FrameInfo("Output", SRC_POS));

    int key = getKey(ofs);
    if (key != -1)
    {
      switch(key)
      {

        //Controll arm
        case 10: //1
          armControllerArmSim->setBasePos(10, true);
          LINFO("Sim Base: %d",armControllerArmSim->getBase());
          break;
        case 24: //q
          armControllerArmSim->setBasePos(-10, true);
          break;
        case 11: //2
          armControllerArmSim->setSholderPos(-10, true);
          break;
        case 25: //w
          armControllerArmSim->setSholderPos(10, true);
          break;
        case 12: //3
          armControllerArmSim->setElbowPos(10, true);
          break;
        case 26: //e
          armControllerArmSim->setElbowPos(-10, true);
          break;
        case 13: //4
          armControllerArmSim->setWrist1Pos(10, true);
          break;
        case 27: //r
          armControllerArmSim->setWrist1Pos(-10, true);
          break;
        case 14: //5
          armControllerArmSim->setWrist2Pos(10, true);
          break;
        case 28: //t
          armControllerArmSim->setWrist2Pos(-10, true);
          break;
        case 15: //6
          armControllerArmSim->setGripper(0);
          break;
        case 29: //y
          armControllerArmSim->setGripper(1);
          break;
        case 65: //space
          armControllerArmSim->killMotors();
          break;
        case 41: //f
          break;
        case 55: //v
          break;
        case 42: //g roll
          break;
        case 56: //b roll
          break;
        case 57: //n
          isGibbs = ~isGibbs;
          break;

      }//End Switch
      output();
      if(gibbsSampling() && isGibbs)
        sync();

      LINFO("Key = %i", key);
    }
    if(isGibbs){
      gibbsSampling();
      //isGibbs = false;
    }

  }//End While(1)
  mgr->stop();

  return 0;

}

