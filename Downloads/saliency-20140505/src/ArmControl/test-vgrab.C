/*! @file ArmControl/test-vgrab.C  grab slient objects */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/test-vgrab.C $
// $Id: test-vgrab.C 10794 2009-02-08 06:21:09Z itti $
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
#include "ArmControl/ArmSim.H"
#include "ArmControl/RobotArm.H"
#include "ArmControl/ArmController.H"
#include "Devices/DeviceOpts.H"
#include "Util/MathFunctions.H"
#include <unistd.h>
#include <stdio.h>
#include <signal.h>

#ifdef LWPR
#define USE_EXPAT
#include <lwpr/lwpr.h>
#include <lwpr/lwpr_xml.h>
#endif
  ModelManager *mgr = new ModelManager("Test ObjRec");
  nub::soft_ref<Scorbot> scorbot(new Scorbot(*mgr,"Scorbot", "Scorbot", "/dev/ttyUSB0"));
  nub::soft_ref<ArmSim> armSim(new ArmSim(*mgr));
  nub::soft_ref<ArmController> armControllerScorbot(new ArmController(*mgr, "ArmControllerScorbot", "ArmControllerScorbot", scorbot));
  nub::soft_ref<ArmController> armControllerArmSim(new ArmController(*mgr,
        "ArmControllerArmSim", "ArmControllerArmSim", armSim));

  double desireCup[3] = {-1*1.125/4-0.15,0.15,0.25};//cup
  double desireBlock[3] = {-1*1.125/4,0,0.05};//red spot

bool gradient(double *desire,double errThres);
//! Signal handler (e.g., for control-C)
void terminate(int s)
{

      armControllerScorbot->setBasePos( 0 , false);
      armControllerScorbot->setSholderPos( 0 , false);
      armControllerScorbot->setElbowPos( 0 , false);
      armControllerScorbot->setWrist1Pos(0 , false);
      armControllerScorbot->setWrist2Pos(0 , false);
      while(!armControllerScorbot->isFinishMove())
      {
        usleep(1000);
      }

        LERROR("*** INTERRUPT ***");
        exit(0);
}
double getDistance(const double* pos, const double* desire)
{

  //double *pos = armSim->getEndPos();
  double sum = 0;
  for(int i=0;i<3;i++)
    sum+= (pos[i]-desire[i])*(pos[i]-desire[i]);
  return sqrt(sum);
}
//double getDistance(double desire[3])
//{
//  getDistance(armSim->getEndPos(), desire);
//}
//Predict the joint angle required for this position
ArmController::JointPos getIK(LWPR_Model& ik_model, const double* desiredPos)
{
  double joints[5];
#ifdef LWPR
  lwpr_predict(&ik_model, desiredPos, 0.001, joints, NULL, NULL);
#endif

  ArmController::JointPos jointPos;

  jointPos.base = (int)joints[0];
  jointPos.sholder = (int)joints[1];
  jointPos.elbow = (int)joints[2];
  jointPos.wrist1 = (int)joints[3];
  jointPos.wrist2 = (int)joints[4];
  jointPos.gripper = 0;

  LDEBUG("Mapping %0.2f %0.2f %0.2f => %i %i %i %i %i",
      desiredPos[0], desiredPos[1], desiredPos[2],
      jointPos.base, jointPos.sholder, jointPos.elbow,
      jointPos.wrist1, jointPos.wrist2);
  return jointPos;
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

void trainArm(LWPR_Model& ik_model, const double* armPos, const ArmController::JointPos& jointPos)
{

  double joints[5];
  double pJoints[5];
  LDEBUG("Training => %0.2f,%0.2f,%0.2f => %i %i %i %i %i",
      armPos[0], armPos[1], armPos[2],
      jointPos.base, jointPos.sholder, jointPos.elbow,
      jointPos.wrist1, jointPos.wrist2);

  joints[0] = jointPos.base;
  joints[1] = jointPos.sholder;
  joints[2] = jointPos.elbow;
  joints[3] = jointPos.wrist1;
  joints[4] = jointPos.wrist2;
#ifdef LWPR
  lwpr_update(&ik_model, armPos, joints, pJoints, NULL);
#endif

}


ArmController::JointPos calcGradient(const double* desiredPos)
{

  ArmController::JointPos jointPos = armControllerArmSim->getJointPos();
  ArmController::JointPos tmpJointPos = jointPos;

  double dist1, dist2;
  double baseGrad, sholderGrad, elbowGrad;
  double err = getDistance(armSim->getEndPos(), desiredPos);

  //get the base gradient
  tmpJointPos.base = jointPos.base + 100;
  armControllerArmSim->setJointPos(tmpJointPos);
  dist1 = getDistance(armSim->getEndPos(), desiredPos);

  tmpJointPos.base = jointPos.base - 100;
  armControllerArmSim->setJointPos(tmpJointPos);
  dist2 = getDistance(armSim->getEndPos(), desiredPos);
  baseGrad = dist1 - dist2;
  tmpJointPos.base = jointPos.base;

  //get the base gradient
  tmpJointPos.sholder = jointPos.sholder + 100;
  armControllerArmSim->setJointPos(tmpJointPos);
  dist1 = getDistance(armSim->getEndPos(), desiredPos);
  tmpJointPos.sholder = jointPos.sholder - 100;
  armControllerArmSim->setJointPos(tmpJointPos);
  dist2 = getDistance(armSim->getEndPos(), desiredPos);
  sholderGrad = dist1 - dist2;
  tmpJointPos.sholder = jointPos.sholder;

  //get the elbow gradient
  tmpJointPos.elbow = jointPos.elbow + 100;
  armControllerArmSim->setJointPos(tmpJointPos);
  dist1 = getDistance(armSim->getEndPos(), desiredPos);
  tmpJointPos.elbow = jointPos.elbow - 100;
  armControllerArmSim->setJointPos(tmpJointPos);
  dist2 = getDistance(armSim->getEndPos(), desiredPos);
  elbowGrad = dist1 - dist2;
  tmpJointPos.elbow = jointPos.elbow;

  int moveThresh = (int)(err*100000);
  jointPos.base -= (int)(randomUpToIncluding(moveThresh)*baseGrad);
  jointPos.sholder -= (int)(randomUpToIncluding(moveThresh)*sholderGrad);
  jointPos.elbow -= (int)(randomUpToIncluding(moveThresh)*elbowGrad);

  return jointPos;
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

      output();

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
bool gibbsSampling(double *desire,double errThres)
{
  //double desire[3] = {-1*1.125/4,0,0.05};//red spot
  //double desire[3] = {-1*1.125/4-0.15,0.15,0.15};//cup
  double current_distance = getDistance(armSim->getEndPos(),desire);
  double prev_distance = current_distance;
  errThres = errThres * errThres;
  int moveThres = 200;
  if(current_distance < 0.02)
    moveThres /= 2;//half 50
  if(current_distance < 0.01)
    moveThres /= 2;//half 25
  if(current_distance < errThres){//close than 5mm
    //output();
    return true;
  }
  do{
    int motor =  randomUpToIncluding(2);//get 0,1,2 motor
    int move =  randomUpToIncluding(moveThres*2)-moveThres;//default get -100~100 move

    LINFO("Move motor %d with %d dist %f",motor,move,current_distance);
    prev_distance = current_distance;
    moveMotor(motor,move);
    current_distance = getDistance(armSim->getEndPos(),desire);
    LINFO("Motor moved %d with %d dist %f",motor,move,current_distance);

    //After random move
    if(current_distance > prev_distance)//if getting far
    {
      moveMotor(motor,-move);
    }
  }while(current_distance > prev_distance);

  return false;
}
bool gibbsControl(double *desire,double d)
{
  double *current =  armSim->getEndPos();
  double distance = getDistance(armSim->getEndPos(),desire);
  //double errThres = 0.005;
  double errThresForSampling = d;
  double v[3],nextPoint[3];
  //if(distance < errThres){//close than 5mm
  //  output();
  //  return true;
  //}
  if(getDistance(armSim->getEndPos(),desire) < 0.01){
    LINFO("Move by gibbs only");
    errThresForSampling = d;
    return gradient(desire,errThresForSampling);
  }else{
    for(int i=0;i<3;i++)
    {
      v[i] = desire[i] - current[i];//line vec
      v[i] = v[i]/distance;//vhat
      nextPoint[i] = current[i]+(distance/2)*v[i];
    }
    while(!gradient(nextPoint,errThresForSampling));
    sync();
    LINFO("Move to next point %f %f %f %f",nextPoint[0],nextPoint[1],nextPoint[2],getDistance(armSim->getEndPos(),nextPoint));

  }

  return false;

}
bool gradient(double *desire,double errThres)
{
  int gradient_base = armSim->getGradientEncoder(RobotArm::BASE,desire);
  int gradient_sholder= armSim->getGradientEncoder(RobotArm::SHOLDER,desire);
  int gradient_elbow = armSim->getGradientEncoder(RobotArm::ELBOW,desire);
  gradient_base =(int)((double)gradient_base*(-483/63));
  gradient_sholder=(int)((double)gradient_sholder*(-2606/354));
  gradient_elbow =(int)((double)gradient_elbow*(-46/399));
  errThres = errThres * errThres;
  LINFO("Gradient: %d %d %d dist %f",gradient_base,gradient_sholder,gradient_elbow,
      getDistance(armSim->getEndPos(),desire));
  if(getDistance(armSim->getEndPos(),desire) > errThres){
    moveMotor(0,gradient_base/10);
    moveMotor(1,gradient_sholder/10);
    moveMotor(2,gradient_elbow/10);

  }else{
    return true;
  }
  return false;
}
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;
  LWPR_Model ik_model;

  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
  mgr->addSubComponent(ofs);

  mgr->addSubComponent(armControllerScorbot);
  mgr->addSubComponent(armControllerArmSim);


  nub::soft_ref<BeoHeadBrain> beoHeadBrain(new BeoHeadBrain(*mgr));
  mgr->addSubComponent(beoHeadBrain);

  nub::soft_ref<GeneralGUI> armSimGUI(new GeneralGUI(*mgr));
  mgr->addSubComponent(armSimGUI);
  nub::soft_ref<GeneralGUI> scorbotGUI(new GeneralGUI(*mgr));
  mgr->addSubComponent(scorbotGUI);

  mgr->exportOptions(MC_RECURSE);

  if (mgr->parseCommandLine(
        (const int)argc, (const char**)argv, "", 0, 0) == false)
    return 1;
  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

#ifdef LWPR
  int numWarnings = 0;
  lwpr_read_xml(&ik_model, "ik_model.xml", &numWarnings);
#endif

  mgr->start();

  initRandomNumbers();


  //start the gui thread
  armSimGUI->startThread(ofs);
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

  armSimGUI->addImage(armControllerArmSim->getPIDImagePtr());

  beoHeadBrain->initHead();

  armControllerScorbot->setMotorsOn(true);
  armControllerArmSim->setMotorsOn(true);
  armControllerScorbot->setPidOn(true);
  armControllerArmSim->setPidOn(true);

  //set The min-max joint pos
  ArmController::JointPos jointPos;
  jointPos.base = 8000;
  jointPos.sholder = 5000;
  jointPos.elbow = 5000;
  jointPos.wrist1 = 0;
  jointPos.wrist2 = 0;
  //jointPos.base = 3000;
  //jointPos.sholder = 3000;
  //jointPos.elbow = 2000;
  //jointPos.wrist1 = 0;
  //jointPos.wrist2 = 0;
  armControllerArmSim->setMaxJointPos(jointPos);

  jointPos.base = -8000;
  jointPos.sholder = -1500;
  jointPos.elbow = -3000;
  jointPos.wrist1 = 0;
  jointPos.wrist2 = 0;
  armControllerArmSim->setMinJointPos(jointPos);




  //Move the arm to 0 pos
  jointPos.base = 0;
  jointPos.sholder = 0;
  jointPos.elbow = 0;
  jointPos.wrist1 = 0;
  jointPos.wrist2 = 0;
  armControllerArmSim->setJointPos(jointPos);

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


  bool isGibbs = false;
  double *desireObject = desireBlock;
  armSim->addObject(ArmSim::BOX,desireObject);

  bool reachComplete = true;
  int numReaches = 0;
  int numTries = 0;
  while(1)
  {
    armSim->simLoop();
    Image<PixRGB<byte> > armCam = flipVertic(armSim->getFrame(-1));

    Image< PixRGB<byte> > inputImg = beoHeadBrain->getLeftEyeImg();

    Image<PixRGB<byte> > simCam = flipVertic(armSim->getFrame(-1));


    if (!inputImg.initialized())
      continue;

    Point2D<int> targetLoc = beoHeadBrain->getTargetLoc();

    if (targetLoc.isValid())
      drawCircle(inputImg, targetLoc, 3, PixRGB<byte>(255,0,0));

    simCam += inputImg/2;

    Layout<PixRGB<byte> > outDisp;
    outDisp = vcat(outDisp, hcat(inputImg, simCam));
    outDisp = hcat(outDisp, armCam);
    ofs->writeRgbLayout(outDisp, "Output", FrameInfo("Output", SRC_POS));

    Point2D<int> clickLoc = getClick(ofs);
    if (clickLoc.isValid())
    {
     LINFO("clickPos %ix%i", clickLoc.i, clickLoc.j);
     int screenx = clickLoc.i%320;
     double loc[3];
     armSim->getObjLoc(screenx, clickLoc.j, loc);
     LINFO("Loc %f,%f,%f", loc[0], loc[1], loc[2]);
     desireObject[0] = loc[0];
     desireObject[1] = loc[1];
     desireObject[2] = 0.04;//loc[2];
     armSim->moveObject(desireObject);
     //armSim->drawLine(desireObject);

     //Predict the joint angle required for this position
     ArmController::JointPos jointPos = getIK(ik_model, desireObject);

     //move the arm
     armControllerArmSim->setJointPos(jointPos);

    }

     numTries++;

    //get the arm end effector position
    double *armPos = armSim->getEndPos();
    LDEBUG("Pos %f %f %f", armPos[0], armPos[1], armPos[2]);

    //compute the error
    double err = getDistance(armPos, desireObject);
    if (!reachComplete)
      LINFO("Err %f tries %i: reaches=%i", err, numTries, numReaches);

    if (err > 0.01)
    {
      //Learn by following the gradient
      jointPos = calcGradient(desireObject);

      armControllerArmSim->setJointPos(jointPos);

      jointPos =  armControllerArmSim->getJointPos();
      armPos = armSim->getEndPos();

      trainArm(ik_model, armPos, jointPos);
    } else {
      reachComplete = true;
      numReaches++;
      //write the network to a file

#ifdef LWPR
      lwpr_write_xml(&ik_model, "ik_model.xml");
#endif
    }

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
          //armControllerArmSim->setGripper(0);
          armControllerScorbot->setGripper(0);//close
          break;
        case 29: //y
          //armControllerArmSim->setGripper(1);
          armControllerScorbot->setGripper(1);//open
          break;
        case 65: //space
          armControllerArmSim->killMotors();
          armControllerScorbot->killMotors();
          break;
        case 41: //f
          armControllerScorbot->setWrist1Pos(10, true);
          armControllerScorbot->setWrist2Pos(10, true);
          break;
        case 55: //v
          armControllerScorbot->setWrist1Pos(-10, true);
          armControllerScorbot->setWrist2Pos(-10, true);
          break;
        case 42: //g
          gradient(desireObject,0.005);
          break;
        case 56: //b roll
          armControllerScorbot->setWrist1Pos(-10, true);
          armControllerScorbot->setWrist2Pos(10, true);
          break;
        case 57: //n
          LINFO("Set gibbs true");
          isGibbs = true;
          break;
        case 39: //s
          LINFO("Sync robot");
          sync();
          break;
        case 43:// h
          armControllerScorbot->setElbowPos( 0 , false);
          sleep(2);
          armControllerScorbot->setWrist1Pos( 0, false);
          armControllerScorbot->setWrist2Pos( 0 ,false);
          armControllerScorbot->setSholderPos( 0 , false);
          armControllerScorbot->setBasePos( 0 , false);
          armControllerScorbot->setGripper(1);//open

          armControllerArmSim->setWrist1Pos( 0, false);
          armControllerArmSim->setWrist2Pos( 0 ,false);
          armControllerArmSim->setElbowPos( 0 , false);
          armControllerArmSim->setSholderPos( 0 , false);
          armControllerArmSim->setBasePos( 0 , false);
          break;

      }//End Switch
      output();

      LINFO("Key = %i", key);
    }
    if(isGibbs){
      //if(gibbsControl(desireObject,0.005)){
      if(gradient(desireObject,0.005)){

        sync();

        if(desireObject == desireBlock){
        //desireObject = desireCup;
        //sync();
        //while(!armControllerScorbot->isFinishMove())
        //  usleep(1000);
        //armControllerScorbot->setGripper(0);//close
        //sleep(1);
        //armControllerScorbot->setSholderPos(-1500, true);
        //armControllerScorbot->setElbowPos(1500, true);
        }else{
          //move on the cup
        //sync();
        //desireObject = desireBlock;
        //while(!armControllerScorbot->isFinishMove())
        //  usleep(1000);
        //armControllerScorbot->setGripper(1);//Open
        //sleep(1);
        //armControllerScorbot->setSholderPos(-1500, true);
        //armControllerScorbot->setElbowPos(1500, true);

        //isGibbs = false;

        }
      }

    }

  }//End While(1)
  mgr->stop();

  return 0;

}

