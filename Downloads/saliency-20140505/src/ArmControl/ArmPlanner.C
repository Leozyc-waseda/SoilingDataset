/*! @file ArmControl/ArmPlanner.C  planning how to grab objects */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/ArmPlanner.C $
// $Id: ArmPlanner.C 11001 2009-03-08 06:28:55Z mundhenk $
//
//#define USE_LWPR
#include "ArmControl/ArmPlanner.H"


ArmPlanner::ArmPlanner( OptionManager& mgr,
          const std::string& descrName,
          const std::string& tagName,
          nub::soft_ref<ArmController> realController,
          nub::soft_ref<ArmController> controller,
          nub::soft_ref<ArmSim> armSim):
  ModelComponent(mgr, descrName, tagName),
  numWarnings(0),
  itsRealArmController(realController),
  itsArmController(controller),
  itsArmSim(armSim),
  itsImage(512,256, ZEROS)
{
  addSubComponent(itsArmController);
  addSubComponent(itsArmSim);

#ifdef USE_LWPR
  lwpr_read_xml(&ik_model, "ik_model.xml", &numWarnings);
#endif
}


#ifdef USE_LWPR
ArmController::JointPos ArmPlanner::getIK(LWPR_Model& ik_model, const double* desiredPos)
{
  double joints[5];
  lwpr_predict(&ik_model, desiredPos, 0.001, joints, NULL, NULL);

  ArmController::JointPos jointPos;

  jointPos.base = (int)joints[0];
  jointPos.sholder = (int)joints[1];
  jointPos.elbow = (int)joints[2];
  jointPos.wrist1 = (int)joints[3];
  jointPos.wrist2 = (int)joints[4];
//  jointPos.gripper = 0;

  LDEBUG("Mapping %0.2f %0.2f %0.2f => %i %i %i %i %i",
      desiredPos[0], desiredPos[1], desiredPos[2],
      jointPos.base, jointPos.sholder, jointPos.elbow,
      jointPos.wrist1, jointPos.wrist2);
  return jointPos;
}
#endif
void ArmPlanner::getFK(double *endPos)
{
  double jointsAng[5],armLoc[3];
  itsArmSim->getArmLoc(armLoc);
  ArmController::JointPos jointPos = itsArmController->getJointPos();
  jointsAng[0] = itsArmSim->encoder2ang(jointPos.base,RobotArm::BASE);
  jointsAng[1] = itsArmSim->encoder2ang(jointPos.sholder,RobotArm::SHOLDER);
  jointsAng[2] = itsArmSim->encoder2ang(jointPos.elbow,RobotArm::ELBOW);
  jointsAng[3] = itsArmSim->encoder2ang(jointPos.wrist1,RobotArm::WRIST1);
  jointsAng[4] = itsArmSim->encoder2ang(jointPos.wrist2,RobotArm::WRIST2);
  //Have not done yet
}
ArmController::JointPos ArmPlanner::getIK2(const double* desiredPos)
{
  double joints[5],armLoc[3];
  itsArmSim->getArmLoc(armLoc);
  ArmController::JointPos jointPos = itsArmController->getJointPos();
  //joints[0] = atan2(desiredPos[1]-armLoc[1],desiredPos[0]-armLoc[0]);
  joints[0] = atan2(desiredPos[1]-armLoc[1],desiredPos[0]-armLoc[0]);
  /*
   |
   |     /\
   |<W> /  \
   |---/    |
   |^
   |BaseH+bodyH
   |v
   +-----------o-----
   |<----r ---->
   */
//  itsArmParam.base[0] = 0.230/2; //radius
//  itsArmParam.base[1] = 0.210; //length
//
//  itsArmParam.body[0] = 0.200; //x
//  itsArmParam.body[1] = 0.200; //y
//  itsArmParam.body[2] = 0.174; //z
//  itsArmParam.upperarm[0] = 0.065; //x
//  itsArmParam.upperarm[1] = 0.15; //y
//  itsArmParam.upperarm[2] = 0.220; //z
//
//  itsArmParam.forearm[0] = 0.06; //x
//  itsArmParam.forearm[1] = 0.13; //y
//  itsArmParam.forearm[2] = 0.220; //z

  double r = getDistance2D(armLoc,desiredPos);
  double x = r - 0.03;//upperarm to body offset
  double y = -1*(0.210+0.174) + desiredPos[2]; // + 0.11;//base length + body height
  double l1 = 0.220 ;//upperarm
  double l2 = 0.220 + 0.145;// - 0.06;//forearm + wrist

  double c2 = (sq(x)+sq(y)-sq(l1)-sq(l2)) / (2*l1*l2);
  if(c2 < 1 && c2 > -1){
    jointPos.base = itsArmSim->ang2encoder(joints[0],RobotArm::BASE); //set the base
    double s2 = -1* sqrt(1-sq(c2));

    double th2 = atan2(-1*s2,c2);
    jointPos.elbow = itsArmSim->ang2encoder(th2-M_PI/2,RobotArm::ELBOW);

    double k1 = l1+l2*c2;
    double k2 = l2*s2;
    double th1 = atan2(y,x) - atan2(k2,k1);
    //th1 = M_PI;
    jointPos.sholder = itsArmSim->ang2encoder(-1*th1+M_PI/2,RobotArm::SHOLDER);
    double th3 = M_PI/2 - th1 - th2;
    jointPos.wrist2 = itsArmSim->ang2encoder(th3,RobotArm::WRIST2);
    jointPos.wrist2 = 0;//itsArmSim->ang2encoder(th3,RobotArm::WRIST2);
    jointPos.wrist1 = 0;

    LINFO("th1 %f,th2 %f , th3 %f, r %f ,x %f,y %f, c2 %f ,s2 %f",th1,th2,th3, r,x,y,c2,s2);
//    jointPos.gripper = 0;
    jointPos.reachable = true;
  }else{

    LINFO("Unreachable point x %f y %f z %f",desiredPos[0],desiredPos[1],desiredPos[2]);
    jointPos.reachable = false;
  }
  LDEBUG("Mapping %0.2f %0.2f %0.2f => %i %i %i %i %i",
      desiredPos[0], desiredPos[1], desiredPos[2],
      jointPos.base, jointPos.sholder, jointPos.elbow,
      jointPos.wrist1, jointPos.wrist2);
  return jointPos;
}

#ifdef USE_LWPR
void ArmPlanner::trainArm(LWPR_Model& ik_model, const double* armPos, const ArmController::JointPos& jointPos)
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

  lwpr_update(&ik_model, armPos, joints, pJoints, NULL);

}
#endif
ArmController::JointPos ArmPlanner::calcGradient(const double* desiredPos)
{

  ArmController::JointPos jointPos = itsArmController->getJointPos();
  ArmController::JointPos tmpJointPos = jointPos;

  double dist1, dist2;
  double baseGrad, sholderGrad, elbowGrad;
  double err = getDistance(itsArmSim->getEndPos(), desiredPos);

  //get the base gradient
  tmpJointPos.base = jointPos.base + 100;
  itsArmController->setJointPos(tmpJointPos);
  dist1 = getDistance(itsArmSim->getEndPos(), desiredPos);

  tmpJointPos.base = jointPos.base - 100;
  itsArmController->setJointPos(tmpJointPos);
  dist2 = getDistance(itsArmSim->getEndPos(), desiredPos);
  baseGrad = dist1 - dist2;
  tmpJointPos.base = jointPos.base;

  //get the base gradient
  tmpJointPos.sholder = jointPos.sholder + 100;
  itsArmController->setJointPos(tmpJointPos);
  dist1 = getDistance(itsArmSim->getEndPos(), desiredPos);
  tmpJointPos.sholder = jointPos.sholder - 100;
  itsArmController->setJointPos(tmpJointPos);
  dist2 = getDistance(itsArmSim->getEndPos(), desiredPos);
  sholderGrad = dist1 - dist2;
  tmpJointPos.sholder = jointPos.sholder;

  //get the elbow gradient
  tmpJointPos.elbow = jointPos.elbow + 100;
  itsArmController->setJointPos(tmpJointPos);
  dist1 = getDistance(itsArmSim->getEndPos(), desiredPos);
  tmpJointPos.elbow = jointPos.elbow - 100;
  itsArmController->setJointPos(tmpJointPos);
  dist2 = getDistance(itsArmSim->getEndPos(), desiredPos);
  elbowGrad = dist1 - dist2;
  tmpJointPos.elbow = jointPos.elbow;

  int moveThresh = (int)(err*100000);
  jointPos.base -= (int)(randomUpToIncluding(moveThresh)*baseGrad);
  jointPos.sholder -= (int)(randomUpToIncluding(moveThresh)*sholderGrad);
  jointPos.elbow -= (int)(randomUpToIncluding(moveThresh)*elbowGrad);

  return jointPos;
}


bool ArmPlanner::gibbsSampling(double *desire,double errThres)
{
  double current_distance = getDistance(itsArmSim->getEndPos(),desire);
  double prev_distance = current_distance;
  errThres = errThres * errThres;
  int moveThres = 200;
  if(current_distance < 0.02)
    moveThres /= 2;//half 50
  if(current_distance < 0.01)
    moveThres /= 2;//half 25
  if(current_distance < errThres){//close than 5mm
    return true;
  }
  do{
    int motor =  randomUpToIncluding(2);//get 0,1,2 motor
    int move =  randomUpToIncluding(moveThres*2)-moveThres;//default get -100~100 move

    LINFO("Move motor %d with %d dist %f",motor,move,current_distance);
    prev_distance = current_distance;
    moveMotor(motor,move);
    current_distance = getDistance(itsArmSim->getEndPos(),desire);
    LINFO("Motor moved %d with %d dist %f",motor,move,current_distance);

    //After random move
    if(current_distance > prev_distance)//if getting far
    {
      //
      moveMotor(motor,-move);
    }
  }while(current_distance > prev_distance);

  //Get joint pos from armSim and sync with real arm
  ArmController::JointPos jointPos = itsArmController->getJointPos();
  itsRealArmController->setJointPos(sim2real(jointPos),false);
  return false;
}

void ArmPlanner::moveMotor(int motor,int move)
{
    switch(motor)
    {
      case 0:
        itsArmController->setBasePos(move, true);
        break;
      case 1:
        itsArmController->setSholderPos(move, true);
        break;
      case 2:
        itsArmController->setElbowPos(move, true);
        break;
      default:
        break;
    }
    while(!itsArmController->isFinishMove())
    {
      itsArmSim->simLoop();
      usleep(1000);
    }
}
bool ArmPlanner::gibbsControl(double *desire,double d)
{
  double *current =  itsArmSim->getEndPos();
  double distance = getDistance(itsArmSim->getEndPos(),desire);
  //double errThres = 0.005;
  double errThresForSampling = d;
  double v[3],nextPoint[3];
  if(getDistance(itsArmSim->getEndPos(),desire) < 0.01){
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
    LINFO("Move to next point %f %f %f %f",nextPoint[0],nextPoint[1],nextPoint[2],getDistance(itsArmSim->getEndPos(),nextPoint));

  }

  return false;

}

bool ArmPlanner::gradient(double *desire,double errThres)
{
  int gradient_base = itsArmSim->getGradientEncoder(RobotArm::BASE,desire);
  int gradient_sholder= itsArmSim->getGradientEncoder(RobotArm::SHOLDER,desire);
  int gradient_elbow = itsArmSim->getGradientEncoder(RobotArm::ELBOW,desire);
  gradient_base =(int)((double)gradient_base*(-483/63));
  gradient_sholder=(int)((double)gradient_sholder*(-2606/354));
  gradient_elbow =(int)((double)gradient_elbow*(-46/399));
  errThres = errThres * errThres;
  LINFO("Gradient: %d %d %d dist %f",gradient_base,gradient_sholder,gradient_elbow,
      getDistance(itsArmSim->getEndPos(),desire));
  if(getDistance(itsArmSim->getEndPos(),desire) > errThres){
//    moveMotor(0,gradient_base/10);
//    moveMotor(1,gradient_sholder/10);
//    moveMotor(2,gradient_elbow/10);

  }else{
    return true;
  }
  return false;
}

bool ArmPlanner::moveRel(double x,double y,double z,double errThres)
{

  double err;
  double* current = itsArmSim->getEndPos();
  double desire[3];
  int numTries = 0;

  desire[0] =  current[0] + x;
  desire[1] =  current[1] + y;
  desire[2] =  current[2] + z;
  LINFO("Current %f,%f,%f De: %f,%f,%f",
      current[0], current[1], current[2],
      desire[0], desire[1], desire[2]);
  getchar();

  //Predict the joint angle required for this position
  ArmController::JointPos jointPos = getIK2(desire);
  LINFO("Got Joint POs");
  if(!jointPos.reachable){
    LINFO("Can't reach point");
    return false;//can't not reach those point
  }
  //move the arm
  CtrlPolicy xCP;
  CtrlPolicy yCP;
  CtrlPolicy zCP;
  xCP.setGoalPos(desire[0]);
  yCP.setGoalPos(desire[1]);
  zCP.setGoalPos(desire[2]);


  xCP.setCurrentPos(current[0]);
  yCP.setCurrentPos(current[1]);
  zCP.setCurrentPos(current[2]);
  LINFO("Moving");
  err = getDistance(itsArmSim->getEndPos(), desire);
  int numTriesThres = 10000;
  while(err > errThres && numTries < numTriesThres)
  {

    numTries++;
    double cp[3];
    current = itsArmSim->getEndPos();
    cp[0] = xCP.getPos(current[0]);
    cp[1] = yCP.getPos(current[1]);
    cp[2] = zCP.getPos(current[2]);
    //Showing the moving path, it may slow down the computer
    //itsArmSim->addDrawObject(ArmSim::SPHERE, cp);
    //LINFO("cpx %f y %f z %f",cp[0],cp[1],cp[2]);
    jointPos = getIK2(cp);
    updateDataImg();
    if(jointPos.reachable){
      itsArmController->setJointPos(jointPos,true);
      //itsRealArmController->setJointPos(sim2real(jointPos),false);
    } else {
      LINFO("Can't reach point");
    }
    //LINFO("numTries %d",numTries);
    err = getDistance(itsArmSim->getEndPos(), desire);
  }
  if(numTries <= numTriesThres)
  {
    jointPos = getIK2(desire);
    itsArmController->setJointPos(jointPos);
    //If arm doesn't fellow the cp , than don't wait arm finish
    //itsRealArmController->setJointPos(sim2real(jointPos),false);
  }
  else{
    LINFO("Can't reach point");

  }
  //gibbsSampling(desire,errThres);
  //Check the final error
  err = getDistance(itsArmSim->getEndPos(), desire);

  LINFO("Moving Done,err %f errThres %f",err,errThres);
  return (err < errThres);

}
bool ArmPlanner::move(double *desire,double errThres)
{
    double err;
    int numTries = 0;
    //Predict the joint angle required for this position
    ArmController::JointPos jointPos = getIK2(desire);
    LINFO("Got Joint POs");
    if(!jointPos.reachable){
      LINFO("Can't reach point");
      return false;//can't not reach those point
    }
    //move the arm
    CtrlPolicy xCP;
    CtrlPolicy yCP;
    CtrlPolicy zCP;
    xCP.setGoalPos(desire[0]);
    yCP.setGoalPos(desire[1]);
    zCP.setGoalPos(desire[2]);


    double* current = itsArmSim->getEndPos();
    xCP.setCurrentPos(current[0]);
    yCP.setCurrentPos(current[1]);
    zCP.setCurrentPos(current[2]);
    LINFO("Moving");
    err = getDistance(itsArmSim->getEndPos(), desire);
    int numTriesThres = 10000;
    while(err > errThres && numTries < numTriesThres)
    {

            numTries++;
      double cp[3];
      current = itsArmSim->getEndPos();
      cp[0] = xCP.getPos(current[0]);
      cp[1] = yCP.getPos(current[1]);
      cp[2] = zCP.getPos(current[2]);
      //Showing the moving path, it may slow down the computer
      //itsArmSim->addDrawObject(ArmSim::SPHERE, cp);
      //LINFO("cpx %f y %f z %f",cp[0],cp[1],cp[2]);
      jointPos = getIK2(cp);
      updateDataImg();
      if(jointPos.reachable){
        itsArmController->setJointPos(jointPos,true);
        itsRealArmController->setJointPos(sim2real(jointPos),false);
      } else {
        LINFO("Can't reach point");
      }
      //LINFO("numTries %d",numTries);
            err = getDistance(itsArmSim->getEndPos(), desire);
    }
    if(numTries <= numTriesThres)
    {
    jointPos = getIK2(desire);
    itsArmController->setJointPos(jointPos);
    //If arm doesn't fellow the cp , than don't wait arm finish
      itsRealArmController->setJointPos(sim2real(jointPos),false);
    }
    else{
        LINFO("Can't reach point");

    }
    //gibbsSampling(desire,errThres);
    //Check the final error
    err = getDistance(itsArmSim->getEndPos(), desire);

    LINFO("Moving Done,err %f errThres %f",err,errThres);
    return (err < errThres);
}
void ArmPlanner::minJerk(double *desire,double *nextPoint,double time)
{
  double *currentPos = itsArmSim->getEndPos();
  double distance = getDistance(currentPos,desire);
  double d = distance * 100;//Using distance to calc total time
  //  double t = 1;
  for(int i = 0 ; i < 3 ; i++)
  {
    double td = time/d;
    nextPoint[i] = currentPos[i]+(desire[i] - currentPos[i])
      *(10*pow(td,3) - 15*pow(td,4) + 6*pow(td,5));

  }

}
//#ifdef CPDISPLAY
void ArmPlanner::updateDataImg()
{
  static int x = 0;
  //const double* endpos = itsArmSim->getEndPos();
  if (x < itsImage.getWidth())
  {
    if (!x)
      itsImage.clear();

    //int xpos= (int)(-1.0*((float)endpos[0]*100.0F)*256.0F);
    //LINFO("xpos %i endpos[0] %f",xpos, endpos[0]);
    //drawCircle(img,Point2D<int>(x,basePos), 2, PixRGB<byte>(255,0,0));
    itsImage.setVal(x,0, PixRGB<byte>(255,0,0));
    //itsImage.setVal(x,x/2, PixRGB<byte>(255,0,0));

    x++; // = (x+1)%img.getWidth();
  }

}
//#endif
ArmController::JointPos ArmPlanner::sim2real(ArmController::JointPos armSimJointPos )
{

      ArmController::JointPos scorbotJointPos;

      scorbotJointPos.base    = armSimJointPos.base   ;
      scorbotJointPos.sholder = armSimJointPos.sholder;
      scorbotJointPos.elbow   = armSimJointPos.elbow  ;
      scorbotJointPos.wrist1  = armSimJointPos.wrist1 ;
      scorbotJointPos.wrist2  = armSimJointPos.wrist2 ;
      scorbotJointPos.reachable= armSimJointPos.reachable ;

      //scorbotJointPos.gripper = 1;
      scorbotJointPos.elbow += scorbotJointPos.sholder;

      scorbotJointPos.wrist1 += (int)((0.2496*(float)scorbotJointPos.elbow )-39.746) ;//+ pitch + (roll + offset) ;
      scorbotJointPos.wrist2 -= (int)((0.2496*(float)scorbotJointPos.elbow )-39.746) ;//- pitch + (roll + offset);
      scorbotJointPos.elbow  *=-1;
      //LINFO("sim2real :%d %d %d ",scorbotJointPos.base,scorbotJointPos.sholder,scorbotJointPos.elbow);
      return scorbotJointPos;
}
