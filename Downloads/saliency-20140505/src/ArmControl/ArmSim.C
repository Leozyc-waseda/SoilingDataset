/*!@file ArmControl/ArmSim.C Interfaces to the robot arm */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/ArmSim.C $
// $Id: ArmSim.C 10794 2009-02-08 06:21:09Z itti $
//


#include "ArmControl/RobotArm.H"
#include "ArmControl/ArmSim.H"
#include "Component/OptionManager.H"
#include "Util/MathFunctions.H"
#include "Util/Assert.H"
#include "Util/Angle.H"
#include "rutz/compat_snprintf.h"

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

namespace {

  void nearCallback (void *data, dGeomID o1, dGeomID o2){
    ArmSim *armSim = (ArmSim *)data;
    const int MaxContacts = 10;

    //create a contact joint to simulate collisions
    dContact contact[MaxContacts];
    int nContacts = dCollide (o1,o2,MaxContacts,
        &contact[0].geom,sizeof(dContact));
    for (int i=0; i<nContacts; i++) {
      contact[i].surface.mode = dContactSoftCFM ; //| dContactBounce; // | dContactSoftCFM;
      contact[i].surface.mu = 3.5;
      contact[i].surface.mu2 = 2.0;
      contact[i].surface.bounce = 0; //0.01;
      contact[i].surface.bounce_vel = 0; //0.01;
      contact[i].surface.soft_cfm = 0.001;

      dJointID c = dJointCreateContact (armSim->getWorld(),
          armSim->getContactgroup(),&contact[i]);
      dJointAttach (c,
          dGeomGetBody(contact[i].geom.g1),
          dGeomGetBody(contact[i].geom.g2));
    }
  }
}

// ######################################################################

ArmSim::ArmSim(OptionManager& mgr,

    const std::string& descrName,
    const std::string& tagName,
    const double l0,
    const double l1,
    const double l2,
    const double l3,
    const double l4
    ) :
  RobotArm(mgr, descrName, tagName),
  vp(new ViewPort("ArmSim")
    )
{

  //in meters
  itsTableSize[0] = 1.215;//W
  itsTableSize[1] = 0.75;//L
  itsTableSize[2] = 0.02;

  itsArmParam.armLoc[0] = -1*itsTableSize[0]/2+0.145;
  itsArmParam.armLoc[1] = -1*itsTableSize[1]/2+0.12;
  itsArmParam.armLoc[2] = 0;

  itsArmParam.base[0] = 0.230/2; //radius
  itsArmParam.base[1] = 0.210 + 0.05; //length, the real height is 0.210 but it's too low in armSim
  itsArmParam.baseMass = 9;

  itsArmParam.body[0] = 0.200; //x
  itsArmParam.body[1] = 0.200; //y
  itsArmParam.body[2] = 0.174; //z
  itsArmParam.bodyMass = 0.2;

  itsArmParam.upperarm[0] = 0.065; //x
  itsArmParam.upperarm[1] = 0.15; //y
  itsArmParam.upperarm[2] = 0.220; //z
  itsArmParam.upperarmMass = 0.2;

  itsArmParam.forearm[0] = 0.06; //x
  itsArmParam.forearm[1] = 0.13; //y
  itsArmParam.forearm[2] = 0.220; //z
  itsArmParam.forearmMass = 0.2;

  itsArmParam.wrist1[0] = 0.02; //r
  itsArmParam.wrist1[1] = 0.17; //length
  itsArmParam.wrist1Mass = 0.2;

  itsArmParam.wrist2[0] = 0.032; //x
  itsArmParam.wrist2[1] = 0.115; //y
  itsArmParam.wrist2[2] = 0.067; //z
  itsArmParam.wrist2Mass = 0.2;

  itsArmParam.gripper[0] = 0.015; //x
  itsArmParam.gripper[1] = 0.020; //y
  itsArmParam.gripper[2] = 0.070; //z
  itsArmParam.gripperMass = 0.2;


  vp->setTextures(false);
  vp->setShadows(false);

}

ArmSim::~ArmSim()
{
        dSpaceDestroy(space);
        dWorldDestroy(world);

        delete vp;
}

void ArmSim::start2()
{
//        setDrawStuff();
        world = dWorldCreate();
        space = dHashSpaceCreate(0);
        contactgroup = dJointGroupCreate(0);
        ground = dCreatePlane(space, 0, 0, 1, 0);
  dGeomSetCategoryBits(ground, GROUND_BITFIELD);
        dWorldSetGravity(world,0,0,-9.8);
        //dWorldSetContactMaxCorrectingVel(world,0.9);
        dWorldSetContactSurfaceLayer(world,0.001);
        makeArm();
        //set the viewpoint
  //For fish eye lens
  //double xyz[3]={0.490837, -0.184382, 0.490000};
  //double hpr[3] = {174.500000, -20.500000, 0.000000};
  //For regular lens
  //double xyz[3]={0.087279, -0.112716, 0.300000};
  //double hpr[3] = {163.000000, -46.500000, 0.000000};

  //for world view

  double xyz[3]={0.289603, -1.071132, 0.710000};
  double hpr[3] = {110.500000, -10.500000, 0.000000};


        vp->dsSetViewpoint (xyz,hpr);

}


void ArmSim::setMotor(MOTOR m, int val)
{
  float speed = (float)val/50;
  switch (m)
  {
    case RobotArm::BASE:
      dJointSetHingeParam(itsBody.joint, dParamVel , speed);
      break;
    case RobotArm::SHOLDER:
      dJointSetHingeParam(itsUpperArm.joint, dParamVel , speed);
      break;
    case RobotArm::ELBOW:
      dJointSetHingeParam(itsForearm.joint, dParamVel , speed);
      break;
    case RobotArm::WRIST1:
      dJointSetHingeParam(itsWrist1.joint, dParamVel , speed);
      break;
    case RobotArm::WRIST2:
      dJointSetHingeParam(itsWrist2.joint, dParamVel , speed);
      break;
    case RobotArm::GRIPPER:
      dJointSetHingeParam(itsGripper1.joint, dParamVel , speed);
      dJointSetHingeParam(itsGripper2.joint, dParamVel , -1*speed);
      break;
    default:
      break;
  }

}

void ArmSim::stopAllMotors()
{

  LINFO("Stop all motors");
  dJointSetHingeParam(itsBody.joint, dParamVel , 0);
  dJointSetHingeParam(itsUpperArm.joint, dParamVel , 0);
  dJointSetHingeParam(itsForearm.joint, dParamVel , 0);
  dJointSetHingeParam(itsWrist1.joint, dParamVel , 0);
  dJointSetHingeParam(itsWrist2.joint, dParamVel , 0);
  dJointSetHingeParam(itsGripper1.joint, dParamVel , 0);
  dJointSetHingeParam(itsGripper2.joint, dParamVel , 0);

}

int ArmSim::getEncoder(MOTOR m)
{
  //From real Scorbot, when all joint angle are 0(all arm in one line)
  //The encoder value are 0,-3520,3904,617,-2008
  //The move 90 degree will have encoder value 3013~3091
  //Move 1 degree in the joint, the real encoder should move 34.33
  dReal ang = 0;
  switch(m)
  {
    case RobotArm::BASE:
      ang = dJointGetHingeAngle(itsBody.joint);
      break;
    case RobotArm::SHOLDER:
      ang = dJointGetHingeAngle(itsUpperArm.joint);
      break;
    case RobotArm::ELBOW:
      ang = dJointGetHingeAngle(itsForearm.joint);
      ang -= M_PI/2;
      break;
    case RobotArm::WRIST1:
      ang = dJointGetHingeAngle(itsWrist1.joint);
      break;
    case RobotArm::WRIST2:
      ang = dJointGetHingeAngle(itsWrist2.joint);
      break;
    case RobotArm::GRIPPER:
      ang = dJointGetHingeAngle(itsGripper1.joint); //TODO: the angle betwwen the joints
      break;
    default:
      break;
  }
  int pos = ang2encoder(ang,m);
//  if(m==RobotArm::BASE)
//    pos -=2306;
  return pos;
}
double ArmSim::encoder2ang(int eng,MOTOR m)
{

  switch(m)
  {
    case RobotArm::BASE:
      break;
    case RobotArm::SHOLDER:
      break;
    case RobotArm::ELBOW:
      break;
    case RobotArm::WRIST1:
      break;
    case RobotArm::WRIST2:
      break;
    case RobotArm::GRIPPER:
      break;
    default:
      break;
  }
  double ang = (eng*M_PI)/(2975*2);

//  if(m==RobotArm::BASE)
//    pos -=2306;
  return ang;
}
int ArmSim::ang2encoder(double ang,MOTOR m)
{

  switch(m)
  {
    case RobotArm::BASE:
      break;
    case RobotArm::SHOLDER:
      break;
    case RobotArm::ELBOW:
      break;
    case RobotArm::WRIST1:
      break;
    case RobotArm::WRIST2:
      break;
    case RobotArm::GRIPPER:
      break;
    default:
      break;
  }
  int pos = (int)(ang*2975*2/M_PI);
//  if(m==RobotArm::BASE)
//    pos -=2306;
  return pos;
}
void ArmSim::setSafety(bool val)
{
}

void ArmSim::resetEncoders()
{
}

float ArmSim::getPWM(MOTOR m)
{
        return 0;
}

void ArmSim::homeMotor()
{
  /*
        int pwm = 0;

        switch(m)
        {
                case ELBOW: pwm = -80; break;
                default: pwm = 0; break;
        }

        setMotor(m, pwm);
        for(int i=0; i<100 && pwm != 0; i++)
        {
                pwm = (int) getPWM(m);
                LDEBUG("PWM %i", pwm);
                sleep(1);
        }
        LDEBUG("Done\n");
        setMotor(m, 0);
*/
}

void ArmSim::makeArm()
{
        dMass mass;
        dReal fMax = 100.0;

  const dReal *pos;
  //Base
  itsBase.body = dBodyCreate(world);
  dBodySetPosition(itsBase.body,
      itsArmParam.armLoc[0], itsArmParam.armLoc[1],
      itsArmParam.armLoc[2] + itsArmParam.base[1]/2); //Body initial position
  dMassSetZero(&mass);
  dMassSetCylinderTotal(&mass,itsArmParam.baseMass,3,itsArmParam.base[0],itsArmParam.base[1]);
  dBodySetMass(itsBase.body,&mass);
  //itsBase.geom = dCreateCylinder(space, itsArmParam.base[0],itsArmParam.base[1]);
  //dGeomSetBody(itsBase.geom, itsBase.body);
  dBodySetGravityMode(itsBase.body, ARM_GRAVITYMODE);
  //dGeomSetCategoryBits(itsBase.geom,ARM_BITFIELD);
  //dGeomSetCollideBits(itsBase.geom,!ARM_BITFIELD && !GROUND_BITFIELD); //don't colide with the arm itself and the Ground

  //Body
  itsBody.body = dBodyCreate(world);
  dBodySetPosition(itsBody.body,
      itsArmParam.armLoc[0],itsArmParam.armLoc[1], itsArmParam.armLoc[2] + itsArmParam.base[1] + itsArmParam.body[2]/2);
  dMassSetZero(&mass);
  dMassSetBoxTotal(&mass,itsArmParam.baseMass,
      itsArmParam.body[0], itsArmParam.body[1], itsArmParam.body[2]);
  dBodySetMass(itsBody.body,&mass);
  itsBody.geom = dCreateBox(space,
      itsArmParam.body[0], itsArmParam.body[1], itsArmParam.body[2]);
  dGeomSetBody(itsBody.geom, itsBody.body);
  dBodySetGravityMode(itsBody.body, ARM_GRAVITYMODE);
  dGeomSetCategoryBits(itsBody.geom,ARM_BITFIELD);
  dGeomSetCollideBits(itsBody.geom,!ARM_BITFIELD); //don't colide with the arm itself

  itsBody.joint = dJointCreateHinge(world,0);
  dJointAttach(itsBody.joint,itsBody.body,0);
  dJointSetHingeAnchor(itsBody.joint,
      itsArmParam.armLoc[0],itsArmParam.armLoc[1],
      itsArmParam.base[2]);
  dJointSetHingeAxis(itsBody.joint,0,0,1);
  dJointSetHingeParam(itsBody.joint, dParamFMax, fMax);
  dJointSetHingeParam(itsBody.joint, dParamVel , 0);

  //upperarm
  itsUpperArm.body = dBodyCreate(world);
  pos = dBodyGetPosition(itsBody.body);
  dBodySetPosition(itsUpperArm.body,pos[0]+0.03,pos[1], pos[2] + itsArmParam.body[2]/2 + itsArmParam.upperarm[2]/2);
  dMassSetZero(&mass);
  dMassSetBoxTotal(&mass,itsArmParam.baseMass,
      itsArmParam.upperarm[0], itsArmParam.upperarm[1], itsArmParam.upperarm[2]);
  dBodySetMass(itsUpperArm.body,&mass);
  itsUpperArm.geom = dCreateBox(space,
      itsArmParam.upperarm[0], itsArmParam.upperarm[1], itsArmParam.upperarm[2]);
  dGeomSetBody(itsUpperArm.geom, itsUpperArm.body);
  dBodySetGravityMode(itsUpperArm.body, ARM_GRAVITYMODE);
  dGeomSetCategoryBits(itsUpperArm.geom,ARM_BITFIELD);
  dGeomSetCollideBits(itsUpperArm.geom,!ARM_BITFIELD); //don't colide with the arm itself


  itsUpperArm.joint = dJointCreateHinge(world,0);
  dJointAttach(itsUpperArm.joint,itsUpperArm.body,itsBody.body);
  dJointSetHingeAnchor(itsUpperArm.joint,
      pos[0]+0.03, pos[1], pos[2] + itsArmParam.body[2]/2);
  dJointSetHingeAxis(itsUpperArm.joint,0,1,0);
  dJointSetHingeParam(itsUpperArm.joint, dParamFMax, fMax);
  dJointSetHingeParam(itsUpperArm.joint, dParamVel , 0);

  //forearm
  itsForearm.body = dBodyCreate(world);
  pos = dBodyGetPosition(itsUpperArm.body);
  dBodySetPosition(itsForearm.body,
      pos[0],pos[1], pos[2] + itsArmParam.upperarm[2]/2 + itsArmParam.forearm[2]/2);
  dMassSetZero(&mass);
  dMassSetBoxTotal(&mass,itsArmParam.forearmMass,
      itsArmParam.forearm[0], itsArmParam.forearm[1], itsArmParam.forearm[2]);
  dBodySetMass(itsForearm.body,&mass);
  itsForearm.geom = dCreateBox(space,
      itsArmParam.forearm[0], itsArmParam.forearm[1], itsArmParam.forearm[2]);
  dGeomSetBody(itsForearm.geom, itsForearm.body);
  dBodySetGravityMode(itsForearm.body, ARM_GRAVITYMODE);
  dGeomSetCategoryBits(itsForearm.geom,ARM_BITFIELD);
  dGeomSetCollideBits(itsForearm.geom,!ARM_BITFIELD);

  itsForearm.joint = dJointCreateHinge(world,0);
  dJointAttach(itsForearm.joint,itsForearm.body,itsUpperArm.body);
  dJointSetHingeAnchor(itsForearm.joint,
      pos[0], pos[1], pos[2] + itsArmParam.forearm[2]/2);
  dJointSetHingeAxis(itsForearm.joint,0,1,0);
  dJointSetHingeParam(itsForearm.joint, dParamFMax, fMax);
  dJointSetHingeParam(itsForearm.joint, dParamVel , 0);


  //Wrist
  itsWrist1.body = dBodyCreate(world);
  pos = dBodyGetPosition(itsForearm.body);
  dBodySetPosition(itsWrist1.body,
      pos[0],pos[1], pos[2] + itsArmParam.forearm[2]/2 + 0.04 - itsArmParam.wrist1[1]/2);
  dMassSetZero(&mass);
  dMassSetCylinderTotal(&mass,itsArmParam.wrist1Mass,3,itsArmParam.wrist1[0],itsArmParam.wrist1[1]);
  dBodySetMass(itsWrist1.body,&mass);
  itsWrist1.geom = dCreateCylinder(space,
      itsArmParam.wrist1[0], itsArmParam.wrist1[1]);
  dGeomSetBody(itsWrist1.geom, itsWrist1.body);
  dBodySetGravityMode(itsWrist1.body, ARM_GRAVITYMODE);
  dGeomSetCategoryBits(itsWrist1.geom,ARM_BITFIELD);
  dGeomSetCollideBits(itsWrist1.geom,!ARM_BITFIELD);

  itsWrist1.joint = dJointCreateHinge(world,0);
  dJointAttach(itsWrist1.joint,itsWrist1.body,itsForearm.body);
  dJointSetHingeAnchor(itsWrist1.joint,
      pos[0], pos[1], pos[2] + itsArmParam.forearm[2]/2-0.04);
  dJointSetHingeAxis(itsWrist1.joint,0,1,0);
  dJointSetHingeParam(itsWrist1.joint, dParamFMax, fMax);
  dJointSetHingeParam(itsWrist1.joint, dParamVel , 0);


  //Wrist2
  itsWrist2.body = dBodyCreate(world);
  pos = dBodyGetPosition(itsWrist1.body);
  dBodySetPosition(itsWrist2.body,
      pos[0],pos[1], pos[2] + itsArmParam.wrist1[1]/2 + itsArmParam.wrist2[2]/2);
  dMassSetZero(&mass);
  dMassSetBoxTotal(&mass,itsArmParam.wrist2Mass,
      itsArmParam.wrist2[0], itsArmParam.wrist2[1], itsArmParam.wrist2[2]);
  dBodySetMass(itsWrist2.body,&mass);
  itsWrist2.geom = dCreateBox(space,
      itsArmParam.wrist2[0], itsArmParam.wrist2[1], itsArmParam.wrist2[2]);
  dGeomSetBody(itsWrist2.geom, itsWrist2.body);
  dBodySetGravityMode(itsWrist2.body, ARM_GRAVITYMODE);
  dGeomSetCategoryBits(itsWrist2.geom,ARM_BITFIELD);
  dGeomSetCollideBits(itsWrist2.geom,!ARM_BITFIELD);

  itsWrist2.joint = dJointCreateHinge(world,0);
  dJointAttach(itsWrist2.joint,itsWrist2.body,itsWrist1.body);
  dJointSetHingeAnchor(itsWrist2.joint,
      pos[0], pos[1], pos[2] + itsArmParam.wrist1[1]/2);
  dJointSetHingeAxis(itsWrist2.joint,0,0,1);
  dJointSetHingeParam(itsWrist2.joint, dParamFMax, fMax);
  dJointSetHingeParam(itsWrist2.joint, dParamVel , 0);

  //Grippers
  itsGripper1.body = dBodyCreate(world);
  itsGripper2.body = dBodyCreate(world);

  pos = dBodyGetPosition(itsWrist2.body);
  dBodySetPosition(itsGripper1.body,
      pos[0],pos[1]+itsArmParam.wrist2[0], pos[2] + itsArmParam.wrist2[1]/2); // + itsArmParam.gripper[2]/2);
  dBodySetPosition(itsGripper2.body,
      pos[0],pos[1]-itsArmParam.wrist2[0], pos[2] + itsArmParam.wrist2[1]/2); // + itsArmParam.gripper[2]/2);
  dMassSetZero(&mass);
  dMassSetBoxTotal(&mass,itsArmParam.gripperMass,
      itsArmParam.gripper[0], itsArmParam.gripper[1], itsArmParam.gripper[2]);

  dBodySetMass(itsGripper1.body,&mass);
  dBodySetMass(itsGripper2.body,&mass);

  itsGripper1.geom = dCreateBox(space,
      itsArmParam.gripper[0], itsArmParam.gripper[1], itsArmParam.gripper[2]);
  dGeomSetBody(itsGripper1.geom, itsGripper1.body);
  itsGripper2.geom = dCreateBox(space,
      itsArmParam.gripper[0], itsArmParam.gripper[1], itsArmParam.gripper[2]);
  dGeomSetBody(itsGripper2.geom, itsGripper2.body);

  dBodySetGravityMode(itsGripper1.body, ARM_GRAVITYMODE);
  dGeomSetCategoryBits(itsGripper1.geom,GRIPPER_BITFIELD);
  dGeomSetCollideBits(itsGripper1.geom,!ARM_BITFIELD);
  dBodySetGravityMode(itsGripper2.body, ARM_GRAVITYMODE);
  dGeomSetCategoryBits(itsGripper2.geom,GRIPPER_BITFIELD);
  dGeomSetCollideBits(itsGripper2.geom,!ARM_BITFIELD);

  itsGripper1.joint = dJointCreateHinge(world,0);
  dJointAttach(itsGripper1.joint,itsGripper1.body,itsWrist2.body);
  dJointSetHingeAnchor(itsGripper1.joint,
      pos[0], pos[1] + itsArmParam.wrist2[0], pos[2]); // + itsArmParam.wrist2[1]/2);
  dJointSetHingeAxis(itsGripper1.joint,1,0,0);
  dJointSetHingeParam(itsGripper1.joint, dParamFMax, fMax);
  dJointSetHingeParam(itsGripper1.joint, dParamVel , 0);

  itsGripper2.joint = dJointCreateHinge(world,0);
  dJointAttach(itsGripper2.joint,itsGripper2.body,itsWrist2.body);
  dJointSetHingeAnchor(itsGripper2.joint,
      pos[0], pos[1] - itsArmParam.wrist2[0], pos[2]); // + itsArmParam.wrist2[1]/2);
  dJointSetHingeAxis(itsGripper2.joint,1,0,0);
  dJointSetHingeParam(itsGripper2.joint, dParamFMax, fMax);
  dJointSetHingeParam(itsGripper2.joint, dParamVel , 0);

}

void ArmSim::drawLine(double pos[3])
{

  vp->dsSetColor(0,255,255);
  vp->dsDrawLine(pos,getEndPos());
  //LINFO("Line Length %f ",getDistance(pos,getEndPos()));
}
void ArmSim::drawArm()
{

  double boxSize[3]={0.01,0.01,0.01};
  double R[12];
  //Base
  vp->dsSetColor(0,0,0);
  vp->dsDrawCylinderD(
      dBodyGetPosition(itsBase.body),
      dBodyGetRotation(itsBase.body),
      itsArmParam.base[1],
      itsArmParam.base[0]);

  //Body
  vp->dsSetColor(1,0.5,0);
  vp->dsDrawBox(
      dBodyGetPosition(itsBody.body),
      dBodyGetRotation(itsBody.body),
      itsArmParam.body);

  ////Upperarm
  vp->dsSetColor(1,0.5,0);
  vp->dsDrawBox(
      dBodyGetPosition(itsUpperArm.body),
      dBodyGetRotation(itsUpperArm.body),
      itsArmParam.upperarm);


  //Forearm
  vp->dsSetColor(1,0.5,0);
  vp->dsDrawBox(
      dBodyGetPosition(itsForearm.body),
      dBodyGetRotation(itsForearm.body),
      itsArmParam.forearm);

  //Wrist
  vp->dsSetColor(0,0,0);
  vp->dsDrawCylinderD(
      dBodyGetPosition(itsWrist1.body),
      dBodyGetRotation(itsWrist1.body),
      itsArmParam.wrist1[1],
      itsArmParam.wrist1[0]);

  //Wrist2
  vp->dsSetColor(0,0,0);
  vp->dsDrawBox(
      dBodyGetPosition(itsWrist2.body),
      dBodyGetRotation(itsWrist2.body),
      itsArmParam.wrist2);

  //Gripper
  vp->dsSetColor(0,0,0);
  vp->dsDrawBox(
      dBodyGetPosition(itsGripper1.body),
      dBodyGetRotation(itsGripper1.body),
      itsArmParam.gripper);

  vp->dsSetColor(0,0,0);
  vp->dsDrawBox(
      dBodyGetPosition(itsGripper2.body),
      dBodyGetRotation(itsGripper2.body),
      itsArmParam.gripper);

  //Draw The end point
  dRFromAxisAndAngle (R,0,0,1,0);
  vp->dsSetColor(1,0,0);

  vp->dsDrawBox(getEndPos(), R, boxSize);
}

void ArmSim::getArmLoc(double loc[3])
{

  loc[0] = itsArmParam.armLoc[0];
  loc[1] = itsArmParam.armLoc[1];
  loc[2] = itsArmParam.armLoc[2];

}
//We get end pos by 3 object position
//
//                   @ = p[i] = middle of two gripper
//    _  #   _       # = endPos
//   | |    | |
// g1|*| @  |*|g2    _
//   |_|____|_|      | length l
//   |        |      |
//   |   * m  |      V
//   |________|
//

dReal * ArmSim::getEndPos()
{
  const dReal*g1 = dBodyGetPosition(itsGripper1.body);//gripper 1 center point
  const dReal*g2 = dBodyGetPosition(itsGripper2.body);//gripper 2 center point
  const dReal*m = dBodyGetPosition(itsWrist2.body);//wrist center point

  //Half gripper length + half wrist length
  //const dReal x = itsArmParam.wrist2[2]/2 ;
  const dReal l = itsArmParam.gripper[2]/2 +itsArmParam.wrist2[2]/2 ;
  const dReal w = itsArmParam.gripper[2]/2;//half gripper length
        dReal v[3],p[3];
  for(int i=0;i<3;i++){
    p[i] = (g1[i]+g2[i])/2;
    v[i] = p[i]-m[i];
    v[i] = v[i]/l;//v hat, unit vector
    endPoint[i]=m[i]+((l+w)*v[i]);
    //x = half length of wrist2
    //l+w = length from center of wrist to the end of gripper
  }
  return endPoint;

}
void ArmSim::drawTable()
{

  double ori = 0;
  double pos[3] = {0,0,0};
  double R[12];
  double pt0[3] = {0,0,0.0};
  double pt1[3] = {-1*itsTableSize[0]/4,0,0.0};
  double pt2[3] = {-1*itsTableSize[0]/4,itsTableSize[0]/4,0.0};
  double pt3[3] = {0,itsTableSize[0]/4,0.0};
  double pt4[3] = {-1*itsTableSize[0]/4,-1*itsTableSize[0]/4,0.0};
  double pt5[3] = {0,-1*itsTableSize[0]/4,0.0};
  double boxSize[3]={0.03,0.03,0.03};

  dRFromAxisAndAngle (R,0,0,1,ori);

  vp->dsSetColor(1,1,1);
  vp->dsDrawBox(pos, R, itsTableSize); // sides[3] = {75, 1.215, 0.001};

  vp->dsSetColor(0,0,255);//Blue
  vp->dsDrawBox(pt0, R, boxSize);
  vp->dsSetColor(1,0,0);//Read
  vp->dsDrawBox(pt1, R, boxSize);
  vp->dsSetColor(0,255,255);//cyan
  vp->dsDrawBox(pt2, R, boxSize);
  vp->dsSetColor(255,0,255);//magenta
  vp->dsDrawBox(pt3, R, boxSize);
  vp->dsSetColor(1,0,1);//black
  vp->dsDrawBox(pt4, R, boxSize);
  vp->dsSetColor(1,1,0);//black
  vp->dsDrawBox(pt5, R, boxSize);
//  Table Layout
//  T = itsTableSize[0] = 1.215
//  W = itsTableSize[1] = 0.75
//  Arm location (-T/2+0.145,-W/2+0.12)
//  ---------------------------------------
//  |RobotArm |      |                |   |
//  |         |      |                |   |
//  |  (O)    |      |                |   |
//  |         |      |                |   |
//  |_________|      |                |   |
//  | |pt4           |pt1(-T/4,0)     |pt2|(-T/4,T/4)
//  |-X--------------X----------------X----
//  | |(-T/4,-T/4)   |                |   |
//  | |              |                |   |
//  | |              |                |   |
//  | |              |                |   |
//  | |              |                |   |
//  | |pt5(0,-T/4)   |pt0(0,0)        |pt3| (0,T/4)
//  |-X--------------X----------------X----
//  |                                     |
//  |                                     |
//  |                      /(((())))\     |
//  |                    ((          ))   |
//  |                   ((  (0)  (0)  ))  |
//  |                    ((          ))   |
//  |                     ((][][][][))    |
//  |                          ][         |
//  |                       __/][\__      |
//  |                      Robot Head     |
//  |                        Camera       |
//  |                                     |
//  ---------------------------------------
}



void ArmSim::simLoop()
{
         dSpaceCollide(space,this,&nearCallback);
  dWorldStep(world,0.01);
  dJointGroupEmpty(contactgroup);

  ////For get Current xyz & hpr test only
  //double xyz[3],hpr[3];
  //vp->dsGetViewpoint (xyz,hpr);
  //LINFO("double xyz[3]={%f, %f, %f};\ndouble hpr[3] = {%f, %f, %f};",
  //    xyz[0], xyz[1], xyz[2],
  //    hpr[0], hpr[1], hpr[2]);

}

Image<PixRGB<byte> > ArmSim::getFrame(int camera)
{

  XEvent event = vp->initFrame();
  drawArm();
  drawTable();
  drawObjects();
  vp->updateFrame();

  return vp->getFrame();

}

void ArmSim::drawObjects()
{

  for(uint i=0; i<itsObjects.size(); i++)
  {
    Object obj = itsObjects[i];

    vp->dsSetColor(obj.color[0],obj.color[1],obj.color[2]);
    switch(obj.type)
    {
      case BOX:
        dReal size[3];
        dGeomBoxGetLengths(obj.geom,size);
        vp->dsDrawBox(
            dBodyGetPosition(obj.body),
            dBodyGetRotation(obj.body),
            size);
        drawLine((double*)dBodyGetPosition(obj.body));
        break;
      default:
        break;
    }

  }

  for(uint i=0; i<itsDrawObjects.size(); i++)
  {
    DrawObject obj = itsDrawObjects[i];
    switch(obj.type)
    {
      case SPHERE:
        vp->dsSetColor(1.0,0,0);
        vp->dsDrawSphere(obj.loc, obj.rot, 0.005);
        break;
      default:
        break;
    }
  }

}

void ArmSim::getObjLoc(const int x, const int y, double loc[3])
{
  LINFO("Getting obj loc from %ix%i\n", x, y);

  vp->unProjectPoint(x,y,loc);

}
void ArmSim::moveObject(double pos[3],int objID)
{
  if(itsObjects.size() == 0)
    return ;
  Object obj = itsObjects[objID];
    dBodySetPosition(obj.body,pos[0],pos[1],pos[2]);


}
void ArmSim::addObject(OBJECT_TYPE objType,double initPos[3])
{
      double boxSize[3];
      boxSize[0] = 0.040;
      boxSize[1] = 0.040;
      boxSize[2] = 0.080;
      addObject(objType,initPos,boxSize);
}
void ArmSim::addObject(OBJECT_TYPE objType,double initPos[3],double objSize[3])
{

  //double initPos[3] = {0.3, 0.2, 0.1}; //10cm above flooor and infornt of the robot
  dMass m;
  switch(objType)
  {
    case BOX:
      Object obj;
      obj.body = dBodyCreate(world);
      dBodySetPosition(obj.body, initPos[0], initPos[1], initPos[2]);
      dMassSetBoxTotal(&m, 1, objSize[0], objSize[1], objSize[2]);
      dBodySetMass(obj.body, &m);
      obj.geom = dCreateBox(space, objSize[0], objSize[1], objSize[2]);
      dGeomSetBody(obj.geom, obj.body);
      dGeomSetCategoryBits(obj.geom,OBJ_BITFIELD);
      dGeomSetCollideBits(obj.geom,!GRIPPER_BITFIELD && !ARM_BITFIELD);
      //50 216 73 green block
      obj.color[0] = 0; obj.color[1] = 196; obj.color[2] = 127; //1 0 0 Red object
      obj.type = BOX;
      itsObjects.push_back(obj);
      break;
    default:
      LINFO("Unknown object");
      break;
  }

}

void ArmSim::addDrawObject(OBJECT_TYPE objType,double pos[3])
{
  switch(objType)
  {
    case SPHERE:
      DrawObject obj;
      obj.type = SPHERE;
      obj.loc[0] = pos[0];
      obj.loc[1] = pos[1];
      obj.loc[2] = pos[2];

      dRSetIdentity(obj.rot);
      itsDrawObjects.push_back(obj);
      break;
    default:
      LINFO("Unknown object");
      break;
  }

}


int ArmSim::getGradientEncoder(MOTOR m,double *targetPos)
{
  return ang2encoder(getGradient(m,targetPos),m);
}
double ArmSim::getGradient(MOTOR m,double *targetPos)
{
  dReal axis[3],joint_center[3];
  dReal *tip;
  dReal toTip[3],toTarget[3];
  dReal movement[3];
  double gradient;
  dJointID joint = getJointID(m);
  dJointGetHingeAxis(joint,axis);
  dJointGetHingeAnchor(joint,joint_center);
  tip = getEndPos();
  for(int i = 0;i < 3;i++)
  {
    toTip[i] = tip[i] -joint_center[i];
    toTarget[i] = targetPos[i] - tip[i];
  }
  crossproduct(toTip,axis,movement);
  gradient = (double)dotproduct(movement,toTarget);

  return gradient;
}
void ArmSim::getJointAxis(MOTOR m,dReal axis[3])
{
  dJointGetHingeAxis(getJointID(m),axis);
}
void ArmSim::getJointCenter(MOTOR m,dReal joint[3])
{
  dJointGetHingeAnchor(getJointID(m),joint);
}
dJointID ArmSim::getJointID(MOTOR m)
{
  dJointID joint = 0;
  switch(m)
  {
    case RobotArm::BASE:
      joint = itsBody.joint;
      break;
    case RobotArm::SHOLDER:
      joint = itsUpperArm.joint;
      break;
    case RobotArm::ELBOW:
      joint = itsForearm.joint;
      break;
    case RobotArm::WRIST1:
      joint = itsWrist1.joint;
      break;
    case RobotArm::WRIST2:
      joint = itsWrist2.joint;
      break;
    case RobotArm::GRIPPER:
      joint = itsGripper1.joint;
      break;
    default:
      break;
  }
  return joint;

}
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
