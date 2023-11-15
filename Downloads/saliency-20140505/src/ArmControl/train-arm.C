/*! @file ArmControl/train-arm.C  train the arm */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/train-arm.C $
// $Id: train-arm.C 10794 2009-02-08 06:21:09Z itti $
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

#define USE_EXPAT
#include <lwpr/lwpr.h>
#include <lwpr/lwpr_xml.h>


ModelManager *mgr;
nub::soft_ref<ArmSim> armSim;
nub::soft_ref<ArmController> armControllerArmSim;

//generate a random target location
void getTargetPos(double* desiredPos)
{
  desiredPos[0] = -1.214/2+0.145 + randomDouble()*0.20;
  desiredPos[1] = -0.75/2+0.12 + randomDouble()*0.20;
  desiredPos[2] = 0.05; //0.60207 + randomDouble()*0.10;

  LDEBUG("New target: %0.2f %0.2f %0.2f",
      desiredPos[0],
      desiredPos[1],
      desiredPos[2]);

}

//Predict the joint angle required for this position
ArmController::JointPos getIK(LWPR_Model& ik_model, const double* desiredPos)
{
  double joints[5];
  lwpr_predict(&ik_model, desiredPos, 0.001, joints, NULL, NULL);

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

double getDistance(const double* pos, const double* desiredPos)
{
  double sum=0;
  for(int i=0; i<3; i++)
    sum += (pos[i]-desiredPos[i])*(pos[i]-desiredPos[i]);
  return sqrt(sum);

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

  lwpr_update(&ik_model, armPos, joints, pJoints, NULL);

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

int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;
  LWPR_Model ik_model;

  initRandomNumbers();
  mgr = new ModelManager("Test ObjRec");
  armSim = nub::soft_ref<ArmSim>(new ArmSim(*mgr));
  armControllerArmSim = nub::soft_ref<ArmController>(new ArmController(*mgr,
        "ArmControllerArmSim", "ArmControllerArmSim", armSim));
  mgr->addSubComponent(armControllerArmSim);

  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
  mgr->addSubComponent(ofs);

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


  ////INit the lwpr model
  //lwpr_init_model(&ik_model,3,5,"Arm_IK");

  ///* Set initial distance metric to 50*(identity matrix) */
  //lwpr_set_init_D_spherical(&ik_model,50);

  ///* Set init_alpha to 250 in all elements */
  //lwpr_set_init_alpha(&ik_model,250);

  ///* Set w_gen to 0.2 */
  //ik_model.w_gen = 0.2;

  int numWarnings = 0;
  lwpr_read_xml(&ik_model, "ik_model.xml", &numWarnings);

  mgr->start();

  initRandomNumbers();

  armControllerArmSim->setMotorsOn(true);
  armControllerArmSim->setPidOn(true);

  //set The min-max joint pos
  ArmController::JointPos jointPos;
  jointPos.base = 3000;
  jointPos.sholder = 3000;
  jointPos.elbow = 2000;
  jointPos.wrist1 = 0;
  jointPos.wrist2 = 0;
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
  jointPos.gripper = 0;
  armControllerArmSim->setJointPos(jointPos);

  bool reachComplete = true;
  int numReaches = 0;
  int numTries = 0;
  double desiredPos[3] = {0,0,0.4};
  while(1)
  {
    armSim->simLoop();
    Image<PixRGB<byte> > armCam = flipVertic(armSim->getFrame(-1));

    ofs->writeRGB(armCam, "Output", FrameInfo("Output", SRC_POS));

    //generate a random target location
    //if (reachComplete || numTries > 500)
    {
      Point2D<int> clickLoc = getClick(ofs);
      if (clickLoc.isValid())
      {
        //getTargetPos(desiredPos);
        armSim->getObjLoc(clickLoc.i, clickLoc.j, desiredPos);
        reachComplete = false;
        numTries = 0;

        //Predict the joint angle required for this position
        ArmController::JointPos jointPos = getIK(ik_model, desiredPos);

        //move the arm
        armControllerArmSim->setJointPos(jointPos);
      }
    }

    numTries++;

    //get the arm end effector position
    double *armPos = armSim->getEndPos();
    LDEBUG("Pos %f %f %f", armPos[0], armPos[1], armPos[2]);

    //compute the error
    double err = getDistance(armPos, desiredPos);
    if (!reachComplete)
      LINFO("Err %f tries %i: reaches=%i", err, numTries, numReaches);

    if (err > 0.01)
    {
      //Learn by following the gradient
      jointPos = calcGradient(desiredPos);

      armControllerArmSim->setJointPos(jointPos);

      jointPos =  armControllerArmSim->getJointPos();
      armPos = armSim->getEndPos();

      trainArm(ik_model, armPos, jointPos);
    } else {
      reachComplete = true;
      numReaches++;
      //write the network to a file
      //lwpr_write_xml(&ik_model, "ik_model.xml");
    }

  }
  mgr->stop();

  return 0;

}

