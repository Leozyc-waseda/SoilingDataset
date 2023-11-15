/*!@file ArmControl/vgrab-server.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/vgrab-server.C $
// $Id: vgrab-server.C 10794 2009-02-08 06:21:09Z itti $
//

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Media/FrameSeries.H"
#include "NeovisionII/nv2_common.h"
#include "NeovisionII/nv2_label_server.h"
#include <iostream> // for std::cin
#include <signal.h>
#include "ArmControl/CtrlPolicy.H"
#include "ArmControl/ArmSim.H"
#include "ArmControl/RobotArm.H"
#include "ArmControl/ArmController.H"
#include "ArmControl/ArmPlanner.H"

bool terminate = false;
struct nv2_label_server* server;
nub::soft_ref<Scorbot> scorbot;
nub::soft_ref<ArmController> armControllerScorbot;

void terminateProc(int s)
{
  LINFO("Ending application\n");
  ArmController::JointPos jointPos;
  //set The min-max joint pos
  jointPos.base = 0;
  jointPos.sholder = 0;
  jointPos.elbow = 0;
  jointPos.wrist1 = 0;
  jointPos.wrist2 = 0;

  armControllerScorbot->setJointPos(jointPos);
  armControllerScorbot->setControllerOn(true);
  sleep(1);
  while(!armControllerScorbot->isFinishMove())
  {
    usleep(1000);
  }
  LINFO("Reached home position");
  armControllerScorbot->setControllerOn(false);

        scorbot->stopAllMotors();
  nv2_label_server_destroy(server);
  terminate = true;
  exit(0);
}

PID<float> basePID(-0.75f, 0.55, 0.0,
    -20, -20,
    10,   //error_threshold
    -35,   //no move pos threshold
    35   //no move neg threshold
    );


PID<float> sholderPID(-20.0f, 0.0, 0.0,
    -20, -20,
    10,   //error_threshold
    -20,   //no move pos threshold
    20,   //no move neg threshold
    100, //max motor
    -100, //min motor
    150, //no move thresh
    true, //run pid
    1.0, //speed
    0.001, //pos static err thresh
    -0.001 //neg startic err thresh
    );

PID<float> elbowPID(-1.0f, 0.25, 0.0,
    -20, -20,
    10,   //error_threshold
    -40,   //no move pos threshold
    40   //no move neg threshold
    );


#define Z_THRESH -0.180
bool moveToObject(const Point2D<int> fix,
    const nub::soft_ref<Scorbot>& scorbot)
{

  bool moveDone = false;
  LINFO("Moving to %ix%i", fix.i, fix.j);

  int baseVal = (int)basePID.update(320/2, fix.i);
  //Get the z value
  float ef_x, ef_y, ef_z;

  scorbot->getEF_pos(ef_x, ef_y, ef_z);
  LINFO("Z %f X %f", ef_z, ef_x);
  sholderPID.update(Z_THRESH, ef_z);

  if (baseVal) //move base first
  {
    LINFO("Base val %i", baseVal);
    scorbot->setMotor(RobotArm::BASE, baseVal);
    scorbot->setMotor(RobotArm::SHOLDER, 0);
    //scorbot->setMotor(RobotArm::ELBOW, 0);
  } else {
    scorbot->setMotor(RobotArm::BASE, 0);


    if (ef_z > Z_THRESH)
    {
      int sholderVal = 0;
      if (ef_z > -0.1)
      {
        float ang = scorbot->getEncoderAng(RobotArm::SHOLDER);
        LINFO("Ang %f", ang);
        if (ang > -M_PI/4  && ang < M_PI/4)
          sholderVal = 60;
        else
          sholderVal = 13;
      }
      else
      {
        sholderVal = (int)sholderPID.update(Z_THRESH, ef_z);
        LINFO("Sholder PID %f", sholderPID.getErr());
      }
      LINFO("Sholder val %i", sholderVal);
      scorbot->setMotor(RobotArm::SHOLDER, sholderVal);

    } else {
      int sholderVal = (int)sholderPID.update(Z_THRESH, ef_z);
      scorbot->setMotor(RobotArm::SHOLDER, sholderVal);


      if (fabs(elbowPID.getErr()) < 10 &&
          fabs(sholderPID.getErr()) < 0.10 &&
          fabs(basePID.getErr()) < 10)
        moveDone = true;

    }


  }
  int elbowVal = (int)elbowPID.update(240/2, fix.j);
  LINFO("Elbow val %i",elbowVal);
  if (ef_x > 0.2)
  {
        scorbot->setMotor(RobotArm::ELBOW, elbowVal);
  } else {
        scorbot->setMotor(RobotArm::ELBOW, 0);
  }


  LINFO("ERR: %f %f %f", elbowPID.getErr(),
          sholderPID.getErr(),
          basePID.getErr());

  return moveDone;
}

int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager *mgr = new ModelManager("Test ObjRec");

  //nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
  //mgr->addSubComponent(ofs);

  scorbot = nub::soft_ref<Scorbot>(new Scorbot(*mgr,"Scorbot", "Scorbot"
        , "/dev/ttyUSB0"));

  armControllerScorbot = nub::soft_ref<ArmController>(new ArmController(*mgr,
        "ArmControllerScorbot", "ArmControllerScorbot", scorbot));
  mgr->addSubComponent(armControllerScorbot);


  if (mgr->parseCommandLine(
        (const int)argc, (const char**)argv, "<server ip>", 1, 1) == false)
    return 1;
  mgr->start();


  armControllerScorbot->setMotorsOn(true);
  armControllerScorbot->setPidOn(true);
  armControllerScorbot->setControllerOn(false);
  ArmController::JointPos jointPos;

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


  //// catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminateProc); signal(SIGINT, terminateProc);
  signal(SIGQUIT, terminateProc); signal(SIGTERM, terminateProc);
  signal(SIGALRM, terminateProc);


  //get command line options
  const char *server_ip = mgr->getExtraArg(0).c_str();

  server = nv2_label_server_create(9930,
        server_ip,
        9931);

  nv2_label_server_set_verbosity(server,0); //allow warnings


  armControllerScorbot->setGripper(1);
  LINFO("Starting");
  LINFO("Hit return to grab");
  getchar();
  while(!terminate)
  {

    struct nv2_image_patch p;
    const enum nv2_image_patch_result res =
      nv2_label_server_get_current_patch(server, &p);

    std::string objName = "nomatch";
    if (res == NV2_IMAGE_PATCH_END)
    {
      fprintf(stdout, "ok, quitting\n");
      break;
    }
    else if (res == NV2_IMAGE_PATCH_NONE)
    {
      usleep(10000);
      continue;
    }
    else if (res == NV2_IMAGE_PATCH_VALID &&
       p.type == NV2_PIXEL_TYPE_RGB24)
    {

      const Image<PixRGB<byte> > im((const PixRGB<byte>*) p.data,
          p.width, p.height);
      bool moveDone = moveToObject(Point2D<int>(p.fix_x, p.fix_y), scorbot);
      LINFO("MoveDone %i", moveDone);
      if (moveDone)
      {
        sleep(2);
        LINFO("Grasping Object");
        //TODO

        //Move wrist to object
        ArmController::JointPos currentJointPos =
          armControllerScorbot->getJointPos();
        ArmController::JointPos touchJointPos =
          armControllerScorbot->getJointPos();
        currentJointPos.wrist1 = -250;
        currentJointPos.wrist2 = 250;
        currentJointPos.gripper = 0;
        currentJointPos.reachable = true;
        LINFO("Set Pos");
        armControllerScorbot->setJointPos(currentJointPos, false);
        armControllerScorbot->setControllerOn(true);
        sleep(1);
        while(!armControllerScorbot->isFinishMove())
        {
          usleep(1000);
        }

        //Grasp the object
        LINFO("Ready to Grasp object");
        armControllerScorbot->setGripper(0);
        while(!armControllerScorbot->isFinishMove())
        {
          usleep(1000);
        }

                int gripperVal = scorbot->getEncoder(RobotArm::GRIPPER);
                LINFO("Gripper Value %i", gripperVal);
                if (gripperVal < 700)
                {

                        LINFO("Object grasped");

                        //Do somthing with the block (script commands);
                        currentJointPos.sholder = 1000;
                        currentJointPos.elbow = -1000;
                        currentJointPos.base = -1000;
                        currentJointPos.gripper = 0;
                        currentJointPos.reachable = true;
                        LINFO("Set Pos");
                        armControllerScorbot->setJointPos(currentJointPos, false);
                        armControllerScorbot->setControllerOn(true);
                        sleep(1);
                        while(!armControllerScorbot->isFinishMove())
                        {
                                usleep(1000);
                        }

                        //Tell the brain we grasped the object
                        struct nv2_patch_label l;
                        l.protocol_version = NV2_LABEL_PROTOCOL_VERSION;
                        l.patch_id = p.id;
                        snprintf(l.source, sizeof(l.source), "%s", "VGrab");
                        snprintf(l.name, sizeof(l.name), "%s", "Object Grasped");
                        snprintf(l.extra_info, sizeof(l.extra_info),
                                        "auxiliary information");

                        nv2_label_server_send_label(server, &l);
                        LINFO("Send Msg to head");


                        sleep(2);
                        //Show the block to the head
                        currentJointPos.sholder = 1000;
                        currentJointPos.elbow = -1000;
                        currentJointPos.base = -2000;
                        currentJointPos.gripper = 0;
                        currentJointPos.wrist1 = 900;
                        currentJointPos.wrist2 = 900;
                        currentJointPos.reachable = true;
                        LINFO("Set Pos");
                        armControllerScorbot->setJointPos(currentJointPos, false);
                        armControllerScorbot->setControllerOn(true);
                        sleep(1);
                        while(!armControllerScorbot->isFinishMove())
                        {
                                usleep(1000);
                        }

                        sleep(2);

                        //Move to the place where is pick up
                        armControllerScorbot->setJointPos(touchJointPos, false);
                        armControllerScorbot->setControllerOn(true);
                        sleep(1);
                        while(!armControllerScorbot->isFinishMove())
                        {
                                usleep(1000);
                        }
                        //Left up a little bit
                        currentJointPos =
                                armControllerScorbot->getJointPos();

                        currentJointPos.sholder = 700;
                        currentJointPos.elbow = -700;
                        armControllerScorbot->setJointPos(currentJointPos, false);
                        armControllerScorbot->setControllerOn(true);
                        sleep(1);
                        //drop the object
                        armControllerScorbot->setGripper(1);
                        //Move to home position
                        currentJointPos.base = 0;
                        currentJointPos.sholder = 0;
                        currentJointPos.elbow = 0;
                        currentJointPos.wrist1 = 0;
                        currentJointPos.wrist2 = 0;
                        currentJointPos.gripper = 0;
                        armControllerScorbot->setJointPos(currentJointPos);
                        sleep(1);
                        while(!armControllerScorbot->isFinishMove())
                        {
                                usleep(1000);
                        }

                        LINFO("Hit return to grab");
                        getchar();
                        armControllerScorbot->setControllerOn(false);
                } else {
                        armControllerScorbot->setGripper(1);
                        LINFO("Did not grab object. try again");
                        ArmController::JointPos currentJointPos =
                                armControllerScorbot->getJointPos();
                        currentJointPos.sholder -= 300;
                        currentJointPos.elbow -= -300;
                        currentJointPos.gripper = 0;
                        currentJointPos.wrist1 = 0;
                        currentJointPos.wrist2 = 0;
                        currentJointPos.reachable = true;
                        armControllerScorbot->setJointPos(currentJointPos, false);
                        armControllerScorbot->setControllerOn(true);
                        sleep(1);

                        armControllerScorbot->setControllerOn(false);
                        moveDone = false;
                }



      }


    }

    nv2_image_patch_destroy(&p);

    sleep(1);
  }

  nv2_label_server_destroy(server);

}
