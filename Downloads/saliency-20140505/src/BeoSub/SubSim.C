/*!@file BeoSub/SubSim.C Sub Simulator */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         // // See http://iLab.usc.edu for information about this project.          //
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/SubSim.C $
// $Id: SubSim.C 11565 2009-08-09 02:14:40Z rand $
//

#include "BeoSub/SubSim.H"
#include "Component/OptionManager.H"
#include "Util/MathFunctions.H"
#include "Util/Assert.H"
#include "rutz/compat_snprintf.h"
#include "Image/MatrixOps.H"
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>


namespace {

  void nearCallback (void *data, dGeomID o1, dGeomID o2){
    SubSim *subSim = (SubSim *)data;
    const int MaxContacts = 10;

    //create a contact joint to simulate collisions
    dContact contact[MaxContacts];
    int nContacts = dCollide (o1,o2,MaxContacts,
        &contact[0].geom,sizeof(dContact));
    for (int i=0; i<nContacts; i++) {
      contact[i].surface.mode = dContactSoftCFM ; //| dContactBounce; // | dContactSoftCFM;
      contact[i].surface.mu = 0.5;
      contact[i].surface.mu2 = 0.5;
      contact[i].surface.bounce = 0.01;
      contact[i].surface.bounce_vel = 0.01;
      contact[i].surface.soft_cfm = 0.001;

      dJointID c = dJointCreateContact (subSim->getWorld(),
          subSim->getContactgroup(),&contact[i]);
      dJointAttach (c,
          dGeomGetBody(contact[i].geom.g1),
          dGeomGetBody(contact[i].geom.g2));
    }
  }
}

// ######################################################################
SubSim::SubSim(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName, bool showWorld) :
  ModelComponent(mgr, descrName, tagName),
  itsWaterLevel(4.57), //15 feet of water
  itsSubLength(1.5), //the length of sub in meters
  itsSubRadius(0.2), //the radius of the sub in meters
  itsSubWeight(30), // the weight of the sub in kg
  vp(new ViewPort("SubSim")),
  itsPanTruster(0),
  itsTiltTruster(0),
  itsForwardTruster(0),
  itsUpTruster(0),
  itsXPos(0),
  itsYPos(0),
  itsDepth(0),
  itsRoll(0),
  itsPitch(0),
  itsYaw(0),
  itsWorldView(true),
  itsShowWorld(showWorld),
  itsWorldDisp(NULL)
{
  vp->isSubSim = true;

  if (itsShowWorld)
  {
    itsWorldDisp = new XWinManaged(vp->getDims(), -1, -1, "World View");
  }

  pthread_mutex_init(&itsDispLock, NULL);

}

SubSim::~SubSim()
{
  dSpaceDestroy(space);
  dWorldDestroy(world);

  delete vp;
  delete itsWorldDisp;
  pthread_mutex_destroy(&itsDispLock);
}

void SubSim::start2()
{
  //        setDrawStuff();
  world = dWorldCreate();
  space = dHashSpaceCreate(0);
  contactgroup = dJointGroupCreate(0);
  ground = dCreatePlane(space, 0, 0, 1, 0);
  dWorldSetGravity(world,0,0,0);

  dWorldSetCFM (world,1e-6);
  dWorldSetERP (world,1);
  //dWorldSetAutoDisableFlag (world,1);
  dWorldSetContactMaxCorrectingVel (world,0.1);
  //set the contact penetration
  dWorldSetContactSurfaceLayer(world, 0.001);


  makeSub();

  //set the viewpoint
  double xyz[3] = {0 , -3.0, 15};
  double hpr[3] = {90.0,-45,0.0};
  vp->dsSetViewpoint (xyz,hpr);

}

void SubSim::makeSub()
{
  dMass mass;
  itsSubBody = dBodyCreate(world);
  //pos[0] = 0; pos[1] = 1.0*5; pos[2] = 1.80;

  //  dBodySetPosition(itsSubBody,1.85,4.91*5,3.04); //Sub is 10 feet underwater
  dBodySetPosition(itsSubBody,0,0.2*5,1.94); //Sub is 10 feet underwater

  dMatrix3 R;
  dRFromAxisAndAngle (R,1,0,0,-M_PI/2);
  dBodySetRotation(itsSubBody, R);
  dMassSetZero(&mass);
  dMassSetCappedCylinderTotal(&mass,itsSubWeight,3,itsSubRadius,itsSubLength);
  dMassRotate(&mass, R);
  dBodySetMass(itsSubBody,&mass);
  itsSubGeom = dCreateCCylinder(space, itsSubRadius, itsSubLength);
  dGeomSetRotation(itsSubGeom, R);
  dGeomSetBody(itsSubGeom, itsSubBody);
}

void SubSim::drawSub()
{
  //double r, length;
  dReal r, length;

  dGeomCCylinderGetParams(itsSubGeom,&r,&length);

  vp->dsSetColor(1,1,0);
  vp->dsDrawCappedCylinder(
      dBodyGetPosition(itsSubBody),
      dBodyGetRotation(itsSubBody),
      length,
      r);

}

void SubSim::drawArena()
{

  double pos[3];

  pos[0] = 0; pos[1] = 1.0*5; pos[2] = 2.80;
  drawGate(pos);

  pos[0] = 0.16*5; pos[1] = 3.8*5; pos[2] = 1.83;
  drawBuoy(pos);

  pos[0] = 0.66*5; pos[1] = 4.33*5; pos[2] = 0.6;
  drawPipeline(0, pos);

  pos[0] = 1.16*5; pos[1] = 4.66*5; pos[2] = 0.6;
  drawBin(0, pos);

  pos[0] = 1.75*5; pos[1] = 4.91*5; pos[2] = 0.6;
  drawPipeline(M_PI/4, pos);

  pos[0] = 2.33*5; pos[1] = 5.41*5; pos[2] = 0.6;
  drawPipeline(0, pos);

  pos[0] = 3.0*5; pos[1] = 5.25*5; pos[2] = 0.6;
  drawPipeline(-M_PI/4, pos);

  pos[0] = 3.5*5; pos[1] = 4.083*5; pos[2] = 0.6;
  drawBin(M_PI/2, pos);

  pos[0] = 3.5*5; pos[1] = 4.083*5; pos[2] = 1.83;
  drawBuoy(pos);

  pos[0] = 3.83*5; pos[1] = 4.33*5; pos[2] = 0.6;
  drawPipeline(-M_PI/4, pos);


  pos[0] = 4.75*5; pos[1] = 3.5*5; pos[2] = 0.6;
  drawPinger(pos);
}


void SubSim::drawGate(const double *gatePos)
{
  double pos[3];
  double R[12];
  //Gate
  vp->dsSetColor(0,0,0);
  //Top
  pos[0] = gatePos[0]; pos[1] = gatePos[1]; pos[2] = gatePos[2];

  dRFromAxisAndAngle (R,0,1,0,-M_PI/2);
  vp->dsDrawCappedCylinder(pos, R, 3.05f, 0.1f);

  //side
  dRSetIdentity(R);
  pos[0] = gatePos[0]-(3.05/2); pos[1] = gatePos[1]; pos[2] = gatePos[2]/2;
  vp->dsDrawCappedCylinder(pos, R, 1.83, 0.1);

  pos[0] = gatePos[0]+(3.05/2); pos[1] = gatePos[1]; pos[2] = gatePos[2]/2;
  vp->dsDrawCappedCylinder(pos, R, 1.83, 0.1);
}


void SubSim::drawBuoy(const double *bouyPos)
{
  static int frame = 0;
  static bool bouyOn = false;
  double pos[3];
  double R[12];

  //start Buoy
  //flash buoy
  if (frame++ > 5) //this sets the frame rate
  {
    bouyOn = !bouyOn;
    frame = 0;
  }

  if (bouyOn)
    vp->dsSetColor(1,0,0);
  else
    vp->dsSetColor(0.5,0,0);
  pos[0] = bouyPos[0]; pos[1] = bouyPos[1]; pos[2] = bouyPos[2];
  dRSetIdentity(R);
  vp->dsDrawSphere(pos, R, 0.20);
  double pos1[3];
  vp->dsSetColor(0,0,0);
  pos1[0] = pos[0]; pos1[1] = pos[1]; pos1[2] = 0;
  vp->dsDrawLine(pos, pos1);

}

void SubSim::drawPipeline(const double ori, const double *pipePos)
{

  double sides[3] = {1.2, 0.15, 0.1};
  double R[12];

  dRFromAxisAndAngle (R,0,0,1,ori);

  vp->dsSetColor(1,0.5,0);

  vp->dsDrawBox(pipePos, R, sides);
}

void SubSim::drawBin(const double ori, const double *binPos)
{

  double sides[3];
  double R[12];

  dRFromAxisAndAngle (R,0,0,1,ori);

  vp->dsSetColor(1,1,1);
  sides[0] = 0.6; sides[1] = 0.8; sides[2] = 0.1;
  vp->dsDrawBox(binPos, R, sides);

  vp->dsSetColor(0,0,0);
  sides[0] = 0.3; sides[1] = 0.6; sides[2] = 0.15;
  vp->dsDrawBox(binPos, R, sides);
}

void SubSim::drawPinger(const double *pingerPos)
{
  double pos[3];
  double R[12];

  vp->dsSetColor(1,1,1);
  pos[0] = pingerPos[0]; pos[1] = pingerPos[1]; pos[1] = pingerPos[2] + 1.2/2;
  vp->dsDrawCappedCylinder(pos, R, 1.2, 0.1);

}


void SubSim::handleWinEvents(XEvent& event)
{
}

//void SubSim::updateSensors(const double *pos, const double *R)
void SubSim::updateSensors(const dReal *pos, const dReal *R)
{
  itsXPos = pos[0];
  itsYPos = pos[1];
  itsDepth = pos[2];
  itsRoll  = atan2(R[9], R[10]) + M_PI/2;     //phi correct for initial rotation
  itsPitch = asin(-R[8]);            //theta
  itsYaw   = atan2(R[4], R[0]);      //greek Y

  if (itsYaw < 0) itsYaw += M_PI*2;
  // LINFO("(%f,%f) Depth %f, roll %f pitch %f yaw %f",
  //     itsXPos, itsYPos, itsDepth,
  //     itsRoll, itsPitch, itsYaw);

}

void SubSim::simLoop()
{

  //set the trusters
  dBodyAddRelForceAtRelPos(itsSubBody,0,itsUpTruster,0, 0,0,0);
  dBodyAddRelTorque(itsSubBody,itsTiltTruster, 0, 0);
  dBodyAddRelTorque(itsSubBody,0, itsPanTruster, 0);
  dBodyAddRelForceAtRelPos(itsSubBody,0,0,itsForwardTruster, 0,0,0);

  //Apply a viscosity water force
  applyHydrodynamicForces(0.5);

  //Folow the sub with the camera
  //  const double *bodyPos = dBodyGetPosition(itsSubBody);
  //const double *bodyR = dBodyGetRotation(itsSubBody);
  const dReal *bodyPos = dBodyGetPosition(itsSubBody);
  const dReal *bodyR = dBodyGetRotation(itsSubBody);

  updateSensors(bodyPos, bodyR);

  dSpaceCollide (space,this,&nearCallback); //check for collisions

  dWorldStep(world,0.1);

  dJointGroupEmpty (contactgroup); //delete the contact joints

  if (itsShowWorld)
  {
    itsWorldDisp->drawImage(flipVertic(getFrame(0)));
  }
}


//! Calculate the water forces on the object
// Obtained from http://ode.org/pipermail/ode/2005-January/014929.html
void SubSim::applyHydrodynamicForces(dReal viscosity)
{
  const dReal *lvel = dBodyGetLinearVel(itsSubBody);
  const dReal *avel = dBodyGetAngularVel(itsSubBody);
  const dReal *R = dBodyGetRotation(itsSubBody);


  //Should be the area of the sub
  dReal AreaX = 10;
  dReal AreaY = 10;
  dReal AreaZ = 10;

  dReal nx = (R[0] * lvel[0] + R[4] * lvel[1] + R[8] * lvel[2]) *  AreaX;
  dReal ny = (R[1] * lvel[0] + R[5] * lvel[1] + R[9] * lvel[2]) * AreaY;
  dReal nz = (R[2] * lvel[0] + R[6] * lvel[1] + R[10] * lvel[2]) * AreaZ;

  dReal temp = -nx * viscosity;
  dBodyAddForce(itsSubBody, temp * R[0], temp * R[4], temp * R[8]);

  temp = -ny * viscosity;
  dBodyAddForce(itsSubBody, temp * R[1], temp * R[5], temp * R[9]);

  temp =-nz * viscosity;
  dBodyAddForce(itsSubBody, temp * R[2], temp * R[6], temp * R[10]);

  nx = (R[0] * avel[0] + R[4] * avel[1] + R[8] * avel[2]) * AreaZ;
  ny = (R[1] * avel[0] + R[5] * avel[1] + R[9] * avel[2]) * AreaX;
  nz = (R[2] * avel[0] + R[6] * avel[1] + R[10] * avel[2]) * AreaY;

  temp = -nx * viscosity; // * 500; //seems to strong
  dBodyAddTorque(itsSubBody, temp * R[0], temp * R[4], temp * R[8]);

  temp = -ny * viscosity; // * 500;
  dBodyAddTorque(itsSubBody, temp * R[1], temp * R[5], temp * R[9]);

  temp = -nz * viscosity; // * 500;
  dBodyAddTorque(itsSubBody, temp * R[2], temp * R[6], temp * R[10]);

}

Image<PixRGB<byte> > SubSim::getFrame(int camera)
{
  const dReal *bodyPos = dBodyGetPosition(itsSubBody);
  const dReal *bodyR = dBodyGetRotation(itsSubBody);

  double cam_xyz[3], cam_hpr[3] = {0.0,0.0,0.0};

  switch (camera)
  {
    case 0: //world camera
      cam_xyz[0] = bodyPos[0];
      cam_xyz[1] =  bodyPos[1]-5;
      cam_xyz[2] =  10;

      cam_hpr[0]   = 90.0;
      cam_hpr[1]   = -45.0;
      cam_hpr[2]   = 0.0;

      break;
    case 1:
      cam_xyz[0] = bodyPos[0];
      cam_xyz[1] = bodyPos[1];
      cam_xyz[2] = bodyPos[2];

      cam_hpr[0]   = (atan2(bodyR[4], bodyR[0])*180/M_PI) + 90; //yaw

      break;
    case 2:
      cam_xyz[0] = bodyPos[0];
      cam_xyz[1] = bodyPos[1];
      cam_xyz[2] = bodyPos[2];

      cam_hpr[0]   = (atan2(bodyR[4], bodyR[0])*180/M_PI) + 90; //yaw
      cam_hpr[1]   = -90; //yaw


      break;
  }
  pthread_mutex_lock(&itsDispLock);
  if (camera != -1)
    vp->dsSetViewpoint (cam_xyz,cam_hpr);
  vp->initFrame();
  drawSub();
  drawArena();
  vp->updateFrame();
  pthread_mutex_unlock(&itsDispLock);


  return vp->getFrame();

}

void SubSim::getSensors(float &xPos, float &yPos, float &depth,
    float &roll, float &pitch, float &yaw)
{

  xPos = itsXPos;
  yPos = itsYPos;
  depth = itsDepth;
  roll = itsRoll;
  pitch = itsPitch;
  yaw = itsYaw;

}

void SubSim::setTrusters(float panTruster, float tiltTruster, float forwardTruster, float upTruster)
{

  itsPanTruster =      panTruster;
  itsTiltTruster =     tiltTruster;
  itsForwardTruster =  forwardTruster;
  itsUpTruster =       upTruster;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
