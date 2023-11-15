/*!@file ArmControl/ArmSim.H Interface to robot arm */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/ArmSim.H $
// $Id: ArmSim.H 10794 2009-02-08 06:21:09Z itti $
//

#ifndef ArmSim_H_DEFINED
#define ArmSim_H_DEFINED
#include "ArmControl/RobotArm.H"
#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Devices/Serial.H"
#include "Controllers/PID.H"
#include "GUI/ViewPort.H"
#include "Util/Types.H"
#include <deque>
#include <pthread.h>
#include <stdarg.h>
#include <ode/ode.h>

#define MAX_PWM 90

#define NUM 5

#define INIT_ELBOW_ANGLE 50.0
#define INIT_SHOLDER_ANGLE 75.0



class ArmSim : public RobotArm
{
public:
#define ARM_GRAVITYMODE 0
#define ARM_BITFIELD 1
#define GRIPPER_BITFIELD 2
#define GROUND_BITFIELD 4
#define OBJ_BITFIELD 8

  //servo definition
//  enum MOTOR {BASE, ELBOW, SHOLDER, WRIST1, WRIST2, GRIPPER};

  struct ArmParam {
    double armLoc[3];

    double base[2]; //0-r 1-length
    double baseMass;

    double body[3];
    double bodyMass;

    double upperarm[3];
    double upperarmMass;

    double forearm[3];
    double forearmMass;

    double wrist1[2];
    double wrist1Mass;

    double wrist2[3];
    double wrist2Mass;

    double gripper[3];
    double gripperMass;

  };

  enum OBJECT_TYPE {BOX, CYLINDER, SPHERE};

  //Constructor
  ArmSim(OptionManager& mgr,
          const std::string& descrName = "ArmSim",
          const std::string& tagName = "ArmSim",
          const double l0 = 0.8,const double l1 = 0.9,const double l2 = 1.00,
    const double l3 = 1.00,const double l4 = 1.00
          );
  ~ArmSim();

  //###########################################################
  void start2();


  void setMotor(MOTOR m, int val);
  void setMotorPos(MOTOR m, int pos){}
  int getEncoder(MOTOR m);
  void resetEncoders();
  void resetMotorPos(){}
  void stopAllMotors();
  void homeMotor();


  void setSafety(bool val);
  //! Bring motor to a predefined position
  void simLoop();
  void startSim(void);
  float getPWM(MOTOR m);

  dWorldID        getWorld() { return world; }
  dJointGroupID getContactgroup() { return contactgroup; }

  float toDegree(float r){return (180*r)/M_PI;}
  float toRadius(float d){return (d*M_PI)/180;}
  Image<PixRGB<byte> > getFrame(int camera);

  //! Get the arm location from 2d to 3d
  void getArmLoc(double loc[3]);
  //! Get the object location from 2d to 3d
  void getObjLoc(const int x, const int y, double loc[3]);
  //! Get the gradient movement from joint in angle value
  double getGradient(MOTOR m,double *targetPos);
  //! Get the gradient movement from joint in encoder value
  int getGradientEncoder(MOTOR m,double *targetPos);
  //! Add an object of type OBJECT_TYPE
  void addObject(OBJECT_TYPE objType,double initPos[3]);
  void addObject(OBJECT_TYPE objType,double initPos[3],double objSize[3]);

  void addDrawObject(OBJECT_TYPE objType,double pos[3]);
  //! Draw the objects we know of
  void drawObjects();
  //! Move the first add objects
  void moveObject(double pos[3],int objID = 0);
  dReal * getEndPos();
  //! Convert angle value to encoder value
  int ang2encoder(double ang,MOTOR m);
  //! Convert encoder value to angle value
  double encoder2ang(int eng,MOTOR m);
  //! Get joint axis direction by motor
  void getJointAxis(MOTOR m,dReal axis[3]);
  //! Get the value of the joint center position
  void getJointCenter(MOTOR m,dReal joint[3]);
  //! get joint id by motor
  dJointID getJointID(MOTOR m);
  //! Draw a line from arm end position to target
  void drawLine(double pos[3]);
private:

  void makeArm();
  void drawArm();
  void drawTable();

  void pwmControl();

  dWorldID        world;
  dSpaceID        space;
  dGeomID          ground;
  dJointGroupID contactgroup;

  typedef struct{
          dBodyID body;
          dGeomID geom;
    dJointID joint;
    OBJECT_TYPE type;
    float color[3];
  } Object;

  ArmParam itsArmParam;
  Object itsBase;
  Object itsBody;
  Object itsUpperArm;
  Object itsForearm;
  Object itsWrist1;
  Object itsWrist2;
  Object itsGripper1;
  Object itsGripper2;

  double itsTableSize[3];
  dReal endPoint[3];
  std::vector<Object> itsObjects;

  typedef struct {
    OBJECT_TYPE type;
    double loc[3];
    double rot[12];
  } DrawObject;
  std::vector<DrawObject> itsDrawObjects;

  ViewPort *vp;
  void crossproduct(dReal *v1,dReal *v2,dReal *result)
  {
    result[0] = v1[1]*v2[2]-v1[2]*v2[1];
    result[1] = v1[2]*v2[0]-v1[0]*v2[2];
    result[2] = v1[0]*v2[1]-v1[1]*v2[0];
  }
  dReal dotproduct(dReal *v1,dReal *v2)
  {
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
  }
  double getDistance(const double* pos, const double* desire)
  {
    double sum = 0;
    for(int i = 0; i < 3; i++)
            sum+= (pos[i]-desire[i])*(pos[i]-desire[i]);
    return sqrt(sum);
  }
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
