/*!@file ArmControl/ArmController.H  Control motors and pid */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// Primary maintainer for this file: Lior Elazary
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/ArmController.H $
// $Id: ArmController.H 10794 2009-02-08 06:21:09Z itti $
//

#ifndef ArmController_H_DEFINED
#define ArmController_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "rutz/shared_ptr.h"
#include "Util/Types.H"
#include "Util/Timer.H"
#include "Util/WorkThreadServer.H"
#include "Util/JobWithSemaphore.H"
#include "Controllers/PID.H"
#include "Devices/Scorbot.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "ArmControl/ArmSim.H"

#include <list>



// ######################################################################
class ArmController : public ModelComponent
{
public:

//NOTE: Do not change these motor mappings! Just plug in the cables correctly, please.
//      If for some reason these mappings must be changed, then you need to also change the order
//      in which ArmController::setMotor and ArmController::updatePID sendd their motor packets
#define ERR_THRESH 10
#define BASE_THRESH_POS 30
#define BASE_THRESH_NEG -30

#define SHOLDER_THRESH_POS 30
#define SHOLDER_THRESH_NEG -60

#define ELBOW_THRESH_POS 44
#define ELBOW_THRESH_NEG -25

#define WRIST1_THRESH_POS 30
#define WRIST1_THRESH_NEG -30

#define WRIST2_THRESH_POS 30
#define WRIST2_THRESH_NEG -30


  struct JointPos
  {
    int base;
    int sholder;
    int elbow;
    int wrist1;
    int wrist2;
    int gripper;
    bool reachable;
  };

  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  ArmController(OptionManager& mgr,
             const std::string& descrName = "ArmController",
             const std::string& tagName = "ArmController",
             nub::soft_ref<RobotArm> robotArm = nub::soft_ref<RobotArm>()); //(new RobotArm(mgr)));

  //! Destructor
  ~ArmController();


  bool setBasePos(int base, bool rel=false); //maintain a base
  bool setSholderPos(int sholder, bool rel=false); //maintain a sholder
  bool setElbowPos(int elbow, bool rel=false); //maintain a elbow
  bool setWrist1Pos(int wrist1, bool rel=false);  //maintain a wrist1
  bool setWrist2Pos(int wrist2, bool rel=false);  //maintain a wrist2
  bool setSpeed(int speed);  //maintain a speed

  void setGripper(int pos);

  void start2();

  int getElbow() { return itsCurrentElbow; }
  int getElbowErr();
  int getBase() { return itsCurrentBase; }
  int getSholder() { return itsCurrentSholder; }
  int getWrist1() { return itsCurrentWrist1; }
  int getWrist2() { return itsCurrentWrist2; }

  bool getKillSwitch() { return false; }

  void setMotor(int motor, int val);

  const int* getBasePtr() { return &itsCurrentBase; }
  const int* getSholderPtr() { return &itsCurrentSholder; }
  const int* getElbowPtr() { return &itsCurrentElbow; }
  const int* getWrist1Ptr() { return &itsCurrentWrist1; }
  const int* getWrist2Ptr() { return &itsCurrentWrist2; }

  const int* getMotor_Base_Ptr() { return &itsCurrentMotor_Base; }
  const int* getMotor_Sholder_Ptr() { return &itsCurrentMotor_Sholder; }
  const int* getMotor_Elbow_Ptr() { return &itsCurrentMotor_Elbow; }
  const int* getMotor_Wrist1_Ptr() { return &itsCurrentMotor_Wrist1; }
  const int* getMotor_Wrist2_Ptr() { return &itsCurrentMotor_Wrist2; }

  nub::soft_ref<RobotArm> getRobotArm(){return itsRobotArm;}

  void setMotorsOn(bool val) { motorsOn.setVal(val);}
  void setPidOn(bool val) {pidOn.setVal(val); }
  void setControllerOn(bool val) {controllerOn.setVal(val); }
  bool isControllerOn() {return controllerOn.getVal(); }

  void updateBase(int currentBase);
  void updateSholder(int currentSholder);
  void updateElbow(unsigned int currentElbow);
  void updateWrist1(unsigned int currentWrist1);
  void updateWrist2(unsigned int currentWrist2);
  void updatePID();
  void sendHeartBeat();

  //! Wait until all joint finish move
  bool isFinishMove();
  //! Get the joint positions
  JointPos getJointPos();

  //! Reset the joint positions to a particuler val
  void resetJointPos(JointPos &jointPos, int val = 0);

  //! Set the joint positions
  bool setJointPos(const JointPos &jointPos, bool block = true);

  //! Set the min joint positions
  void setMinJointPos(const JointPos &jointPos);

  //! Set the max joint positions
  void setMaxJointPos(const JointPos &jointPos);

  //!Stop all motors
  void killMotors();

  void genPIDImage();
  const Image<PixRGB<byte> >* getPIDImagePtr() { return &itsPIDImage; }

  //@}

  // ######################################################################
  /*! @name member functions */
  //@{


private:
  int itsDesiredBase;
  int itsDesiredSholder;
  int itsDesiredElbow;
  int itsDesiredWrist1;
  int itsDesiredWrist2;
  int itsDesiredGripper;
  int itsDesiredSpeed;

  int itsCurrentBase;
  int itsCurrentSholder;
  int itsCurrentElbow;
  int itsCurrentWrist1;
  int itsCurrentWrist2;
  int itsCurrentGripper;
  int itsCurrentSpeed;


  PID<float> itsBasePID;
  PID<float> itsSholderPID;
  PID<float> itsElbowPID;
  PID<float> itsWrist1PID;
  PID<float> itsWrist2PID;

  JointPos itsMaxJointPos;
  JointPos itsMinJointPos;

  int itsCurrentMotor_Base;
  int itsCurrentMotor_Sholder;
  int itsCurrentMotor_Elbow;
  int itsCurrentMotor_Wrist1;
  int itsCurrentMotor_Wrist2;

  int itsMoveCount_Base;
  int itsMoveCount_Sholder;
  int itsMoveCount_Elbow;
  int itsMoveCount_Wrist1;
  int itsMoveCount_Wrist2;


  void paramChanged(ModelParamBase* const param, const bool valueChanged, ParamClient::ChangeStatus* status);

  NModelParam<int> setCurrentMotor_Base;
  NModelParam<int> setCurrentMotor_Sholder;
  NModelParam<int> setCurrentMotor_Elbow;
  NModelParam<int> setCurrentMotor_Wrist1;
  NModelParam<int> setCurrentMotor_Wrist2;

  NModelParam<float> baseP;
  NModelParam<float> baseI;
  NModelParam<float> baseD;

  NModelParam<float> sholderP;
  NModelParam<float> sholderI;
  NModelParam<float> sholderD;

  NModelParam<float> elbowP;
  NModelParam<float> elbowI;
  NModelParam<float> elbowD;

  NModelParam<float> wrist1P;
  NModelParam<float> wrist1I;
  NModelParam<float> wrist1D;

  NModelParam<float> wrist2P;
  NModelParam<float> wrist2I;
  NModelParam<float> wrist2D;

  NModelParam<bool>  motorsOn;
  NModelParam<bool>  pidOn;
  NModelParam<bool>  guiOn;
  NModelParam<bool>  controllerOn;
  NModelParam<int>   motorsSpeed;
  NModelParam<bool>  basePIDDisplay;
  NModelParam<bool>  sholderPIDDisplay;
  NModelParam<bool>  elbowPIDDisplay;
  NModelParam<bool>  wrist1PIDDisplay;
  NModelParam<bool>  wrist2PIDDisplay;


  NModelParam<int>  basePos;
  NModelParam<int>  sholderPos;
  NModelParam<int>  elbowPos;
  NModelParam<int>  wrist1Pos;
  NModelParam<int>  wrist2Pos;



  /// GUI Related
  Image<PixRGB<byte> > itsPIDImage;


  rutz::shared_ptr<WorkThreadServer> itsThreadServer;

  nub::soft_ref<RobotArm> itsRobotArm;

  /// Running Time Related
  int itsAvgn;
  uint64 itsAvgtime;
  double itsLps;
  Timer itsTimer;
  void getLps(){
          itsAvgn++;
        itsAvgtime += itsTimer.getReset();
        if(itsAvgn == 20)
        {
                itsLps = 1000.0F / double(itsAvgtime) * double(itsAvgn);
                itsAvgtime = 0;
                itsAvgn = 0;

        }
        LINFO("Loop per sec %.1f lps",itsLps);

  }

};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
