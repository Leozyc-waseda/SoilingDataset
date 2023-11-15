/*!@file ArmControl/ArmPlanner.H Interface to robot arm */

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
// Primary maintainer for this file: Chin-Kai Chang<chinkaic@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/ArmPlanner.H $
// $Id: ArmPlanner.H 10794 2009-02-08 06:21:09Z itti $
//

#ifndef ArmPlanner_H_DEFINED
#define ArmPlanner_H_DEFINED
#include "Component/ModelParam.H"
#include "Component/ModelOptionDef.H"
#include "Component/ModelComponent.H"
#include "ArmControl/CtrlPolicy.H"
#include "ArmControl/ArmSim.H"
#include "ArmControl/RobotArm.H"
#include "ArmControl/ArmController.H"
#include "Util/Types.H"
#include "Util/MathFunctions.H"
#include <deque>
#include <stdarg.h>
#include <unistd.h>
#include <stdio.h>

//#define USE_LWPR
#ifdef USE_LWPR
#define USE_EXPAT
#include <lwpr/lwpr.h>
#include <lwpr/lwpr_xml.h>
#endif
class ArmPlanner : public ModelComponent
{
public:


  //Constructor
  ArmPlanner(OptionManager& mgr,
      const std::string& descrName = "ArmPlanner",
      const std::string& tagName = "ArmPlanner",
      nub::soft_ref<ArmController> realControler = nub::soft_ref<ArmController>(),
      nub::soft_ref<ArmController> controler = nub::soft_ref<ArmController>(),
      nub::soft_ref<ArmSim> armSim = nub::soft_ref<ArmSim>()
      );

  //Distructor
  ~ArmPlanner(){};

  //! Move the arm to desire point with the error closer than errThres
  bool move(double *desire,double errThres);//{}
  bool moveRel(double x,double y,double z,double errThres);
  //! Compute the gradient to get the next movement direction and distance
  bool gradient(double *desire,double errThres);//{return false;}
  //! Useing the gibbs sampling to approach the desire point
  bool gibbsSampling(double *desire,double errThres);//{return false;}
  //! Useing the line segment approaching between desire and arm
  bool gibbsControl(double *desire,double d);//{return false;}
  //! Move ArmSim motor
  void moveMotor(int motor,int move);
#ifdef USE_LWPR
  //! Traing the arm to learn how to approach the desire point
  void trainArm(LWPR_Model& ik_model, const double* armPos, const ArmController::JointPos& jointPos);
  //! Using current learning data to get IK
  ArmController::JointPos getIK(LWPR_Model& ik_model, const double* desiredPos);
#endif
  ArmController::JointPos getIK2(const double* desiredPos);
  void getFK(double *endPos);
  //!
  //!
  ArmController::JointPos calcGradient(const double* desiredPos);

  void minJerk(double *desire,double *nextPoint,double time);
  void updateDataImg();
  const Image<PixRGB<byte> >* getImagePtr() { return &itsImage; }
  ArmController::JointPos sim2real(ArmController::JointPos armSimJointPos );
  //###########################################################
private:
#ifdef USE_LWPR
  LWPR_Model ik_model;
#endif
  int numWarnings ;
  nub::soft_ref<ArmController> itsRealArmController;
  nub::soft_ref<ArmController> itsArmController;
  nub::soft_ref<ArmSim> itsArmSim;
  Image<PixRGB<byte> > itsImage;

  double getDistance(const double* pos, const double* desire)
  {
    double sum = 0;
    for(int i = 0; i < 3; i++)
            sum+= (pos[i]-desire[i])*(pos[i]-desire[i]);
    return sqrt(sum);
  }
  double getDistance2D(const double* pos, const double* desire)
  {
    double sum = 0;
    for(int i = 0; i < 2; i++)
            sum+= (pos[i]-desire[i])*(pos[i]-desire[i]);
    return sqrt(sum);
  }
  double sq(const double x)
  {
    return x*x;
  }

};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
