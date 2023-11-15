/*! @file ArmControl/CtrlPolicy.C  control policy for robot arm */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/CtrlPolicy.C $
// $Id: CtrlPolicy.C 10794 2009-02-08 06:21:09Z itti $
//

#include "Util/MathFunctions.H"
#include "ArmControl/CtrlPolicy.H"

CtrlPolicy::CtrlPolicy() :
  dt(.01),
  alpha_z(4),
  alpha_py(1),
  beta_z(2),
  y(0),
  y_vel(0),
  z(0),
  z_vel(0),
  goal(0)
{
}


void CtrlPolicy::setCurrentPos(double pos)
{
  y = pos;
  z = 0;

}
void CtrlPolicy::setGoalPos(double pos)
{
  goal = pos;
}
double CtrlPolicy::getPos(const double y_actual)
{

    //x_acc = alpha_x*(beta_x*(goal-x)-x_vel);
    //x_vel += dt*x_acc;
    //x += dt*x_vel;

    z_vel = alpha_z*(beta_z*(goal-y)-z);
    z+= dt*z_vel;

    //float x_tilde=(x-x_0)/(goal-x_0);
    float f = 0;
    y_vel= (z+f) + alpha_py*(y_actual-y);
    y+= dt*y_vel;
    return y;


 // z+=dt;

 // double minJerk = y_actual + (goal - y_actual)*
 //   (10*pow(z,3) - 15*pow(z,4) + 6*pow(z,5));

  //return minJerk;
}
bool CtrlPolicy::moveDone()
{
  return (fabs(goal-y)<0.001);

}
