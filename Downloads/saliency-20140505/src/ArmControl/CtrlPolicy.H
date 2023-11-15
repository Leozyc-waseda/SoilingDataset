/*! @file ArmControl/CtrlPolicy.H  control policy for robot arm */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/CtrlPolicy.H $
// $Id: CtrlPolicy.H 10794 2009-02-08 06:21:09Z itti $
//
#ifndef CtrlPolicy_H_DEFINED
#define CtrlPolicy_H_DEFINED

class CtrlPolicy
{

  public:
  //! Constructor
  CtrlPolicy();

  //! Destructor
 // ~CtrlPolicy();

  void setCurrentPos(double pos);
  void setGoalPos(double pos);
  double getPos(const double y_actual);
  bool moveDone();

  private:
  double dt;
  double alpha_z; // transformation alpha
  double alpha_py;
  double beta_z;  // transformation beta
  double y; // position
  double y_vel; // velocity
  double z; // transformation system
  double z_vel; // transformation system velocity
  double goal;
  //bool moveDone;
  //float alpha_x=8; // canonical alpha
  //float beta_x=0.5; // canonical beta
  //float x=0, x_0=0; //canonical system
  //float x_vel=0; // canonical system velocity
  //float x_acc=0; // canonical system acceleration
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
