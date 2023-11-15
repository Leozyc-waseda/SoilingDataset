/*!@file Beobot/LocParticle.H localizer particle for probalistic 
  localization estimator */
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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/LocParticle.H $
// $Id: LocParticle.H 15277 2012-05-07 22:37:01Z kai $
//

// ######################################################################

#ifndef BEOBOT_LOCPARTICLE_H_DEFINED
#define BEOBOT_LOCPARTICLE_H_DEFINED


// ######################################################################

// ######################################################################
//! Thread with localizer, Object database, search priority list
//! takes in salient object and gist vector as input
class LocParticle
{
public:

  //! The default constructor for the topological map
  //! initializes the segnum to 0, length traveled to 0.0
  //! make the grid coord to invalid: -1, -1, -1
  inline LocParticle(uint snum = 0, float ltrav = 0.0F);

  //! initializes with x,y, and with or without z cartesian-coordinate 
  //! default z coordinate to 0 on the ground
  inline LocParticle(float in_x, float in_y, float in_z = 0.0F);

  //! initializes with x,y,z grid-coordinate 
  inline LocParticle
  (uint snum, float ltrav, float in_x, float in_y, float in_z = 0.0F);

  //! topological map coordinate 
  uint  segnum;
  float lentrav;

  //! cartesian coordinate
  float x;
  float y;

  //! keep z unitiliazed if want to keep it 2D
  //! can be used as a descriptor for which floor the robot is on
  //! in a building
  float z; 

  //! weights are initialized separately, here default is 0.5
  //! for programs that do not need it do not have to worry about it
  float weight; 
};

// ######################################################################
inline LocParticle::LocParticle(uint snum, float ltrav):
  segnum(snum), lentrav(ltrav)
{ 
  x = -1.0F;
  y = -1.0F;
  z = -1.0F;
  
  weight = 0.5F;
}

// ######################################################################
inline LocParticle::LocParticle(float in_x, float in_y, float in_z):
  x(in_x), y(in_y)
{   
  segnum  = 0;  
  lentrav = -1.0F;
  
  weight = 0.5F;
}

// ######################################################################
inline LocParticle::LocParticle
(uint snum, float ltrav, float in_x, float in_y, float in_z):
  segnum(snum), lentrav(ltrav),
  x(in_x), y(in_y), z(in_z)
{  
  weight = 0.5F;
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
