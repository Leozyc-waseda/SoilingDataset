/*!@file Beobot/BeobotAction.H Motor actions for the Beobots */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotAction.H $
// $Id: BeobotAction.H 4663 2005-06-23 17:47:28Z rjpeters $
//

#ifndef BEOBOTACTION_H_DEFINED
#define BEOBOTACTION_H_DEFINED

#include "Util/Types.H"
#include "Util/log.H"

//! This class defined motor actions for the Beobots
/*! Turn is between -1.0 (full left) and 1.0 (full right), speed is
    between -1.0 (full reverse) and 1.0 (full forward), and gear is 0
    (neutral), 1 or 2. */

class BeobotAction {
public:
  //! no arg constructor
  inline BeobotAction();

  //! constructor
  inline BeobotAction(const float t, const float s, const int g);

  //! copy constructor
  inline BeobotAction( const BeobotAction &other );

  //! Assignment constructor
  inline BeobotAction& operator=(const BeobotAction& other);

  //! access function
  inline float getTurn() const;

  //! access function
  inline float getSpeed() const;

  //! access function
  inline int getGear() const;

  //! access function
  inline void setTurn(const float t);

  //! access function
  inline void setSpeed(const float s);

  //! access function
  inline void setGear(const int g);

  //! destructor
  inline ~BeobotAction();

private:
  float turn;
  float speed;
  int gear;
};

// ######################################################################
// ######################################################################
// ######################################################################
// INLINE FUNCTIONS:
// ######################################################################
// ######################################################################
// ######################################################################

// ######################################################################
inline BeobotAction::BeobotAction()
{ turn = 0.0F; speed = 0.0F; gear = 1; }

// ######################################################################
inline BeobotAction::BeobotAction( const BeobotAction& other )
{ turn = other.turn; speed = other.speed; gear = other.gear; }

// ######################################################################
inline  BeobotAction::BeobotAction(const float t, const float s, const int g)
{ setTurn(t); setSpeed(s); setGear(g); }

// ######################################################################
inline BeobotAction& BeobotAction::operator=(const BeobotAction& other)
{ turn = other.turn; speed = other.speed; gear = other.gear; return *this; }

// ######################################################################
inline BeobotAction::~BeobotAction()
{ }

// ######################################################################
inline float BeobotAction::getTurn() const
{ return turn; }

// ######################################################################
inline float BeobotAction::getSpeed() const
{ return speed; }

// ######################################################################
inline int BeobotAction::getGear() const
{ return gear; }

// ######################################################################
inline void BeobotAction::setTurn(const float t)
{
  if (t < -1.0F || t > 1.0F) LERROR("Invalid value %f -- IGNORED", t);
  else turn = t;
}

// ######################################################################
inline void BeobotAction::setSpeed(const float s)
{
  if (s < -1.0F || s > 1.0F) LERROR("Invalid value %f -- IGNORED", s);
  else speed = s;
}

// ######################################################################
inline void BeobotAction::setGear(const int g)
{
  if (g < 0 || g > 2) LERROR("Invalid value %d -- IGNORED", g);
  else gear = g;
}


#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
