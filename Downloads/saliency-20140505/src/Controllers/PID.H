/*!@file Controllers/PID.H Basic PID for the BeoSub */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Controllers/PID.H $
// $Id: PID.H 15264 2012-05-05 01:10:08Z kai $
//

#ifndef PID_H_DEFINED
#define PID_H_DEFINED

#include "Util/Angle.H"

//! Definition and access functions for a BeoSub PID
template <class T> class PID
{
public:

  //! Constructor
  PID(const float pGain, const float iGain, const float dGain,
      const T& iMin, const T& iMax,
      const T& errThresh = 0,
      const T& posThresh = 0, const T& negThresh = 0,
      const T& maxMotor = 100,const T& minMotor = -100,
      const int noMoveThresh = 150,
      const bool runPID = true,
      const float speed = 1,
      const T& posStaticErrThresh = 50,
      const T& negStaticErrThresh = -50
      );

  //! Destructor
  ~PID();

  //! Set the gains
  void setPIDPgain(float p);
  void setPIDIgain(float i);
  void setPIDDgain(float d);
  void setSpeed(float s);
  void setPIDOn(bool val);

  float getPIDPgain() { return itsPgain; }
  float getPIDIgain() { return itsIgain; }
  float getPIDDgain() { return itsDgain; }

  T getVel() { return itsVel; }
  T getErr() { return itsErr; }
  T getVal() { return itsVal; }
  T getIErr() { return itsIstate; }
  T getTarget() { return itsTarget; }
  T getPTerm() { return itsPTerm; }
  T getRunPID() { return itsRunPID; }

  //! Receive a new measurement value and get the new recommended action
  T update(const T& target, const T& val);

  //! Receive a new measurement value and get the new recommended action
  T update(const T& targetPos, const T& targetVel, const T& currentVal);

private:
  T itsIstate, itsDstate;
  float itsPgain, itsIgain, itsDgain;
  T itsImin, itsImax;
  T itsErrThresh, itsPosStaticErrThresh, itsNegStaticErrThresh, itsPosThresh, itsNegThresh;
  T itsMaxMotor, itsMinMotor;
  int itsNoMoveThresh;
  bool itsRunPID;
  float itsSpeed;
  int itsNoMoveCount;
  T itsVel;
  T itsErr;
  T itsVal; //current value
  T itsTarget; //current target
  T itsPTerm; //current correction
  T itsLastVal; //current correction
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
