/*!@file BeoSub/BeoSubAction.H Helper class for BeoSub motor actions */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubAction.H $
// $Id: BeoSubAction.H 4663 2005-06-23 17:47:28Z rjpeters $
//

#ifndef BEOSUBACTION_H_DEFINED
#define BEOSUBACTION_H_DEFINED

class BeoSub;

// ######################################################################
//! Helper base class for BeoSub motor actions
/*! The base class defines an interface that will then be used in
  various derived forms in order to perfom a variety of actions */
class BeoSubAction
{
public:
  //! Constructor
  BeoSubAction(BeoSub *bs, const bool pulseWidthControl = false);

  //! Destructor
  virtual ~BeoSubAction();

  //! Execute the action
  /*! This will attempt to reach the target reading by iteratively
    outputing motor commands and monitoring the instruments. If stabil
    is true, we will attempt to maintain the sub stable while
    executing the action. Returns true on success. */
  virtual bool execute(const float target, const bool stabil, const int itsMaxIter = 100);

  virtual bool pulseWidthExecute(const float target, const bool stabil, const int itsMaxIter = 100);


protected:
  //! Activate the action for one step
  /*! @param incr if true, the one step of activation should tend to
    increase the value returned by getPosition(), otherwise it should
    tend to decrease it
    @return true on success */
  virtual bool activate(const bool incr) = 0;

  //turn on the motor for corresponding action
  virtual bool turnOnMotor(const bool incr) = 0;

  //turn off the motor for corresponding action
  virtual bool turnOffMotor(const bool incr) = 0;

  //! Get a reading of the current position
  virtual float getPosition() const = 0;

  //! Stabilize whatever is not being modified by the action
  virtual bool stabilize(const bool pulseWidthControl = false) = 0;

  //! Calculate current and target value differences
  virtual float difference(float pos, float target) = 0;

  BeoSub *itsBeoSub; //! our BeoSub, which holds our actuators & instruments
  float itsDeadBand; //! deadband
  bool pulseWidthControl; //! true when using pulse width control, number of pusle control when false

  float itsGainP;    //! Gain for Propositional Control
  float itsGainD;    //! Gain for Derivative Control
  float itsGainI;    //! Gain for Integral Control

  int turnOnTime; //! Number of time steps (iterations) for the motor to turn on when going pule interval control

  float itsErrorHistory[20]; //! storing pass 20 iterations value
  int itsErrIndex; //! index top iterate the history
  float u; //! motor command in number of pulses
};

// ######################################################################
//! Helper base class for BeoSub dive actions
class BeoSubActionDive : public BeoSubAction
{
public:
  //! Constructor
  BeoSubActionDive(BeoSub *bs, bool pulseWidthControl = false);

  //! Destructor
  virtual ~BeoSubActionDive();


protected:
  //! Dive for one step (go deeper if incr=true)
  virtual bool activate(const bool incr);

  //turn on the motor for corresponding action
  virtual bool turnOnMotor(const bool incr);

  //turn off the motor for corresponding action
  virtual bool turnOffMotor(const bool incr);

  //! Get a reading of the current depth
  virtual float getPosition() const;

  //! Stabilize the heading
  virtual bool stabilize(const bool pulseWidthControl = false);

  //! Calculate current and target value differences
  virtual float difference(float pos, float target);


private:
  float itsHeading;  // ref heading (yaw only) in case we want to stabilize it
};

// ######################################################################
//! Helper base class for BeoSub turn actions
class BeoSubActionTurn : public BeoSubAction
{
public:
  //! Constructor
  BeoSubActionTurn(BeoSub *bs, bool pulseWidthControl = false);

  //! Destructor
  virtual ~BeoSubActionTurn();

protected:
  //! Turn for one step (turn right if incr=true)
  virtual bool activate(const bool incr);

  //turn on the motor for corresponding action
  virtual bool turnOnMotor(const bool incr);

  //turn off the motor for corresponding action
  virtual bool turnOffMotor(const bool incr);


  //! Get a reading of the current heading
  virtual float getPosition() const;

  //! Stabilize the depth
  virtual bool stabilize(const bool pulseWidthControl = false);

  //! Calculate current and target value differences
  //! Returning shortest turning angle with clockwise direction (positive)
  virtual float difference(float pos, float target);

private:
  float itsDepth;    // our reference depth in case we want to stabilize it
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
