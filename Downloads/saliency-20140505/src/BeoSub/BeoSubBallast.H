/*!@file BeoSub/BeoSubBallast.H Ballast for a BeoSub */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubBallast.H $
// $Id: BeoSubBallast.H 5130 2005-07-30 06:07:25Z itti $
//

#ifndef BEOSUBBALLAST_H_DEFINED
#define BEOSUBBALLAST_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include <pthread.h>

class BeoChip;

//! BeoSub Ballast Class - responsible for the lower level movement of the Sub
/*! This is the class that does the actual workhorse of moving ballast
  This class gets instantiated in BeoSub constructor. */

class BeoSubBallast : public ModelComponent
{
public:
  //! Constructor
  /*! Ballast Class Constructor
   */
  BeoSubBallast(OptionManager& mgr,
                const std::string& descrName = "BeoSubBallast",
                const std::string& tagName = "BeoSubBallast");

  //! Set our BeoChip
  void setBeoChip(nub::soft_ref<BeoChip>& bc);

  //! Destructor
  ~BeoSubBallast();

  //! Initialize the ballast by probing its endpoints
  void mechanicalInitialize();

  //! Set to a given fill level, between 0.0 (empty) and 1.0 (full)
  void set(const float val, const bool blocking = false);

  //! Get the current level, between 0.0 (empty) and 1.0 (full)
  float get() const;

  //! Get the current level, in number of pulses
  int getPulses() const;

  //! Something changed on the BeoChip's inputs
  void input(const int val);

  //! Are we moving?
  bool moving() const;

protected:
  NModelParam<int> itsPulsesPerFill; //!< how many pulses per full travel?
  NModelParam<int> itsOutRed; //!< BeoChip digital out number for red wire
  NModelParam<int> itsOutWhite; //!< BeoChip digital out number for wht wire
  NModelParam<int> itsInYellow; //!< BeoChip keyboard in number for yellow wire
  NModelParam<int> itsInWhite; //!< BeoChip keyboard in number for yellow wire

private:
  nub::soft_ref<BeoChip> itsBeoChip;
  volatile int itsDesiredPulses;  // our desired value
  volatile int itsCurrentPulses;  // our current value
  enum MotorDirection { Idle = 0, Filling = 1, Emptying = 2 };
  MotorDirection itsDirection;
  MotorDirection itsPreviousDirection;
  bool itsIsFull, itsIsEmpty;
  bool itsInitialized;

  int itsPreviousInputs;  // copy of previous inputs

  void move(const MotorDirection dir);
  void inputDuringInit(const int val);
  bool itsGotEndstop;

  pthread_mutex_t itsLock;
};


#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
