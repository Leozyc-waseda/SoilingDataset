/*!@file BeoSub/SeaBee.H An autonomous submarine with one ballast */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/SeaBee.H $
// $Id: SeaBee.H 8521 2007-06-28 17:45:49Z rjpeters $
//

#ifndef BEOSUBTWO_H_DEFINED
#define BEOSUBTWO_H_DEFINED

#include "BeoSub/BeoSub.H"
#include "Devices/BeoChip.H"


//! Definition and access functions for the BeoSub
/*! BeoSub is an autonomous visually-guided submarine. This class
  impoements the highl-evel functionality, relying on low-level
  drivers to handle motor activation, compass reading, etc */
class SeaBee: public BeoSub
{
public:
  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  SeaBee(OptionManager& mgr);

  //! Destructor
  ~SeaBee();

  //@}

        void test();

  // ######################################################################
  /*! @name mid-level overloads */
  //@{

  //! Advance the sub by some distance in meters
  /*! This is an overload of BeoSub::advanceRel() which ensures we
    turn off the heading PID during the advance */
  virtual void advanceRel(const float relDist, const bool stop = true);

  //! Strafes the sub by some distance in meters
  /*! This is an overload of BeoSub::strafeRel() which ensures we turn
    the heading PID off during the strafe */

  ////////////////////////////////////////////////////////////////
  void turnOpen(Angle ang, bool b) ;

  void dropMarker(bool b) ;
  Image<PixRGB <byte> > grabImage(BeoSubCamera bsc) const;
  /////////////////////////////////////////////////
  //@}

  // ######################################################################
  /*! @name Low-level access functions */
  //@{

  //! Activate the thrusters
  /*! Values should be between -1.0 (full reverse) and 1.0 (full
    forward), 0.0 being no motion */
  void thrust(const float leftval, const float rightval);
  void dive(const float leftval, const float rightval);

  //! Get the current thruster settings
  void getThrusters(float& leftval, float& rightval) const;
  void getDiveThrusters(float& leftval, float& rightval) const;


  //@}

  // **********************************************************************
  /*! closed-loop control */
private:
  void updateThrusters();
  void updateDepth(const float depth);
  float PIDTransVel, TransVelCommand;
public:
  void setTransVel(const float desired);

  // ######################################################################
  /*! @name PID control */
  //@{

  //@}

  // ######################################################################
  /*! @name Miscellaneous */
  //@{


  //@}

  // ######################################################################
  /*! @name For internal use by our listeners only -- do not call */
  //@{

protected:
//   NModelParam<int> itsLeftVThrusterServoNum;  //!< servo num for left thruster
//   NModelParam<int> itsRightVThrusterServoNum; //!< servo num for right thruster

//   NModelParam<int> itsLeftHThrusterServoNum; //!< servo num for left-horizontal thruster

//   NModelParam<int> itsRightHThrusterServoNum; //!< servo num for right-horizontal thruster

  nub::soft_ref<BeoChip> itsBeo;             //!< our beochip

  float itsThrustLeftV, itsThrustRightV, itsThrustLeftH, itsThrustRightH;       //!< our current thruster settings

  float itsDiveSetting;

  virtual void start1(); //!< get started, before our subcomponents start
  virtual void start2(); //!< get started, after our subcomponents have started
  virtual void stop1();

private:
  //! Dispatch a BeoChip event
  void dispatchBeoChipEvent(const BeoChipEventType t,
                            const int valint,
                            const float valfloat);

  friend class BeoSubListener;
};






#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
