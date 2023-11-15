/*!@file Controllers/PID.C Encapsulation of PID */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Controllers/PID.C $
// $Id: PID.C 11533 2009-07-28 04:35:43Z lior $
//

#include "Controllers/PID.H"

//hack to adjust for buoyancy
#define BUOYANCY_OFFSET 0

// ######################################################################
template <class T>
PID<T>::PID(const float pGain, const float iGain,
    const float dGain, const T& iMin, const T& iMax,
    const T& errThresh,
    const T& posThresh, const T& negThresh,
    const T& maxMotor, const T& minMotor,
    const int noMoveThresh,
    const bool runPID,
    const float speed,
    const T& posStaticErrThresh,
    const T& negStaticErrThresh
    ) :
  itsIstate(0), itsDstate(0),
  itsPgain(pGain), itsIgain(iGain), itsDgain(dGain),
  itsImin(iMin), itsImax(iMax),
  itsErrThresh(errThresh), itsPosStaticErrThresh(posStaticErrThresh),
  itsNegStaticErrThresh(negStaticErrThresh),
  itsPosThresh(posThresh), itsNegThresh(negThresh),
  itsMaxMotor(maxMotor), itsMinMotor(minMotor) ,
  itsNoMoveThresh(noMoveThresh),
  itsRunPID(runPID),
  itsSpeed(speed),
  itsNoMoveCount(0)

{ }

// ######################################################################
template <class T>
PID<T>::~PID()
{ }

// ######################################################################
template <class T>
void PID<T>::setPIDPgain(float p)
{
  itsPgain = p;
}

template <class T>
void PID<T>::setPIDIgain(float i)
{
  itsIgain = i;
}

template <class T>
void PID<T>::setPIDDgain(float d)
{
  itsDgain = d;
}

template <class T>
void PID<T>::setSpeed(float s)
{
  itsSpeed = s;
}

template <class T>
void PID<T>::setPIDOn(bool val)
{
  itsRunPID = val;
  itsNoMoveCount = 0;
}

// ######################################################################
template <class T>
T PID<T>::update(const T& target, const T& val)
{
  // NOTE: we only use unary operators on T for compatibility with Angle...

  itsVal = val;
  itsTarget = target;

  if (!itsRunPID) return 0;

  // compute the error:
  T err = target - val;
  itsErr = err;
  T pterm = err; pterm *= itsPgain;

  // update the integral state:
  itsIstate += err;
  if (itsIstate > itsImax) itsIstate = itsImax;
  else if (itsIstate < itsImin) itsIstate = itsImin;
  T iterm = itsIstate; iterm *= itsIgain;

  // update the derivative term:
  T diff = (val - itsDstate);
  itsVel = diff;
  itsDstate = val;
  T dterm = diff; dterm *= itsDgain;

  // Command is just the P-I-D combination of the above:
  pterm += iterm; pterm -= dterm;

  //process some non linear controls
  //if (itsErrThresh > 0)
  //{
  //  if (itsErr > itsPosStaticErrThresh  || itsErr < itsNegStaticErrThresh)
  //  {
  //    if(itsVel != 0)
  //    {
  //      itsNoMoveCount = 0;
  //    }
  //    else
  //    {
  //      if (pterm != 0)
  //      {
  //        if (itsErr > 0)
  //          pterm = itsPosThresh;
  //        else
  //          pterm = itsNegThresh;
  //      }

  //      itsNoMoveCount++;
  //    }
  // }
   // else
   // {
   //   pterm = 0;
   //   itsNoMoveCount = 0;
   // }
    //Over move Handling
    //TODO Fix it
    //if (itsNoMoveCount > itsNoMoveThresh)
    //{
    //  //Stop Motor
    //  pterm = 0;
    //  itsRunPID = false;
    //}
  //}


  if (pterm > itsSpeed) pterm = itsSpeed;
  if (pterm < -itsSpeed) pterm = -itsSpeed;

  if (pterm > itsMaxMotor) pterm = itsMaxMotor;
  if (pterm < itsMinMotor) pterm = itsMinMotor;
  itsPTerm = pterm;


  return pterm;
}

template <class T>
T PID<T>::update(const T& targetPos, const T& targetVel, const T& currentVal)
{
  // NOTE: we only use unary operators on T for compatibility with Angle...


  if (!itsRunPID) return 0;

  // compute the error:
  T posErr = targetPos - currentVal;
  itsErr = posErr;

  // update the integral state:
  //itsIstate += err;
  //if (itsIstate > itsImax) itsIstate = itsImax;
  //else if (itsIstate < itsImin) itsIstate = itsImin;
  //T iterm = itsIstate; iterm *= itsIgain;

  // update the velocity term:

  T vel = (currentVal - itsLastVal);
  T velErr = (targetVel - vel);

  T pTerm = posErr; //To avoid angle multiplication
  pTerm *= itsPgain;

  T dTerm = velErr;
  dTerm *= itsDgain;

  T pid = pTerm + dTerm;

  //if (fabs(velErr) > 0 && posErr > 0 && vel == 0 && pid < itsPosThresh)
  //  pid = itsPosThresh;
  //else if (fabs(velErr) > 0 && posErr < 0 && vel == 0 && pid > itsNegThresh)
  //  pid = itsNegThresh;

  itsLastVal = currentVal;


  if (pid > itsMaxMotor) pid = itsMaxMotor;
  if (pid < itsMinMotor) pid = itsMinMotor;

  return pid;
}

// Instantiations:
template class PID<float>;
template class PID<Angle>;

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
