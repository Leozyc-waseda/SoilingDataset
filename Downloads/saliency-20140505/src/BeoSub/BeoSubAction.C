/*!@file BeoSub/BeoSubAction.C Helper class for BeoSub motor actions */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubAction.C $
// $Id: BeoSubAction.C 4663 2005-06-23 17:47:28Z rjpeters $
//

#include "BeoSub/BeoSubAction.H"
#include "BeoSub/BeoSub.H"
#include <cmath>

// ######################################################################
// ########## BeoSubAction member functions
// ######################################################################
BeoSubAction::BeoSubAction(BeoSub *bs, const bool pulseWidthControl) :
  itsBeoSub(bs), itsDeadBand(10.0f), pulseWidthControl(pulseWidthControl)
{ }

// ######################################################################
BeoSubAction::~BeoSubAction()
{ }

// ######################################################################
bool BeoSubAction::execute(const float target, const bool stabil, const int itsMaxIter)
{

  if(pulseWidthControl){
    return pulseWidthExecute(target, stabil, itsMaxIter);
  }
  int iter = 0;
  itsErrIndex = 0;
  itsErrorHistory[itsErrIndex] = difference(getPosition(), target);


  while(iter < itsMaxIter)
    {
            float pos = getPosition();
        //difference between current pos and target pos
        itsErrorHistory[(itsErrIndex+1)%20] = difference(pos,  target);
        //difference between current error and error in the previous iteration
        float errDiff = itsErrorHistory[(itsErrIndex+1)%20] - itsErrorHistory[itsErrIndex];

        LDEBUG("pos = %f, target = %f", pos, target);
        LDEBUG("Current error = %f, Previous Error = %f, errDiff = %f", itsErrorHistory[(itsErrIndex+1)%20], itsErrorHistory[itsErrIndex], errDiff);

        // have we achieved the target position?
        if (fabsf(difference(pos, target)) < itsDeadBand) return true; // done

        //calculate the total error for integral control
        float totalErr  = 0; // total error over last 20 iterations
        for(int i = 0; i < 20; i ++)
          totalErr += itsErrorHistory[i];

        LDEBUG("Total error = %f", totalErr);

        //calculating number of pulses based on PID control
        u = itsGainP*itsErrorHistory[(itsErrIndex+1)%20]
          + itsGainD*errDiff
          + itsGainI*totalErr;
        LDEBUG("u = %f", u);
        int numOfPulses;
        if(fabsf(u) > 8 )
          numOfPulses = 8;
        else
          numOfPulses = int(fabsf(u));
        // activate according to difference between current and target pos:
        for(float i = 0; i < numOfPulses; i++){
           activate(u > 0);
           // if desired, stabilize a bit:
           if (stabil) stabilize();
        }

        iter ++;
        itsErrIndex = (itsErrIndex +1)%20;
        usleep(50000);
    }

  LINFO("Max number (%d) of iterations reached -- GIVING UP", itsMaxIter);
  return false;
}


// ######################################################################
bool BeoSubAction::pulseWidthExecute(const float target, const bool stabil, const int itsMaxIter)
{
  int iter = 0;
  itsErrIndex = 0;
  itsErrorHistory[itsErrIndex] = difference(getPosition(), target);


  while(iter < itsMaxIter)
    {
            float pos = getPosition();
        //difference between current pos and target pos
        itsErrorHistory[(itsErrIndex+1)%20] = difference(pos,  target);
        //difference between current error and error in the previous iteration
        float errDiff = itsErrorHistory[(itsErrIndex+1)%20] - itsErrorHistory[itsErrIndex];

        LDEBUG("Pulse Width: pos = %f, target = %f", pos, target);
        LDEBUG("Pulse Width: Current error = %f, Previous Error = %f, errDiff = %f", itsErrorHistory[(itsErrIndex+1)%20], itsErrorHistory[itsErrIndex], errDiff);

        // have we achieved the target position?
        if (fabsf(difference(pos, target)) < itsDeadBand) return true; // done

        //calculate the total error for integral control
        float totalErr  = 0; // total error over last 20 iterations
        for(int i = 0; i < 20; i ++)
          totalErr += itsErrorHistory[i];

        LDEBUG("Pulse Width: Total error = %f", totalErr);

        //calculating number of pulses based on PID control
        u = itsGainP*itsErrorHistory[(itsErrIndex+1)%20]
          + itsGainD*errDiff
          + itsGainI*totalErr;
        LDEBUG("u = %f", u);
        if(turnOnTime < 0)
          turnOnTime ++;

        if(turnOnTime == 0){
          turnOnTime = int(fabsf(u));
          turnOnMotor(u > 0);
        }

        if(turnOnTime > 0){
          turnOnTime --;
          if(turnOnTime == 0)
            turnOffMotor(u > 0);
          turnOnTime = -3;
        }

        // if desired, stabilize a bit:
        if (stabil) stabilize(pulseWidthControl);
        iter ++;
        itsErrIndex = (itsErrIndex +1)%20;
        usleep(50000);
    }

  LINFO("Max number (%d) of iterations reached -- GIVING UP", itsMaxIter);
  return false;
}


// ######################################################################
// ########## BeoSubActionDive member functions
// ######################################################################
BeoSubActionDive::BeoSubActionDive(BeoSub *bs, const bool pulseWidthControl) :
  BeoSubAction(bs, pulseWidthControl)
{
  itsDeadBand = 5.0f; //FIXME: what are the units???

  if(pulseWidthControl){
    itsGainP = 10.0f/5;
    itsGainD = 0;
    itsGainI = 0.01f/1;
  }else{
    itsGainP = 13.0f/5;
    itsGainD = 0;
    itsGainI = 0.01f/1;
  }
  // get current heading in case we want to stabilize it:
  itsHeading = itsBeoSub->getHeading();
}

// ######################################################################
BeoSubActionDive::~BeoSubActionDive()
{ }

// ######################################################################
bool BeoSubActionDive::activate(const bool incr)
{
  // one pulse on vertical thruster
  itsBeoSub->pulseMotor(BSM_UPDOWN, !incr);
  return true;
}

// ######################################################################
bool BeoSubActionDive::turnOnMotor(const bool incr){
  itsBeoSub->turnOnMotor(BSM_UPDOWN, !incr);
  return true;
}

// ######################################################################
bool BeoSubActionDive::turnOffMotor(const bool incr){
  itsBeoSub->turnOffMotor(BSM_UPDOWN);
  return true;
}

// ######################################################################
float BeoSubActionDive::getPosition() const
{ return itsBeoSub->getDepth(); }

// ######################################################################
bool BeoSubActionDive::stabilize(const bool pulseWidthControl)
{
  BeoSubActionTurn t(itsBeoSub, pulseWidthControl);

  return t.execute(itsHeading, false, 1);
}

// ######################################################################
float BeoSubActionDive::difference(float pos, float target)
{
  return pos - target;
}


// ######################################################################
// ########## BeoSubActionTurn member functions
// ######################################################################
BeoSubActionTurn::BeoSubActionTurn(BeoSub *bs, const bool pulseWidthControl) :
  BeoSubAction(bs, pulseWidthControl)
{
  itsDeadBand = 10.0f; //FIXME: what are the units???

  if(pulseWidthControl){
    itsGainP = 5.0f/20;
    itsGainD = 0;
    itsGainI = 0.01f/5;
  }
  else{
    itsGainP = 5.0f/20;
    itsGainD = 0;
    itsGainI = 0.01f/5;
  }
  // get current depth in case we want to stabilize it:
  itsDepth = itsBeoSub->getDepth();
}

// ######################################################################
BeoSubActionTurn::~BeoSubActionTurn()
{ }

// ######################################################################
bool BeoSubActionTurn::activate(const bool incr)
{
  // one pulse on lateral thrusters
  itsBeoSub->pulseMotor(BSM_LEFTRIGHT, incr);
  return true;
}
// ######################################################################
bool BeoSubActionTurn::turnOnMotor(const bool incr){
  itsBeoSub->turnOnMotor(BSM_UPDOWN, !incr);
  return true;
}

// ######################################################################
bool BeoSubActionTurn::turnOffMotor(const bool incr){
  itsBeoSub->turnOffMotor(BSM_UPDOWN);
  return true;
}

// ######################################################################
float BeoSubActionTurn::getPosition() const
{ return itsBeoSub->getHeading(); }

// ######################################################################
bool BeoSubActionTurn::stabilize(const bool pulseWidthControl)
{
  BeoSubActionDive d(itsBeoSub, pulseWidthControl);
  return d.execute(itsDepth, false, 1);
}

// ######################################################################
float BeoSubActionTurn::difference(float pos, float target)
{
  //assuming turning clockwise is positive
  if(fabsf(pos - target) > 180){
    if(pos > target)
      return target + 360 - pos;
    else
        return pos + 360 - target;
  }
  else
    return (pos - target);

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
