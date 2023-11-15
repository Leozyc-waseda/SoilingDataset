/*!@file SeaBee/MovementController.C  Control seabee movement */

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
// Primary maintainer for this file: Michael Montalbo
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/MovementController.C $
// $Id: MovementController.C 10794 2009-02-08 06:21:09Z itti $
//

#include "SeaBee/MovementController.H"
#include "Component/ModelOptionDef.H"


// ######################################################################
MovementController::MovementController(OptionManager& mgr,
                                       nub::soft_ref<SubController> subController,
                                       const std::string& descrName,
                                       const std::string& tagName):

  ModelComponent(mgr, descrName, tagName),
  itsDepthErrThresh(&OPT_DepthErrThresh, this, ALLOW_ONLINE_CHANGES),
  itsHeadingErrThresh(&OPT_HeadingErrThresh, this, ALLOW_ONLINE_CHANGES),
  pipeP("pipeP", this, 0, ALLOW_ONLINE_CHANGES),
  pipeI("pipeI", this, 0, ALLOW_ONLINE_CHANGES),
  pipeD("pipeD", this, 0, ALLOW_ONLINE_CHANGES),
  itsPipePID(pipeP.getVal(), pipeI.getVal(), pipeD.getVal(), -20, 20,
             5, 0, 0, 100, -100, 150, true, 1, 25, -25),
  setDiveValue(&OPT_DiveValue, this, ALLOW_ONLINE_CHANGES),
  setGoStraightTimeValue(&OPT_GoStraightTime, this, ALLOW_ONLINE_CHANGES),
  setSpeedValue(&OPT_SpeedValue, this, ALLOW_ONLINE_CHANGES),
  setHeadingValue(&OPT_HeadingValue, this, ALLOW_ONLINE_CHANGES),
  setRelative("setRelative", this, false, ALLOW_ONLINE_CHANGES),
  setTimeout(&OPT_TimeoutValue, this, ALLOW_ONLINE_CHANGES),
  itsSubController(subController)
{

}

// ######################################################################
MovementController::~MovementController()
{

}

// ######################################################################
bool MovementController::dive(int depth, bool relative, int timeout)
{
  LINFO("Dive to depth: %d", depth);
  if(relative)
    {
      int currentDepth = itsSubController->getDepth();

      // wait for depth to initialize
      if(currentDepth == -1)
        sleep(1);

      currentDepth = itsSubController->getDepth();
      if(currentDepth == -1)
        return false;


      itsSubController->setDepth(depth + currentDepth);
      LINFO("Diving to depth: %d", currentDepth + depth);
    }
  else
    {
      itsSubController->setDepth(depth);
    }

  int time = 0;

  while(itsSubController->getDepthErr() > itsDepthErrThresh.getVal())
    {
      //      LINFO("err: %d, thresh: %d",itsSubController->getDepthErr(),itsDepthErrThresh.getVal());
      usleep(3000);
      time++;
      //      if(time++ > timeout) return false;
      //<TODO mmontalbo> turn off diving
      //      LINFO("err: %d, thresh: %d",itsSubController->getDepthErr(),itsDepthErrThresh.getVal());
    }

  return true;
}

// ######################################################################
bool MovementController::goStraight(int speed, int time)
{
  LINFO("Go Straight");
  itsSubController->setHeading(itsSubController->getHeading());
  itsSubController->setSpeed(speed);
  sleep(time);
  itsSubController->setSpeed(0);
  return true;
}

// ######################################################################
bool MovementController::setHeading(int heading, bool relative, int timeout)
{
  if(relative)
    {
      itsSubController->setHeading(heading + itsSubController->getHeading());
    }
  else
    {
      itsSubController->setHeading(heading);
    }

  int time = 0;

  while(itsSubController->getHeadingErr() > itsHeadingErrThresh.getVal())
    {
      //      usleep(10000);
      if(time++ > timeout) return false;
      //<TODO mmontalbo> turn off turning
    }

  return true;
}

// ######################################################################
int MovementController::trackPipe(const Point2D<int>& pointToTrack,
                                  const Point2D<int>& desiredPoint)
{
  float pipeCorrection = (float)itsPipePID.update(pointToTrack.i, desiredPoint.i);

  //  itsSubController->setHeading(itsSubController->getHeading()
  //                          + pipeCorrection);

  itsSubController->setTurningSpeed(pipeCorrection);

  return abs(pointToTrack.i - desiredPoint.i);
}

// ######################################################################
void MovementController::paramChanged(ModelParamBase* const param,
                                 const bool valueChanged,
                                 ParamClient::ChangeStatus* status)
{
//   if (param == &setDiveValue && valueChanged)
//     {
//       if(dive(setDiveValue.getVal(), setRelative.getVal(), setTimeout.getVal()))
//         LINFO("Dive completed successfully");
//       else
//         LINFO("Dive failed");
//     }
//   else if(param == &setGoStraightTimeValue && valueChanged)
//     {
//       if(goStraight(setSpeedValue.getVal(), setGoStraightTimeValue.getVal()))
//         LINFO("Go straight completed successfully");
//       else
//         LINFO("Go straight failed");
//     }
//   else if(param == &setHeadingValue && valueChanged)
//     {
//       if(setHeading(setHeadingValue.getVal(), setRelative.getVal(), setTimeout.getVal()))
//         LINFO("Turn completed successfully");
//       else
//         LINFO("Turn failed");
//     }
  //////// Pipe PID constants/gain change ////////
  if (param == &pipeP && valueChanged)
    itsPipePID.setPIDPgain(pipeP.getVal());
  else if(param == &pipeI && valueChanged)
    itsPipePID.setPIDIgain(pipeI.getVal());
  else if(param == &pipeD && valueChanged)
    itsPipePID.setPIDDgain(pipeD.getVal());

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
