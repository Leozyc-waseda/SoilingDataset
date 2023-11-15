/*!@file SeaBee/MovementController.H  Control motors and pid */

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
// Primary maintainer for this file: Mike Montalbo mmontalbo[at]gmail[dot].com
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/MovementController.H $
// $Id: MovementController.H 10794 2009-02-08 06:21:09Z itti $
//

#ifndef MovementController_H_DEFINED
#define MovementController_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelOptionDef.H"
#include "Component/ModelParam.H"
#include "Component/OptionManager.H"
#include "Image/Point2D.H"
#include "SubController.H"

const ModelOptionCateg MOC_SeaBee_Movement_Controller = {
  MOC_SORTPRI_3, "SeaBee Movement Controller related options" };

static const ModelOptionDef OPT_DepthErrThresh =
  { MODOPT_ARG(int), "DepthErrThresh", &MOC_SeaBee_Movement_Controller, OPTEXP_SAVE,
    "Error threshold for depth",
    "depth-err-thresh", '\0', "<int>", "0" };

static const ModelOptionDef OPT_HeadingErrThresh =
  { MODOPT_ARG(int), "HeadingErrThresh", &MOC_SeaBee_Movement_Controller, OPTEXP_SAVE,
    "Error threshold for heading",
    "heading-err-thresh", '\0', "<int>", "0" };

static const ModelOptionDef OPT_DiveValue =
  { MODOPT_ARG(int), "DiveValue", &MOC_SeaBee_Movement_Controller, OPTEXP_SAVE,
    "Depth to dive",
    "depth-value", '\0', "<int>", "0" };

static const ModelOptionDef OPT_GoStraightTime =
  { MODOPT_ARG(int), "GoStraightTime", &MOC_SeaBee_Movement_Controller, OPTEXP_SAVE,
    "Time to go straight for",
    "straight-time", '\0', "<int>", "0" };

static const ModelOptionDef OPT_SpeedValue =
  { MODOPT_ARG(int), "SpeedValue", &MOC_SeaBee_Movement_Controller, OPTEXP_SAVE,
    "Value for speed",
    "speed-value", '\0', "<int>", "0" };

static const ModelOptionDef OPT_HeadingValue =
  { MODOPT_ARG(int), "HeadingValue", &MOC_SeaBee_Movement_Controller, OPTEXP_SAVE,
    "Value for heading",
    "heading-value", '\0', "<int>", "0" };

static const ModelOptionDef OPT_TimeoutValue =
  { MODOPT_ARG(int), "TimeoutValue", &MOC_SeaBee_Movement_Controller, OPTEXP_SAVE,
    "Value for timeout",
    "timeout-value", '\0', "<int>", "0" };


// ######################################################################
class MovementController : public ModelComponent
{
public:

  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  MovementController(OptionManager& mgr,
                     nub::soft_ref<SubController> subController,
                     const std::string& descrName = "MovementController",
                     const std::string& tagName = "MovementController");

  //! Destructor
  ~MovementController();

  bool dive(int depth, bool relative = false, int timeout = 0);
  bool goStraight(int speed, int time);
  bool setHeading(int heading, bool relative = false, int timeout = 0);
  int trackPipe(const Point2D<int>& pointToTrack, const Point2D<int>& desiredPoint);

  void paramChanged(ModelParamBase* const param,
                    const bool valueChanged,
                    ParamClient::ChangeStatus* status);

  //@}

  // ######################################################################
  /*! @name member functions */
  //@{

  OModelParam<int>  itsDepthErrThresh;
  OModelParam<int>  itsHeadingErrThresh;

  NModelParam<float> pipeP;
  NModelParam<float> pipeI;
  NModelParam<float> pipeD;
  PID<float> itsPipePID;

  OModelParam<int>  setDiveValue;
  OModelParam<int>  setGoStraightTimeValue;
  OModelParam<int>  setSpeedValue;
  OModelParam<int>  setHeadingValue;
  NModelParam<bool>  setRelative;
  OModelParam<int>  setTimeout;

private:

  nub::soft_ref<SubController> itsSubController;

};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
