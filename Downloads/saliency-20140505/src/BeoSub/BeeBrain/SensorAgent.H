/*!@file BeoSub/BeeBrain/SensorAgent.H Sensor Agent superclass        */
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/SensorAgent.H $
// $Id: SensorAgent.H 8623 2007-07-25 17:57:51Z rjpeters $
//
//////////////////////////////////////////////////////////////////////////

#ifndef __SENSOR_AGENT_H__
#define __SENSOR_AGENT_H__

#include "BeoSub/BeeBrain/Globals.H"

#include "BeoSub/BeeBrain/Agent.H"
#include "BeoSub/BeeBrain/OceanObject.H"
//#include "BeoSub/BeeBrain/PreFrontalCortex.H"

#include "rutz/shared_ptr.h"

#include <list>
#include <cstdlib>
#include <string>

class SensorAgent : public Agent
{

public:

  SensorAgent(std::string name);

  //!
  void msgFindAndTrackObject
  (uint oceanObjectId,
   OceanObject::OceanObjectType oceanObjectType,
   DataTypes dataType);

  //!
  void msgStopLookingForObject
  (uint oceanObjectId,
   DataTypes dataType);

  //   virtual bool pickAndExecuteAnAction();

  //!
  uint getNumJobs();

  //   void setPreFrontalCortex(PreFrontalCortex* c);

  //!
  void cleanJobs();

protected:

  enum JobStatus { NOT_STARTED, IN_PROGRESS, IGNORE };

  struct Job
  {
    rutz::shared_ptr<OceanObject> oceanObject;
    JobStatus status;
    DataTypes dataType;
  };

  std::list<Job*> itsJobs;
  std::list<Job*>::iterator itsJobsItr;

  //   PreFrontalCortex* cortex;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
