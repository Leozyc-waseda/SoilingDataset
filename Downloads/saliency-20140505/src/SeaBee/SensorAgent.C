/*!@file BeoSub/BeeBrain/SensorAgent.C Sensor Agent superclass         */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/SensorAgent.C $
// $Id: SensorAgent.C 10794 2009-02-08 06:21:09Z itti $
//
//////////////////////////////////////////////////////////////////////////

#include "rutz/shared_ptr.h"
#include "Util/log.H"

#include "SensorAgent.H"

// ######################################################################
// Constructor
SensorAgent::SensorAgent(rutz::shared_ptr<AgentManager> ama,
                         const std::string& name)
{ }

SensorAgent::~SensorAgent()
{ }

// ######################################################################
// Messages

void SensorAgent::msgSensorUpdate()
{
  //stateChanged();
}

void SensorAgent::msgFindAndTrackObject
(uint sensorResultId,
 SensorResult::SensorResultType sensorResultType,
 DataType dataType)
{
  //stateChanged();
  //Create a new Job
  Job j = Job();
  rutz::shared_ptr<SensorResult> sensorResult(new SensorResult());
  sensorResult->setId(sensorResultId);
  sensorResult->setType(sensorResultType);
  j.sensorResult = sensorResult;
  j.dataType = dataType;
  j.status = NOT_STARTED;
  //Add it to the queue of objects to be looked for
  itsJobs.push_back(j);
}

// ######################################################################
void SensorAgent::msgStopLookingForObject
( uint sensorResultId,
  DataType dataType)
{
  //stateChanged();
  // Iterate through all the jobs and find the object which is to be ignored
  for(itsJobsItr = itsJobs.begin(); itsJobsItr != itsJobs.end(); ++itsJobsItr)
  {
    rutz::shared_ptr<SensorResult> currentSensorResult = (*itsJobsItr).sensorResult;
    DataType currentDataType = (*itsJobsItr).dataType;
    Job currentJob = *itsJobsItr;
    // When found, set status so as to ignore that job from now on
    if(currentSensorResult->getId() == sensorResultId && currentDataType == dataType)
    {
      currentJob.status = IGNORE;
    }
  }
}

// ######################################################################
// Dummy scheduler. Descendent agents implement something meaningful here
// bool SensorAgent::pickAndExecuteAnAction()
// {
//   return false;
// }

// ######################################################################
// Actions
void SensorAgent::cleanJobs()
{
  itsJobsItr = itsJobs.begin();

  while(itsJobsItr != itsJobs.end())
  {
    Job currentJob = *itsJobsItr;
    // If a job is marked as ignore, remove it from the job list
    if(currentJob.status == IGNORE)
      {
        LINFO("Removing Sensor Result: %d from Job list", currentJob.sensorResult->getId());
        itsJobsItr = itsJobs.erase(itsJobsItr);
      }
    else
      {
        itsJobsItr++;
      }
  }
}

// Misc

// void SensorAgent::setPreFrontalCortex(PreFrontalCortex* c)
// {
//   itsCortex = c;
// }

// ######################################################################
uint SensorAgent::getNumJobs()
{
  return itsJobs.size();
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
