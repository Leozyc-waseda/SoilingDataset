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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/SensorAgent.C $
// $Id: SensorAgent.C 8623 2007-07-25 17:57:51Z rjpeters $
//
//////////////////////////////////////////////////////////////////////////

#include "rutz/shared_ptr.h"
#include "Util/log.H"

#include "BeoSub/BeeBrain/SensorAgent.H"

// ######################################################################
// Constructor
SensorAgent::SensorAgent(std::string name) : Agent(name)
{ }

// ######################################################################
// Messages
void SensorAgent::msgFindAndTrackObject
(uint oceanObjectId,
 OceanObject::OceanObjectType oceanObjectType,
 DataTypes dataType)
{
  stateChanged();
  //Create a new Job
  Job* j = new Job();
  rutz::shared_ptr<OceanObject> oceanObject(new OceanObject());
  oceanObject->setId(oceanObjectId);
  oceanObject->setType(oceanObjectType);
  j->oceanObject = oceanObject;
  j->dataType = dataType;
  j->status = NOT_STARTED;
  //Add it to the queue of objects to be looked for
  itsJobs.push_back(j);
}

// ######################################################################
void SensorAgent::msgStopLookingForObject
( uint oceanObjectId,
  DataTypes dataType)
{
  stateChanged();
  // Iterate through all the jobs and find the object which is to be ignored
  for(itsJobsItr = itsJobs.begin(); itsJobsItr != itsJobs.end(); ++itsJobsItr)
  {
    rutz::shared_ptr<OceanObject> currentOceanObject = (*itsJobsItr)->oceanObject;
    DataTypes currentDataType = (*itsJobsItr)->dataType;
    Job* currentJob = *itsJobsItr;
    // When found, set status so as to ignore that job from now on
    if(currentOceanObject->getId() == oceanObjectId && currentDataType == dataType)
    {
      currentJob->status = IGNORE;
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
    Job* currentJob = *itsJobsItr;
    // If a job is marked as ignore, remove it from the job list
    if(currentJob->status == IGNORE)
      {
        LINFO("Removing Ocean Object: %d from Job list", currentJob->oceanObject->getId());
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
