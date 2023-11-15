/*!@file BeoSub/BeeBrain/SonarListen.C Sonar Listen Agent         */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/SonarListen.C $
// $Id: SonarListen.C 8623 2007-07-25 17:57:51Z rjpeters $
//
//////////////////////////////////////////////////////////////////////////

#include "BeoSub/BeeBrain/SonarListen.H"

// ######################################################################
SonarListenAgent::SonarListenAgent(std::string name) : SensorAgent(name) { }

// ######################################################################
SonarListenAgent::SonarListenAgent
( std::string name,
  rutz::shared_ptr<AgentManagerB> amb) : SensorAgent(name)
{
  itsAgentManager = amb;
}

// ######################################################################
// Scheduler
bool SonarListenAgent::pickAndExecuteAnAction()
{
  bool listenedForObject = false;

  if(!itsJobs.empty())
    {
      //first clean out any jobs which are to be ignored
      cleanJobs();

      for(itsJobsItr = itsJobs.begin(); itsJobsItr != itsJobs.end(); ++itsJobsItr)
        {
          rutz::shared_ptr<OceanObject> currentOceanObject = (*itsJobsItr)->oceanObject;
          Job* currentJob = *itsJobsItr;

          //find it based on its type
          if(currentOceanObject->getType()  == OceanObject::PINGER)
            {
              listenForPinger(currentJob);
            }

          listenedForObject = true;
        }

      return listenedForObject;
    }
  else
    {
      return false;
    }

}

// ######################################################################
// Actions
void SonarListenAgent::listenForPinger(Job* j)
{
  if(j->status == NOT_STARTED) { j->status = IN_PROGRESS; }

  DataTypes jobDataType = j->dataType;
  rutz::shared_ptr<OceanObject> jobOceanObject = j->oceanObject;

  bool isFound = false;

  if(jobDataType == ORIENTATION)
    {
      Do("Listening for pinger heading");

      isFound = true;
    }
  else if(jobDataType == DISTANCE)
    {
      Do("Listening for approximate pinger distance");

      isFound = true;
    }

  oceanObjectUpdate(jobOceanObject, jobDataType, isFound);
}

// ######################################################################
void SonarListenAgent::oceanObjectUpdate
( rutz::shared_ptr<OceanObject> o,
  DataTypes dataType,
  bool isFound)
{
  if(isFound)
    {
      if(o->getStatus() == OceanObject::NOT_FOUND)
        {
          o->setStatus(OceanObject::FOUND);
          itsAgentManager->pushResult((CommandType)(OCEAN_OBJECT_STATUS), dataType, o);
        }
    }
  else
    {
      if(o->getStatus() == OceanObject::FOUND)
        {
          o->setStatus(OceanObject::LOST);
          itsAgentManager->pushResult((CommandType)(OCEAN_OBJECT_STATUS), dataType, o);
        }
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
