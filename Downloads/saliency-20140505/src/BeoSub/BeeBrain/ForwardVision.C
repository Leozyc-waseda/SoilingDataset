/*!@file BeoSub/BeeBrain/ForwardVision.C Forward Vision Agent         */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/ForwardVision.C $
// $Id: ForwardVision.C 9412 2008-03-10 23:10:15Z farhan $
//
//////////////////////////////////////////////////////////////////////////

#include "BeoSub/BeeBrain/ForwardVision.H"

// ######################################################################
ForwardVisionAgent::ForwardVisionAgent(std::string name) : SensorAgent(name)
{
//   pipeRecognizer = BeoSubPipe();
//   crossRecognizer = BeoSubPipe();
//   binRecognizer = BeoSubPipe();

}

// ######################################################################
//Scheduler
bool ForwardVisionAgent::pickAndExecuteAnAction()
{
  bool lookedForObject = false;

  if(!itsJobs.empty())
    {
      //first clean out any jobs which are to be ignored
      cleanJobs();

      for(itsJobsItr = itsJobs.begin(); itsJobsItr != itsJobs.end(); ++itsJobsItr)
        {
          rutz::shared_ptr<OceanObject> currentOceanObject = (*itsJobsItr)->oceanObject;
          Job* currentJob = *itsJobsItr;

          //find it based on its type
          if(currentOceanObject->getType()  == OceanObject::PIPE)
            {
              lookForPipe(currentJob);
            }
          else if(currentOceanObject->getType() == OceanObject::CROSS)
            {
              lookForCross(currentJob);
            }
          else if(currentOceanObject->getType() == OceanObject::BIN)
            {
              lookForBin(currentJob);
            }
          else if(currentOceanObject->getType() == OceanObject::BUOY)
            {
              lookForBuoy(currentJob);
            }

          lookedForObject = true;
        }

      return lookedForObject;
    }
  else
    {
      return false;
    }

}

// ######################################################################
//Actions
void ForwardVisionAgent::lookForPipe(Job* j)
{
  if(j->status == NOT_STARTED) { j->status = IN_PROGRESS; }

  DataTypes jobDataType = j->dataType;
  rutz::shared_ptr<OceanObject> jobOceanObject = j->oceanObject;

  bool isFound = false;

//   std::vector<LineSegment2D> lines = getHoughLines(cameraImage, outputImage);

  if(jobDataType == POSITION)
    {
      Do("Looking for pipe blob center");
//       uint stalePointCount = 0;
//       Point2D<int> projPoint = pipeRecognizer->getPipeProjPoint(lines, stalePointCount);
//       if(projPoint.isValid() && stalePointCount < 10)
//         {
          isFound = true;
//           jobOceanObject->setPosition(Point3D(projPoint.i, projPoint.j, -1));
//         }
    }

  oceanObjectUpdate(jobOceanObject, isFound);
}

// ######################################################################
void ForwardVisionAgent::lookForBin(Job* j)
{
  if(j->status == NOT_STARTED) { j->status = IN_PROGRESS; }

  DataTypes jobDataType = j->dataType;
  rutz::shared_ptr<OceanObject> jobOceanObject = j->oceanObject;

  bool isFound = false;

//   std::vector<LineSegment2D> lines = getHoughLines(cameraImage, outputImage);

  if(jobDataType == POSITION)
    {
      Do("Looking for bin center");
//       Point2D<int> binCenter = binRecognizer->getBinCenter(lines);
//       if(pipeCenter.isValid())
//         {
          isFound = true;
//           jobOceanObject->setPosition(Point3D(binCenter.i, binCenter.j, -1));
//         }
    }

  oceanObjectUpdate(jobOceanObject, isFound);

}

// ######################################################################
void ForwardVisionAgent::lookForCross(Job* j)
{
  j->status = IN_PROGRESS;
  DataTypes jobDataType = j->dataType;
  rutz::shared_ptr<OceanObject> jobOceanObject = j->oceanObject;

  bool isFound = false;

//   std::vector<LineSegment2D> lines = getHoughLines(cameraImage, outputImage);

  if(jobDataType == POSITION)
    {
      Do("Looking for cross blob center");
 //      uint stalePointCount = 0;
//       Point2D<int> crossCenter = crossRecognizer->getCrossCenter(lines,stalePointCount);
//       if(pipeCenter.isValid() && stalePointCount < 10)
//         {
          isFound = true;
//           jobOceanObject->setPosition(Point3D(projPoint.i, projPoint.j, -1));
//         }
    }

  oceanObjectUpdate(jobOceanObject, isFound);
}

// ######################################################################
void ForwardVisionAgent::lookForBuoy(Job* j)
{
  j->status = IN_PROGRESS;
  DataTypes jobDataType = j->dataType;
  rutz::shared_ptr<OceanObject> jobOceanObject = j->oceanObject;

  bool isFound = false;

//   std::vector<LineSegment2D> lines = getHoughLines(cameraImage, outputImage);

  if(jobDataType == POSITION)
    {
      Do("Looking for buoy center");
 //      uint stalePointCount = 0;
//       Point2D<int> crossCenter = crossRecognizer->getCrossCenter(lines,stalePointCount);
//       if(pipeCenter.isValid() && stalePointCount < 10)
//         {
          isFound = true;
//           jobOceanObject->setPosition(Point3D(projPoint.i, projPoint.j, -1));
//         }
    }
  else if(jobDataType == MASS)
    {
      Do("Looking for buoy mass");
//       float crossMass = pipeRecognizer->getCrossMass();
//       if(crossMass >= 0)
//         {
          isFound = true;
//           jobOceanObject->setMass(crossMass);
//         }
    }
  else if(jobDataType == FREQUENCY)
    {
      Do("Looking for buoy frequency");
//       float crossMass = pipeRecognizer->getCrossMass();
//       if(crossMass >= 0)
//         {
          isFound = true;
//           jobOceanObject->setMass(crossMass);
//         }
    }

  oceanObjectUpdate(jobOceanObject, isFound);
}

// ######################################################################
void ForwardVisionAgent::oceanObjectUpdate( rutz::shared_ptr<OceanObject> o, bool isFound)
{
  if(isFound)
    {
      if(o->getStatus() == OceanObject::NOT_FOUND)
        {
          o->setStatus(OceanObject::FOUND);
//           cortex->msgObjectUpdate();
        }
    }
  else
    {
      if(o->getStatus() == OceanObject::FOUND)
        {
          o->setStatus(OceanObject::LOST);
//           cortex->msgObjectUpdate();
        }
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

