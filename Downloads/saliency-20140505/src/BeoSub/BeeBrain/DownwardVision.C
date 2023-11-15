/*!@file BeoSub/BeeBrain/DownwardVision.C Downward Vision Agent         */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/DownwardVision.C $
// $Id: DownwardVision.C 9412 2008-03-10 23:10:15Z farhan $
//
//////////////////////////////////////////////////////////////////////////

#include "BeoSub/BeeBrain/DownwardVision.H"

// ######################################################################
DownwardVisionAgent::DownwardVisionAgent(std::string name) : SensorAgent(name)
{
//   pipeRecognizer = BeoSubPipe();
  itsCrossRecognizer.reset(new BeoSubCross());
//   binRecognizer = BeoSubPipe();

}

// ######################################################################
DownwardVisionAgent::DownwardVisionAgent
( std::string name,
  rutz::shared_ptr<AgentManagerB> amb) : SensorAgent(name)
{
  itsAgentManager = amb;
  itsCrossRecognizer.reset(new BeoSubCross());
//   pipeRecognizer = BeoSubPipe();
//   binRecognizer = BeoSubPipe();

}

// ######################################################################
//Scheduler
bool DownwardVisionAgent::pickAndExecuteAnAction()
{
  bool lookedForObject = false;

  itsCameraImage = itsAgentManager->getCurrentImage();

  if(!itsJobs.empty())
    {
      //first clean out any jobs which are to be ignored
      cleanJobs();

      for(itsJobsItr = itsJobs.begin(); itsJobsItr != itsJobs.end(); ++itsJobsItr)
        {
          rutz::shared_ptr<OceanObject> currentOceanObject = (*itsJobsItr)->oceanObject;
          Job* currentJob = *itsJobsItr;

          if(currentJob->status != IGNORE)
            {

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

              lookedForObject = true;
            }
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
void DownwardVisionAgent::lookForPipe(Job* j)
{
  if(j->status == NOT_STARTED) { j->status = IN_PROGRESS; }

  DataTypes jobDataType = j->dataType;
  rutz::shared_ptr<OceanObject> jobOceanObject = j->oceanObject;

  bool isFound = false;

//   std::vector<LineSegment2D> lines = getHoughLines(itsCameraImage, outputImage);

  if(jobDataType == POSITION)
    {
      Do("Looking for pipe center");
//       uint stalePointCount = 0;
//       Point2D<int> projPoint = pipeRecognizer->getPipeProjPoint(lines, stalePointCount);
//       if(projPoint.isValid() && stalePointCount < 10)
//         {
          isFound = true;
//           jobOceanObject->setPosition(Point3D(projPoint.i, projPoint.j, -1));
//         }
    }
  else if(jobDataType == ORIENTATION)
    {
      Do("Looking for pipe orientation");
//       uint staleAngleCount = 0;
//       Angle pipeOrientation = pipeRecognizer->getPipeDir(lines, staleAngleCount);
//       if(pipeOrientation.getVal() < 361 && staleAngleCount < 10)
//         {
          isFound = true;
//           jobOceanObject->setOrientation(pipeOrientation);
//         }
    }

  oceanObjectUpdate(jobOceanObject, jobDataType, isFound);
}

// ######################################################################
void DownwardVisionAgent::lookForBin(Job* j)
{
  if(j->status == NOT_STARTED) { j->status = IN_PROGRESS; }

  DataTypes jobDataType = j->dataType;
  rutz::shared_ptr<OceanObject> jobOceanObject = j->oceanObject;

  bool isFound = false;

//   std::vector<LineSegment2D> lines = getHoughLines(itsCameraImage, outputImage);

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

  oceanObjectUpdate(jobOceanObject, jobDataType, isFound);
}

// ######################################################################
void DownwardVisionAgent::lookForCross(Job* j)
{
  j->status = IN_PROGRESS;
  DataTypes jobDataType = j->dataType;
  rutz::shared_ptr<OceanObject> jobOceanObject = j->oceanObject;

  Image<PixRGB<byte> > outputImage;

  bool isFound = false;

  std::vector<LineSegment2D> centerPointLines;

   std::vector<LineSegment2D> lines =
     itsCrossRecognizer->getHoughLines(itsCameraImage, outputImage);

  if(jobDataType == POSITION)
    {
      Do("Looking for cross center");
      uint stalePointCount = 0;
      Point2D<int> crossCenter =
        itsCrossRecognizer->getCrossCenter(lines,
                                           centerPointLines,
                                           stalePointCount);

      if(crossCenter.isValid() && stalePointCount < 10)
        {

          Image<PixRGB<byte> > crossOverlayImage = itsCameraImage;

          PixRGB <byte> crossColor;

          if(stalePointCount <= 20)
            {
              crossColor =  PixRGB <byte> (0, 255,0);
            }
          else
            {
              crossColor =  PixRGB <byte> (255, 0 ,0);
            }

          drawCrossOR(crossOverlayImage,
                      crossCenter,
                      crossColor,
                      20,5, 0);

          isFound = true;
          itsAgentManager->drawImage(crossOverlayImage,
                                Point2D<int>(crossOverlayImage.getWidth(),0));
          jobOceanObject->setPosition(Point3D(crossCenter.i, crossCenter.j, -1));
          //jobOceanObject->setPosition(Point3D(123,23, -1));
          itsAgentManager->pushResult((CommandType)(SEARCH_OCEAN_OBJECT_CMD),
                                      jobDataType,
                                      jobOceanObject);
        }
    }
  else if(jobDataType == ORIENTATION)
    {
      Do("Looking for cross orientation");
//       uint staleAngleCount = 0;
//       Angle crossOrientation = crossRecognizer->getCrossDir(lines, staleAngleCount);
//       if(crossOrientation.getVal() < 361 && staleAngleCount < 10)
//         {
          isFound = true;
//           jobOceanObject->setOrientation(crossOrientation);
//         }
    }
  else if(jobDataType == MASS)
    {
      Do("Looking for cross mass");
//       float crossMass = pipeRecognizer->getCrossMass();
//       if(crossMass >= 0)
//         {
          isFound = true;
//           jobOceanObject->setMass(crossMass);
//         }
    }

  oceanObjectUpdate(jobOceanObject, jobDataType, isFound);
}

// ######################################################################
void DownwardVisionAgent::oceanObjectUpdate
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
