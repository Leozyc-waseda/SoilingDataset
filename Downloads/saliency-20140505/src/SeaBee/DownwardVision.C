/*!@file SeaBee/DownwardVision.C Downward Vision Agent                  */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/DownwardVision.C $
// $Id: DownwardVision.C 10794 2009-02-08 06:21:09Z itti $
//
//////////////////////////////////////////////////////////////////////////

#include "BeoSub/IsolateColor.H"
#include "DownwardVision.H"
#include "GUI/DebugWin.H"

// ######################################################################
DownwardVisionAgent::DownwardVisionAgent(OptionManager& mgr,
                                         nub::soft_ref<AgentManager> ama,
                                         const std::string& name):
  SensorAgent(mgr, ama, name)
{
  itsPipeRecognizer.reset(new PipeRecognizer());
  itsFrameNumber = 0;
}

// ######################################################################
//Scheduler
bool DownwardVisionAgent::pickAndExecuteAnAction()
{
  //  bool lookedForObject = false;
  //itsCameraImage = itsAgentManager->getCurrentImage();

  uint fNum = itsAgentManager->getCurrentFrameNumber();

  if(1)
  {
    itsFrameNumber = fNum;
    switch(itsCurrentMission->missionName)
      {

      case Mission::FIRST_BIN:
        {
          //rutz::shared_ptr<Image<byte> > segImg(new Image<byte>(320,240, ZEROS));
          rutz::shared_ptr<Image<PixRGB<byte> > > currImg ( new Image<PixRGB<byte> > ((itsAgentManager->getCurrentDownwardImage())));

          //          SHOWIMG(*currImg);
          //          isolateOrange(*currImg,*segImg);

          lookForPipe(currImg, currImg);

          break;
        }

      case Mission::SECOND_BIN:
        {
          break;
        }

      case Mission::GET_TREASURE:
        {
          break;
        }
      default:
        {
          //        LINFO("Uknown mission: %d",itsCurrentMission->missionName);
        }

      }


    SensorResult saliency = SensorResult(SensorResult::SALIENCY);
    //  runSaliency(saliency);

    SensorResult stereo = SensorResult(SensorResult::STEREO);
    //  runStereo(stereo);
    return true;
  }

  return false;

}

// ######################################################################
//Actions
void DownwardVisionAgent::runStereo()
{
  Do("Running Stereo Vision");
}

void DownwardVisionAgent::runSaliency()
{
  Do("Running Saliency");
}

void DownwardVisionAgent::lookForPipe(rutz::shared_ptr<Image<PixRGB<byte> > > segImg,
                                      rutz::shared_ptr<Image<PixRGB<byte> > > currImg)
{
  Do("Looking for Pipe");

  //  rutz::shared_ptr<Image<PixRGB<byte> > > outputImage(new Image<PixRGB<byte> >(320,240, ZEROS));

  rutz::shared_ptr<SensorResult> pipe(new SensorResult(SensorResult::PIPE));
  pipe->setStatus(SensorResult::NOT_FOUND);

  std::vector<LineSegment2D> pipelines = itsPipeRecognizer->getPipeLocation(segImg,
                                                                            currImg,
                                                                            PipeRecognizer::HOUGH);


  int minY = -1; //minimum midpoint y coordinate found
  int followLineIndex = -1; //index of pipeline with minimum y coordinate

  //iterates through pipelines and finds the topmost one in the image
  for(uint i = 0; i < pipelines.size(); i++)
    {
      LineSegment2D pipeline = pipelines[i];

      if(pipeline.isValid())
        {
          Point2D<int> midpoint = (pipeline.point1() + pipeline.point2())/2;

          if(midpoint.j < minY || minY == -1)
            {
              minY = midpoint.j;
              followLineIndex = i;
            }
        }
    }

  //if we found a pipeline
  if(followLineIndex != -1)
    {
      LineSegment2D followLine = pipelines[followLineIndex];
      Point2D<int> midpoint = (followLine.point1() + followLine.point2())/2;
      Point2D<int> projPoint(Point2D<int>(0,0));

      projPoint.i = (int)(midpoint.i+30*cos(followLine.angle()));
      projPoint.j = (int)(midpoint.j+30*sin(followLine.angle()));

      drawLine(*currImg, midpoint, projPoint, PixRGB <byte> (255, 255,0), 3);

      Point3D pipePos (midpoint.i,-1,midpoint.j);
      Angle pipeOri = Angle(followLine.angle());

      pipe->setPosition(pipePos);
      pipe->setOrientation(pipeOri);
      pipe->setFrameNum(itsFrameNumber);


      pipe->setStatus(SensorResult::FOUND);
      itsAgentManager->updateSensorResult(pipe);
    }


  itsAgentManager->drawDownwardImage(currImg);

}

// ######################################################################
void DownwardVisionAgent::lookForBin(rutz::shared_ptr<Image<byte> > segImg)
{
  Do("Looking for Bin");

  //  itsAgentManager->pushResult(s);

//   if(j.status == NOT_STARTED) { j.status = IN_PROGRESS; }

//   DataType jobDataType = j.dataType;
//   rutz::shared_ptr<SensorResult> jobSensorResult = j.sensorResult;

//   bool isFound = false;

// //   std::vector<LineSegment2D> lines = getHoughLines(itsCameraImage, outputImage);

//   if(jobDataType == POSITION)
//     {
//       Do("Looking for bin center");
// //       Point2D<int> binCenter = binRecognizer->getBinCenter(lines);
// //       if(pipeCenter.isValid())
// //         {
//           isFound = true;
// //           jobSensorResult->setPosition(Point3D(binCenter.i, binCenter.j, -1));
// //         }
//     }

//   sensorResultUpdate(jobSensorResult, jobDataType, isFound);
}

// ######################################################################
void DownwardVisionAgent::lookForCross(rutz::shared_ptr<Image<byte> > segImg)
{
  Do("Looking for Cross");

}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
