/*!@file SeaBee/ForwardVision.C Forward Vision Agent         */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/ForwardVision.C $
// $Id: ForwardVision.C 10794 2009-02-08 06:21:09Z itti $
//
//////////////////////////////////////////////////////////////////////////

#include "ForwardVision.H"

// ######################################################################
ForwardVisionAgent::ForwardVisionAgent(
                                       //    rutz::shared_ptr<AgentManager> ama,
                nub::soft_ref<EnvVisualCortex> evc,
                const std::string& name) :
        itsEVC(evc)
{
  itsDebugImage = Image<PixRGB<byte> >(320,240,ZEROS);
  itsFrameNumber = 0;
}

// ######################################################################
//Scheduler
bool ForwardVisionAgent::pickAndExecuteAnAction()
{
  //
  //if(itsCurrentMission->missionName != Mission::NONE)
  //  {
  //    rutz::shared_ptr<Image<PixRGB<byte> > >
  //      currImg( new Image<PixRGB<byte> >(itsAgentManager->getCurrentForwardImage()));
  //
  //    if(currImg->initialized())
  //      {
  //        lookForSaliency(currImg);
  //      }
  //    else
  //      {
  //        usleep(10000);
  //      }
  //
  //  }
  return true;

}

// ######################################################################
//Actions
Point2D<int> ForwardVisionAgent::lookForBuoy(const Image<PixRGB<byte> >& img)
{
  LINFO("Look for buoy");

  itsDebugImage = img;
  int smap_level = itsEVC->getMapLevel();

  //look for the most salient point and go toward it
  itsEVC->input(img);
  Image<float> vcxmap = itsEVC->getVCXmap();

  Point2D<int> maxPos(-1,-1); float maxVal = -1;
  findMax(vcxmap, maxPos, maxVal);

  Point2D<int> buoyLoc (maxPos.i<<smap_level,
                        maxPos.j<<smap_level);
  if (img.getVal(buoyLoc).red > img.getVal(buoyLoc).green && img.getVal(buoyLoc).red > img.getVal(buoyLoc).blue)
    {
      if(buoyLoc.isValid())
        {
          drawCircle(itsDebugImage,
                     buoyLoc,
                     10, PixRGB<byte>(255,0,0));
        }

    }

  LINFO("%d %d",buoyLoc.i,buoyLoc.j);
  return buoyLoc;

}

void ForwardVisionAgent::lookForSaliency(rutz::shared_ptr<Image<PixRGB<byte> > > img)
{
//   LINFO("Looking for salient features...");

//   rutz::shared_ptr<SensorResult> saliency(new SensorResult(SensorResult::SALIENCY));
//   saliency->setStatus(SensorResult::NOT_FOUND);

//   int smap_level = itsEVC->getMapLevel();

//   //look for the most salient point and go toward it
//   itsEVC->input(*img);
//   Image<float> vcxmap = itsEVC->getVCXmap();

//   Point2D<int> maxPos(-1,-1); float maxVal = -1;
//   findMax(vcxmap, maxPos, maxVal);

//   if (maxVal > 150)
//     {
//       drawCircle(*img,
//                  Point2D<int>(maxPos.i<<smap_level,
//                               maxPos.j<<smap_level),
//                  10, PixRGB<byte>(255,0,0));
//       saliency->setStatus(SensorResult::FOUND);

//       itsAgentManager->updateSensorResult(saliency);
//     }

//   itsAgentManager->drawForwardImage(img);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

