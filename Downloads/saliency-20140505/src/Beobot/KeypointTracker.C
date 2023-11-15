/*!@file Beobot/KeypointTracker.C a list of keypoints */

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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/KeypointTracker.C $
// $Id: KeypointTracker.C 12074 2009-11-24 07:51:51Z itti $
//

#include "Beobot/KeypointTracker.H"

#include <cstdio>

// ######################################################################
KeypointTracker::KeypointTracker(const std::string& name):
  itsName(name),
  itsKeypoints(),
  itsOffsets(),
  itsFrameNums()
{
}

// ######################################################################
KeypointTracker::~KeypointTracker()
{ }

// ######################################################################
void KeypointTracker::add(rutz::shared_ptr<Keypoint> kp, Point2D<int> offset,
                          uint fNum)
{
  itsKeypoints.push_back(kp);
  itsOffsets.push_back(offset);
  itsFrameNums.push_back(fNum);
}

// ######################################################################
const rutz::shared_ptr<Keypoint>& KeypointTracker::getLastKeypoint() const
{
  return itsKeypoints[itsKeypoints.size()-1];
}

// ######################################################################
const rutz::shared_ptr<Keypoint>& KeypointTracker::getKeypointInFrame(uint index) const
{
  ASSERT(hasKeypointInFrame(index) && (itsKeypoints.size() > 0));

  for(uint i = 0; i < itsFrameNums.size(); i++)
    {
      if(itsFrameNums[i] == index)
        return itsKeypoints[i];
    }
  return itsKeypoints[0];
}

// ######################################################################
const bool KeypointTracker::hasKeypointInFrame(uint index) const
{
  for(uint i = 0; i < itsFrameNums.size(); i++)
    {
      if(itsFrameNums[i] == index)
        return true;
    }
  return false;
}

// ######################################################################
bool KeypointTracker::isInactiveSince(uint fNum)
{
  // if the last frames
  return (itsFrameNums[itsFrameNums.size() - 1] < fNum);
}

// ######################################################################
Point2D<int> KeypointTracker::getAbsLoc()
{
  Point2D<int> offset = itsOffsets[itsFrameNums.size() - 1];
  rutz::shared_ptr<Keypoint>  kp =  itsKeypoints[itsFrameNums.size() - 1];
  const float x = kp->getX();
  const float y = kp->getY();

  return Point2D<int>(offset.i + int(x + 0.5F), offset.j + int(y + 0.5F));
}

// ######################################################################
void KeypointTracker::print()
{
  printf("%s: ",itsName.c_str());
  for(uint i = 0; i < itsFrameNums.size(); i++)
    printf("%3d -",itsFrameNums[i]);
  printf("\n");
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

