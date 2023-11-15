/*!@file Beobot/KeypointTracker.H a list of keypoints */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/KeypointTracker.H $
// $Id: KeypointTracker.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef KEYPOINTTRACKER_DEFINED
#define KEYPOINTTRACKER_DEFINED

#include "Image/Point2D.H"
#include "SIFT/Keypoint.H"
#include <vector>

//! Landmark class to take into account temporal info
class KeypointTracker
{
public:

  //! Constructor
  KeypointTracker(const std::string& name = "");

  //! Destructor
  ~KeypointTracker();

  //! add a keypoint to the list
  void add(rutz::shared_ptr<Keypoint> kp, Point2D<int> offset, uint fNum);

  //! print the current list
  void print();

  //! get the last keypoints
  const rutz::shared_ptr<Keypoint>& getLastKeypoint() const;

  //! do we have a keypoint from frame index
  const bool hasKeypointInFrame(uint index) const;

  //! get the keypoint from frame index
  const rutz::shared_ptr<Keypoint>& getKeypointInFrame(uint index) const;

  //! check to see if the tracker has been inactive since
  //! frame fNum
  bool isInactiveSince(uint fNum);

  //! get the top left corner location + the KP location
  Point2D<int> getAbsLoc();

  // ######################################################################
  /*! @name Iterators and access functions
  The iterators iterate on our feature vector data only, while the
   other access functions can be used to access the other data. */
  //@{

  //! name getter
  inline const std::string& getName() const;

  //! get all SIFT keypoints
  inline const std::vector< rutz::shared_ptr<Keypoint> >& getKeypoints() const;

  //! get number of SIFT keypoints
  inline uint numKeypoints() const;

  //! standard iterator
  typedef std::vector<rutz::shared_ptr<Keypoint> >::iterator iterator;

  //! const iterator
  typedef std::vector<rutz::shared_ptr<Keypoint> >::const_iterator const_iterator;

  //! Returns a read-only iterator to the beginning of the keypoint list
  inline const_iterator begin() const;

  //! Returns a read-only iterator to one-past-the-end of the keypoint list
  inline const_iterator end() const;

   //! Returns a read-write iterator to the beginning of the keypoint list
  inline iterator beginw();

   //! Returns a read-write iterator to one-past-the-end of the keypoint list
  inline iterator endw();

private:
  std::string itsName;                             // name of this keypoint tracker
  std::vector<rutz::shared_ptr<Keypoint> > itsKeypoints;
  std::vector<Point2D<int> > itsOffsets;
  std::vector<uint> itsFrameNums;
};

// ######################################################################
// Inlined member functions
// ######################################################################
inline const std::string& KeypointTracker::getName() const
{ return itsName; }

inline const std::vector< rutz::shared_ptr<Keypoint> >&
KeypointTracker::getKeypoints() const
{ return itsKeypoints; }

inline uint KeypointTracker::numKeypoints() const
{ return itsKeypoints.size(); }

inline KeypointTracker::const_iterator KeypointTracker::begin() const
{ return itsKeypoints.begin(); }

inline KeypointTracker::const_iterator KeypointTracker::end() const
{ return itsKeypoints.end(); }

inline KeypointTracker::iterator KeypointTracker::beginw()
{ return itsKeypoints.begin(); }

inline KeypointTracker::iterator KeypointTracker::endw()
{ return itsKeypoints.end(); }

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
