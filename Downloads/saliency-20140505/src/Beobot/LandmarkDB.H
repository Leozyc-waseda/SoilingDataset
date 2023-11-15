/*!@file Beobot/LandmarkDB.H manages groups of landmarks, which includes
  spatial,geographical, temporal and episodic information               */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/LandmarkDB.H $
// $Id: LandmarkDB.H 15454 2013-01-31 02:17:53Z siagian $
//
//////////////////////////////////////////////////////////////////////////
/* LandmarkDB manages groups of landmarks, which includes
   spatial, geographical, temporal, and episodic information
*/
#ifndef BEOBOT_LANDMARKDB_H_DEFINED
#define BEOBOT_LANDMARKDB_H_DEFINED

#include "Beobot/Landmark.H"

//! difference in frame number before landmark is considered inactive
#define NFDIFF 20

//! to store ground truth for accuracy evaluation
struct GroundTruth
{
  GroundTruth() { };

  GroundTruth(const int inFnum, const int inSegNum, const float inLtrav) :
    fnum(inFnum), snum(inSegNum), ltrav(inLtrav) {  }

  uint fnum;  int snum;  float ltrav;
};

//! LandmarkDB class
class LandmarkDB
{
public:

  // ######################################################################
  /*! @name Constructors, assignment, and Destructors */
  //

  //! Constructor
  LandmarkDB(uint nSegment = 0);

  //! Destructor
  ~LandmarkDB();

  //! set the window to debug
  inline void setWindow(rutz::shared_ptr<XWinManaged> win);

  //! clear and resize the DB to the passed in number
  inline void resize(uint nSegment);

  //! add landmark to segment snum
  inline void addLandmark(uint snum, rutz::shared_ptr<Landmark> lmk);

  //@}

  // ######################################################################
  /*! @name access functions */
  //@{

  //! get the number of segments
  inline uint getNumSegment();

  //! get the number of landmark in segNum
  inline uint getNumSegLandmark(uint segNum);

  //! get the requested landmark
  inline rutz::shared_ptr<Landmark> getLandmark(uint segNum, uint lmkNum);

  //! get the length traveled given the object index in the database
  float getLenTrav(uint snum, uint lnum, uint index);

  //! get the location range of a landmark
  inline std::pair<float,float> getLocationRange(uint snum, uint lnum);

  //! get the session length (in frame numbers) of a session
  //uint getSessionLength(std::string sname);

  //! print and display the database
  void display();

  //@}

  // ######################################################################
  /*! @name member functions */
  //@{

  /*! build and finishBuild are 2 functions that needs to be called
    to create landmarks.
    build is called repeatedly every time new evidence comes in.
    finishBuild is called at the end to wrap up the process.
    It prunes all the ephemeral landmarks.

    Usually build and actual use are done on seperate session.

    Before use, need to call setSession function to get the landmarks
    sorted and all the session related info reset
  */

  //! build the landmarkDB by inputting the Visual Object, etc information
  void build
  ( std::vector<rutz::shared_ptr<VisualObject> > &inputVO,
    std::vector<Point2D<int> > &objOffset, uint fNum, uint currSegNum,
    rutz::shared_ptr<VisualObject> scene);

  //! wrap up the building process by pruning small landmarks
  void finishBuild(uint rframe);

  //! set the session information
  //! must call this after finished building a landmark db
  //! to sort and then set session related information
  void setSession(std::string sessionFName, bool sort = true);

  //! sort landmarks according to the session names order
  void sortLandmarks();

  //@}

private:

  //! its input full image scenes
  std::vector<rutz::shared_ptr<VisualObject> > itsScenes;

  //! its landmark array
  std::vector<std::vector<rutz::shared_ptr<Landmark> > > itsLandmarks;

  //! store the information if the landmark is currently being built
  std::vector<std::vector<bool> > itsActiveLmkList;

  //! all the visual objects skipped in building process
  std::vector<rutz::shared_ptr<VisualObject> > itsSkipList;

  //! session information
  std::vector<std::string> itsSessionNames;
  std::vector<uint> itsSessionLength;

  //! the landmark database for each sessions
  //! NOTE: this information is current as of the last setSessionInfo
  std::vector<std::vector<std::vector<rutz::shared_ptr<Landmark> > > >
  itsSessionLandmarks;

  std::vector<std::vector<std::pair<float,float> > >
  itsLandmarkLocationRange;

  //! store the ground truth location for each frame in each session
  std::vector<std::vector<GroundTruth> > itsSessionGroundTruth;
  std::vector<float> itsSessionGroundTruthSegmentLength;

  //! display window to debug
  rutz::shared_ptr<XWinManaged> itsWin;

  // ######################################################################
  //! @name Compute functions (all the inner-working functions)
  //@{

  //! filter out visual objects that have 5 keypoints or less
  void kpFilter
  ( std::vector<rutz::shared_ptr<VisualObject> > &inputVO,
    std::vector<Point2D<int> > &objOffset);

  //! get the offset distance score
  float getOffsetDistScore
  ( rutz::shared_ptr<Landmark> landmark, int indb, int intdb, int tindex,
    rutz::shared_ptr<VisualObject> vo, Point2D<int> offset, Dims sDims,
    rutz::shared_ptr<VisualObjectMatch> cmatch);

  //! get the overlap score
  float getOverlapScore(rutz::shared_ptr<VisualObjectMatch> cmatch);

  //! print scores used in processing input
  void printScores
  ( std::vector<rutz::shared_ptr<VisualObject> > inputVO,
    int currSegNum, std::vector<std::vector<float> > inscores);

  //! calculate the best/2ndbest match ratio between landmarks
  //! find the best inputVO to Landmark match using the passed in ratio
  void findBestMatch
  (std::vector<std::vector<float> > scores, std::vector<float> &ratio,
   std::vector<bool> &inserted, std::vector<bool> &lmatched,
   float &mratio, int &mind, std::vector<int> &mlist);

  //! classify which landmark is no longer active
  //! once a landmark is inactive, it can be re-activate
  void classifyInactiveLandmarks(uint fNum, uint nfDiff, bool print = false);

  //! prune landmarks that have less that minimal number of objects
  void pruneLandmarks();

  //! display all the images that have small number of keypoints
  void printSkiplist();

  //! set the landmark range for each session
  void setSessionInfo();

  //! create a session based landmark access
  void setSessionLandmarks();

  //! set the location range for each landmark
  void setLocationRange();

  //! extract and store ground truth from inputted ground truth file
  std::vector<GroundTruth> getGroundTruth
  (std::string gt_filename, uint segnum, float& seg_length);

  //@}

};

// ######################################################################
// Implementation for LandmarkDB inline functions
// ######################################################################

inline void LandmarkDB::setWindow(rutz::shared_ptr<XWinManaged> win)
{
  itsWin = win;

  // set window for each landmark as well
  for(uint i = 0; i < itsLandmarks.size(); i++)
    for(uint j = 0; j < itsLandmarks[i].size(); j++)
    itsLandmarks[i][j]->setMatchWin(win);
}

inline void LandmarkDB::resize(uint nSegment)
{
  itsLandmarks.clear();
  itsLandmarks.resize(nSegment);
  itsActiveLmkList.clear();
  itsActiveLmkList.resize(nSegment);
  itsSkipList.clear();
}

inline void LandmarkDB::addLandmark(uint snum, rutz::shared_ptr<Landmark> lmk)
{
  ASSERT(snum < itsLandmarks.size());
  itsLandmarks[snum].push_back(lmk);
  itsActiveLmkList[snum].push_back(true);
}

inline uint LandmarkDB::getNumSegment()
{
  return itsLandmarks.size();
}

inline uint LandmarkDB::getNumSegLandmark(uint segNum)
{
  ASSERT(segNum < itsLandmarks.size());
  return itsLandmarks[segNum].size();
}

inline rutz::shared_ptr<Landmark>
LandmarkDB::getLandmark(uint segNum, uint lmkNum)
{
  ASSERT(segNum < itsLandmarks.size() &&
         lmkNum < itsLandmarks[segNum].size());
  return itsLandmarks[segNum][lmkNum];
}

inline std::pair<float,float>
LandmarkDB::getLocationRange(uint snum, uint lnum)
{
  ASSERT(snum < itsLandmarks.size());
  ASSERT(lnum < itsLandmarks[snum].size());

  return itsLandmarkLocationRange[snum][lnum];
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
