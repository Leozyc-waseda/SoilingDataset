/*!@file Beobot/Landmark.H Landmark class for localization */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/Landmark.H $
// $Id: Landmark.H 15454 2013-01-31 02:17:53Z siagian $
//

#ifndef BEOBOT_LANDMARK_DEFINED
#define BEOBOT_LANDMARK_DEFINED

#include "SIFT/VisualObjectDB.H"
#include "GUI/XWinManaged.H"
#include "Beobot/KeypointTracker.H"
#include "SIFT/CameraIntrinsicParam.H"

//! a Visual object-base Landmark class

/*  Work done to take into account temporal info available
    Can be used to prune out bad keypoints
 */
class Landmark
{
public:
  // ######################################################################
  //! @name Constructor, assigment and destructor
  //@{

  //! Constructor assuming precomputed Visual object
  /*! @param obj initial Visual object of landmark
      @param objOffset the coordinate of the topleft corner of the image
      used by the visual object
      @param fNum the initial frame number of the object it came from
      @param name name of the landmark
  */
  Landmark(rutz::shared_ptr<VisualObject>& obj,
           Point2D<int> objOffset,
           uint fNum,
           const std::string& name = "");

  //! blank constructor with an optional name entry
  Landmark(const std::string& name = "");

  //! function to populate the initial landmark evidence
  void init(rutz::shared_ptr<VisualObject>& obj,
            Point2D<int> objOffset, uint fNum);

  //! function to load an existing landmark file
  bool loadFrom(const std::string& fname);

  //! function to save the current landmark
  bool saveTo(const std::string& fname);

  //! Destructor
  ~Landmark();

  //@}

  // ######################################################################
  //! @name Access functions
  //@{

  //! name getter
  inline const std::string& getName() const;

  //! name setter
  inline void setName(const std::string name);

  //! set the match window to view the matching
  inline void setMatchWin(rutz::shared_ptr<XWinManaged> matchWin);

  //! get the match window to view the matching
  inline rutz::shared_ptr<XWinManaged> getMatchWin();

  //! number of visual objects in DB
  inline const uint numObjects() const;

  //! number of visual objects in Temp
  inline const uint numTempObjects() const;

  //! latest frame number on DB list
  inline const uint getLatestVisualObjectFNum() const;

  //! latest frame number on temp list
  inline const uint getLatestTempVisualObjectFNum() const;

  //! latest frame number on either list (temp or DB)
  inline const uint getLatestFNum() const;

  //! latest offset coordinates on either list (temp or DB)
  inline const Point2D<int> getLatestOffsetCoords() const;

  //! get the latest visualObject
  inline const rutz::shared_ptr<VisualObject>&
  getLatestObject(bool& indb, bool& intdb, int& tindex) const;

  //! object getter
  inline const rutz::shared_ptr<VisualObject>&
  getObject(const uint index) const;

  //! get the frame Number of object[index]
  inline const uint getVisualObjectFNum(uint index) const;

  //! get the offset coordinates of object[index]
  inline const Point2D<int> getOffsetCoords(uint index) const;

  //! check whether object[index] is salient
  inline const bool isVisualObjectSal(uint index) const;

  //! temp object getter
  inline const rutz::shared_ptr<VisualObject>&
  getTempObject(const uint index) const;

  //! get the frame Number of temp object[index]
  inline const uint getTempVisualObjectFNum(uint index) const;

  //! get the offset coordinates of temp object[index]
  inline const Point2D<int> getTempOffsetCoords(uint index) const;

  //! get the visual object database
  inline rutz::shared_ptr<VisualObjectDB> getVisualObjectDB() const;

  //! get the bounding box rectangle for the object[index]
  Rectangle getObjectRect(uint index);

  //! does the landmark have evidence VO from this session
  inline bool haveSessionVO(std::string session);

  //! get the number of sessions it has
  inline uint getNumSession();

  //! get a session name of the passed in index
  inline std::string getSession(const uint index);

  //! get the session index range
  inline std::pair<uint,uint> getSessionIndexRange(const uint index);
  //@}

  // ######################################################################
  //! @name Compute functions
  //@{

  //! check if visual object(index) is salient
  inline const bool isSalVisualObject(uint index) const;

  //! build the landmark by adding a new salient Visual Object
  rutz::shared_ptr<VisualObjectMatch>
  build(rutz::shared_ptr<VisualObject> obj,
        Point2D<int> objOffset, uint fNum);

  //! build the landmark by adding visual object with no salient point
  //! (usually whole frame) -> will crop out the non-overlapping area
  rutz::shared_ptr<VisualObjectMatch>
  build(rutz::shared_ptr<VisualObject> obj, uint fNum);

  //! just like build except it does not add the object
  //! only return where it would have been placed (in DB or Temp DB)
  //! or if it is rejected altogether
  rutz::shared_ptr<VisualObjectMatch>
  buildCheck(rutz::shared_ptr<VisualObject> obj,
             Point2D<int> objOffset, uint fNum,
             bool &inDB, bool &inTDB, int &tIndex);

  //! corresponding non salient object input
  rutz::shared_ptr<VisualObjectMatch>
  buildCheck(rutz::shared_ptr<VisualObject> obj, uint fNum,
             bool &inDB, bool &inTDB, int &tIndex);

  //! build the landmark using salient object
  //! based on information passed by build check
  rutz::shared_ptr<VisualObjectMatch>
  build(rutz::shared_ptr<VisualObject> obj,
        Point2D<int> objOffset, uint fNum,
        bool inDB, bool inTDB, int tIndex,
        rutz::shared_ptr<VisualObjectMatch> cmatch);

  //! build the landmark using non-salient object
  //! based on information passed by build check
  rutz::shared_ptr<VisualObjectMatch>
  build(rutz::shared_ptr<VisualObject> obj, uint fNum,
        bool inDB, bool inTDB, int tIndex,
        rutz::shared_ptr<VisualObjectMatch> cmatch);

  //! get the current velocity of the landmark
  Point2D<int> getVelocity();
  // ===> USE VELOCITY LATER

  //! get the position of the landmark
  Point2D<int> getPosition();

  //! get the keypoints in the current frames that are likely to be
  //! as determined using the trackers
  std::vector<rutz::shared_ptr<Keypoint> > getActiveKeypoints();

  //! get current active tracker with the longest chain
  rutz::shared_ptr<KeypointTracker> getFittestKPtr();

  //! set the camera intrinsic param value
  inline void setCameraIntrinsicParam
  (rutz::shared_ptr<CameraIntrinsicParam> cip);

  //! move the last object on the temp DB to the store DB
  void moveLatestTempVisualObjectToDB();

  //! find number of object matches between the two landmarks
  uint numMatch(rutz::shared_ptr<Landmark> lmk,
                float maxPixDist = 10.0F, float minfsim = .75F,
                float minscore = 2.5F, uint minmatch = 4);

  //! combine the two landmarks
  void combine(rutz::shared_ptr<Landmark> lmk1,
               rutz::shared_ptr<Landmark> lmk2);

  //! append the input landmark to the DB
  void append(rutz::shared_ptr<Landmark> lmk);

  //! match the visual object to the objects in the landmark
  int match(rutz::shared_ptr<VisualObject> obj,
            rutz::shared_ptr<VisualObjectMatch> &cmatch,
            int start = -1, int end = -1,
            float maxPixDist = 10.0F,
            float minfsim = .75F,
            float minscore = 2.5F,
            uint minmatch = 4,
            float maxRotate = M_PI/4,
            float maxScale = 5.0F,
            float maxShear = 0.25F);

  //! transfer evidence from landmark2 to here
  void transferEvidence
  ( rutz::shared_ptr<Landmark> landmark2,
    bool indb2, bool intdb2, int tIndex2,
    rutz::shared_ptr<VisualObjectMatch> cmatch);

  //! match the salient features
  //! return a normalized [ 0.0 ... 1.0 ]
  float matchSalientFeatures(rutz::shared_ptr<VisualObject> object);

  //! sort the objects in the landmark using the passed in session order
  void sort(std::vector<std::string> sessionNames);

  //! reset the session related information
  void setSessionInfo();

  //@}

private:

  //! name of the landmark
  std::string itsName;

  //! TAKE OUT: for display purposes
  rutz::shared_ptr<XWinManaged> itsMatchWin;

  //! for calculating match
  rutz::shared_ptr<VisualObjectMatchAlgo> itsVoma;

  //! list of objects as evidence of its exitance
  rutz::shared_ptr<VisualObjectDB> itsVisualObjectDB;

  //! offset from the origin of the scene frame
  std::vector<Point2D<int> > itsOffsetCoords;

  //! frame number of object
  std::vector<uint> itsVisualObjectFNum;

  //! temporary storage for list of objects
  //! to limit actual number of objects stored
  rutz::shared_ptr<VisualObjectDB> itsTempVisualObjectDB;

  //! offset from the origin of the scene frame
  std::vector<Point2D<int> > itsTempOffsetCoords;

  //! frame number of temp object
  std::vector<uint> itsTempVisualObjectFNum;

  //! NOTE: not used for beobot-GSnav
  //! set of keypoints extracted from the objects
  std::vector<rutz::shared_ptr<KeypointTracker> > itsKPtrackers;

 //! camera intrinsic param calibration values
  rutz::shared_ptr<CameraIntrinsicParam> itsCip;

  //! latest frame number in both DB list
  uint itsLatestVisualObjectFNum;

  //! latest frame number in both temp list
  uint itsLatestTempVisualObjectFNum;

  //! summarizing salient features of the visual objects in the landmark
  //! first value is the average of the feature values
  //! second value is standard deviation of the feature values
  std::vector<std::pair<double,double> > itsSalientFeatures;

  //! the sessions that make up the landmark
  std::vector<std::string> itsSession;
  std::vector<std::pair<uint,uint> > itsSessionIndexRange;

  //! the range locations of the landmarkDB
  //! this is the min and max for all session
  //! if need individual, it can be accessed through itsSessionIndexRange
  //! NOTE: not available as it need session information
  //std::pair<float,float> itsLandmarkLocation;

  // ######################################################################
  //! @name Compute functions (all the inner-working functions)
  //@{

  //! match visual object a and b
  //! return all the computed statistics
  rutz::shared_ptr<VisualObjectMatch>
  match(rutz::shared_ptr<VisualObject> a,
        rutz::shared_ptr<VisualObject> b, bool &isFit,
        float maxPixDist = 10.0F,
        float minfsim = .75F,
        float minscore = 2.5F,
        uint minmatch = 4,
        float maxRotate = M_PI/4,
        float maxScale = 5.0F,
        float maxShear = 0.25F,
        bool showMatch = false);

  //! track the landmark using the new Visual Object
  rutz::shared_ptr<VisualObjectMatch>
  input(rutz::shared_ptr<VisualObject> obj,
        Point2D<int> objOffset, uint fNum);

  //! track check the landmark using the new Visual Object
  rutz::shared_ptr<VisualObjectMatch>
  inputCheck(rutz::shared_ptr<VisualObject> obj,
             Point2D<int> objOffset, uint fNum,
             bool &inDB, bool &inTDB, int &tIndex);

  // track the landmark using the new Visual Object
  // with the information passed by buildcheck
  rutz::shared_ptr<VisualObjectMatch>
  input(rutz::shared_ptr<VisualObject> obj,
        Point2D<int> objOffset, uint fNum,
        bool inDB, bool inTDB, int tIndex,
        rutz::shared_ptr<VisualObjectMatch> cmatch);

  //! crop the input image so that the parts
  //! not included by the landmark reference object
  rutz::shared_ptr<VisualObjectMatch>
  cropInputImage(rutz::shared_ptr<VisualObject> &obj, Point2D<int> &objOffset,
                 int mInd, rutz::shared_ptr<VisualObjectMatch> cmatch);

  //! track the keypoints
  void trackKeypoints(rutz::shared_ptr<VisualObjectMatch> match, int mInd);

  //! prune the keypoints
  void temporalPrune(uint index);

  //! get the index of object with this FNum
  int getFNumIndex(uint fNum);

  //! add the visual object to the VisualObjectDB
  void addObject(rutz::shared_ptr<VisualObject> obj,
                 Point2D<int> objOffset, uint fNum);

  //! add the visual object to the TempVisualObjectDB
  void tAddObject(rutz::shared_ptr<VisualObject> obj,
                  Point2D<int> objOffset, uint fNum);

  //! find a match between object and the Visual Object DB
  int findDBmatch(rutz::shared_ptr<VisualObject> obj,
                  rutz::shared_ptr<VisualObjectMatch> &cmatch,
                  uint nFrames,
                  bool isForward = false, int start = -1,
                  float maxPixDist = 10.0F, float minfsim = .75F,
                  float minscore = 2.5F, uint minmatch = 4,
                  float maxRotate = M_PI/4,
                  float maxScale = 5.0F,
                  float maxShear = 0.25F);

  //! find a match between object and the Temp Visual Object DB
  int findTempDBmatch(rutz::shared_ptr<VisualObject> obj,
                      rutz::shared_ptr<VisualObjectMatch> &cmatch,
                      uint nFrames,
                      float maxPixDist = 10.0F, float minfsim = .75F,
                      float minscore = 2.5F, uint minmatch = 4);

  //! move the Temp DB object of index
  //! to the back of the Visual Object DB
  void moveTempVisualObjectToDB(int index);

  //! delete object/fnum/objOffset and cleanly relink everything else
  void cleanDelete(bool indb, bool intdb, int tIndex);

  //! erase object at the passed in index
  inline void eraseObject(const uint index);

  //! erase temp object at the passed in index
  inline void eraseTempObject(const uint index);

  //! erase offset coordinate at the passed in index
  inline void eraseOffsetCoords(const uint index);

  //! erase temp offset coordinate at the passed in index
  inline void eraseTempOffsetCoords(const uint index);

  //! erase VO frame number at the passed in index
  inline void eraseVisualObjectFNum(const uint index);

  //! erase temp VO frame number at the passed in index
  inline void eraseTempVisualObjectFNum(const uint index);

  //! compute a summarizing landmark salient features
  void computeSalientFeatures();

  //@}
};

// ######################################################################
// Implementation for Landmark inline functions
// ######################################################################
inline const std::string& Landmark::getName() const
{ return itsName; }

inline void Landmark::setName(const std::string name)
{ itsName = name; }

inline void Landmark::setMatchWin(rutz::shared_ptr<XWinManaged> matchWin)
{ itsMatchWin = matchWin; }

inline rutz::shared_ptr<XWinManaged> Landmark::getMatchWin()
{ return itsMatchWin; }

inline const uint Landmark::numObjects() const
{ return itsVisualObjectDB->numObjects(); }

inline const uint Landmark::numTempObjects() const
{ return itsTempVisualObjectDB->numObjects(); }

inline const uint Landmark::getLatestVisualObjectFNum() const
{ return itsLatestVisualObjectFNum; }

inline const uint Landmark::getLatestTempVisualObjectFNum() const
{ return itsLatestTempVisualObjectFNum; }

inline const uint Landmark::getLatestFNum() const
{
  if(itsLatestVisualObjectFNum > itsLatestTempVisualObjectFNum)
    return itsLatestVisualObjectFNum;
  else
    return itsLatestTempVisualObjectFNum;
}

inline const rutz::shared_ptr<VisualObject>&
Landmark::getLatestObject(bool &indb, bool &intdb, int &tindex) const
{
  if(itsLatestVisualObjectFNum >= itsLatestTempVisualObjectFNum)
    {
      indb = true;
      intdb = false;
      int nObject = numObjects();
      for(int i = nObject-1; i >= 0; i--)
        {
          if(itsVisualObjectFNum[i] == itsLatestVisualObjectFNum)
            {
              tindex = i;
              return itsVisualObjectDB->getObject(i);
            }
        }
    }
  else
    {
      indb = false;
      intdb = true;
      int nTObject = numTempObjects();
      for(int i = nTObject-1; i >= 0; i--)
        {
          if(itsTempVisualObjectFNum[i] == itsLatestTempVisualObjectFNum)
            {
              tindex = i;
              return itsTempVisualObjectDB->getObject(i);
            }
        }
    }

  LFATAL("latest object not found; something is wrong");
  return itsVisualObjectDB->getObject(0);
}

//! latest offset coordinates on either list (temp or DB)
inline const Point2D<int> Landmark::getLatestOffsetCoords() const
{
  if(itsLatestVisualObjectFNum >= itsLatestTempVisualObjectFNum)
    {
      for(int i = numObjects()-1; i >= 0; i--)
        if(itsVisualObjectFNum[i] == itsLatestVisualObjectFNum)
          return itsOffsetCoords[i];
    }
  else
    {
      for(int i = numTempObjects()-1; i >= 0; i--)
        if(itsTempVisualObjectFNum[i] == itsLatestTempVisualObjectFNum)
          return itsTempOffsetCoords[i];
    }

  LFATAL("latest object not found; something is wrong");
  return Point2D<int>(-1,-1);
}

//! object getter
inline const rutz::shared_ptr<VisualObject>&
Landmark::getObject(const uint index) const
{
  ASSERT(index < numObjects());
  return itsVisualObjectDB->getObject(index);
}

//! get the frame Number of object[index]
inline const uint Landmark::getVisualObjectFNum(uint index) const
{
  ASSERT(index < itsVisualObjectFNum.size());
  return itsVisualObjectFNum[index];
}

//! get offset coordinates of object[index]
inline const Point2D<int> Landmark::getOffsetCoords(uint index) const
{
  ASSERT(index < itsOffsetCoords.size());
  return itsOffsetCoords[index];
}

//! check whether object[index] is salient
inline const bool Landmark::isSalVisualObject(uint index) const
{
  ASSERT(index < numObjects());
  return (itsVisualObjectDB->getObject(index)->getSalPoint().i != -1);
}

//! temp object getter
inline const rutz::shared_ptr<VisualObject>&
Landmark::getTempObject(const uint index) const
{
  ASSERT(index < numTempObjects());
  return itsTempVisualObjectDB->getObject(index);
}

//! get the frame Number of temp object[index]
inline const uint Landmark::getTempVisualObjectFNum(uint index) const
{
  ASSERT(index < itsTempVisualObjectFNum.size());
  return itsTempVisualObjectFNum[index];
}

//! get offset coordinates of temp object[index]
inline const Point2D<int> Landmark::getTempOffsetCoords(uint index) const
{
  ASSERT(index < itsTempOffsetCoords.size());
  return itsTempOffsetCoords[index];
}

//! return a pointer to the visual object DB
inline rutz::shared_ptr<VisualObjectDB>
Landmark::getVisualObjectDB() const
{ return itsVisualObjectDB; }

//! erase object at the passed in index
inline void Landmark::eraseObject(const uint index)
{
  ASSERT(index < numObjects());
  itsVisualObjectDB->eraseObject(index);
}

//! erase temp object at the passed in index
inline void Landmark::eraseTempObject(const uint index)
{
  ASSERT(index < numTempObjects());
  itsTempVisualObjectDB->eraseObject(index);
}

//! erase offset coordinate at the passed in index
inline void Landmark::eraseOffsetCoords(const uint index)
{
  ASSERT(index < itsOffsetCoords.size());
  std::vector<Point2D<int> >::iterator begin
    = itsOffsetCoords.begin();
  itsOffsetCoords.erase(begin + index);
}

//! erase temp offset coordinate at the passed in index
inline void Landmark::eraseTempOffsetCoords(const uint index)
{
  ASSERT(index < itsTempOffsetCoords.size());
  std::vector<Point2D<int> >::iterator begin
    = itsTempOffsetCoords.begin();
  itsTempOffsetCoords.erase(begin + index);
}

//! erase VO frame number at the passed in index
inline void Landmark::eraseVisualObjectFNum(const uint index)
{
  ASSERT(index < itsVisualObjectFNum.size());
  std::vector<uint>::iterator begin
    = itsVisualObjectFNum.begin();
  itsVisualObjectFNum.erase(begin + index);
}

//! erase temp VO frame number at the passed in index
inline void Landmark::eraseTempVisualObjectFNum(const uint index)
{
  ASSERT(index < itsTempVisualObjectFNum.size());
  std::vector<uint>::iterator begin
    = itsTempVisualObjectFNum.begin();
  itsTempVisualObjectFNum.erase(begin + index);
}

//! set the camera intrinsic parameters
inline void Landmark::setCameraIntrinsicParam
(rutz::shared_ptr<CameraIntrinsicParam> cip)
{ itsCip = cip; }

//! check whether it has the passed in session
inline bool Landmark::haveSessionVO(std::string session)
{
  for(uint i = 0; i < itsSession.size(); i++)
    if(!itsSession[i].compare(session)) return true;
  return false;
}

//! get the session index range
inline std::pair<uint,uint> Landmark::getSessionIndexRange(const uint index)
{
  ASSERT(index < itsSessionIndexRange.size());
  return itsSessionIndexRange[index];
}

//! get the number of sessions it has
inline uint Landmark::getNumSession()
{
  return itsSession.size();
}

//! get a session name of the passed in index
inline std::string Landmark::getSession(const uint index)
{
  ASSERT(index < itsSession.size());
  return itsSession[index];
}


#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
