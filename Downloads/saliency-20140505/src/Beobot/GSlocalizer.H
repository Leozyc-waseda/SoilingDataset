/*!@file Beobot/GSlocalizer.H takes in salient object and  gist vector
  to localize in the map it has. It also takes in command to go to a
  target location                                                       */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/GSlocalizer.H $
// $Id: GSlocalizer.H 15454 2013-01-31 02:17:53Z siagian $
//

// ######################################################################


#ifndef BEOBOT_GSLOCALIZER_H_DEFINED
#define BEOBOT_GSLOCALIZER_H_DEFINED

#define NUM_GSL_THREAD      8   //16
#define N_OBJECT_BLOCK      40

#include "Beowulf/Beowulf.H"
#include "Beobot/beobot-GSnav-def.H"

#include "Component/ModelComponent.H"
#include "Beobot/Landmark.H"
#include "SIFT/Histogram.H"
#include "Beobot/LocParticle.H"
#include "Beobot/Environment.H"
#include "Util/Timer.H"

#include <list>
#include <pthread.h>


// ######################################################################

struct GSlocJobData
{
  GSlocJobData() { };

  GSlocJobData(const int inObjNum,
               const int inSegNum,
               const int inLmkNum,
               const int inVOstartNum,
               const int inVOendNum) :
    objNum(inObjNum),
    segNum(inSegNum),
    lmkNum(inLmkNum),
    voStartNum(inVOstartNum),
    voEndNum(inVOendNum)
  {
    pVal = 0.0;
    segVal = 0.0;
    salVal = 0.0;
    locVal = 0.0;
  }

  int objNum;
  int segNum;
  int lmkNum;
  int voStartNum;
  int voEndNum;

  //! priority value of the job
  float pVal;
  float segVal;
  float salVal;
  float locVal;

  bool operator < (const GSlocJobData& rhs)
  {
    return pVal < rhs.pVal;
  }

};


// ######################################################################
//! Thread with localizer, Object database, search priority list
//! takes in salient object and gist vector as input
class GSlocalizer : public ModelComponent
{
public:

  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  GSlocalizer(OptionManager& mgr,
              const std::string& descrName = "Gist and Saliency localizer",
              const std::string& tagName = "GSlocalizer");

  //! Destructor
  virtual ~GSlocalizer();

  //! set the prefix of file to save data  - has to be done
  void setSavePrefix(std::string prefix);

  //! set the environment - has to be done
  void setEnvironment(rutz::shared_ptr<Environment> env);

  //@}

  // ######################################################################
  //! @name Access functions
  //@{

  //! get number of objects compared in the search
  uint getNumObjectSearch(uint index);

  //! get the environment information
  rutz::shared_ptr<Environment> getEnvironment();

  //! set the window to display results
  void setWindow(rutz::shared_ptr<XWinManaged> inputWin);

  //! set beowulf access
  void setBeoWulf(nub::soft_ref<Beowulf> beo);

  //! get the input image
  Image<PixRGB<byte> > getInputImage();

  //! get the number of objects inputted
  uint getNumInputObject();

  //! get the visual object that we try to match
  rutz::shared_ptr<VisualObject> getInputVO(uint index);

  //! get the input gist
  Image<double> getInputGist();

  //! get the visual object match for the found object
  rutz::shared_ptr<VisualObjectMatch> getVOmatch(uint index);

  //! get the rectangle of database visual object match for the found object
  Rectangle getInputVORect(uint index);

  //! get the visual object match for the found object
  Rectangle getDBmatchVORect(uint index);

  //! get the segment number of the object match found
  uint getSegmentNumberMatch(uint index);

  //! get the length traveled of the object match found
  float getLengthTraveledMatch(uint index);

  //! get the object offset of the visual object
  //! that we try to match
  Point2D<int> getInputObjOffset(uint index);

  //! get the last input frame number
  int getInputFnum();

  //! get the last input frame number where search is started
  int getSearchInputFnum();

  //! get the segment histogram from the segment classifier
  rutz::shared_ptr<Histogram> getSegmentHistogram();

  //! get our geographical location
  Point2D<int> getLocation();

  //! get our segment location
  uint getSegmentLocation();

  //! get the length traveled within the segment
  float getSegmentLengthTraveled();

  //! set ground truth
  void setGroundTruth(uint snum, float ltrav);

  //! get ground truth
  void getGroundTruth(uint &snum, float &ltrav);

  //@}

  // ######################################################################
  /*! @name member functions */
  //@{

  //! initialize the localization particles
  void initParticles(std::string belFName = std::string(""));

  //! get the belief particles (usually for recovering crashes)
  std::vector<LocParticle> getBeliefParticles();

  //! check if the serach is finished
  bool outputReady();

  //! return the result of the matching search
  bool isMatchFound(uint index);

  //! input the image, visual object and gist feature for search
  //! also add the odometry change
  void input(Image<PixRGB<byte> > ima,
             std::vector<rutz::shared_ptr<VisualObject> > inputVO,
             std::vector<Point2D<int> > inputObjOffset,
             int inputFnum, Image<double> cgist,
             float dx = -1.0F, float dy = -1.0F);

  //! the belief histogram for segment only localization
  rutz::shared_ptr<Histogram> getSegmentBeliefHistogram();

  //! For internal thread use: Compute a conspicuity map from an image
  void threadCompute();

  //! stop search by cleaning up the queue
  //! NOTE: this is a hard stop (blocking operation)
  //!       may take time (500ms, or even longer)
  void stopSearch();

  //! stop search by flipping a stop-search bit
  //! NOTE: this is a soft/non-blocking operation
  void stopSearch2();

  //! update belief using the input just processed
  //! update our likely location
  void updateBelief();

  //! move the object from the previous location
  void actionUpdateBelief();

  //! update belief using the segment prediction
  void segmentUpdateBelief();

  //! update belief using all the objects found
  void objectUpdateBelief();

  //! update belief using object 'index'
  void objectUpdateBelief(uint index);

  //! set the most likely location
  void setLocation();

  //! get the belief image (it is put on top of a map)
  Image<PixRGB<byte> > getBeliefImage(uint w, uint h, float &scale);

  //! get the match image
  Image<PixRGB<byte> > getMatchImage(uint index, Dims d);

  //! get motor signal
  /*! motor signal can be used (using PID, for example) to obtain
    motor command. Motor signal is a delta signal (in image coordinate)
    of where the robot should go to get to the goal state.
   */
  Point2D<int> getMotorSignal();

  //@}

protected:
  void start1(); //!< get started
  void stop2();  //!< get stopped

private:

  // ######################################################################
  /*! @name private functions */
  //@{

  //! set the segment and object search priority for landmark DB
  void setSearchPriority();

  //! add the search order preference randomly
  void addRandomPriority();

  //! add the search order preference based on segment
  void addSegmentPriority();

  //! add the search order preference based on saliency match
  void addSaliencyPriority();

  //! add the search order preference based on current belief location
  void addLocationPriority();

  //! get the match
  GSlocJobData getMatch(uint index);

  //@}

  //!  file prefix to save data
  std::string itsSavePrefix;

  //! localization particles for beliefs
  std::vector<LocParticle> itsBeliefParticles;
  std::vector<Point2D<int> > itsBeliefLocations;

  //! all the environment information
  rutz::shared_ptr<Environment> itsEnvironment;

  //! from its environment: topological map
  rutz::shared_ptr<TopologicalMap> itsTopologicalMap;

  //! from its environment: visual landmark database
  rutz::shared_ptr<LandmarkDB> itsLandmarkDB;

  //! the input image, visual objects, and gist
  Image<PixRGB<byte> > itsInputImage;
  std::vector<rutz::shared_ptr<VisualObject> > itsInputVO;
  std::vector<bool> itsVOKeypointsComputed;
  std::vector<Point2D<int> > itsInputObjOffset;
  Image<double> itsInputGist;
  int itsInputFnum;
  int itsSearchInputFnum;

  //! ground truth information - default to (0,0.0)
  uint  itsSnumGT;
  float itsLtravGT;

  //! the current robot movement
  float itsRobotDx;
  float itsRobotDy;

  //! segment histogram from the classifier
  rutz::shared_ptr<Histogram> itsSegmentHistogram;

  //! result of search
  std::vector<rutz::shared_ptr<VisualObjectMatch> > itsVOmatch;
  std::vector<Rectangle> itsInputVORect;
  std::vector<Rectangle> itsDBmatchVORect;
  std::vector<GSlocJobData> itsLmkMatch;
  std::vector<uint> itsSegNumMatch;
  std::vector<float> itsLenTravMatch;
  std::vector<bool> itsMatchFound;
  std::vector<uint> itsNumObjectSearch;

  //! resulting geographic location
  uint itsSegmentLocation;
  float itsSegmentLengthTraveled;
  Point2D<int> itsLocation;

  //! job queue and number of jobs to do
  //! Note: they are on jobLock
  std::list<GSlocJobData> itsJobQueue;
  bool itsIsQueueSorted;         //!< note if the queue is sorted
  uint itsNumJobsProcessed;      //!< number of jobs that has been processed
  uint itsLastSuccessfulJob;     //!< job index last found
  uint itsNumObjectFound;        //!< number of objects found
  uint itsNumJobs;               //!< original number of jobs
  bool itsStopSearch;            //!< stop search request

  uint itsNumWorking;            //!< the number of threads that are working

  //! master node to send to
  nub::soft_ref<Beowulf> itsBeowulf;

  //! segment histogram from the belief particles
  rutz::shared_ptr<Histogram> itsSegmentBeliefHistogram;

  //! especially for input
  bool itsOutputReady2;

  //! thread stuff
  pthread_t *worker;
  pthread_mutex_t jobLock;       //!< locking jobQueue
  pthread_mutex_t fnumLock;      //!< locking frame number
  pthread_mutex_t or2Lock;       //!< locking itsOutputReady2
  pthread_mutex_t stopSearchLock;//!< locking stop search
  pthread_mutex_t resLock;       //!< locking results
  pthread_mutex_t workLock;      //!< locking number of working threads
  pthread_mutex_t particleLock;  //!< locking belief particles
  pthread_cond_t jobCond;
  uint numWorkers;

  //! debug window and timer
  rutz::shared_ptr<XWinManaged> itsWin;
  rutz::shared_ptr<Timer> itsTimer;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
