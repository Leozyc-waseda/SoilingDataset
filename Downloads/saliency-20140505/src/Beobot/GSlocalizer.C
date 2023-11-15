/*!@file Beobot/GSlocalizer.C takes in salient object and gist vector
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/GSlocalizer.C $
// $Id: GSlocalizer.C 15454 2013-01-31 02:17:53Z siagian $
//

// ######################################################################

#include "Beobot/GSlocalizer.H"
#include "Image/CutPaste.H"       // for inplacePaste()
#include "Image/MatrixOps.H"      // for transpose()
#include "Image/DrawOps.H"        // for drawing

// number of particles used
#define NUM_PARTICLES          100

// maximum allowable localization error (in unit map)
#define MAX_LOC_ERROR          5.0

// standard deviation for odometry error (in feet)
#define STD_ODO_ERROR          0.02

// standard deviation for length traveled error (in ltrav [0.0 ... 1.0])
#define STD_LTRAV_ERROR        0.02

#define GIST_PRIORITY_WEIGHT   0.5
#define SAL_PRIORITY_WEIGHT    0.2
#define LOCN_PRIORITY_WEIGHT   0.3

// ######################################################################
void *GSlocalizer_threadCompute(void *gsl)
{
  GSlocalizer *gsl2 = (GSlocalizer *)gsl;
  gsl2->threadCompute();
  return NULL;
}

// ######################################################################
GSlocalizer::GSlocalizer(OptionManager& mgr,
                         const std::string& descrName,
                         const std::string& tagName)
  :
  ModelComponent(mgr, descrName, tagName),
  itsSegmentBeliefHistogram(new Histogram())
{
  // default start and ground truth location:
  // the two formats are not the same
  itsSegmentLocation       = 0;
  itsSegmentLengthTraveled = 0.0F;
  itsLocation              = Point2D<int>(-1,-1);
  itsSnumGT                = 0;
  itsLtravGT               = 0.0;

  itsIsQueueSorted = false;

  itsOutputReady2 = true;

  itsStopSearch = false;

  itsTimer.reset(new Timer(1000000));
}

// ######################################################################
void GSlocalizer::setEnvironment(rutz::shared_ptr<Environment> env)
{
  itsEnvironment = env;

  //! from its environment: topological map
  itsTopologicalMap = env->getTopologicalMap();

  //! from its environment: visual landmark database
  itsLandmarkDB = env->getLandmarkDB();
}

// ######################################################################
void GSlocalizer::setSavePrefix(std::string prefix)
{
  itsSavePrefix = prefix;
}

// ######################################################################
rutz::shared_ptr<Environment> GSlocalizer::getEnvironment()
{
  return itsEnvironment;
}

// ######################################################################
void GSlocalizer::setBeoWulf(nub::soft_ref<Beowulf> beo)
{
  itsBeowulf = beo;
}

// ######################################################################
void GSlocalizer::start1()
{
  // start threads. They should go to sleep on the condition since no
  // jobs have been queued up yet:
  pthread_mutex_init(&jobLock, NULL);
  pthread_mutex_init(&fnumLock, NULL);
  pthread_mutex_init(&or2Lock, NULL);
  pthread_mutex_init(&stopSearchLock, NULL);
  pthread_mutex_init(&resLock, NULL);
  pthread_mutex_init(&workLock, NULL);
  pthread_mutex_init(&particleLock, NULL);
  pthread_cond_init(&jobCond, NULL);

  LINFO("Starting with %d threads...", NUM_GSL_THREAD);

  // get our processing threads started:
  worker = new pthread_t[NUM_GSL_THREAD];
  for (uint i = 0; i < NUM_GSL_THREAD; i ++)
    {
      pthread_create(&worker[i], NULL, GSlocalizer_threadCompute,
                     (void *)this);

      // all threads should go and lock against our job condition.
      // Sleep a bit to make sure this really happens:
      usleep(100000);
    }

  itsNumWorking = 0;
}

// ######################################################################
void GSlocalizer::stop2()
{
  // should cleanup the threads, mutexes, etc...
  pthread_cond_destroy(&jobCond);

  //for (uint i = 0; i < NUM_GSL_THREAD; i ++)
  //  pthread_delete(&worker[i].....

  delete [] worker;
}

// ######################################################################
GSlocalizer::~GSlocalizer()
{ }

// ######################################################################
void GSlocalizer::setWindow(rutz::shared_ptr<XWinManaged> inputWin)
{
  itsWin = inputWin;
}

// ######################################################################
void GSlocalizer::initParticles(std::string belFName)
{
  uint nsegment = itsTopologicalMap->getSegmentNum();
  itsSegmentBeliefHistogram->resize(nsegment);
  LINFO("number of segment : %d", nsegment);

  itsBeliefParticles.clear();
  itsBeliefLocations.clear();

  // check if the file does not exist or it's a blank entry
  FILE *fp; if((fp = fopen(belFName.c_str(),"rb")) == NULL)
    {
      LINFO("Belief file %s not found", belFName.c_str());
      LINFO("Create random particles");

      // create initial random particles
      for(uint i = 0; i < NUM_PARTICLES; i++)
        {



          float t  = 0.0;//rand()/(RAND_MAX + 1.0);
          float t2 = 0.0;//rand()/(RAND_MAX + 1.0);




          uint  snum  = uint ((0)    + ((nsegment) * t ));
          float ltrav = float((0.0F) + ((1.0F    ) * t2));
          itsBeliefParticles.push_back(LocParticle(snum, ltrav));
        }
    }
  else
    {
      LINFO("Belief file %s found", belFName.c_str());

      // get the particles
      for(uint i = 0; i < NUM_PARTICLES; i++)
        {
          char inLine[200]; if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");
          uint snum; float ltrav;
          sscanf(inLine, "%d %f", &snum, &ltrav);
          itsBeliefParticles.push_back(LocParticle(snum, ltrav));
        }
      Raster::waitForKey();
    }

  // fill in the locations as well
  for(uint i = 0; i < NUM_PARTICLES; i++)
    {
      uint  snum  = itsBeliefParticles[i].segnum;
      float ltrav = itsBeliefParticles[i].lentrav;

      // convert to Point2D<int>
      Point2D<int> loc = itsTopologicalMap->getLocation(snum, ltrav);
      itsBeliefLocations.push_back(loc);

      LDEBUG("particle[%4u]: (%3u, %10.6f) = (%4d %4d)",
             i, snum, ltrav, loc.i, loc.j);
    }
}

// ######################################################################
std::vector<LocParticle> GSlocalizer::getBeliefParticles()
{
  std::vector<LocParticle> beliefParticles(itsBeliefParticles.size());

  pthread_mutex_lock(&particleLock);
  for(uint i = 0; i < itsBeliefParticles.size(); i++)
    {
      beliefParticles[i] =
        LocParticle(itsBeliefParticles[i].segnum,
                   itsBeliefParticles[i].lentrav);
    }
  pthread_mutex_unlock(&particleLock);

  return beliefParticles;
}

// ######################################################################
rutz::shared_ptr<Histogram> GSlocalizer::getSegmentBeliefHistogram()
{
  itsSegmentBeliefHistogram->clear();

  for(uint i = 0; i < NUM_PARTICLES; i++)
    {
      itsSegmentBeliefHistogram->
        addValue(itsBeliefParticles[i].segnum, 1.0F);
    }

  //! print the histogram profile
  uint nsegment = itsTopologicalMap->getSegmentNum();
  for(uint i = 0; i < nsegment; i++)
    LDEBUG("[%d]: %d", i, uint(itsSegmentBeliefHistogram->getValue(i)));
  return itsSegmentBeliefHistogram;
}

// ######################################################################
bool GSlocalizer::outputReady()
{
  bool ret = false;
  uint njobs; uint nworking;
  pthread_mutex_lock(&jobLock);
  njobs = itsJobQueue.size();
  pthread_mutex_unlock(&jobLock);

  pthread_mutex_lock(&workLock);
  nworking = itsNumWorking;
  pthread_mutex_unlock(&workLock);

  ret = (njobs == 0U && nworking == 0U);
  LDEBUG("jobs left: %u, still working: %d", njobs, nworking);
  return ret;
}

// ######################################################################
bool GSlocalizer::isMatchFound(uint index)
{
  bool ret = false;
  pthread_mutex_lock(&resLock);
  ASSERT(index < itsMatchFound.size());
  ret = itsMatchFound[index];
  pthread_mutex_unlock(&resLock);

  return ret;
}

// ######################################################################
GSlocJobData GSlocalizer::getMatch(uint index)
{
  GSlocJobData lmkMatch;
  pthread_mutex_lock(&resLock);
  ASSERT(index < itsLmkMatch.size());
  lmkMatch = itsLmkMatch[index];
  pthread_mutex_unlock(&resLock);

  return lmkMatch;
}

// ######################################################################
Image<PixRGB<byte> > GSlocalizer::getInputImage()
{
  return itsInputImage;
}

// ######################################################################
uint GSlocalizer::getNumInputObject()
{
  return itsInputVO.size();
}

// ######################################################################
rutz::shared_ptr<VisualObject> GSlocalizer::getInputVO(uint index)
{
  rutz::shared_ptr<VisualObject> retvo;
  //pthread_mutex_lock(&resLock);
  ASSERT(index < itsInputVO.size());
  retvo = itsInputVO[index];
  //pthread_mutex_unlock(&resLock);

  return retvo;
}

// ######################################################################
Image<double> GSlocalizer::getInputGist()
{
  Image<double> retigist;
  //pthread_mutex_lock(&resLock);
  retigist = itsInputGist;
  //pthread_mutex_unlock(&resLock);

  return retigist;
}

// ######################################################################
Point2D<int> GSlocalizer::getInputObjOffset(uint index)
{
  Point2D<int> retpt;
  //pthread_mutex_lock(&resLock);
  ASSERT(index < itsInputObjOffset.size());
  retpt = itsInputObjOffset[index];
  //pthread_mutex_unlock(&resLock);

  return retpt;
}

// ######################################################################
rutz::shared_ptr<VisualObjectMatch> GSlocalizer::getVOmatch(uint index)
{
  rutz::shared_ptr<VisualObjectMatch> retMatch;
  pthread_mutex_lock(&resLock);
  ASSERT(index < itsVOmatch.size());
  retMatch = itsVOmatch[index];
  pthread_mutex_unlock(&resLock);

  return retMatch;
}

// ######################################################################
int GSlocalizer::getInputFnum()
{
  int retFnum;
  pthread_mutex_lock(&fnumLock);
  retFnum = itsInputFnum;
  pthread_mutex_unlock(&fnumLock);

  return retFnum;
}

// ######################################################################
int GSlocalizer::getSearchInputFnum()
{
  int retFnum;
  //pthread_mutex_lock(&resLock);
  retFnum = itsSearchInputFnum;
  //pthread_mutex_unlock(&resLock);

  return retFnum;
}

// ######################################################################
rutz::shared_ptr<Histogram> GSlocalizer::getSegmentHistogram()
{
  return itsSegmentHistogram;
}

// ######################################################################
uint GSlocalizer::getSegmentNumberMatch(uint index)
{
  ASSERT(index < itsMatchFound.size());
  return itsSegNumMatch[index];
}

// ######################################################################
float GSlocalizer::getLengthTraveledMatch(uint index)
{
  ASSERT(index < itsMatchFound.size());
  return itsLenTravMatch[index];
}

// ######################################################################
uint GSlocalizer::getNumObjectSearch(uint index)
{
  uint nObjSearch;
  pthread_mutex_lock(&resLock);
  ASSERT(index < itsNumObjectSearch.size());
  nObjSearch = itsNumObjectSearch[index];
  pthread_mutex_unlock(&resLock);

  return nObjSearch;
}

// ######################################################################
void GSlocalizer::input
( Image<PixRGB<byte> > ima,
  std::vector<rutz::shared_ptr<VisualObject> > inputVO,
  std::vector<Point2D<int> > inputObjOffset, int inputFnum, Image<double> cgist,
  float dx, float dy)
{
  pthread_mutex_lock(&fnumLock);
  itsInputFnum = inputFnum;
  pthread_mutex_unlock(&fnumLock);

  // store the robot movement
  itsRobotDx = dx;
  itsRobotDy = dy;

  // calculate the segment prediction
  itsInputGist = cgist;
  itsSegmentHistogram = itsEnvironment->classifySegNum(itsInputGist);

  // apply action model and segment observation model
  pthread_mutex_lock(&particleLock);
  actionUpdateBelief();
  segmentUpdateBelief();
  pthread_mutex_unlock(&particleLock);

  // if the object recognition system is not running
  bool outputReady2;
  pthread_mutex_lock(&or2Lock);
  outputReady2 = itsOutputReady2;
  pthread_mutex_unlock(&or2Lock);

  // if there is no search or resetting
  if(outputReady2)
    {
      pthread_mutex_lock(&or2Lock);
      itsOutputReady2 = false;
      pthread_mutex_unlock(&or2Lock);

      pthread_mutex_lock(&jobLock);
      itsJobQueue.clear();
      itsSearchInputFnum = inputFnum;
      LDEBUG("[%6d] NEW salregs", itsSearchInputFnum);

      // store the input image, visual-objects
      itsInputImage = ima;              // FIX: <- need to be updated
      itsInputVO.clear();
      itsVOKeypointsComputed.clear();
      itsInputObjOffset.clear();
      itsInputVORect.clear();
      uint inputSize = inputVO.size();
      for(uint i = 0; i < inputSize; i++)
        {
          itsInputVO.push_back(inputVO[i]);
          itsVOKeypointsComputed.push_back(false);
          itsInputObjOffset.push_back(inputObjOffset[i]);
          Dims d = inputVO[i]->getImage().getDims();
          itsInputVORect.push_back(Rectangle(inputObjOffset[i], d));
        }
      pthread_mutex_unlock(&jobLock);

      // resize the result storage
      pthread_mutex_lock(&resLock);
      itsMatchFound.clear();
      itsVOmatch.clear();         itsVOmatch.resize(inputSize);
      itsDBmatchVORect.clear();   itsDBmatchVORect.resize(inputSize);
      itsLmkMatch.clear();        itsLmkMatch.resize(inputSize);
      itsSegNumMatch.clear();     itsSegNumMatch.resize(inputSize);
      itsLenTravMatch.clear();    itsLenTravMatch.resize(inputSize);
      itsNumObjectSearch.clear(); itsNumObjectSearch.resize(inputSize);
      for(uint i = 0; i < inputSize; i++) itsMatchFound.push_back(false);
      for(uint i = 0; i < inputSize; i++) itsNumObjectSearch[i] = 0;
      pthread_mutex_unlock(&resLock);

      // call the search priority function
      pthread_mutex_lock(&jobLock);
      setSearchPriority();

      // we will prioritize using saliency in the search loop

      // so sort the queue then
      itsIsQueueSorted = false;

      pthread_mutex_unlock(&jobLock);

      // broadcast on job queue condition to wake up worker threads:
      pthread_cond_broadcast(&jobCond);
    }
  else
    {
      LDEBUG("[%6d] NO salregs", inputFnum);

      // FIX: funky project forward stuff
    }
}

// ######################################################################
void GSlocalizer::setGroundTruth(uint snum, float ltrav)
{
  itsSnumGT  = snum;
  itsLtravGT = ltrav;
}

// ######################################################################
void GSlocalizer::getGroundTruth(uint &snum, float &ltrav)
{
  snum  = itsSnumGT;
  ltrav = itsLtravGT;
}

// ######################################################################
//! set the search priority for landmark DB
void GSlocalizer::setSearchPriority()
{
  // search priority is:
  // GIST_PRIORITY_WEIGHT * segment priority +
  // LOCN_PRIORITY_WEIGHT * current location priority +
  // SAL_PRIORITY_WEIGHT * saliency priority
  //   (sal is done in the search loop)

  itsTimer->reset();
  // create jobs for each landmark - object combination
  for(uint i = 0; i < itsEnvironment->getNumSegment(); i++)
    {
      uint nlmk = itsLandmarkDB->getNumSegLandmark(i);
      LDEBUG("itsLandmarkDB[%d]: %d", i, nlmk);
      for(uint j = 0; j < nlmk; j++)
        {
          // check each object
          for(uint l = 0; l < itsInputVO.size(); l++)
            {
              uint nObj = itsLandmarkDB->getLandmark(i,j)->numObjects();
              uint k = 0;
              while(k < nObj)
                {
                  uint k2 = k + N_OBJECT_BLOCK - 1;
                  if(k2 > nObj-1) k2 = nObj - 1;
                  itsJobQueue.push_back(GSlocJobData(l, i, j, k, k2));
                  LDEBUG("match obj[%d] lDB[%3d][%3d]:[%3d,%3d]",
                         l, i,j,k,k2);
                  k = k2 + 1;
                }
            }
        }
    }
  LDEBUG("setting jobs %11.5f", itsTimer->get()/1000.0F);

  // FIX: GOOD LANDMARKS ARE FOUND IN MULTIPLE RUNS !

  // set the order of search to random values
  // not actually used, just for baseline to put in ICRA08
  //addRandomPriority();

  // FIX: always load the last 10 matched landmarks first
  // we do this by adding a value of 3.0

  // add the segment priority
  itsTimer->reset();
  addSegmentPriority();
  LDEBUG("segment      %11.5f", itsTimer->get()/1000.0F);

  // add the current location priority
  itsTimer->reset();
  addLocationPriority();
  LDEBUG("location     %11.5f", itsTimer->get()/1000.0F);

  // information for when to quit
  itsNumJobs           = itsJobQueue.size();
  itsNumJobsProcessed  = 0;
  itsLastSuccessfulJob = 0;
  itsNumObjectFound    = 0;
}

// ######################################################################
void GSlocalizer::addRandomPriority()
{
  // add a random value to the priority value
  std::list<GSlocJobData>::iterator itr = itsJobQueue.begin();
  while (itr != itsJobQueue.end())
    {
      // flip the value
      float val = float(rand()/(RAND_MAX + 1.0));
      (*itr).pVal += val;
      itr++;
    }
}

// ######################################################################
void GSlocalizer::addSegmentPriority()
{
  // add the segment value to the priority value
  std::list<GSlocJobData>::iterator itr = itsJobQueue.begin();
  while (itr != itsJobQueue.end())
    {
      // flip the value
      float val = GIST_PRIORITY_WEIGHT *
        (1.0 - itsSegmentHistogram->getValue((*itr).segNum));

      (*itr).pVal += val;
      (*itr).segVal = val;
      itr++;
    }
}

// ######################################################################
void GSlocalizer::addSaliencyPriority()
{
  uint nObj = itsInputVO.size();

  // go through each landmark - object combination
  itsTimer->reset();
  uint nSeg = itsEnvironment->getNumSegment();
  std::vector<std::vector<std::vector<float> > > salVal(nSeg);
  for(uint i = 0; i < itsEnvironment->getNumSegment(); i++)
    {
      uint nlmk = itsLandmarkDB->getNumSegLandmark(i);
      salVal[i].resize(nlmk);
      for(uint j = 0; j < nlmk; j++)
        {
          // check each object
          salVal[i][j].resize(nObj);
          for(uint k = 0; k < nObj; k++)
            {
              LDEBUG("sal seg[%3d] lmk[%3d] obj[%3d]", i,j,k);
              salVal[i][j][k] = SAL_PRIORITY_WEIGHT *
                itsLandmarkDB->getLandmark(i,j)
                ->matchSalientFeatures(itsInputVO[k]);
            }

          // display the database landmark object
          //Image<PixRGB<byte> > img = itsLandmarkDB->getLandmark(i,j)
          //  ->getObject(0)->getSalAndKeypointImage();
          //itsWin->drawImage(img, 0,0);
          //Raster::waitForKey();
        }
    }
  LDEBUG("compute saliency dist %11.5f", itsTimer->get()/1000.0F);

  // add the saliency value to the priority value
  std::list<GSlocJobData>::iterator itr = itsJobQueue.begin();
  while (itr != itsJobQueue.end())
    {
      float t = (*itr).pVal;
      float val = salVal[(*itr).segNum][(*itr).lmkNum][(*itr).objNum];
      (*itr).pVal += val;
      (*itr).salVal = val;

      LDEBUG("pval[%3d][%3d][%3d]: %f + %f = %f",
             (*itr).segNum, (*itr).lmkNum, (*itr).objNum, t, val, (*itr).pVal);
      itr++;
    }
}

// ######################################################################
void GSlocalizer::addLocationPriority()
{
  // normalizer: setup weight and sigma for decision boundary
  Dims mDims = itsTopologicalMap->getMapDims();
  Point2D<int> brMap(mDims.w(), mDims.h());
  float mDiag = brMap.distance(Point2D<int>(0,0));
  float sigma = .1 * mDiag;
  LDEBUG("map diagonal: %f -> sigma: %f",  mDiag, sigma);
  LDEBUG("curr loc: %d, %f ", itsSegmentLocation, itsSegmentLengthTraveled);

  // get the distance to all landmarks from current belief location
  uint nSeg =itsEnvironment->getNumSegment();
  std::vector<std::vector<float> > locVal(nSeg);
  for(uint i = 0; i < itsEnvironment->getNumSegment(); i++)
    {
      uint nlmk = itsLandmarkDB->getNumSegLandmark(i);
      locVal[i].resize(nlmk);
      for(uint j = 0; j < nlmk; j++)
        {
          std::pair<float,float> locRange =
            itsLandmarkDB->getLocationRange(i,j);
          LDEBUG("lmk[%3d][%3d]: (%f,%f)", i,j,
                 locRange.first, locRange.second);

          // if the location is within the range of the landmark
          if(itsSegmentLocation == i &&
             itsSegmentLengthTraveled >= locRange.first &&
             itsSegmentLengthTraveled <= locRange.second  )
            {
              locVal[i][j] = 0.0;
              LDEBUG("dist[%d][%d]: within -> %f",i,j, locVal[i][j]);
            }
          else
            {
              // get distance to the first seen location
              float fdist = itsTopologicalMap->
                getDistance(itsSegmentLocation, itsSegmentLengthTraveled,
                            i, locRange.first);

              // get distance to the last seen location
              float ldist = itsTopologicalMap->
                getDistance(itsSegmentLocation, itsSegmentLengthTraveled,
                            i, locRange.second);

              // get the minimum distance
              float dist = std::min(fdist,ldist);

              LDEBUG("f - l: %d [%f -> %f][%f -> %f]",
                     i, locRange.first, fdist, locRange.second, ldist);

              // get distances to nodes in between
              std::vector<std::pair<uint,float> > betLocs =
                itsTopologicalMap->getNodeLocationsInInterval
                (i, locRange.first, locRange.second);
              for(uint k = 0; k < betLocs.size(); k++)
                {
                  float bdist = itsTopologicalMap->
                    getDistance(itsSegmentLocation, itsSegmentLengthTraveled,
                                i, betLocs[k].second);
                  if(dist > bdist) dist = bdist;

                  LDEBUG("bet: %d [%f -> %f]", i, betLocs[k].second, bdist);
                }

              // normalize gaussian val to 1.0 invert the values
              locVal[i][j] = LOCN_PRIORITY_WEIGHT *
                (1.0 - pow(M_E, -dist*dist/(2.0*sigma*sigma)));
              LDEBUG("dist[%d][%d]: %f ->%f",i,j, dist, locVal[i][j]);
            }
        }
    }

  // add the location value to the priority value
  std::list<GSlocJobData>::iterator itr = itsJobQueue.begin();
  while (itr != itsJobQueue.end())
    {
      float t = (*itr).pVal;
      float val = locVal[(*itr).segNum][(*itr).lmkNum];
      (*itr).pVal += val;
      (*itr).locVal = val;

      LDEBUG("pval[%3d][%3d][%3d]: %f -> %f",
             (*itr).segNum, (*itr).lmkNum, (*itr).objNum, t, (*itr).pVal);
      itr++;
    }
}

// ######################################################################
void GSlocalizer::stopSearch()
{
  //! stop search by cleaning up the queue
  pthread_mutex_lock(&jobLock);
  itsJobQueue.clear();
  pthread_mutex_unlock(&jobLock);

  //! wait until all workers stop working
  uint nworking = 1;
  while (nworking != 0U)
    {
      pthread_mutex_lock(&workLock);
      nworking = itsNumWorking;
      pthread_mutex_unlock(&workLock);
      LINFO("still working: %d", nworking);
      usleep(1000);
    }
}

// ######################################################################
void GSlocalizer::stopSearch2()
{
  pthread_mutex_lock(&stopSearchLock);
  itsStopSearch = true;
  pthread_mutex_unlock(&stopSearchLock);
}

// ######################################################################
// The threaded function
void GSlocalizer::threadCompute()
{
  pthread_mutex_lock(&resLock);
  uint myNum = numWorkers ++;
  pthread_mutex_unlock(&resLock);
  LINFO("  ... worker %u ready.", myNum);

  while(true)
    {
      // wait until there's a job to do
      pthread_mutex_lock(&jobLock);
      GSlocJobData cjob(0, 0, 0, 0, 0); bool nojobs = true;
      if (itsJobQueue.empty() == false)
        {
          if(!itsIsQueueSorted)
            {
              // doing saliency prioritization is slow
              // has to be done outside the input loop

              // add the saliency priority
              addSaliencyPriority();

              // push the jobs based on the biasing values
              itsTimer->reset();
              itsJobQueue.sort();
              LDEBUG("sal prior %11.5f", itsTimer->get()/1000.0F);

              // print the priority values
              std::list<GSlocJobData>::iterator itr = itsJobQueue.begin();
              uint count = 0;
              while (itr != itsJobQueue.end())
                {
                  LDEBUG("[%5d] pval[%3d][%3d][%3d]: %f + %f + %f = %f", count,
                         (*itr).segNum, (*itr).lmkNum, (*itr).objNum,
                         (*itr).segVal, (*itr).salVal, (*itr).locVal,
                         (*itr).pVal);
                  itr++; count++;
                }

              itsIsQueueSorted = true;
            }

          cjob = itsJobQueue.front();
          itsJobQueue.pop_front();
          nojobs = false;
        }
      else
        pthread_cond_wait(&jobCond, &jobLock);
      pthread_mutex_unlock(&jobLock);

      // if we don't have a job to do, just wait more:
      if (nojobs) continue;

      // else we have something
      LDEBUG("T[%4d] match object[%d] itsLandmarkDB[%d][%d]: [ %d, %d ]",
             myNum, cjob.objNum, cjob.segNum, cjob.lmkNum,
             cjob.voStartNum, cjob.voEndNum);

      // add to busy thread count
      pthread_mutex_lock(&workLock);
      itsNumWorking++;
      pthread_mutex_unlock(&workLock);

      // make sure the VO keypoints are computed
      pthread_mutex_lock(&jobLock);
      if(!itsVOKeypointsComputed[cjob.objNum])
        {
          itsInputVO[cjob.objNum]->computeKeypoints();
          itsVOKeypointsComputed[cjob.objNum] = true;
        }
      pthread_mutex_unlock(&jobLock);

      // match the object with the range of database
      // we limit the maxScale range to [2/3 ... 3/2] by entering 1.5
      rutz::shared_ptr<VisualObjectMatch> cmatch;
      int ind = itsLandmarkDB->getLandmark(cjob.segNum, cjob.lmkNum)->
        match(itsInputVO[cjob.objNum],
              cmatch, cjob.voStartNum, cjob.voEndNum,
              15.0F, 0.5F, 2.5F, 4, M_PI/4, 1.5F, .25F);

      pthread_mutex_lock(&jobLock);

      // if match is found
      if(ind != -1)
        {
          // NOTE: to print matches
          LDEBUG("-> found match[%d]: %s with itsLandmarkDB[%d][%d]\n %s : %s",
                 cjob.objNum, itsInputVO[cjob.objNum]->getName().c_str(),
                 cjob.segNum, cjob.lmkNum,
                 cmatch->getVoRef()->getName().c_str(),
                 cmatch->getVoTest()->getName().c_str());

          // clear queue of jobs for this object
          uint njobs = 0; uint njobsremoved = 0;
          std::list<GSlocJobData>::iterator itr = itsJobQueue.begin();
          while (itr != itsJobQueue.end())
            {
              if((*itr).objNum == cjob.objNum)
                { itr = itsJobQueue.erase(itr); njobsremoved++; }
              else
                { itr++; njobs++; }
            }
          itsNumObjectSearch[cjob.objNum] += (ind - cjob.voStartNum + 1);
          itsLastSuccessfulJob = itsNumJobsProcessed;
          LDEBUG("removing %d jobs left to do: %" ZU ,
                 njobsremoved, itsJobQueue.size());

          // store the match information
          // since there is a possibility that there are concurrent threads
          // that also just found it we will choose the first match
          pthread_mutex_lock(&resLock);
          //if(!itsMatchFound[cjob.objNum])
          //  {
              itsNumObjectFound++;
              itsVOmatch[cjob.objNum] = cmatch;
              itsLmkMatch[cjob.objNum] =
                GSlocJobData(cjob.objNum, cjob.segNum, cjob.lmkNum, ind, ind);
              itsSegNumMatch[cjob.objNum] = cjob.segNum;
              itsLenTravMatch[cjob.objNum] =
                itsLandmarkDB->getLenTrav(cjob.segNum, cjob.lmkNum, ind);
              itsMatchFound[cjob.objNum] = true;

              Point2D<int> mOffset =
                itsLandmarkDB->getLandmark(itsLmkMatch[cjob.objNum].segNum,
                                           itsLmkMatch[cjob.objNum].lmkNum)
                ->getOffsetCoords(itsLmkMatch[cjob.objNum].voStartNum);
              Dims mDims = 
                itsLandmarkDB->getLandmark(itsLmkMatch[cjob.objNum].segNum,
                                           itsLmkMatch[cjob.objNum].lmkNum)
                ->getObject(itsLmkMatch[cjob.objNum].voStartNum)->getImage().getDims();
              itsDBmatchVORect[cjob.objNum] = Rectangle(mOffset, mDims);

              // update the belief using individual object
              // NOTE: this is for individual object updating
              //LINFO("updating objBelief[%d]", cjob.objNum);
              //pthread_mutex_lock(&particleLock);
              //objectUpdateBelief(cjob.objNum);
              //pthread_mutex_unlock(&particleLock);
              //  }
          pthread_mutex_unlock(&resLock);
        }
      else
        {
          pthread_mutex_lock(&resLock);
          itsNumObjectSearch[cjob.objNum]
            += (cjob.voEndNum - cjob.voStartNum + 1);
          pthread_mutex_unlock(&resLock);
        }

      // check if stop search is requested
      bool stopSearch = false;
      pthread_mutex_lock(&stopSearchLock);
      stopSearch = itsStopSearch;
      pthread_mutex_unlock(&stopSearchLock);

      // update last successful job count
      itsNumJobsProcessed++;
      int dlast = itsNumJobsProcessed - itsLastSuccessfulJob;

      // stop after matching 3 of 5
      // check if the number of matches since last successful one
      //   is bigger than a percentage of the queue size
      bool earlyExit =
        itsNumObjectFound > 2 ||
        (itsNumObjectFound == 2 && dlast > (int)(.0167 * itsNumJobs)) ||
        (itsNumObjectFound == 1 && dlast > (int)(.033  * itsNumJobs)) ||
        (itsNumObjectFound == 0 && dlast > (int)(.05   * itsNumJobs));

      if(itsJobQueue.size() > 0 && (earlyExit || stopSearch))
        {
          // NOTE: print to display performance
          LDEBUG("EE: %d SS: %d [found: %d, dlast: %d] clear: %" ZU ","
                " jobs processed: %d/%d = %f",
                earlyExit, stopSearch,
                itsNumObjectFound, dlast, itsJobQueue.size(),
                itsNumJobsProcessed, itsNumJobs,
                (float)itsNumJobsProcessed/itsNumJobs);
          itsJobQueue.clear();
        }
      uint njobs = itsJobQueue.size();
      pthread_mutex_unlock(&jobLock);

      // no longer busy
      pthread_mutex_lock(&workLock);
      itsNumWorking--;

      // when the last thread is done with its work
      // update belief with the object obsevation model


      pthread_mutex_lock(&jobLock);
      njobs = itsJobQueue.size(); 
      pthread_mutex_unlock(&jobLock);
      if(njobs != 0 &&  itsNumWorking == 0) LINFO("Possible error jobs left: %u", njobs);
      if(njobs == 0U && itsNumWorking == 0U)
        {
          // apply object observation model for all objects found
          pthread_mutex_lock(&particleLock);
          objectUpdateBelief();
          pthread_mutex_unlock(&particleLock);

          // save some performance data
          uint cfnum = getInputFnum();

          std::string resFName = itsSavePrefix + sformat("_GS_results.txt");
          FILE *rFile = fopen(resFName.c_str(), "at");
          if (rFile != NULL)
            {
              uint ninput = getNumInputObject();
              std::vector<bool> mfound(ninput);
              std::vector<uint> nObjSearch(ninput);
              for(uint i = 0; i < ninput; i++)
                {
                  mfound[i]     = isMatchFound(i);
                  nObjSearch[i] = getNumObjectSearch(i);
                }
              LDEBUG("saving result to %s", resFName.c_str());
              std::string line = sformat("%5d %d ", cfnum, ninput);
              
              for(uint i = 0; i < ninput; i++)
                if(mfound[i])
                  line += sformat("%3d %6d   %6d %6d %6d %6d  %6d %6d %6d %6d", 
                                  int(mfound[i]), nObjSearch[i],
                                  itsInputVORect[i].top()   , itsInputVORect[i].left(),
                                  itsInputVORect[i].bottomO(), itsInputVORect[i].rightO(),
                                  itsDBmatchVORect[i].top()   , itsDBmatchVORect[i].left(),
                                  itsDBmatchVORect[i].bottomO(), itsDBmatchVORect[i].rightO());
                else
                  line += sformat("%3d %6d   %6d %6d %6d %6d  %6d %6d %6d %6d", 
                                  int(mfound[i]), nObjSearch[i],
                                  itsInputVORect[i].top()    , itsInputVORect[i].left(),
                                  itsInputVORect[i].bottomO(), itsInputVORect[i].rightO(), 
                                  -1, -1, -1, -1);

              LDEBUG("%s", line.c_str());
              line += std::string("\n");

              fputs(line.c_str(), rFile);
              fclose (rFile);
            }
          else LINFO("can't create file: %s", resFName.c_str());

          // signal to master that it is done
          TCPmessage smsg;
          smsg.reset(cfnum, SEARCH_LM_RES);
          smsg.addInt32(int32(getSearchInputFnum()));
          smsg.addInt32(int32(cfnum));
          itsBeowulf->send(-1, smsg);

          pthread_mutex_lock(&or2Lock);
          itsOutputReady2 = true;
          pthread_mutex_unlock(&or2Lock);

          pthread_mutex_lock(&stopSearchLock);
          itsStopSearch = false;
          pthread_mutex_unlock(&stopSearchLock);

          LDEBUG("[%d]  DONE SEARCH_LM: %d\n", cfnum, getSearchInputFnum());
        }
      pthread_mutex_unlock(&workLock);
    }
}

// ######################################################################
void GSlocalizer::updateBelief()
{
  // set the most likely location
  pthread_mutex_lock(&particleLock);
  setLocation();
  pthread_mutex_unlock(&particleLock);
}

// ######################################################################
void GSlocalizer::actionUpdateBelief()
{
  std::vector<int> stotal(itsTopologicalMap->getSegmentNum());
  for(uint i = 0; i < stotal.size(); i++) stotal[i] = 0;

  // apply the motor movement + noise to each particles
  for(uint i = 0; i < NUM_PARTICLES; i++)
    {
      uint  snum  = itsBeliefParticles[i].segnum;
      float ltrav = itsBeliefParticles[i].lentrav;
      LDEBUG("particle[%d]: %d, %f", i, snum, ltrav);

      // apply the motor command
      float nltrav = ltrav + itsRobotDx;

      // assume std odometry error of .02ft
      float mscale = itsTopologicalMap->getMapScale();
      float err    = STD_ODO_ERROR/mscale;

      // add noise from a Gaussian distribution (Box-Muller method)
      float r1 = float(rand()/(RAND_MAX + 1.0));
      float r2 = float(rand()/(RAND_MAX + 1.0));
      double r = err * sqrt( -2.0 * log(r1));
      double phi = 2.0 * M_PI * r2;
      nltrav += (r * cos(phi));

      // if nltrav is now improper
      // FIX: NEED THE SPILL OVER EFFECT
      if(nltrav < 0.0F) nltrav = 0.0F;
      if(nltrav > 1.0F) nltrav = 1.0F;
      itsBeliefParticles[i].lentrav = nltrav;
      //stotal[snum]++;

      // convert to Point2D<int>
      Point2D<int> loc = itsTopologicalMap->
        getLocation(itsBeliefParticles[i].segnum,
                    itsBeliefParticles[i].lentrav);
      itsBeliefLocations[i] = loc;
    }
}

// ######################################################################
void GSlocalizer::segmentUpdateBelief()
{
  // accweight is for easy calculation for choosing a random particle
  std::vector<float> weight(NUM_PARTICLES);
  std::vector<float> accweight(NUM_PARTICLES);

  // for each particles
  float accw = 0.0F;
  for(uint i = 0; i < NUM_PARTICLES; i++)
    {
      uint  snum  = itsBeliefParticles[i].segnum;
      float ltrav = itsBeliefParticles[i].lentrav;
      LDEBUG("particle[%4d]: %d, %f", i, snum, ltrav);

      // get likelihood score
      // score(snum)/sum(score(all segments) * score(snum)
      // with the denominator taken out
      // + 0.25 for the 50% memory of previous time
      float pscore = itsSegmentHistogram->getValue(snum);
      float score  = (pscore * pscore) + 0.25f;
      LDEBUG("score: %f * %f = %f", pscore, pscore, score);

      // update weight
      weight[i] = score;
      accw += score;
      accweight[i] = accw;
    }
  for(uint i = 0; i < NUM_PARTICLES; i++)
    LDEBUG("p[%4d]: w: %f %f ",i, weight[i], accweight[i]);
  LDEBUG("accw: %f",accw);

  // add 1% noise weight
  accw *= 1.01F; LDEBUG("accw+ noise: %f",accw);

  // weighted resample NUM_PARTICLES particles
  std::vector<LocParticle> tbelief(NUM_PARTICLES);
  uint nsegment = itsTopologicalMap->getSegmentNum();
  std::vector<int> stotal(nsegment);
  for(uint i = 0; i < stotal.size(); i++) stotal[i] = 0;

  for(uint i = 0; i < NUM_PARTICLES; i++)
    {
      // draw a random value between 0 and accw
      float rval  = accw * float(rand()/(RAND_MAX + 1.0));

      // get the random index
      uint sind = NUM_PARTICLES;
      for(uint j = 0; j < NUM_PARTICLES; j++)
        if(rval < accweight[j]) { sind = j; j = NUM_PARTICLES; }
      LDEBUG("rval: %f -> %d", rval, sind);

      // if need to create a random particle
      if(sind == NUM_PARTICLES)
        {
          // create initial random particles
          float t  = rand()/(RAND_MAX + 1.0);
          float t2 = rand()/(RAND_MAX + 1.0);

          uint  snum  = uint ((0)    + ((nsegment) * t ));
          float ltrav = float((0.0F) + ((1.0F    ) * t2));
          tbelief[i] = LocParticle(snum, ltrav);
          stotal[snum]++;
          LDEBUG("rand particle[%d]: (%d, %f)", i, snum, ltrav);
        }
      else
        {
          tbelief[i] = itsBeliefParticles[sind];
          stotal[itsBeliefParticles[sind].segnum]++;
          LDEBUG("old  particle[%d]", sind);
        }
   }

  for(uint i = 0; i < stotal.size(); i++) LDEBUG("seg[%d]: %d",i, stotal[i]);

  // copy all the particles to our belief
  for(uint i = 0; i < NUM_PARTICLES; i++)
    {
      itsBeliefParticles[i] = tbelief[i];

      // convert to Point2D<int>
      Point2D<int> loc = itsTopologicalMap->
        getLocation(tbelief[i].segnum, tbelief[i].lentrav);
      itsBeliefLocations[i] = loc;
    }
}

// ######################################################################
void GSlocalizer::objectUpdateBelief()
{
  // make sure at least 1 object is found
  uint c = 0;
  for(uint i = 0; i < itsMatchFound.size(); i++) if(itsMatchFound[i]) c++;
  if(c == 0) return;

  // setup weight and sigma for decision boundary
  Dims mDims = itsTopologicalMap->getMapDims();
  Point2D<int> brMap(mDims.w(), mDims.h());
  float mDiag = brMap.distance(Point2D<int>(0,0));
  float sigma = .05*mDiag;
  LDEBUG("map diagonal: %f -> sigma: %f", mDiag, sigma);

  // accweight is for easy calculation for choosing a random particle
  std::vector<float> weight(NUM_PARTICLES);
  std::vector<float> accweight(NUM_PARTICLES);

  // for each particles
  float accw = 0.0F;
  for(uint i = 0; i < NUM_PARTICLES; i++)
    {
      uint  snum  = itsBeliefParticles[i].segnum;
      float ltrav = itsBeliefParticles[i].lentrav;
      LDEBUG("particle[%d]: %d, %f", i, snum, ltrav);

      // go through each object found
      float pObjObs = 1.0;
      for(uint index = 0; index < itsMatchFound.size(); index++)
        {
          if(itsMatchFound[index])
            {
              // get the location of the object found
              uint  snumMatch  = itsSegNumMatch[index];
              float ltravMatch = itsLenTravMatch[index];
              LDEBUG("Match[%d]: [%d %f]", index, snumMatch, ltravMatch);

              // get the distance between the two points
              float dist = itsTopologicalMap->
                getDistance(snum, ltrav, snumMatch, ltravMatch);

              float pOMatch = 1.0/(sigma * sqrt(2.0 * M_PI)) *
                pow(M_E, -dist*dist/(2.0*sigma*sigma));
              pObjObs *= pOMatch;

              LDEBUG("dist: %f -> pOMatch: %f -> %f", dist, pOMatch, pObjObs);
            }
        }

      // update weight
      weight[i] = pObjObs;
      accweight[i] = weight[i] + accw;
      accw = accweight[i];
    }

  LDEBUG("accw: %f",accw);
  // add 20% weight - because objects are more exacting
  accw *= 1.20F;
  LDEBUG("accw+ noise: %f",accw);

  // weighted resample NUM_PARTICLES particles
  std::vector<LocParticle> tbelief;
  std::vector<int> stotal(itsTopologicalMap->getSegmentNum());
  for(uint i = 0; i < stotal.size(); i++) stotal[i] = 0;

  uint nsegment = itsTopologicalMap->getSegmentNum();

  for(uint i = 0; i < NUM_PARTICLES; i++)
    LDEBUG("p[%d]: %f %f ",i, weight[i], accweight[i]);

  for(uint i = 0; i < NUM_PARTICLES; i++)
    {
      // draw a random value between 0 and accw
      float rval  = accw * float(rand()/(RAND_MAX + 1.0));

      // get the random index
      uint sind = NUM_PARTICLES;
      for(uint j = 0; j < NUM_PARTICLES; j++)
        if(rval < accweight[j]) { sind = j; j = NUM_PARTICLES; }
      LDEBUG("rval: %f -> %d", rval, sind);

      // if need to create a random particle
      if(sind == NUM_PARTICLES)
        {
          // create initial random particles
          float t  = rand()/(RAND_MAX + 1.0);
          float t2 = rand()/(RAND_MAX + 1.0);

          uint  snum  = uint ((0)    + ((nsegment) * t ));
          float ltrav = float((0.0F) + ((1.0F    ) * t2));
          tbelief.push_back(LocParticle(snum, ltrav));
          stotal[snum]++;
          LDEBUG("rand particle[%d]: (%d, %f)", i, snum, ltrav);
        }
      else
        {
          tbelief.push_back(itsBeliefParticles[sind]);
          stotal[itsBeliefParticles[sind].segnum]++;
          LDEBUG("old  particle[%d]", sind);
        }
    }

  for(uint i = 0; i < stotal.size(); i++) LDEBUG("[%d]: %d",i, stotal[i]);

  // copy all the particles to our belief
  for(uint i = 0; i < NUM_PARTICLES; i++)
    {
      itsBeliefParticles[i] = tbelief[i];

      // convert to Point2D<int>
      Point2D<int> loc = itsTopologicalMap->
        getLocation(tbelief[i].segnum, tbelief[i].lentrav);
      itsBeliefLocations[i] = loc;
    }
}

// ######################################################################
void GSlocalizer::objectUpdateBelief(uint index)
{
  // make sure this object is found
  if(!itsMatchFound[index]) return;

  // setup weight and sigma for decision boundary
  Dims mDims = itsTopologicalMap->getMapDims();
  Point2D<int> brMap(mDims.w(), mDims.h());
  float mDiag = brMap.distance(Point2D<int>(0,0));
  float sigma = .05*mDiag;
  LDEBUG("map diagonal: %f -> sigma: %f", mDiag, sigma);

  // accweight is for easy calculation for choosing a random particle
  std::vector<float> weight(NUM_PARTICLES);
  std::vector<float> accweight(NUM_PARTICLES);

  // for each particles
  float accw = 0.0F;
  for(uint i = 0; i < NUM_PARTICLES; i++)
    {
      uint  snum  = itsBeliefParticles[i].segnum;
      float ltrav = itsBeliefParticles[i].lentrav;
      LDEBUG("particle[%d]: %d, %f", i, snum, ltrav);

      // get the location of the object found
      uint  snumMatch  = itsSegNumMatch[index];
      float ltravMatch = itsLenTravMatch[index];
      LDEBUG("Match[%d]: [%d %f]", index, snumMatch, ltravMatch);

      // get the distance between the two points
      float dist = itsTopologicalMap->
        getDistance(snum, ltrav, snumMatch, ltravMatch);

      float pObjObs = 1.0/(sigma * sqrt(2.0 * M_PI)) *
        pow(M_E, -dist*dist/(2.0*sigma*sigma));
      LDEBUG("dist: %f -> %f", dist, pObjObs);

      // update weight
      weight[i] = pObjObs;
      accweight[i] = weight[i] + accw;
      accw = accweight[i];
    }

  LDEBUG("accw: %f",accw);
  // add 20% weight - because objects are more exacting
  accw *= 1.20F;
  LDEBUG("accw+ noise: %f",accw);

  // weighted resample NUM_PARTICLES particles
  std::vector<LocParticle> tbelief;
  std::vector<int> stotal(itsTopologicalMap->getSegmentNum());
  for(uint i = 0; i < stotal.size(); i++) stotal[i] = 0;

  uint nsegment = itsTopologicalMap->getSegmentNum();

  for(uint i = 0; i < NUM_PARTICLES; i++)
    LDEBUG("p[%d]: %f %f ",i, weight[i], accweight[i]);

  for(uint i = 0; i < NUM_PARTICLES; i++)
    {
      // draw a random value between 0 and accw
      float rval  = accw * float(rand()/(RAND_MAX + 1.0));

      // get the random index
      uint sind = NUM_PARTICLES;
      for(uint j = 0; j < NUM_PARTICLES; j++)
        if(rval < accweight[j]) { sind = j; j = NUM_PARTICLES; }
      LDEBUG("rval: %f -> %d", rval, sind);

      // if need to create a random particle
      if(sind == NUM_PARTICLES)
        {
          // create initial random particles
          float t  = rand()/(RAND_MAX + 1.0);
          float t2 = rand()/(RAND_MAX + 1.0);

          uint  snum  = uint ((0)    + ((nsegment) * t ));
          float ltrav = float((0.0F) + ((1.0F    ) * t2));
          tbelief.push_back(LocParticle(snum, ltrav));
          stotal[snum]++;
          LDEBUG("rand particle[%d]: (%d, %f)", i, snum, ltrav);
        }
      else
        {
          tbelief.push_back(itsBeliefParticles[sind]);
          stotal[itsBeliefParticles[sind].segnum]++;
          LDEBUG("old  particle[%d]", sind);
        }
    }

  for(uint i = 0; i < stotal.size(); i++) LDEBUG("[%d]: %d",i, stotal[i]);

  // copy all the particles to our belief
  for(uint i = 0; i < NUM_PARTICLES; i++)
    {
      itsBeliefParticles[i] = tbelief[i];

      // convert to Point2D<int>
      Point2D<int> loc = itsTopologicalMap->
        getLocation(tbelief[i].segnum, tbelief[i].lentrav);
      itsBeliefLocations[i] = loc;
    }
}

// ######################################################################
void GSlocalizer::setLocation()
{
  // for each point
  float maxscore = 0.0F;
  for(uint i = 0; i < itsBeliefLocations.size(); i++)
    {
      // get distances with the neighbors closer than MAX_ERROR_DIST
      float score = 0.0F;
      Point2D<int> a = itsBeliefLocations[i];
      uint aseg = itsBeliefParticles[i].segnum;
      for(uint j = 0; j < itsBeliefLocations.size(); j++)
        {
          Point2D<int> b = itsBeliefLocations[j];
          float dist = a.distance(b);

          uint bseg = itsBeliefParticles[j].segnum;
          float cscore = 0.0; float sthresh = MAX_LOC_ERROR/2.0;
          if(dist < sthresh) // (0.0  ... 2.5] -> (1.0 -> 0.8]
            {
              cscore = (1.0  - (dist - 0.0)/sthresh * 0.2);
            }
          else if(dist < sthresh*2) // 2.5 ... 5.0] -> [0.8 ... 0.2]
            {
              cscore = (0.8 - (dist - sthresh)/sthresh * 0.6);
            }
          if(aseg != bseg) cscore *= .5;
          score += cscore;
        }

      // update max location
      if(score > maxscore)
        {
          maxscore = score;
          itsLocation = itsBeliefLocations[i];
          itsSegmentLocation = itsBeliefParticles[i].segnum;
          itsSegmentLengthTraveled = itsBeliefParticles[i].lentrav;
        }
    }

  LDEBUG("max score: %f: (%d, %d) = [%d %f]", maxscore,
         itsLocation.i, itsLocation.j,
         itsSegmentLocation, itsSegmentLengthTraveled);
}

// ######################################################################
Point2D<int> GSlocalizer::getLocation()
{
  return itsLocation;
}

// ######################################################################
uint GSlocalizer::getSegmentLocation()
{
  return itsSegmentLocation;
}

// ######################################################################
float GSlocalizer::getSegmentLengthTraveled()
{
  return itsSegmentLengthTraveled;
}

// ######################################################################
Image<PixRGB<byte> > GSlocalizer::getBeliefImage(uint w, uint h, float &scale)
{
  Image< PixRGB<byte> > res(w,h,ZEROS);

  // get the map from the topolagical map class
  Image< PixRGB<byte> > mapImg = itsTopologicalMap->getMapImage(w, h);
  Dims d = itsTopologicalMap->getMapDims();

  // add the particles on the map
  scale = float(mapImg.getWidth())/float(d.w());
  for(uint i = 0; i < itsBeliefParticles.size(); i++)
    {
      // get the point
      Point2D<int> loc(int(itsBeliefLocations[i].i * scale),
                       int(itsBeliefLocations[i].j * scale) );
      LDEBUG("point: %d %d", loc.i, loc.j);

      drawDisk(mapImg, loc, 2, PixRGB<byte>(0,255,255));
    }

  // draw circle to the most likely location of the object
  Point2D<int> loc_int(itsLocation.i*scale, itsLocation.j*scale);
  drawDisk  (mapImg, loc_int, 2, PixRGB<byte>(0,0,255));
  drawCircle(mapImg, loc_int, int(MAX_LOC_ERROR*scale),
             PixRGB<byte>(0,0,255), 1);

  inplacePaste(res, mapImg, Point2D<int>(0,0));

  // get the segment belief histogram
  rutz::shared_ptr<Histogram> shist = getSegmentBeliefHistogram();

  // check which side the histogram is going to be appended to
  uint wslack = w - mapImg.getWidth();
  uint hslack = h - mapImg.getHeight();
  if(hslack >= wslack)
    {
      Image<byte> sHistImg =
        shist->getHistogramImage(w, hslack, 0.0F, float(NUM_PARTICLES));
      inplacePaste(res, Image<PixRGB<byte> >(sHistImg),
                   Point2D<int>(0,mapImg.getHeight()));
    }
  else
    {
      Image<byte> sHistImg =
        shist->getHistogramImage(h, wslack, 0.0F, float(NUM_PARTICLES));

      Image<PixRGB<byte> >
        t = Image<PixRGB<byte> >(flipHoriz(transpose(sHistImg)));
      inplacePaste(res, t, Point2D<int>(mapImg.getWidth(), 0));
    }
  return res;
}

// ######################################################################
Image<PixRGB<byte> > GSlocalizer::getMatchImage(uint index, Dims d)
{
  Image< PixRGB<byte> > result;

  ASSERT(index < itsMatchFound.size());
  Point2D<int> objOffset1 = itsInputObjOffset[index];
  Point2D<int> objOffset2 =
    itsLandmarkDB->getLandmark(itsLmkMatch[index].segNum,
                               itsLmkMatch[index].lmkNum)
    ->getOffsetCoords(itsLmkMatch[index].voStartNum);

  bool isODmatch = (itsInputVO[index] == itsVOmatch[index]->getVoRef());
  bool isDOmatch = (itsInputVO[index] == itsVOmatch[index]->getVoTest());

  if(isODmatch)
    result = itsVOmatch[index]->getMatchImage(d, objOffset1, objOffset2);
  else if(isDOmatch)
    result = itsVOmatch[index]->getMatchImage(d, objOffset2, objOffset1);
  else
    {
      LINFO("obj[%d] %s : %s, %s",
            index, itsInputVO[index]->getName().c_str(),
            itsVOmatch[index]->getVoRef()->getName().c_str(),
            itsVOmatch[index]->getVoTest()->getName().c_str());
      LFATAL("object neither ref nor tst");
    }

  return result;
}

// ######################################################################
Point2D<int> GSlocalizer::getMotorSignal()
{

//       LINFO("Zero out [%d,%d]", diff.i, diff.j);
//       // horz component: neg: right, pos: left

//       // Fix: store the object just found

//       // provide a clue as to where to go next


  return Point2D<int>(0,0);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
