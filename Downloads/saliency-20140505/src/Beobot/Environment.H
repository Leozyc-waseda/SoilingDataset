/*!@file Beobot/Environment.H all the information describing
  an environment */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/Environment.H $
// $Id: Environment.H 15426 2012-11-02 21:44:22Z siagian $
//

// ######################################################################
/* Environment class has all the information describing an environment
   it connects the map and the landmarks
*/

#ifndef BEOBOT_ENVIRONMENT_H_DEFINED
#define BEOBOT_ENVIRONMENT_H_DEFINED

#include "Beobot/LandmarkDB.H"
#include "Beobot/TopologicalMap.H"
#include "Neuro/gistParams.H"
#include "Gist/FFN.H"
#include "Gist/trainUtils.H"
#include "SIFT/Histogram.H"

//! Environment class
class Environment
{
public:

  // ######################################################################
  /*! @name Constructors, assignment, and Destructors */
  //@{

  //! Constructor: generate a blank environment if file does not exist
  Environment(std::string envFName = std::string(""));

  //! return true if environment has no segments
  bool isBlank();

  //! load the .env file
  bool load(std::string fName);

  //! save the main environment to a file
  bool save(std::string fName, std::string envPrefix);

  //! save the main and the just build environment
  bool save(std::string fName, std::string tfName, std::string envPrefix);

  //! Destructor
  ~Environment();

  //! set the window to view results
  inline void setWindow(rutz::shared_ptr<XWinManaged> win);

  //! reset the number of segments in the environment
  void resetNumSegment(uint nSegment);

  //! set the current
  void setCurrentSegment(uint currSegNum);

  //@}

  // ######################################################################
  //! @name Access functions
  //@{

  //! get the number of segments
  inline uint getNumSegment();

  //! get the segment histogram
  inline rutz::shared_ptr<Histogram> getSegmentHistogram();

  //! get the topological map
  inline rutz::shared_ptr<TopologicalMap> getTopologicalMap();

  //! get the visual landmark database
  inline rutz::shared_ptr<LandmarkDB> getLandmarkDB();

  //! get Point2D<int> location of the (segnum, ltrav) descriptor
  inline Point2D<int> getLocation(uint cseg, float ltrav);

  //! get Point2D<int> location (in float) of the (segnum, ltrav) descriptor
  inline Point2D<float> getLocationFloat(uint cseg, float ltrav);

  //@}

  // ######################################################################
  //! @name Compute functions
  //@{

  //! classify segment number from the input gist
  rutz::shared_ptr<Histogram> classifySegNum(Image<double> cgist);

  //! initialize a new landmarkDB to be build
  void build
  (std::vector<rutz::shared_ptr<VisualObject> > &inputVO,
   std::vector<Point2D<int> > &objOffset, uint fNum,
   rutz::shared_ptr<VisualObject> scene);

  //! build the landmarkDB from the current run
  //! keep calling this every frame
  void startBuild();

  //! finish building the landmarkDB - has all the ending procedure
  void finishBuild(std::string prefix, std::string path, uint rframe);

  //@}

private:

  // ######################################################################
  //! @name Compute functions (all the inner-working functions)
  //@{

  //! setup the segment recognizer
  void setupSegmentRecognizer(std::string segmentRecFName);

  //! save a landmarkDB to a file
  bool save(rutz::shared_ptr<LandmarkDB> landmarkDB,
            std::string fName, std::string envPrefix);

  //! fuse stored landmarks DB and just obtained DB when finished with a run
  rutz::shared_ptr<LandmarkDB> combineLandmarkDBs
  ( rutz::shared_ptr<LandmarkDB> lmks1,
    rutz::shared_ptr<LandmarkDB> lmks2);

  //@}

  //! the landmarkDB to store all the visual objects
  //! along the paths (edges)
  rutz::shared_ptr<LandmarkDB> itsLandmarkDB;

  //! the topological map for localization
  rutz::shared_ptr<TopologicalMap> itsTopologicalMap;

  //! segment recognizer: a feed-forward network
  rutz::shared_ptr<FeedForwardNetwork> itsSegmentRecognizer;
  FFNtrainInfo itsSegRecInfo;
  Image<double> itsPcaIcaMatrix;

  //! the current segment prediction
  rutz::shared_ptr<Histogram> itsSegmentHistogram;
  uint itsCurrentSegment;

  //! the display window to show results
  rutz::shared_ptr<XWinManaged> itsWin;

  //! the current landmarkDB being built
  rutz::shared_ptr<LandmarkDB> itsTempLandmarkDB;

};

// ######################################################################
// Implementation for Environment inline functions
// ######################################################################

inline rutz::shared_ptr<Histogram> Environment::getSegmentHistogram()
{
  return itsSegmentHistogram;
}

inline uint Environment::getNumSegment()
{
  return itsSegmentHistogram->getSize();
}

inline rutz::shared_ptr<TopologicalMap> Environment::getTopologicalMap()
{
  return itsTopologicalMap;
}

inline rutz::shared_ptr<LandmarkDB> Environment::getLandmarkDB()
{
  return itsLandmarkDB;
}

inline void Environment::setWindow(rutz::shared_ptr<XWinManaged> win)
{
  itsWin = win;
  itsLandmarkDB->setWindow(win);
}

inline Point2D<int> Environment::getLocation(uint cseg, float ltrav)
{
  return itsTopologicalMap->getLocation(cseg, ltrav);
}

inline Point2D<float> Environment::getLocationFloat(uint cseg, float ltrav)
{
  return itsTopologicalMap->getLocationFloat(cseg, ltrav);
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
