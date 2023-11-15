/*!@file RCBot/SceneRec.H  Recognize scenes                             */
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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/SceneRec.H $
// $Id: SceneRec.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef SCENEREC_H
#define SCENEREC_H

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "SIFT/ScaleSpace.H"
#include "SIFT/VisualObject.H"
#include "SIFT/Keypoint.H"
#include "SIFT/VisualObjectDB.H"
#include "Image/Image.H"
#include "Image/Pixels.H"

#include <pthread.h>

//! Class to recognize scenes and get landmark positions
class SceneRec : public ModelComponent
{
public:

  SceneRec(OptionManager &mgr,
           const std::string& descrName = "Scene Recognition",
           const std::string& tagName = "SceneRec");

  //! Destructor
  ~SceneRec();

  //! Setup up with image to be processed
  void newInput(const Image< PixRGB<byte> > img);

  //! Is the last given input done processing?
  bool outputReady();

  //! Get the feature point and return the current leg.
  //Used for landmark navigation
  short getLandmarkLoc(Point2D<int> &loc);

  //! Main process for computing locations
  void computeLocation();

  //! Train the system on a given feature location
  void trainFeature(Image<PixRGB<byte> > &img, Point2D<int> loc,
                    Dims window, short leg);

  //! Load the current and the next vdb from position 'leg'
  void loadVisualDB(short leg);

protected:
  virtual void start1();
  virtual void stop2();

private:
  Image< PixRGB<byte> > workImg;     // current color image
  short currentLeg;                  // the current leg in the path
  int objNum;                        // for naming the objects

  VisualObjectDB *currentLegVdb;     // the current landmark we are looking for
  VisualObjectDB *nextLegVdb;        // the next landmark we are looking for

  Point2D<int> landmarkLoc;               // the landmarkLoc location we are after

  // thread stuff
  pthread_t *worker;
  pthread_mutex_t jobLock, locLock, vdbLock;
  pthread_cond_t jobCond;

  bool jobDone;

  // get the information for found landmarks
  void getLandmarkInfo(float &scale, Point2D<int> &loc, unsigned int nmatch,
                       std::vector< rutz::shared_ptr<VisualObjectMatch> > &matches,
                       Image<PixRGB<byte> > &img);
  Dims winsize;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
