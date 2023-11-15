/*!@file Beobot/beobot-GSnav-def.H
  Defined values for robot navigation using saliency and gist.
  Run beobot-GSnav-master at CPU_A to run Gist-Saliency model
  Run beobot-GSnav        at CPU_B to run SIFT object recognition       */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/beobot-GSnav-def.H $
// $Id: beobot-GSnav-def.H 12552 2010-01-13 23:05:00Z siagian $
//
/////////////////////////////////////////////////////////////////////////
//
// This is an on-going project for biologically-plausible
// mobile-robotics navigation.
// It accepts any inputs: video  clip <input.mpg>, camera feed, frames.
//
// The system uses Gist to recognize places and saliency
// to get better localization within the place
// The program also is made to be streamline for fast processing using
// parallel computation. That is the V1 features of different channels are
// computed in parallel
//
// Currently it is able to recognize places through the use of gist features.
// The place classifier uses a neural networks,
// passed in a form of <input_train.txt> -
// the same file is used in the training phase by train-FFN.C.
//
// Related files of interest: GistEstimator.C (and .H) and
// GistEstimatorConfigurator.C (and .H) used by Brain.C to compute
// the gist features.
// test-Gist.C uses GistEstimator to extract gist features from an image.
//
// In parallel we use saliency to get a better spatial resolution
// as well as better place accuracy. The saliency model is used to obtain
// salient locations. We then use ShapeEstimator algorithm to segment out
// the sub-region to get a landmark. Using SIFT we can identify the object,
// create a database, etc.
//
// for localization, path planning we perform landmark-hopping
// to get to the final destination

#ifndef BEOBOT_BEOBOT_GSNAV_DEF_DEFINED
#define BEOBOT_BEOBOT_GSNAV_DEF_DEFINED

#define VENTRAL_NODE         0
#define DORSAL_NODE          1

#define INIT_COMM            10000
#define INIT_DONE            10001
#define SEARCH_LM            10002
#define SEARCH_LM_RES        10003
#define TRACK_LM             10004
#define TRACK_LM_RES         10005
#define ABORT                10006

#define TRAIN_MODE           20000
#define TRAIN_X_MODE         20001
#define TEST_MODE            20002

#define SEARCH_NOT_DONE      30000
#define LOCALIZED            30001
#define NOT_LOCALIZED        30002

#define BC_NO_SIGNAL         40000
#define BC_QUIT_SIGNAL       40001

#define FILE_INPUT           50000
#define CAMERA_INPUT         50001

//! percentage of images that a landmark has to be matches
//! so that we can combine it with the other landmark
#define NMATCH_THRESHOLD    0.25f

//! number of frames over which frame rate is computed
#define NAVG 20

//! Factor to display the sm values as greyscale:
#define SMFAC 0.05F

//! amount of time alloted to search through landmark db
#define SEARCH_TIME_LIMIT    15 //!< 15 frames

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
