/*!@file BeoSub/BeoSubCross.C find pipe     */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubCross.H $
// $Id: BeoSubCross.H 9412 2008-03-10 23:10:15Z farhan $


#ifndef BEOSUB_BEOSUBCROSS_DEFINED
#define BEOSUB_BEOSUBCROSS_DEFINED

#include "GUI/XWinManaged.H"
#include "Image/Image.H"
#include "MBARI/Geometry2D.H"
#include "VFAT/segmentImageTrackMC.H"

#include <list>

//#include <queue>
//END CAMERA STUFF

//canny
#define BOOSTBLURFACTOR 90.0
#define FREE_ARG char*
#define FILLBLEED 4
#define INITIAL_TEMPERATURE 30.0
#define FINAL_TEMPERATURE 0.5
#define ALPHA 0.98
#define STEPS_PER_CHANGE 1000
#define BIN_ANGLE 0.588001425


class BeoSubCross
{
public:

  // ######################################################################
  //! @name Constructor, assigment and destructor
  //@{

  //! Constructor
  BeoSubCross();

  //! Destructor
  ~BeoSubCross();

  //@}

  // ######################################################################
  //! @name Compute functions
  //@{

  void setupOrangeTracker();

  int getOrangeMass(Image< PixRGB<byte> > image,
                    Image< PixRGB<byte> >& display);

  std::vector<LineSegment2D> pruneLines(std::vector<LineSegment2D> lines);

  std::vector<LineSegment2D> getHoughLines  (Image< PixRGB<byte> > &image,
                                             Image< PixRGB<byte> > &outputImage);

  float getCrossDir  (const std::vector<LineSegment2D> lines);

  Point2D<int> getCrossCenter  (const std::vector<LineSegment2D> lines,
                           std::vector<LineSegment2D>& centerPointLines,
                           uint &stalePointCount);
  //@}

  segmentImageTrackMC<float,unsigned int,4> *segmenter;

  readConfig colorConf;

  std::vector<float> color;
  std::vector<float> norm;
  std::vector<float> std;
  std::vector<float> adapt;
  std::vector<float> upperBound;
  std::vector<float> lowerBound;

private:

  std::list<Point2D<int> > centerPointBuff;
  std::list<float> angleBuff;

  int houghThreshold;               //!<
  int minThreshold;                 //!<
  int maxThreshold;
  float sigma;
  float tlow;
  float thigh;
  float linescale;
  int foundCount;
  int avgCrossCenterX;
  int avgCrossCenterY;
  float avgCrossAngle;
  float stdDevCrossX;
  float stdDevCrossY;
  float stdDevCrossAngle;

  uint stdDevAngleCount;
  uint stdDevCenterCount;

  bool itsSetupOrangeTracker;
};

#endif


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
