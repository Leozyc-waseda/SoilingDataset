/*!@file SeaBee/CrossRecognizer.H find pipe     */
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
// Primary maintainer for this file: Kevin Greene <kgreene@usc.edu>  && Eliezer Payzer <epayzer@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/CrossRecognizer.H $
// $Id: CrossRecognizer.H 10794 2009-02-08 06:21:09Z itti $


#ifndef CROSS_RECOGNIZER_DEFINED
#define CROSS_RECOGNIZER_DEFINED

#include "GUI/XWinManaged.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Image/DrawOps.H"

#include "MBARI/Geometry2D.H"
#include "Image/OpenCVUtil.H"
#include "Image/ColorOps.H"

#define LINESCALE 80

class CrossRecognizer
{
public:

  // ######################################################################
  //! @name Constructor, assigment and destructor
  //@{

  //! Constructor
  CrossRecognizer();

  //! Destructor
  ~CrossRecognizer();

  //@}

  // ######################################################################
  //! @name compute function
  //@{

  enum CrossRecognizeMethod { HOUGH, SOMETHING, SOMETHING_ELSE };

  //@}

  // ######################################################################
  //! @name compute function
  //@{

  void getCrossLocation       (rutz::shared_ptr<Image<byte> > colorSegmentedImage,
                               rutz::shared_ptr<Image<PixRGB <byte> > > outputImage,
                               CrossRecognizeMethod method,
                               rutz::shared_ptr<Point2D<int> > crossCenterPoint,
                               rutz::shared_ptr<float> crossAngle,
                               rutz::shared_ptr<uint> stalePointCount
                               );
  //@}

private:
  void calculateHoughTransform(rutz::shared_ptr<Image<byte> > colorSegmentedImage,
                               rutz::shared_ptr<Image<PixRGB<byte> > > outputImage,
                               rutz::shared_ptr<Point2D<int> > crossCenterPoint,
                               rutz::shared_ptr<float> crossAngle,
                               rutz::shared_ptr<uint> stalePointCount
                               );

   Point2D<int> getCrossCenter(const std::vector<LineSegment2D> lines,
                                            std::vector<LineSegment2D> &centerPointLines,
                                            rutz::shared_ptr<uint> stalePointCount
                                            );

  std::vector<LineSegment2D> getHoughLines(rutz::shared_ptr< Image< byte > > colorSegmentedImage,
                                           rutz::shared_ptr< Image< PixRGB <byte> > > outputImage);

  std::vector<LineSegment2D> pruneLines (const std::vector<LineSegment2D> lineSegments);

  float getCrossDir(const std::vector<LineSegment2D> lines);

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
