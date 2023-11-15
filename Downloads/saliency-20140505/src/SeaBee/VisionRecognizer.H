/*!@file SeaBee/VisionRecognizer.C */
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
// Primary maintainers for this file: Kevin Greene <kgreene@usc.edu> & Josh Villbrandt <josh.villbrandt@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/VisionRecognizer.H $
// $Id: VisionRecognizer.H 10794 2009-02-08 06:21:09Z itti $

#ifndef VISION_RECOGNIZER_DEFINED
#define VISION_RECOGNIZER_DEFINED

#ifdef HAVE_OPENCV

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Image/DrawOps.H"

#include "MBARI/Geometry2D.H"
#include "Image/OpenCVUtil.H"
#include "Image/ColorOps.H"
#include "GUI/XWinManaged.H"
#include "Image/CutPaste.H"

class VisionRecognizer
{
public:

  VisionRecognizer();

  std::vector<LineSegment2D> getHoughLines  ( IplImage cannyImage );

  CvSeq* getContours( IplImage* img );

  std::vector<LineSegment2D> pruneHoughLines( const std::vector<LineSegment2D> lineSegments );

  IplImage getCannyImage( Image<byte> colorSegmentedImage );

private:
  //  rutz::shared_ptr<XWinManaged> itsDispWin;

  double angle( CvPoint* pt1, CvPoint* pt2, CvPoint* pt0 );
};

#endif

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
