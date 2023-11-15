/*!@file SeaBee/PipeRecognizer.C find pipelines in image     */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/PipeRecognizer.H $
// $Id: PipeRecognizer.H 12962 2010-03-06 02:13:53Z irock $


#ifndef SEABEE_PIPE_RECOGNIZER_DEFINED
#define SEABEE_PIPE_RECOGNIZER_DEFINED

#ifdef HAVE_OPENCV

#include "SeaBee/VisionRecognizer.H"

#include "Image/Point2D.H"
#include "MBARI/Geometry2D.H"
#include "Image/OpenCVUtil.H"
#include "Image/ColorOps.H"

class PipeRecognizer:public VisionRecognizer
{
public:

  // ######################################################################
  //! @name Constructor, assigment and destructor
  //@{

  //! Constructor
  PipeRecognizer();

  //! Destructor
  ~PipeRecognizer();

  //@}

  // ######################################################################
  //! @name compute function
  //@{

  enum PipeRecognizeMethod { HOUGH, LINE_BEST_FIT, CONTOUR };

  //@}

  // ######################################################################
  //! @name compute function
  //@{

  std::vector<LineSegment2D> getPipeLocation
  (rutz::shared_ptr<Image<PixRGB <byte> > > colorSegmentedImage,
   rutz::shared_ptr<Image<PixRGB <byte> > > outputImage,
   PipeRecognizeMethod method);
   //@}


  std::vector<LineSegment2D> calculateHoughTransform
  (Image<byte>& colorSegmentedImage,
   rutz::shared_ptr<Image<PixRGB<byte> > > outputImage);

private:

  uint calculateLineBestFit
  (Image<byte>  &colorSegmentedImage,
   Image<PixRGB <byte> >  &outputImage,
   Point2D<int> &pipeCenter,
   double &pipeAngle);

  uint calculateContours
  (Image<byte>  &colorSegmentedImage,
   Image<PixRGB <byte> >  &outputImage,
   Point2D<int> &pipeCenter,
   double &pipeAngle);
};

#endif

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
