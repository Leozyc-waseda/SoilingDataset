/*!@file BeoSub/BeoSubPipe.C find pipe     */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubPipe.H $
// $Id: BeoSubPipe.H 8748 2007-09-06 06:33:19Z rjpeters $


#ifndef BEOSUB_BEOSUBPIPE_DEFINED
#define BEOSUB_BEOSUBPIPE_DEFINED

#include "GUI/XWinManaged.H"
#include "Image/Image.H"
#include "MBARI/Geometry2D.H"

#include <list>

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


class BeoSubPipe
{
public:

  // ######################################################################
  //! @name Constructor, assigment and destructor
  //@{

  //! Constructor
  BeoSubPipe();

  //! Destructor
  ~BeoSubPipe();

  //@}

  // ######################################################################
  //! @name Compute functions
  //@{

  float pipeOrientation
  (Image< PixRGB <byte> > &cameraImage,
   Image<PixRGB <byte> >  &outputImg);

  float getPipeDir
  (const std::vector<LineSegment2D> lines,
   const float thresh,
   uint &totaladjustedlinecount);

  std::vector<LineSegment2D> getPipes
  (const std::vector<LineSegment2D> lines
   );

  std::vector<LineSegment2D> getHoughLines  (Image<byte> &image,
                                             Image< PixRGB<byte> > &outputImage);
  //@}

private:

  int houghThreshold;               //!<
  int minThreshold;                 //!<
  int maxThreshold;
  float sigma;
  float tlow;
  float thigh;
  float linescale;
  int foundCount;
  uint fNum;

  std::list<float> angleBuff;

  float avgPipeAngle;
  float stdDevPipeAngle;

  uint stdDevAngleCount;

  //! window to display results
  rutz::shared_ptr<XWinManaged> itsWin;
  bool itsWinInitialized;
};

#endif


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
