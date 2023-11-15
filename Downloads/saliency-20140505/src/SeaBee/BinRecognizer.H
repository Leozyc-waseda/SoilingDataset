/*!@file SeaBee/BinRecognizer.C find bin     */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/BinRecognizer.H $
// $Id: BinRecognizer.H 10794 2009-02-08 06:21:09Z itti $


#ifndef SEABEE_BIN_RECOGNIZER_DEFINED
#define SEABEE_BIN_RECOGNIZER_DEFINED

#include "SeaBee/VisionRecognizer.H"

#include "Image/Point2D.H"
#include "MBARI/Geometry2D.H"
#include "Image/OpenCVUtil.H"
#include "Image/ColorOps.H"

class BinRecognizer : public VisionRecognizer
{
public:

  // ######################################################################
  //! @name Constructor, assigment and destructor
  //@{

  //! Constructor
  BinRecognizer();

  //! Destructor
  ~BinRecognizer();

  //@}

  // ######################################################################
  //! @name compute function
  //@{

  enum BinRecognizeMethod { HOUGH, LINE_BEST_FIT, CONTOUR };

  //@}

  // ######################################################################
  //! @name compute function
  //@{

  uint getBinLocation(rutz::shared_ptr<Image<PixRGB <byte> > > image,
                       rutz::shared_ptr<Image<PixRGB <byte> > > outputImage,
                       BinRecognizeMethod method,
                      rutz::shared_ptr<Point2D<int> > binCenter,
                      uint &staleCount
                       );
   //@}



private:
  void getBinCenter(uint &staleCount,
                    rutz::shared_ptr<Point2D<int> > binCenter,
                    CvSeq* squares);
  Image<PixRGB<byte> > drawSquares( IplImage* in, CvSeq* squares );

};

#endif


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
