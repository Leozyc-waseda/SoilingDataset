/*!@file BeoSub/CannyModel.H Simple shape models */

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
// Primary maintainer for this file: Zack Gossman <gossman@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/CannyModel.H $
// $Id: CannyModel.H 6990 2006-08-11 18:13:51Z rjpeters $
//

#ifndef CANNYMODEL_H_DEFINED
#define CANNYMODEL_H_DEFINED

class XWindow;
#include "Image/Pixels.H"
#include "Image/Image.H"
#include "rutz/shared_ptr.h"

/*************************************************************************
 * NOTES:
 *
 * -Poblems with threshold values:
 *     1) easier to get a "match" for smaller models, so the threshold should take dimensions into account (i.e. larger threshhold needed for larger models)
       2) The threshhold, as is, severely limits matching of partly off-image shapes. PERHAPS this could be solved by making all off-image points have a certain distance match. OR, we could change the code to IGNORE off-image points, up to a certain percentage (maybe half)
 *
 *
 ************************************************************************/

//! The base class for shape models
/*! This base class defines a basic interface by which an shape optimizer
may mathematically represent a shape and get a measurement of how well
the current shape matches with a Canny edge map. */
class ShapeModel {
public:
  //! Constructor
  /*! Constructor
    @param ndims number of dimensions in the model
    @param thresh threshold for a shape match
    @param debug if true, open an XWindow and show debug drawings. */
  ShapeModel(const int ndims, const double thresh, double* dims, const bool debug = false);

  //! Destructor
  virtual ~ShapeModel();

  //! Compute distance between model (defined by p) and distance map
  /*! Programmer's note: calcDist() is a wrapper around getDist()
    which does the actual distance computation. Do not overload
    calcDist(), overload getDist() instead. */
  double calcDist(double p[], const Image<float>& distMap) const;

  //! Get the distance threshold
  double getThreshold() const;

  //! Get the number of dimensions
  int getNumDims() const;

  //! Get the dimension vector
  double* getDimensions() const;

  //!Set the dimension vector
  void setDimensions(double* in);

protected:
  //! get distance value at one location, with optional display
  float getDistVal(const double x, const double y, const Image<float>& distMap,
                   Image< PixRGB<byte> >& xdisp) const;


  //! Core of the distance computation
  /*! This is a pure virtual function, derived classes need to
    implement it. */
  virtual float getDist(double p[], const Image<float>& distMap,
                        Image< PixRGB<byte> >& xdisp) const = 0;

private:
  int itsNumDims;               //!< number of dimensions describing shape
  double itsThreshold;          //!< distance threshold for match
  double* itsDimensions;        //!< dimesion vector for the shape model
  bool itsDebugMode;            //!< do we want debug displays?
  rutz::shared_ptr<XWindow> itsWindow; //!< optional window for display

};


// ######################################################################
//! A Rectangle ShapeModel
class RectangleShape : public ShapeModel {
public:
  //! Constructor
  RectangleShape(const double thresh, double* dims, const bool debug = false);

  //! Destructor
  virtual ~RectangleShape();

protected:
  //! Core of the distance computation
  virtual float getDist(double p[], const Image<float>& distMap,
                        Image< PixRGB<byte> >& xdisp) const;
};


// ######################################################################
//! A Square ShapeModel

class SquareShape : public ShapeModel {
public:
  //! Constructor
  SquareShape(const double thresh, double* dims, const bool debug = false);

  //! Destructor
  virtual ~SquareShape();

protected:
  //! Core of the distance computation
  virtual float getDist(double p[], const Image<float>& distMap,
                        Image< PixRGB<byte> >& xdisp) const;
};


// ######################################################################
//! An Octagon ShapeModel

class OctagonShape : public ShapeModel {
public:
  //! Constructor
  OctagonShape(const double thresh, double* dims, const bool debug = false);

  //! Destructor
  virtual ~OctagonShape();

protected:
  //! Core of the distance computation
  virtual float getDist(double p[], const Image<float>& distMap,
                        Image< PixRGB<byte> >& xdisp) const;
};


// ######################################################################
//! A Circle ShapeModel

class CircleShape : public ShapeModel {
public:
  //! Constructor
  CircleShape(const double thresh, double* dims, const bool debug = false);

  //! Destructor
  virtual ~CircleShape();

protected:
  //! Core of the distance computation
  virtual float getDist(double p[], const Image<float>& distMap,
                        Image< PixRGB<byte> >& xdisp) const;
};


// ######################################################################
//! A Parallel-lines ShapeModel
class ParallelShape : public ShapeModel {
public:
  //! Constructor
  ParallelShape(const double thresh, double* dims, const bool debug = false);

  //! Destructor
  virtual ~ParallelShape();

protected:
  //! Core of the distance computation
  virtual float getDist(double p[], const Image<float>& distMap,
                        Image< PixRGB<byte> >& xdisp) const;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
