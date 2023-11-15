/*!@file Beobot/BeobotSensation.H sensory events for the Beobots */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotSensation.H $
// $Id: BeobotSensation.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef BEOBOTSENSATION_H_DEFINED
#define BEOBOTSENSATION_H_DEFINED

#include "Util/Assert.H"
#include "Util/Types.H"
#include "Image/Pixels.H"
#include "Image/Image.H"
#include "Image/Point2D.H"

//! this class contains all the sensory informations but not the action
/*! it is used to compute an action */
class BeobotSensation
{
public:
  //! no arg constructor
  inline BeobotSensation();

  //! constructor
  inline BeobotSensation(const Image< PixRGB<byte> >& quicklayout,
                         const Point2D<int>& supposedcrackcentroid );

  //! copy constructor
  BeobotSensation( const BeobotSensation& other);

  //! assignment constructor:
  BeobotSensation& operator=(const BeobotSensation& other);

  //! access function
  inline void getQuickLayout(Image< PixRGB<byte> >& im) const;

  //! access function
  inline void getCentroid( Point2D<int> & centroid ) const;

  //! set 'confused' to 'true'
  inline void setConfused( void ) { confused=true; };

  //! return 'confused'
  inline bool isConfused( void ) { return confused; };

  //! return 'initialized'
  inline bool isInitialized( void ) { return initialized; };

  //! destructor
  ~BeobotSensation();

private:
  bool initialized;
  bool confused;
  Image< PixRGB<byte> > quickLayout;
  Point2D<int> supposedTrackCentroid;

  // and at some time the state of the sensors ?
};

// ######################################################################
// ######################################################################
// ######################################################################
// INLINE FUNCTIONS:
// ######################################################################
// ######################################################################
// ######################################################################

// ######################################################################
BeobotSensation::BeobotSensation()
{ initialized = false; confused = false; }

// ######################################################################
BeobotSensation::BeobotSensation(const Image< PixRGB<byte> >& quicklayout,
                                 const Point2D<int>& supposedtrackcentroid )
{
  // note: don't be confused by capitalization :
  // quickLayout (attribute) != quicklayout (arg)
  // the "this->" are just for clarity...

  ASSERT( quicklayout.initialized() );
  this->quickLayout = quicklayout;
  this->supposedTrackCentroid = supposedtrackcentroid;

  this->initialized = true;
}

// ######################################################################
inline void BeobotSensation::getQuickLayout(Image< PixRGB<byte> >& im) const
{
  ASSERT( initialized );
  im = quickLayout;
}

// ######################################################################
inline void BeobotSensation::getCentroid( Point2D<int> & centroid ) const
{
  ASSERT( initialized );
  centroid = supposedTrackCentroid;
}

#endif


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
