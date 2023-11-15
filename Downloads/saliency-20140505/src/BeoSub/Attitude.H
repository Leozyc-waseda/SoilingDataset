/*!@file BeoSub/Attitude.H Attitude class for the submarine */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/Attitude.H $
// $Id: Attitude.H 5494 2005-09-09 01:12:07Z rjpeters $
//

#ifndef ATTITUDE_H_DEFINED
#define ATTITUDE_H_DEFINED

#include "Util/Angle.H"

//! A BeoSub three-dimensional attitude
/*! Conventions and units are as follows:
  heading: ]-180.0 .. 180.0] in degrees, 0.0 is North, 45.0 is NE, etc
           source: compass

  pitch:   ]-180.0 .. 180.0] in degrees, 0.0 is horizontal, 45.0 is ???
           source: compass

  roll:    ]-180.0 .. 180.0] in degrees, 0.0 is horisontal, 45.0 is ???
           source: compass

  depth:   in m, axis oriented Down, positive towards Down, zero at surface
*/
class Attitude
{
public:
  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Default Constructor
  /*! Intializes values to 0 */
  inline Attitude();

  //! Destructor
  inline ~Attitude();

  //@}

  // ######################################################################
  Angle heading, pitch, roll;       //!< Angular position from compass
  float depth;                      //!< depth from pressure sensor
  double compassTime, pressureTime; //!< Times last measures received, in s
};


// ######################################################################
// ########## Inlined methods:
// ######################################################################
inline Attitude::Attitude() :
  heading(0), pitch(0), roll(0), depth(0), compassTime(0.0), pressureTime(0.0)
{ }

// ######################################################################
inline Attitude::~Attitude()
{ }

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
