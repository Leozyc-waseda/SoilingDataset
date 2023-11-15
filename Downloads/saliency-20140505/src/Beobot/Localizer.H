/*!@file Beobot/Localizer.H given a map and a stream of information it
  tries to localize.                                                    */
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
// ///////////////////////////////////////////////////////////////////////
//
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/Localizer.H $
// $Id: Localizer.H 7392 2006-11-02 21:14:09Z rjpeters $
//
//////////////////////////////////////////////////////////////////////////
//
//

#ifndef BEOBOT_LOCALIZER_DEFINED
#define BEOBOT_LOCALIZER_DEFINED

#include "Beobot/TopologicalMap.H"

//! Localizer: given a map and a stream of information it
//!            tries to localize.

/*  Start with topological map and try to do some bayesian
 */
class Localizer
{
public:

  // ######################################################################
  //! @name Constructor, assigment and destructor
  //@{

  //! Constructor with map input
  /*! @param map map of the environment
  */
  Localizer();

  //! Destructor
  ~Localizer();

  // ######################################################################
  //! @name Access functions
  //@{

  //@}

  // ######################################################################
  //! @name Compute functions
  //@{

  //! input: observation
  void obsInput();

  //! input: action
  void actInput();

  //! output: location
  void getLocation();

  //! go to this location
  void goTo();

  //@}

private:

  //! beliefs
  //! storage of all the database of evidences that we are at a segment
  //! map is bare it only provide path generation given 2 points


  // ######################################################################
  //! @name Compute functions (all the inner-working functions)
  //@{

  //! compute the new belief
  void compute();

  //@}
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
