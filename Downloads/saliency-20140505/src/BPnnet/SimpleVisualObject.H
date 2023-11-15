/*!@file SimpleVisualObject.H Header file for visual objects to be recognized */

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
// Primary maintainer for this file: Philip Williams <plw@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BPnnet/SimpleVisualObject.H $
// $Id: SimpleVisualObject.H 6393 2006-03-26 00:57:36Z rjpeters $
//

#ifndef VISUALOBJECT_H_DEFINED
#define VISUALOBJECT_H_DEFINED

//! describes name and attributes of a visual object
class SimpleVisualObject {
public:
  //! Constructor
  SimpleVisualObject();

  //! Constructor
  SimpleVisualObject(const char* n);

  //! Copy constructor
  SimpleVisualObject(const SimpleVisualObject& vo);

  //! initialization
  void init(const char *n);

  //! free memory
  void freeMem();

  //! Destructor
  ~SimpleVisualObject();

  //! Assignment constructor
  SimpleVisualObject& operator=(const SimpleVisualObject& vo);

  //! name getter
  const char* getName() const;

private:
  char* name;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
