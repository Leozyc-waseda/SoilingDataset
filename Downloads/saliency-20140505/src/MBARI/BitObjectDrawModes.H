/*!@file MBARI/BitObjectDrawModes.H definition of the different modes of
  drawing a BitObject into a frame */

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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/BitObjectDrawModes.H $
// $Id: BitObjectDrawModes.H 8827 2007-10-12 17:45:29Z rjpeters $
//

#ifndef BITOBJECTDRAWMODES_H_DEFINED
#define BITOBJECTDRAWMODES_H_DEFINED

#include <string>

//! Possible BitObject draw modes
enum BitObjectDrawMode {
  BODMnone    = 0,   //! don't draw the BitObject
  BODMshape   = 1,   //! draw the shape of the BitObject
  BODMoutline = 2,   //! draw the outline of the BitObject
  BODMbbox    = 3    //! draw the bounding box of the BitObject
  // if you add a new mode here, also update the names in the function below!
};

//! number of known BitObject draw modes:
#define NBBITOBJECTDRAWMODES 4

//! Returns name of given BitObject draw mode
inline const char* bitObjectDrawModeName(const BitObjectDrawMode p)
{
  static const char n[NBBITOBJECTDRAWMODES][15] = {
    "None", "Shape", "Outline", "BoundingBox" };
  return n[int(p)];
}


//! BitObjectDrawMode overload
/*! Format is "name" as defined in BitObjectDrawModes.H */
std::string convertToString(const BitObjectDrawMode val);
//! BitObjectDrawMode overload
/*! Format is "name" as defined in BitObjectDrawModes.H */
void convertFromString(const std::string& str, BitObjectDrawMode& val);

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
