/*!@file Beobot/Map.H abstract (geographical) map class for localization */

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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/Map.H $
// $Id: Map.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef BEOBOT_MAP_H_DEFINED
#define BEOBOT_MAP_H_DEFINED

#include "Image/Image.H"
#include "Image/Pixels.H"

//! Abstract base (2D) map class;
//! see TopologicalMap for the standard implementation.
class Map
{
public:

  //! Constructor: generate a blank map
  Map();

  //! Constructor: retrieve the map from a file
  Map(std::string fileName);

  //! Destructor
  virtual ~Map();

  //! read a map from a file
  virtual bool read(std::string fileName) = 0;

  //! write a map to a file
  virtual bool write(std::string fileName) = 0;

  //! returns an image representation of the map
  virtual Image<PixRGB<byte> > getMapImage(uint w, uint h) = 0;

  //! different maps will have different ways
  //! to generate a path
  //! so, there's no set function
  // virtual std::vector<Point2D<int> > getPath(Point2D<int> a, Point2D<int> b);

  //! edits are implementation specific
  //! between map translation should not be done by a class function
};

#endif // BEOBOT_MAP_H_DEFINED

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
