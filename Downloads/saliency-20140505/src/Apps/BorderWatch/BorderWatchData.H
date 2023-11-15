/*!@file Apps/BorderWatch/BorderWatchData.H */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
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
// Primary maintainer for this file: Laurent Itti
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Apps/BorderWatch/BorderWatchData.H $
// $Id: BorderWatchData.H 13059 2010-03-26 08:14:32Z itti $
//

#ifndef APPS_BORDERWATCH_BORDERWATCHDATA_H_DEFINED
#define APPS_BORDERWATCH_BORDERWATCHDATA_H_DEFINED

#include <iosfwd>
#include "Image/Point2D.H"
#include <string>

//! Log data from the BorderWatch application, used by BorderWatchGui
struct BorderWatchData {
  uint oframe;           //!< Output frame number
  uint iframe;           //!< Input frame number
  float score;           //!< Overall combined score

  Point2D<int> salpoint; //!< Coordinates of most salient point

  float saliency;        //!< Local saliency (or surprise) component of score
  float energy;          //!< Energy component of score
  float uniqueness;      //!< Uniqueness component of score
  float entropy;         //!< Entropy component of score
  float rand;            //!< Rand component of score
  float KLsurprise;      //!< KL surfprise component of score
  float MSDsurprise;     //!< MSD surprise component of score

  std::string itime;     //!< Time of the input frame
};

//! Write BorderWatchData to an ostream
std::ostream& operator<<(std::ostream& os, const BorderWatchData& bwd);

//! Read BorderWatchData from an istream
std::istream& operator>>(std::istream& is, BorderWatchData& bwd);


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPS_BORDERWATCH_BORDERWATCHDATA_H_DEFINED
