/*!@file Beobot/beobot-defs.H Basic definitions for the beobots */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/beobot-defs.H $
// $Id: beobot-defs.H 4663 2005-06-23 17:47:28Z rjpeters $
//

#ifndef BEOBOT_DEFS_H_DEFINED
#define BEOBOT_DEFS_H_DEFINED

#define IMGWIDTH 160
#define IMGHEIGHT 120

// basic constants for low-level image processing:
#define SMLEVEL   2
#define LEVEL_MIN 0
#define LEVEL_MAX 2
#define DELTA_MIN 3
#define DELTA_MAX 4
#define NBORIENTS 4

#define JETLEVEL  3
#define JETDEPTH  3
#define NBNEIGH   24

#define NBCMAP    8
#define NBJMAP    NBCMAP

// The following beowulf actions are defined on the beobots:
#define BEO_LUMFLICK   1
#define BEO_REDGREEN   2
#define BEO_BLUEYELLOW 3
#define BEO_ORI0_45    4
#define BEO_ORI90_135  5
#define BEO_CMAP       6

#define BEO_FMAP_RG    100
#define BEO_FMAP_BY    101
#define BEO_FMAP_I     102
#define BEO_FMAP_F     103
#define BEO_FMAP_O0    104
#define BEO_FMAP_O45   105
#define BEO_FMAP_O90   106
#define BEO_FMAP_O135  107

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
