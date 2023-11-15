/*!@file Parallel/pvisionTCP-defs.H Definitions for pvisionTCP & pvisionTCPmaster */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Parallel/pvisionTCP-defs.H $
// $Id: pvisionTCP-defs.H 13086 2010-03-30 03:24:43Z beobot $
//

/*! Various definitions for the parallel saliency code.
*/

// the following beowulf action definitions are used by pvisionTCP:
#define BEO_RETINA     1
#define BEO_WINNER     2
#define BEO_LUMINANCE  3
#define BEO_REDGREEN   4
#define BEO_BLUEYELLOW 5

#define BEO_ORI0       6
#define BEO_ORI15      7
#define BEO_ORI30      8
#define BEO_ORI45      9
#define BEO_ORI60      10
#define BEO_ORI75      11
#define BEO_ORI90      12
#define BEO_ORI105     13
#define BEO_ORI120     14
#define BEO_ORI135     15
#define BEO_ORI150     16
#define BEO_ORI165     17

#define BEO_CMAP       18
#define BEO_FLICKER    19

//! Number of conspicuity maps for pvisionTCP
#define NBCMAP 7

//! Number of conspicuity maps fo pvisionTCP2
#define NBCMAP2 8

//! Number of conspicuity maps fo pvisionTCP4
#define NBCMAP4 16

//! Node offset for the two processing streams:
#define POFFSET 8

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
