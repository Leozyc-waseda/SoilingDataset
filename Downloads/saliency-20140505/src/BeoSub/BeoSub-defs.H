/*!@file BeoSub/BeoSub-defs.H Defaults for the BeoSub */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSub-defs.H $
// $Id: BeoSub-defs.H 5056 2005-07-27 22:23:10Z beobot $
//

#ifndef BEOSUB_DEFS_H_DEFINED
#define BEOSUB_DEFS_H_DEFINED

//NOTE: Perhaps color defs should go in here?

//! Serial port for the HMR3300 compass:
#define BEOSUB_SERDEV_HMR3300 "/dev/ttyUSB0"

//! Serial port to use for left BeoChip
#define BEOSUB_LEFTBEOCHIP_SERDEV "/dev/ttyS0"

//! Serial port to use for right BeoChip
#define BEOSUB_RIGHTBEOCHIP_SERDEV "/dev/ttyS1"

//NOTE: Why 160*120?

//! Default grab dims
#define BEOSUB_GRAB_W 160
#define BEOSUB_GRAB_H 120

//! Camera numbers
#define BEOSUB_CAM_FRONT 1
#define BEOSUB_CAM_DOWN 0
#define BEOSUB_CAM_UP 2


#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
