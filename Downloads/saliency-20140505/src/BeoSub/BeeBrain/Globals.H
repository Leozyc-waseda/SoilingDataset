/*!@file BeoSub/BeeBrain/Globals.H
  global declaration for AUVSI 2007                                     */
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/Globals.H $
// $Id: Globals.H 8623 2007-07-25 17:57:51Z rjpeters $
//

#ifndef __SEABEE_II_GLOBALS_H__
#define __SEABEE_II_GLOBALS_H__

enum DataTypes { POSITION, ORIENTATION, FREQUENCY, DISTANCE, MASS};
enum NumMoves { ONCE, REPEAT };
enum SearchTypes { SWEEP, SPIN };

#define AUVSI07_COMPETITION_TIME 15*60*1000000

// slave program names
#define COM_B_NODE   0

// signals
#define INIT_COMM                       10000
#define INIT_DONE                       10001

#define SEARCH_OCEAN_OBJECT             20000
#define SEARCH_OCEAN_OBJECT_RES         20001
#define OCEAN_OBJECT_STATUS             20002

#define ABORT                           90000

enum CommandType { SEARCH_OCEAN_OBJECT_CMD = 12212,
                   STOP_SEARCH_OCEAN_OBJECT_CMD = 21121 };


struct motorCommand
{
  // values should all be [ -1.0 ... 1.0 ]

  float l_horz_thrust;
  float r_horz_thrust;

  float l_vert_thrust;
  float r_vert_thrust;

  float b_vert_thrust;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

