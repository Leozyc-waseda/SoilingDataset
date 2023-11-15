/*!@file Script/MediaScript.H */

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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Script/MediaScript.H $
// $Id: MediaScript.H 11876 2009-10-22 15:53:06Z icore $
//

#ifndef SCRIPT_MEDIASCRIPT_H_DEFINED
#define SCRIPT_MEDIASCRIPT_H_DEFINED

#include "Media/FrameSeries.H"
#include "tcl/conversions.h"

class SimTime;
struct Tcl_Interp;

extern "C" int Frameseries_Init(Tcl_Interp* interp);
extern "C" int Inputframeseries_Init(Tcl_Interp* interp);
extern "C" int Outputframeseries_Init(Tcl_Interp* interp);
extern "C" int Siminputframeseries_Init(Tcl_Interp* interp);
extern "C" int Simoutputframeseries_Init(Tcl_Interp* interp);

namespace tcl
{
  tcl::obj aux_convert_from(FrameState s);
  FrameState aux_convert_to(Tcl_Obj* obj, FrameState*);

  tcl::obj aux_convert_from(SimTime s);
  SimTime aux_convert_to(Tcl_Obj* obj, SimTime*);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // SCRIPT_MEDIASCRIPT_H_DEFINED
