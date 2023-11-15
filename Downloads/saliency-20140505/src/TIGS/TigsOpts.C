/*!@file TIGS/TigsOpts.C Shared command-line options for TIGS */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/TigsOpts.C $
// $Id: TigsOpts.C 6701 2006-06-09 03:17:43Z rjpeters $
//

#ifndef TIGS_TIGSOPTS_C_DEFINED
#define TIGS_TIGSOPTS_C_DEFINED

#include "TIGS/TigsOpts.H"

#include "Component/ModelOptionDef.H"

const ModelOptionCateg MOC_TIGS = {
  MOC_SORTPRI_3, "TIGS (Topdown Information Guidance System) Options" };

// Used by: various
const ModelOptionDef OPT_XptSavePrefix =
  { MODOPT_ARG_STRING, "XptSavePrefix", &MOC_TIGS, OPTEXP_CORE,
    "Filename stem name for the psychophysics experiment being tested",
    "xpt-save-prefix", '\0', "<string>", "" };

const ModelOptionDef OPT_FxSaveIllustrations =
  { MODOPT_FLAG, "FxSaveIllustrations", &MOC_TIGS, OPTEXP_CORE,
    "Whether to save fancy illustrations from the feature extractors",
    "fx-save-illustrations", '\0', "", "false" };

const ModelOptionDef OPT_FxSaveRawMaps =
  { MODOPT_FLAG, "FxSaveRawMaps", &MOC_TIGS, OPTEXP_CORE,
    "Whether to save raw maps from the feature extractors",
    "fx-save-raw-maps", '\0', "", "false" };

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_TIGSOPTS_C_DEFINED
