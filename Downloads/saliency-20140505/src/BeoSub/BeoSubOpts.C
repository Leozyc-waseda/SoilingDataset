/*!@file BeoSub/BeoSubOpts.C Command-line options for BeoSub */

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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubOpts.C $
// $Id: BeoSubOpts.C 5969 2005-11-21 23:17:54Z rjpeters $
//

#include "BeoSub/BeoSubOpts.H"
#include "Component/ModelOptionDef.H"

const ModelOptionCateg MOC_BEOSUB = {
  MOC_SORTPRI_4, "BeoSub Related Options" };

// Format here is:
//
// { MODOPT_TYPE, "name", &MOC_CATEG, OPTEXP_CORE,
//   "description of what option does",
//   "long option name", 'short option name', "valid values", "default value" }
//

// alternatively, for MODOPT_ALIAS option types, format is:
//
// { MODOPT_ALIAS, "", &MOC_ALIAS, OPTEXP_CORE,
//   "description of what alias does",
//   "long option name", 'short option name', "", "list of options" }
//

// NOTE: do not change the default value of any existing option unless
// you really know what you are doing!  Many components will determine
// their default behavior from that default value, so you may break
// lots of executables if you change it.

// #################### BeoSub base class options:
const ModelOptionDef OPT_FrontVODBfname =
  { MODOPT_ARG_STRING, "FrontVODBfname", &MOC_BEOSUB, OPTEXP_CORE,
    "File name to use for the VisualObjectDB associated with the front camera",
    "front-vodb-fname", '\0', "<filename>", "/home/tmp/u/beosub/front.vdb" };

const ModelOptionDef OPT_DownVODBfname =
  { MODOPT_ARG_STRING, "DownVODBfname", &MOC_BEOSUB, OPTEXP_CORE,
    "File name to use for the VisualObjectDB associated with the down camera",
    "down-vodb-fname", '\0', "<filename>", "/home/tmp/u/beosub/down.vdb" };

const ModelOptionDef OPT_UpVODBfname =
  { MODOPT_ARG_STRING, "UpVODBfname", &MOC_BEOSUB, OPTEXP_CORE,
    "File name to use for the VisualObjectDB associated with the up camera",
    "up-vodb-fname", '\0', "<filename>", "/home/tmp/u/beosub/up.vdb" };

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
