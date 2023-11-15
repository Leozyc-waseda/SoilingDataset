/*!@file Script/invt.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Script/invt.C $
// $Id: invt.C 11877 2009-10-22 16:01:03Z icore $
//

#ifndef SCRIPT_INVT_C_DEFINED
#define SCRIPT_INVT_C_DEFINED

#include "rutz/prof.h"
#include "tcl/scriptapp.h"

//
// Forward declarations of package init procedures
//

#include "Script/ImageScript.H"
#include "Script/ImageTkScript.H"
#include "Script/MediaScript.H"
#include "Script/ModelScript.H"
#include "Script/NeuroScript.H"
#include "tcl/tclpkg-dlist.h"
#include "tcl/tclpkg-gtrace.h"
#include "tcl/tclpkg-log.h"
#include "tcl/tclpkg-misc.h"
#include "tcl/tclpkg-obj.h"

const tcl::package_info INVT_PKGS[] =
  {

    { "Bimage",              Bimage_Init,              "4.1", false },
    { "Brain",               Brain_Init,               "4.1", false },
    { "Cbimage",             Cbimage_Init,             "4.1", false },
    { "Cfimage",             Cfimage_Init,             "4.1", false },
    { "Dlist",               Dlist_Init,               "4.11876", false },
    { "Fimage",              Fimage_Init,              "4.1", false },
    { "Frameseries",         Frameseries_Init,         "4.1", false },
    { "Gtrace",              Gtrace_Init,              "4.11876", false },
    { "Imagetk",             Imagetk_Init,             "4.1", true  },
    { "Inputframeseries",    Inputframeseries_Init,    "4.1", false },
    { "Log",                 Log_Init,                 "4.11876", false },
    { "Misc",                Misc_Init,                "4.11876", false },
    { "Modelcomponent",      Modelcomponent_Init,      "4.1", false },
    { "Modelmanager",        Modelmanager_Init,        "4.1", false },
    { "Obj",                 Obj_Init,                 "4.11876", false },
    { "Objectdb",            Objectdb_Init,            "4.11876", false },
    { "Outputframeseries",   Outputframeseries_Init,   "4.1", false },
    { "Prof",                Prof_Init,                "4.11876", false },
    { "Raster",              Raster_Init,              "4.1", false },
    { "Simeventqueue",       Simeventqueue_Init,       "4.1", false },
    { "Siminputframeseries", Siminputframeseries_Init, "4.1", false },
    { "Simmodule",           Simmodule_Init,           "4.1", false },
    { "Simoutputframeseries",Simoutputframeseries_Init,"4.1", false },
    { "Stdbrain",            Stdbrain_Init,            "4.1", false },
    { "Winnertakeall",       Winnertakeall_Init,       "4.1", false },
    { "Winnertakeallstd",    Winnertakeallstd_Init,    "4.1", false },

    // WARNING! Keep this entry last
    { 0, 0, 0, 0 }
  };

///////////////////////////////////////////////////////////////////////
//
// main()
//
///////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  GVX_SCRIPT_PROG_BEGIN(app, "iNVT", argc, argv);

  // since our interpreter will catch exceptions and print their
  // what() error messages, we don't need LFATAL() to duplicate its
  // messages to stderr:
  LOG_FLAGS &= (~LFATAL_XTRA_NOISY);

  // don't have LFATAL() print "-- ABORT", since we won't actually be
  // aborting, since the interpreter will catch the exception and
  // return control to the user:
  LOG_FLAGS &= (~LFATAL_PRINTS_ABORT);

  rutz::prof::print_at_exit(true);

  app.splash(PACKAGE_STRING " (" __DATE__ ")\n"
             "\n"
             "Copyright (c) 2001-2005 iLab and the Univ. of Southern California\n"
             "<http://ilab.usc.edu>\n"
             "Copyright (c) 1998-2004 Rob Peters and Caltech\n"
             "<http://ilab.usc.edu/rjpeters/groovx/>\n"
             PACKAGE_NAME " is free software, "
             "covered by the GNU General Public "
             "License, and you are welcome to "
             "change it and/or distribute copies "
             "of it under certain conditions.");

  app.packages(INVT_PKGS);

  app.run();

  GVX_SCRIPT_PROG_END(app);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // SCRIPT_INVT_C_DEFINED
