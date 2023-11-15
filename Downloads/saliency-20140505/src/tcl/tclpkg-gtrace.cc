/** @file tcl/tclpkg-gtrace.cc tcl interface package for tracing and
    profiling facilities */
///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Mon Jan 21 12:52:51 2002
// commit: $Id: tclpkg-gtrace.cc 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/tclpkg-gtrace.cc $
//
// --------------------------------------------------------------------
//
// This file is part of GroovX
//   [http://ilab.usc.edu/rjpeters/groovx/]
//
// GroovX is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// GroovX is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GroovX; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
//
///////////////////////////////////////////////////////////////////////

#ifndef GROOVX_TCL_TCLPKG_GTRACE_CC_UTC20050628161246_DEFINED
#define GROOVX_TCL_TCLPKG_GTRACE_CC_UTC20050628161246_DEFINED

#include "tcl/tclpkg-gtrace.h"

#include "tcl/pkg.h"

#include "rutz/error.h"
#include "rutz/fstring.h"
#include "rutz/sfmt.h"

#include <sstream>

#include "rutz/debug.h"
GVX_DBG_REGISTER
#include "rutz/trace.h"

namespace
{
  rutz::fstring profSummary()
  {
    std::ostringstream oss;
    rutz::prof::print_all_prof_data(oss);
    return rutz::fstring(oss.str().c_str());
  }

  void setOneLevel(int key, int level)
  {
    if (!rutz::debug::is_valid_key(key))
      throw rutz::error(rutz::sfmt("no such debug key '%d'", key),
                        SRC_POS);

    // else...
    rutz::debug::set_level_for_key(key, level);
  }

  void setOneLevelc(const char* fname, int level)
  {
    const int key = rutz::debug::lookup_key(fname);
    if (key == -1)
      throw rutz::error(rutz::sfmt("no debug key for file '%s'", fname),
                        SRC_POS);

    GVX_ASSERT(rutz::debug::is_valid_key(key));

    rutz::debug::set_level_for_key(key, level);
  }
}

extern "C"
int Gtrace_Init(Tcl_Interp* interp)
{
GVX_TRACE("Gtrace_Init");

  GVX_PKG_CREATE(pkg, interp, "Gtrace", "4.$Revision: 11876 $");
  pkg->def("::gtrace", "on_off", &rutz::trace::set_global_trace, SRC_POS);
  pkg->def("::gtrace", "", &rutz::trace::get_global_trace, SRC_POS);
  pkg->def("maxDepth", "level", &rutz::trace::set_max_level, SRC_POS);
  pkg->def("maxDepth", "", &rutz::trace::get_max_level, SRC_POS);
  pkg->def("::dbglevelg", "global_level", &rutz::debug::set_global_level, SRC_POS);
  pkg->def("::dbglevel", "key level", &setOneLevel, SRC_POS);
  pkg->def("::dbglevelc", "filename level", &setOneLevelc, SRC_POS);
  pkg->def("::dbgkey", "filename", &rutz::debug::lookup_key, SRC_POS);

  GVX_PKG_RETURN(pkg);
}

extern "C"
int Prof_Init(Tcl_Interp* interp)
{
GVX_TRACE("Prof_Init");

  GVX_PKG_CREATE(pkg, interp, "Prof", "4.$Revision: 11876 $");
  pkg->def("summary", "", &profSummary, SRC_POS);
  pkg->def("reset", "", &rutz::prof::reset_all_prof_data, SRC_POS);

  GVX_PKG_RETURN(pkg);
}

static const char __attribute__((used)) vcid_groovx_tcl_tclpkg_gtrace_cc_utc20050628161246[] = "$Id: tclpkg-gtrace.cc 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/tclpkg-gtrace.cc $";
#endif // !GROOVX_TCL_TCLPKG_GTRACE_CC_UTC20050628161246_DEFINED
