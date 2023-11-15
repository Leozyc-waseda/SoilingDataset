/** @file tcl/tclpkg-gtrace.h tcl interface package for tracing and
    profiling facilities */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2005-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Sat Jun 25 16:59:51 2005
// commit: $Id: tclpkg-gtrace.h 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/tclpkg-gtrace.h $
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

#ifndef GROOVX_TCL_TCLPKG_GTRACE_H_UTC20050628161246_DEFINED
#define GROOVX_TCL_TCLPKG_GTRACE_H_UTC20050628161246_DEFINED

struct Tcl_Interp;

extern "C" int Gtrace_Init(Tcl_Interp* interp);
extern "C" int Prof_Init(Tcl_Interp* interp);

static const char __attribute__((used)) vcid_groovx_tcl_tclpkg_gtrace_h_utc20050628161246[] = "$Id: tclpkg-gtrace.h 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/tclpkg-gtrace.h $";
#endif // !GROOVX_TCL_TCLPKG_GTRACE_H_UTC20050628161246_DEFINED
