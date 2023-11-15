/** @file tcl/namesp.h Thin wrapper around Tcl_Namespace */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2006-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Fri Feb 17 16:58:18 2006
// commit: $Id: namesp.h 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/namesp.h $
//
// --------------------------------------------------------------------
//
// This file is part of GroovX
//   [http://www.klab.caltech.edu/rjpeters/groovx/]
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

#ifndef GROOVX_TCL_NAMESP_H_UTC20060218005818_DEFINED
#define GROOVX_TCL_NAMESP_H_UTC20060218005818_DEFINED

#include "rutz/fstring.h"

#ifdef HAVE_TCLINT_H
#include <tclInt.h>
#endif

namespace tcl
{
  class interpreter;
  class list;

#ifdef HAVE_TCLINT_H
  class native_namesp
  {
  public:
    native_namesp(tcl::interpreter& interp, const char* name);

    static native_namesp lookup(tcl::interpreter& interp,
                                const char* name);

    void export_cmd(tcl::interpreter& interp,
                    const char* cmdname) const;

    tcl::list get_export_list(tcl::interpreter& interp) const;

  private:
    native_namesp(Tcl_Namespace* ns);

    Tcl_Namespace* m_ns;
  };
#endif

  class emu_namesp
  {
  public:
    emu_namesp(tcl::interpreter& interp, const char* name);

    static emu_namesp lookup(tcl::interpreter& interp,
                             const char* name);

    void export_cmd(tcl::interpreter& interp,
                    const char* cmdname) const;

    tcl::list get_export_list(tcl::interpreter& interp) const;

  private:
    const rutz::fstring m_ns_name;
  };

#ifdef HAVE_TCLINT_H
  typedef native_namesp namesp;
#else
  typedef emu_namesp namesp;
#endif
}


static const char __attribute__((used)) vcid_groovx_tcl_namesp_h_utc20060218005818[] = "$Id: namesp.h 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/namesp.h $";
#endif // !GROOVX_TCL_NAMESP_H_UTC20060218005818DEFINED
