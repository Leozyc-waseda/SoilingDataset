/** @file tcl/namesp.cc Thin wrapper around Tcl_Namespace */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2006-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Fri Feb 17 17:01:19 2006
// commit: $Id: namesp.cc 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/namesp.cc $
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

#ifndef GROOVX_TCL_NAMESP_CC_UTC20060218010119_DEFINED
#define GROOVX_TCL_NAMESP_CC_UTC20060218010119_DEFINED

#include "tcl/namesp.h"

#include "tcl/interp.h"
#include "tcl/list.h"

#include "rutz/error.h"
#include "rutz/sfmt.h"

#include "rutz/debug.h"
GVX_DBG_REGISTER

#ifdef HAVE_TCLINT_H

tcl::native_namesp::native_namesp(Tcl_Namespace* ns)
  :
  m_ns(ns)
{}

tcl::native_namesp::native_namesp(tcl::interpreter& interp, const char* name)
  :
  m_ns(0)
{
  m_ns =
    Tcl_FindNamespace(interp.intp(), name,
                      0 /* namespaceContextPtr*/, TCL_GLOBAL_ONLY);

  if (m_ns == 0)
    {
      m_ns = Tcl_CreateNamespace(interp.intp(),
                                 name,
                                 0 /*clientdata*/,
                                 0 /*delete_proc*/);
    }

  GVX_ASSERT(m_ns != 0);
}

tcl::native_namesp tcl::native_namesp::lookup(tcl::interpreter& interp,
                                              const char* name)
{
  Tcl_Namespace* const ns =
    Tcl_FindNamespace(interp.intp(), name, 0, TCL_GLOBAL_ONLY);

  if (ns == 0)
    throw rutz::error(rutz::sfmt("no Tcl namespace '%s;", name),
                      SRC_POS);

  return tcl::native_namesp(ns);
}

void tcl::native_namesp::export_cmd(tcl::interpreter& interp,
                                    const char* cmdname) const
{
  Tcl_Export(interp.intp(), m_ns, cmdname,
             /*resetExportListfirst*/ false);
}

tcl::list tcl::native_namesp::get_export_list(tcl::interpreter& interp) const
{
  tcl::obj obj;

  Tcl_AppendExportList(interp.intp(), m_ns, obj.get());

  return tcl::list(obj);
}

#endif // HAVE_TCLINT_H


tcl::emu_namesp::emu_namesp(tcl::interpreter& /*interp*/,
                            const char* name)
  :
  m_ns_name(name)
{}

tcl::emu_namesp tcl::emu_namesp::lookup(tcl::interpreter& interp,
                                        const char* name)
{
  return tcl::emu_namesp(interp, name);
}

void tcl::emu_namesp::export_cmd(tcl::interpreter& interp,
                                 const char* cmdname) const
{
  interp.eval(rutz::sfmt("namespace eval %s { namespace export %s }",
                         m_ns_name.c_str(), cmdname));
}

tcl::list tcl::emu_namesp::get_export_list(tcl::interpreter& interp) const
{
  const tcl::obj prevresult = interp.get_result<tcl::obj>();
  interp.reset_result();

  interp.eval(rutz::sfmt("namespace eval %s { namespace export }",
                         m_ns_name.c_str()));

  const tcl::list exportlist = interp.get_result<tcl::list>();

  interp.set_result(prevresult);

  return exportlist;
}

static const char __attribute__((used)) vcid_groovx_tcl_namesp_cc_utc20060218010119[] = "$Id: namesp.cc 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/namesp.cc $";
#endif // !GROOVX_TCL_NAMESP_CC_UTC20060218010119DEFINED
