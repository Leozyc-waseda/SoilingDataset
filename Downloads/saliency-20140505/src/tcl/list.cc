/** @file tcl/list.cc c++ wrapper of tcl list objects; handles ref
    counting and c++/tcl type conversion, offers c++-style list
    iterators */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2001-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Wed Jul 11 12:32:35 2001
// commit: $Id: list.cc 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/list.cc $
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

#ifndef GROOVX_TCL_LIST_CC_UTC20050628162420_DEFINED
#define GROOVX_TCL_LIST_CC_UTC20050628162420_DEFINED

#include "tcl/list.h"

#include "rutz/error.h"

#include <tcl.h>

#include "rutz/trace.h"
#include "rutz/debug.h"
GVX_DBG_REGISTER

tcl::list tcl::aux_convert_to(Tcl_Obj* obj, tcl::list*)
{
GVX_TRACE("tcl::aux_convert_to(tcl::list*)");

  return tcl::list(obj);
}

tcl::obj tcl::aux_convert_from(tcl::list list_value)
{
GVX_TRACE("tcl::aux_convert_from(tcl::list)");

  return list_value.as_obj();
}

tcl::list::list() :
  m_list_obj(Tcl_NewListObj(0,0)),
  m_elements(0),
  m_length(0)
{
GVX_TRACE("tcl::list::list");
  split();
}

tcl::list::list(const tcl::obj& x) :
  m_list_obj(x),
  m_elements(0),
  m_length(0)
{
GVX_TRACE("tcl::list::list");
  split();
}

void tcl::list::split() const
{
GVX_TRACE("tcl::list::split");

  int count;
  if ( Tcl_ListObjGetElements(0, m_list_obj.get(), &count, &m_elements)
       != TCL_OK)
    {
      throw rutz::error("couldn't split Tcl list", SRC_POS);
    }

  GVX_ASSERT(count >= 0);
  m_length = static_cast<unsigned int>(count);
}

void tcl::list::do_append(const tcl::obj& obj, unsigned int times)
{
GVX_TRACE("tcl::list::do_append");

  m_list_obj.make_unique();

  while (times--)
    if ( Tcl_ListObjAppendElement(0, m_list_obj.get(), obj.get())
         != TCL_OK )
      {
        throw rutz::error("couldn't append to Tcl list", SRC_POS);
      }

  invalidate();
}

Tcl_Obj* tcl::list::at(unsigned int index) const
{
GVX_TRACE("tcl::list::at");

  update();

  if (index >= m_length)
    throw rutz::error("index was out of range in Tcl list access",
                      SRC_POS);

  dbg_eval(3, index); dbg_eval_nl(3, m_elements[index]);

  return m_elements[index];
}

unsigned int tcl::list::get_obj_list_length(Tcl_Obj* obj)
{
GVX_TRACE("tcl::list::get_obj_list_length");

  int len;
  if ( Tcl_ListObjLength(0, obj, &len) != TCL_OK)
    {
      throw rutz::error("couldn't get list length of Tcl object",
                        SRC_POS);
    }

  GVX_ASSERT(len >= 0);

  return static_cast<unsigned int>(len);
}

static const char __attribute__((used)) vcid_groovx_tcl_list_cc_utc20050628162420[] = "$Id: list.cc 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/list.cc $";
#endif // !GROOVX_TCL_LIST_CC_UTC20050628162420_DEFINED
