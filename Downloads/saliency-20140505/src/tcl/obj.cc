/** @file tcl/obj.cc c++ wrapper class for Tcl_Obj, to handle
    ref-counting and type conversions */
///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2001-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Wed Jul 11 18:30:47 2001
// commit: $Id: obj.cc 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/obj.cc $
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

#ifndef GROOVX_TCL_OBJ_CC_UTC20050628162420_DEFINED
#define GROOVX_TCL_OBJ_CC_UTC20050628162420_DEFINED

#include "tcl/obj.h"

#include <tcl.h>

tcl::obj::obj() : m_obj(Tcl_NewObj()) { incr_ref(m_obj); }

void tcl::obj::append(const tcl::obj& other)
{
  make_unique();
  Tcl_AppendObjToObj(m_obj, other.m_obj);
}

bool tcl::obj::is_shared() const
{
  return Tcl_IsShared(m_obj);
}

void tcl::obj::make_unique() const
{
  if (is_shared())
    {
      Tcl_Obj* new_obj = Tcl_DuplicateObj(m_obj);
      assign(new_obj);
    }
}

const char* tcl::obj::tcltype_name() const
{
  const Tcl_ObjType* type = m_obj->typePtr;

  return type ? type->name : "(none)";
}

void tcl::obj::incr_ref(Tcl_Obj* obj)
{
  Tcl_IncrRefCount(obj);
}

void tcl::obj::decr_ref(Tcl_Obj* obj)
{
  Tcl_DecrRefCount(obj);
}

static const char __attribute__((used)) vcid_groovx_tcl_obj_cc_utc20050628162420[] = "$Id: obj.cc 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/obj.cc $";
#endif // !GROOVX_TCL_OBJ_CC_UTC20050628162420_DEFINED
