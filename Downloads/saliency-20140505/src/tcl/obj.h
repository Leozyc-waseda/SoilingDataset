/** @file tcl/obj.h c++ wrapper class for Tcl_Obj, to handle
    ref-counting and type conversions */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 1999-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Tue May 11 13:44:19 1999
// commit: $Id: obj.h 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/obj.h $
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

#ifndef GROOVX_TCL_OBJ_H_UTC20050628162421_DEFINED
#define GROOVX_TCL_OBJ_H_UTC20050628162421_DEFINED

typedef struct Tcl_Obj Tcl_Obj;

namespace tcl
{
  class obj;
}

///////////////////////////////////////////////////////////////////////
/**
 *
 * This class acts as a reference counted smart pointer to Tcl_Obj. It
 * manages the reference count of a held Tcl_Obj*, and also defines a
 * conversion operator back to Tcl_Obj* in order to provide access to
 * Tcl_Obj's interface.
 *
 **/
///////////////////////////////////////////////////////////////////////

class tcl::obj
{
public:
  /// Default constructor with a shared and empty Tcl_Obj*.
  obj();

  /// Construct with a Tcl_Obj*.
  obj(Tcl_Obj* obj) : m_obj(obj) { incr_ref(m_obj); }

  /// Destructor.
  ~obj() { decr_ref(m_obj); }

  /// Copy constructor.
  obj(const obj& x) :
    m_obj(x.m_obj)
    {
      incr_ref(m_obj);
    }

  /// Assignment operator from obj.
  obj& operator=(const obj& x)
    {
      assign(x.m_obj); return *this;
    }

  /// Assignment operator from Tcl_Obj*.
  obj& operator=(Tcl_Obj* x)
    {
      assign(x); return *this;
    }

  Tcl_Obj* get() const { return m_obj; }

  void append(const tcl::obj& other);

  bool is_shared() const;
  bool is_unique() const { return !is_shared(); }

  void make_unique() const;

  const char* tcltype_name() const;

private:
  static void incr_ref(Tcl_Obj* obj);
  static void decr_ref(Tcl_Obj* obj);

  void assign(Tcl_Obj* x) const
    {
      if (m_obj != x)
        {
          decr_ref(m_obj);
          m_obj = x;
          incr_ref(m_obj);
        }
    }

  mutable Tcl_Obj* m_obj;
};

static const char __attribute__((used)) vcid_groovx_tcl_obj_h_utc20050628162421[] = "$Id: obj.h 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/obj.h $";
#endif // !GROOVX_TCL_OBJ_H_UTC20050628162421_DEFINED
