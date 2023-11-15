/** @file tcl/objpkg.h tcl package helpers for defining basic
    class-level tcl commands (e.g. find all objects of a given type) */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Fri Nov 22 17:05:11 2002
// commit: $Id: objpkg.h 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/objpkg.h $
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

#ifndef GROOVX_TCL_OBJPKG_H_UTC20050626084018_DEFINED
#define GROOVX_TCL_OBJPKG_H_UTC20050626084018_DEFINED

#include "nub/objfactory.h"

#include "rutz/shared_ptr.h"

namespace rutz
{
  struct file_pos;
}

namespace tcl
{
  class obj_caster;
  class pkg;
}

namespace nub
{
  class object;
}

// ########################################################
/// obj_caster class encapsulates casts to see if objects match a given type.

class tcl::obj_caster
{
protected:
  obj_caster();

public:
  virtual ~obj_caster();

  virtual bool is_my_type(const nub::object* obj) const = 0;

  virtual unsigned int get_sizeof() const = 0;

  bool is_not_my_type(const nub::object* obj) const { return !is_my_type(obj); }

  bool is_id_my_type(nub::uid uid) const;

  bool is_id_not_my_type(nub::uid uid) const { return !is_id_my_type(uid); }
};

namespace tcl
{
  /// cobj_caster implements obj_caster with dynamic_cast.
  template <class C>
  class cobj_caster : public obj_caster
  {
  public:
    virtual unsigned int get_sizeof() const
    {
      return sizeof(C);
    }

    virtual bool is_my_type(const nub::object* obj) const
    {
      return (obj != 0 && dynamic_cast<const C*>(obj) != 0);
    }
  };

  void def_basic_type_cmds(pkg* pkg, rutz::shared_ptr<obj_caster> caster,
                           const rutz::file_pos& src_pos);

  template <class C>
  void def_basic_type_cmds(pkg* pkg, const rutz::file_pos& src_pos)
  {
    rutz::shared_ptr<obj_caster> caster(new cobj_caster<C>);
    def_basic_type_cmds(pkg, caster, src_pos);
  }

  template <class C>
  void def_creator(pkg*, const char* alias_name = 0)
  {
    const char* origName =
      nub::obj_factory::instance().register_creator(&C::make);

    if (alias_name != 0)
      nub::obj_factory::instance().register_alias(origName, alias_name);
  }
}

static const char __attribute__((used)) vcid_groovx_tcl_objpkg_h_utc20050626084018[] = "$Id: objpkg.h 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/objpkg.h $";
#endif // !GROOVX_TCL_OBJPKG_H_UTC20050626084018_DEFINED
