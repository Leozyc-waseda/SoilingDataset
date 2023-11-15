/** @file tcl/objpkg.cc tcl package helpers for defining basic
    class-level tcl commands (e.g. find all objects of a given type) */
///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Fri Nov 22 17:06:50 2002
// commit: $Id: objpkg.cc 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/objpkg.cc $
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

#ifndef GROOVX_TCL_OBJPKG_CC_UTC20050626084017_DEFINED
#define GROOVX_TCL_OBJPKG_CC_UTC20050626084017_DEFINED

#include "objpkg.h"

#include "nub/objdb.h"
#include "nub/object.h"

#include "tcl/list.h"
#include "tcl/pkg.h"

#include "rutz/iter.h"

#include "rutz/trace.h"
#include "rutz/debug.h"
GVX_DBG_REGISTER

using rutz::shared_ptr;

//---------------------------------------------------------------------
//
// obj_caster
//
//---------------------------------------------------------------------

tcl::obj_caster::obj_caster() {}

tcl::obj_caster::~obj_caster() {}

bool tcl::obj_caster::is_id_my_type(nub::uid uid) const
{
  nub::soft_ref<nub::object> item(uid);
  return (item.is_valid() && is_my_type(item.get()));
}

namespace
{
  int count_all(shared_ptr<tcl::obj_caster> caster)
  {
    nub::objectdb& instance = nub::objectdb::instance();
    int count = 0;
    for (nub::objectdb::iterator itr(instance.objects()); itr.is_valid(); ++itr)
      {
        if (caster->is_my_type(*itr))
          ++count;
      }
    return count;
  }

  tcl::list find_all(shared_ptr<tcl::obj_caster> caster)
  {
    nub::objectdb& instance = nub::objectdb::instance();

    tcl::list result;

    for (nub::objectdb::iterator itr(instance.objects()); itr.is_valid(); ++itr)
      {
        if (caster->is_my_type(*itr))
          result.append((*itr)->id());
      }

    return result;
  }

  void remove_all(shared_ptr<tcl::obj_caster> caster)
  {
    nub::objectdb& instance = nub::objectdb::instance();
    for (nub::objectdb::iterator itr(instance.objects());
         itr.is_valid();
         /* increment done in loop body */)
      {
        dbg_eval(3, (*itr)->id());
        dbg_dump(3, *(*itr)->get_counts());

        if (caster->is_my_type(*itr) && (*itr)->is_unshared())
          {
            nub::uid remove_me = (*itr)->id();
            ++itr;
            instance.remove(remove_me);
          }
        else
          {
            ++itr;
          }
      }
  }

  bool is_my_type(shared_ptr<tcl::obj_caster> caster, nub::uid id)
  {
    return caster->is_id_my_type(id);
  }

  unsigned int get_sizeof(shared_ptr<tcl::obj_caster> caster)
  {
    return caster->get_sizeof();
  }
}

void tcl::def_basic_type_cmds(tcl::pkg* pkg,
                              shared_ptr<tcl::obj_caster> caster,
                              const rutz::file_pos& src_pos)
{
GVX_TRACE("tcl::def_basic_type_cmds");

  const int flags = tcl::NO_EXPORT;

  pkg->def_vec( "is", "objref(s)",
                rutz::bind_first(is_my_type, caster), 1, src_pos, flags );
  pkg->def( "count_all", "",
            rutz::bind_first(count_all, caster), src_pos, flags );
  pkg->def( "find_all", "",
            rutz::bind_first(find_all, caster), src_pos, flags );
  pkg->def( "remove_all", "",
            rutz::bind_first(remove_all, caster), src_pos, flags );
  pkg->def( "sizeof", "",
            rutz::bind_first(get_sizeof, caster), src_pos, flags );
}

static const char __attribute__((used)) vcid_groovx_tcl_objpkg_cc_utc20050626084017[] = "$Id: objpkg.cc 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/objpkg.cc $";
#endif // !GROOVX_TCL_OBJPKG_CC_UTC20050626084017_DEFINED
