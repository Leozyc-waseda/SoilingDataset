/** @file tcl/tclpkg-obj.cc tcl interface packages for nub::object and
    nub::objectdb */
///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Fri Jun 14 16:24:33 2002
// commit: $Id: tclpkg-obj.cc 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/tclpkg-obj.cc $
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

#ifndef GROOVX_TCL_TCLPKG_OBJ_CC_UTC20050628161246_DEFINED
#define GROOVX_TCL_TCLPKG_OBJ_CC_UTC20050628161246_DEFINED

#include "tcl/tclpkg-obj.h"

#include "nub/objdb.h"
#include "nub/objmgr.h"

#include "tcl/objpkg.h"
#include "tcl/pkg.h"
#include "tcl/list.h"
#include "tcl/interp.h"

#include "rutz/demangle.h"
#include "rutz/sfmt.h"

#include <tcl.h>

#include "rutz/trace.h"

using nub::soft_ref;
using nub::object;

namespace
{
  void dbClear() { nub::objectdb::instance().clear(); }
  void dbPurge() { nub::objectdb::instance().purge(); }
  void dbRelease(nub::uid id) { nub::objectdb::instance().release(id); }
  void dbClearOnExit() { nub::objectdb::instance().clear_on_exit(); }

  // This is just here to select between the const char* +
  // rutz::fstring versions of new_obj().
  soft_ref<object> objNew(const char* type)
  {
    return nub::obj_mgr::new_obj(type);
  }

  soft_ref<object> objNewArgs(const char* type,
                             tcl::list init_args,
                             tcl::interpreter interp)
  {
    soft_ref<object> obj(nub::obj_mgr::new_obj(type));

    for (unsigned int i = 0; i+1 < init_args.length(); i+=2)
      {
        tcl::list cmd_str;
        cmd_str.append("::->");
        cmd_str.append(obj.id());
        cmd_str.append(init_args[i]);
        cmd_str.append(init_args[i+1]);
        interp.eval(cmd_str.as_obj());
      }

    return obj;
  }

  tcl::list objNewArr(const char* type, unsigned int array_size)
  {
    tcl::list result;

    while (array_size-- > 0)
      {
        soft_ref<object> item(nub::obj_mgr::new_obj(type));
        result.append(item.id());
      }

    return result;
  }

  void objDelete(tcl::list objrefs)
  {
    tcl::list::iterator<nub::uid>
      itr = objrefs.begin<nub::uid>(),
      stop = objrefs.end<nub::uid>();
    while (itr != stop)
      {
        nub::objectdb::instance().remove(*itr);
        ++itr;
      }
  }

  void arrowDispatch(tcl::call_context& ctx)
  {
    /* old implementation was this:

       pkg->eval("proc ::-> {args} {\n"
                 "  set ids [lindex $args 0]\n"
                 "  set namesp [Obj::type [lindex $ids 0]]\n"
                 "  set cmd [lreplace $args 0 1 [lindex $args 1] $ids]\n"
                 "  namespace eval $namesp $cmd\n"
                 "}");

       but the problem was that it involved a string conversion cycle
       of the trailing args, which meant that we lost the internal rep
    */

    // e.g.      "-> {3 4} foo 4.5"
    // becomes   "Namesp::foo {3 4} 4.5"

    if (ctx.objc() < 3)
      throw rutz::error("bad objc", SRC_POS);

    Tcl_Obj* const* origargs = ctx.get_raw_args();

    tcl::list objrefs(origargs[1]);

    const rutz::fstring namesp =
      objrefs.get<soft_ref<object> >(0)->obj_typename();

    rutz::fstring origcmdname = ctx.get_arg<rutz::fstring>(2);

    rutz::fstring newcmdname =
      rutz::sfmt("%s::%s", namesp.c_str(), origcmdname.c_str());

    tcl::list newargs;

    newargs.append(newcmdname);
    newargs.append(objrefs);

    for (unsigned int i = 3; i < ctx.objc(); ++i)
      {
        newargs.append(origargs[i]);
      }

    // use eval_objv() instead of eval(), so that we don't break any
    // objects with fragile internal representations:
    ctx.interp().eval_objv(newargs);
  }
}

extern "C"
int Objectdb_Init(Tcl_Interp* interp)
{
GVX_TRACE("Objectdb_Init");

  GVX_PKG_CREATE(pkg, interp, "Objectdb", "4.$Revision: 11876 $");

  pkg->on_exit( &dbClearOnExit );

  pkg->def( "clear", 0, &dbClear, SRC_POS );
  pkg->def( "purge", 0, &dbPurge, SRC_POS );
  pkg->def( "release", 0, &dbRelease, SRC_POS );

  GVX_PKG_RETURN(pkg);
}

extern "C"
int Obj_Init(Tcl_Interp* interp)
{
GVX_TRACE("Obj_Init");

  GVX_PKG_CREATE(pkg, interp, "Obj", "4.$Revision: 11876 $");
  tcl::def_basic_type_cmds<object>(pkg, SRC_POS);

  pkg->def_getter("refCount", &object::dbg_ref_count, SRC_POS);
  pkg->def_getter("weakRefCount", &object::dbg_weak_ref_count, SRC_POS);
  pkg->def_action("incr_ref_count", &object::incr_ref_count, SRC_POS);
  pkg->def_action("decr_ref_count", &object::decr_ref_count, SRC_POS);

  pkg->def_getter( "type", &object::obj_typename, SRC_POS );
  pkg->def_getter( "realType", &object::real_typename, SRC_POS );

  pkg->def( "new", "typename", &objNew, SRC_POS );
  pkg->def( "new", "typename {cmd1 arg1 cmd2 arg2 ...}",
            rutz::bind_last(&objNewArgs, tcl::interpreter(interp)),
            SRC_POS );
  pkg->def( "newarr", "typename array_size=1", &objNewArr, SRC_POS );
  pkg->def( "delete", "objref(s)", &objDelete, SRC_POS );

  pkg->def_raw( "::->", tcl::arg_spec(3).nolimit(),
                "objref(s) cmdname ?arg1 arg2 ...?",
                &arrowDispatch, SRC_POS );

  pkg->namesp_alias("::", "new");
  pkg->namesp_alias("::", "newarr");
  pkg->namesp_alias("::", "delete");

  GVX_PKG_RETURN(pkg);
}

static const char __attribute__((used)) vcid_groovx_tcl_tclpkg_obj_cc_utc20050628161246[] = "$Id: tclpkg-obj.cc 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/tclpkg-obj.cc $";
#endif // !GROOVX_TCL_TCLPKG_OBJ_CC_UTC20050628161246_DEFINED
