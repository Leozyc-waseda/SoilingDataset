/** @file tcl/vecdispatch.cc apply vectorized dispatching to a
    tcl::command */
///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2001-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Thu Jul 12 12:15:46 2001
// commit: $Id: vecdispatch.cc 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/vecdispatch.cc $
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

#ifndef GROOVX_TCL_VECDISPATCH_CC_UTC20050628162420_DEFINED
#define GROOVX_TCL_VECDISPATCH_CC_UTC20050628162420_DEFINED

#include "tcl/vecdispatch.h"

#include "tcl/command.h"
#include "tcl/list.h"

#include "rutz/error.h"
#include "rutz/shared_ptr.h"

#include <vector>

#include "rutz/trace.h"
#include "rutz/debug.h"
GVX_DBG_REGISTER

namespace tcl
{
  class vec_context;
}

///////////////////////////////////////////////////////////////////////
//
// tcl::vec_context implements the tcl::call_context interface in such
// a way as to treat each of the arguments as lists, and provide
// access to slices across those lists, thus allowing "vectorized"
// command invocations.
//
///////////////////////////////////////////////////////////////////////

class tcl::vec_context : public tcl::call_context
{
public:
  vec_context(tcl::interpreter& interp, unsigned int objc,
              Tcl_Obj* const objv[]) :
    call_context(interp, objc, objv),
    m_arg0(objv[0]),
    m_args(),
    m_result()
  {
    for (unsigned int i = 1; i < objc; ++i)
      {
        tcl::list arg(objv[i]);
        if (arg.length() == 0)
          {
            throw rutz::error("argument was empty", SRC_POS);
          }
        m_args.push_back( arg.begin<Tcl_Obj*>() );
      }
  }

  virtual ~vec_context() throw() {}

  void flush_result()
  {
    tcl::call_context::set_obj_result(m_result.as_obj());
  }

  void next()
  {
    for (unsigned int i = 0; i < m_args.size(); ++i)
      {
        if (m_args[i].has_more())
          ++(m_args[i]);
      }
  }

protected:
  virtual Tcl_Obj* get_objv(unsigned int argn) throw()
  {
    if (argn == 0) return m_arg0;

    return *(m_args.at(argn-1));
  }

  virtual void set_obj_result(const tcl::obj& obj)
  {
    m_result.append(obj);
  }

private:
  typedef tcl::list::iterator<Tcl_Obj*> Iter;

  Tcl_Obj* m_arg0;
  std::vector<Iter> m_args;
  tcl::list m_result;
};

namespace tcl
{
  class vec_dispatcher;
}

///////////////////////////////////////////////////////////////////////
/**
 *
 * \c tcl::vec_dispatcher reimplements dispatch() to use a specialized
 * \c tcl::call_context class that treats each of the arguments as
 * lists, and provide access to slices across those lists, thus
 * allowing "vectorized" command invocations.
 *
 **/
///////////////////////////////////////////////////////////////////////

class tcl::vec_dispatcher : public tcl::arg_dispatcher
{
public:
  vec_dispatcher(unsigned int key_argn) : m_key_argn(key_argn) {}

  virtual ~vec_dispatcher() throw() {}

  virtual void dispatch(tcl::interpreter& interp,
                        unsigned int objc, Tcl_Obj* const objv[],
                        tcl::function& callback);

private:
  unsigned int m_key_argn;
};


void tcl::vec_dispatcher::dispatch(tcl::interpreter& interp,
                                   unsigned int objc,
                                   Tcl_Obj* const objv[],
                                   tcl::function& callback)
{
GVX_TRACE("tcl::vec_dispatcher::dispatch");

  const unsigned int ncalls
    = tcl::list::get_obj_list_length(objv[m_key_argn]);

  if (ncalls > 1)
    {
      vec_context cx(interp, objc, objv);

      for (unsigned int c = 0; c < ncalls; ++c)
        {
          callback.invoke(cx);
          cx.next();
        }

      cx.flush_result();
    }
  else if (ncalls == 1)
    {
      tcl::call_context cx(interp, objc, objv);
      callback.invoke(cx);
    }
  else // (ncalls == 0)
    {
      ;// do nothing, so we gracefully handle empty lists
    }
}


void tcl::use_vec_dispatch(tcl::command& cmd, unsigned int key_argn)
{
  cmd.set_dispatcher(rutz::make_shared(new vec_dispatcher(key_argn)));
}

static const char __attribute__((used)) vcid_groovx_tcl_vecdispatch_cc_utc20050628162420[] = "$Id: vecdispatch.cc 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/vecdispatch.cc $";
#endif // !GROOVX_TCL_VECDISPATCH_CC_UTC20050628162420_DEFINED
