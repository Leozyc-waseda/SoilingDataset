/** @file tcl/pkg.cc tcl package class, holds a set of commands, wraps
    calls to Tcl_PkgProvide(), etc. */
///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 1999-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Tue Jun 15 12:33:54 1999
// commit: $Id: pkg.cc 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/pkg.cc $
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

#ifndef GROOVX_TCL_PKG_CC_UTC20050628162420_DEFINED
#define GROOVX_TCL_PKG_CC_UTC20050628162420_DEFINED

#include "tcl/pkg.h"

#include "tcl/command.h"
#include "tcl/interp.h"
#include "tcl/list.h"
#include "tcl/namesp.h"

#include "rutz/error.h"
#include "rutz/fstring.h"
#include "rutz/sfmt.h"
#include "rutz/shared_ptr.h"

#include <tcl.h>
#ifdef HAVE_TCLINT_H
#include <tclInt.h> // for Tcl_FindNamespace() etc.
#endif
#include <cctype>
#include <iostream>
#include <typeinfo>
#include <string>
#include <vector>

#include "rutz/trace.h"
#include "rutz/debug.h"
GVX_DBG_REGISTER

#if (TCL_MAJOR_VERSION > 8) || (TCL_MAJOR_VERSION == 8 && TCL_MINOR_VERSION >= 5)
#  define HAVE_TCL_NAMESPACE_API
#else
#  undef  HAVE_TCL_NAMESPACE_API
#endif

using std::string;
using std::vector;
using rutz::shared_ptr;

namespace
{
  bool VERBOSE_INIT = false;

  int INIT_DEPTH = 0;

  // Construct a capitalization-correct version of the given name that
  // is just how Tcl likes it: first character uppercase, all others
  // lowercase.
  string make_clean_pkg_name(const string& name)
  {
    string clean;

    clean += char(toupper(name[0]));

    for (size_t i = 1; i < name.length(); ++i)
      {
        if (name[i] != '-' && name[i] != '_')
          clean += char(tolower(name[i]));
      }

    return clean;
  }

  string make_clean_version_string(const string& s)
  {
    string::size_type dollar1 = s.find_first_of("$");
    string::size_type dollar2 = s.find_last_of("$");

    if (dollar1 == dollar2)
      return s;

    const string r = s.substr(dollar1,dollar2+1-dollar1);

    string::size_type n1 = r.find_first_of("0123456789");
    string::size_type n2 = r.find_last_of("0123456789");

    string result(s);

    if (n1 != string::npos)
      {
        const string n = r.substr(n1,n2+1-n1);
        result.replace(dollar1, dollar2+1-dollar1, n);
      }
    else
      {
        result.replace(dollar1, dollar2+1-dollar1, "0");
      }

    return result;
  }

  void export_into(tcl::interpreter& interp,
                   const char* from, const char* to,
                   const char* pattern)
  {
  GVX_TRACE("export_into");
    const rutz::fstring cmd =
      rutz::sfmt("namespace eval %s { namespace import ::%s::%s }",
                 to, from, pattern);

    interp.eval(cmd);
  }

  tcl::list get_command_list(tcl::interpreter& interp, const char* namesp)
  {
    tcl::obj saveresult = interp.get_result<tcl::obj>();
    rutz::fstring cmd = rutz::sfmt("info commands ::%s::*", namesp);
    interp.eval(cmd);
    tcl::list cmdlist = interp.get_result<tcl::list>();
    interp.set_result(saveresult);
    return cmdlist;
  }

  const char* get_name_tail(const char* name)
  {
    const char* p = name;
    while (*p != '\0') ++p; // skip to end of string
    while (--p > name) {
      if ((*p == ':') && (*(p-1) == ':')) {
        ++p;
        break;
      }
    }
    GVX_ASSERT(p >= name);
    return p;
  }
}

const int tcl::pkg::STATUS_OK = TCL_OK;
const int tcl::pkg::STATUS_ERR = TCL_ERROR;

///////////////////////////////////////////////////////////////////////
//
// Helper functions that provide typesafe access to Tcl_LinkVar
//
///////////////////////////////////////////////////////////////////////

struct tcl::pkg::impl
{
private:
  impl(const impl&);
  impl& operator=(const impl&);

public:
  impl(Tcl_Interp* interp, const char* name, const char* version);

  ~impl() throw();

  tcl::interpreter                     interp;
  string                         const namesp_name;
  string                         const pkg_name;
  string                         const version;
  int                                  init_status;
  std::vector<shared_ptr<int> >        owned_ints;
  std::vector<shared_ptr<double> >     owned_doubles;
  exit_callback*                       on_exit;

  static void c_exit_handler(void* clientdata)
  {
    GVX_TRACE("tcl::pkg-c_exit_handler");
    tcl::pkg* pkg = static_cast<tcl::pkg*>(clientdata);
    dbg_eval_nl(3, pkg->namesp_name());
    delete pkg;
  }
};

tcl::pkg::impl::impl(Tcl_Interp* intp,
                     const char* name, const char* vers) :
  interp(intp),
  namesp_name(name ? name : ""),
  pkg_name(make_clean_pkg_name(namesp_name)),
  version(make_clean_version_string(vers)),
  init_status(TCL_OK),
  owned_ints(),
  owned_doubles(),
  on_exit(0)
{
GVX_TRACE("tcl::pkg::impl::impl");
}

tcl::pkg::impl::~impl() throw()
{
GVX_TRACE("tcl::pkg::impl::~impl");
  if (on_exit != 0)
    on_exit();
}

tcl::pkg::pkg(Tcl_Interp* interp,
              const char* name, const char* version) :
  rep(0)
{
GVX_TRACE("tcl::pkg::pkg");

  rep = new impl(interp, name, version);

  ++INIT_DEPTH;
}

tcl::pkg::~pkg() throw()
{
GVX_TRACE("tcl::pkg::~pkg");

  // To avoid double-deletion:
  Tcl_DeleteExitHandler(&impl::c_exit_handler, static_cast<void*>(this));

  try
    {
#ifndef HAVE_TCL_NAMESPACE_API
      tcl::list cmdnames = get_command_list(rep->interp,
                                            rep->namesp_name.c_str());

      for (unsigned int i = 0; i < cmdnames.length(); ++i)
        {
          Tcl_DeleteCommand(rep->interp.intp(),
                            cmdnames.get<const char*>(i));
        }
#else
      Tcl_Namespace* namesp =
        Tcl_FindNamespace(rep->interp.intp(), rep->namesp_name.c_str(),
                          0, TCL_GLOBAL_ONLY);
      if (namesp != 0)
        Tcl_DeleteNamespace(namesp);
#endif
    }
  catch (...)
    {
      rep->interp.handle_live_exception("tcl::pkg::~pkg", SRC_POS);
    }

  delete rep;
}

void tcl::pkg::on_exit(exit_callback* callback)
{
GVX_TRACE("tcl::pkg::on_exit");
  rep->on_exit = callback;
}

int tcl::pkg::destroy_on_unload(Tcl_Interp* intp, const char* pkgname)
{
GVX_TRACE("tcl::pkg::destroy_on_unload");
  tcl::interpreter interp(intp);
  tcl::pkg* pkg = tcl::pkg::lookup(interp, pkgname);
  if (pkg != 0)
    {
      delete pkg;
      return 1; // TCL_OK
    }
  // else...
  return 0; // TCL_ERROR
}

tcl::pkg* tcl::pkg::lookup(tcl::interpreter& interp, const char* name,
                           const char* version) throw()
{
GVX_TRACE("tcl::pkg::lookup");

  void* clientdata = 0;

  const string clean_name = make_clean_pkg_name(name);

  tcl::obj saveresult = interp.get_result<tcl::obj>();

  const char* ver =
    Tcl_PkgRequireEx(interp.intp(), clean_name.c_str(),
                     version, 0, &clientdata);

  interp.set_result(saveresult);

  if (ver != 0)
    {
      tcl::pkg* result = static_cast<tcl::pkg*>(clientdata);

      result = dynamic_cast<tcl::pkg*>(result);

      return result;
    }

  return 0;
}

int tcl::pkg::init_status() const throw()
{
GVX_TRACE("tcl::pkg::init_status");
  if (rep->interp.get_result<const char*>()[0] != '\0')
    {
      rep->init_status = TCL_ERROR;
    }
  return rep->init_status;
}

tcl::interpreter& tcl::pkg::interp() throw()
{
GVX_TRACE("tcl::pkg::interp");
  return rep->interp;
}

void tcl::pkg::handle_live_exception(const rutz::file_pos& pos) throw()
{
GVX_TRACE("tcl::pkg::handle_live_exception");
  rep->interp.handle_live_exception(rep->pkg_name.c_str(), pos);
  this->set_init_status_error();
}

void tcl::pkg::namesp_alias(const char* namesp, const char* pattern)
{
GVX_TRACE("tcl::pkg::namesp_alias");

  export_into(rep->interp, rep->namesp_name.c_str(), namesp, pattern);
}

void tcl::pkg::inherit_namesp(const char* namesp, const char* pattern)
{
GVX_TRACE("tcl::pkg::inherit_namesp");

  // (1) export commands from 'namesp' into this tcl::pkg's namespace
  export_into(rep->interp, namesp, rep->namesp_name.c_str(), pattern);

  // (2) get the export patterns from 'namesp' and include those as
  // export patterns for this tcl::pkg's namespace
  const tcl::namesp otherns = tcl::namesp::lookup(rep->interp, namesp);

  const tcl::list exportlist = otherns.get_export_list(rep->interp);

  const tcl::namesp thisns(rep->interp, rep->namesp_name.c_str());

  for (unsigned int i = 0; i < exportlist.size(); ++i)
    {
      thisns.export_cmd(rep->interp, exportlist.get<const char*>(i));
    }
}

void tcl::pkg::inherit_pkg(const char* name, const char* version)
{
GVX_TRACE("tcl::pkg::inherit_pkg");

  tcl::pkg* other = lookup(rep->interp, name, version);

  if (other == 0)
    throw rutz::error(rutz::sfmt("no tcl::pkg named '%s'", name),
                      SRC_POS);

  inherit_namesp(other->namesp_name());
}

const char* tcl::pkg::namesp_name() throw()
{
  return rep->namesp_name.c_str();
}

const char* tcl::pkg::pkg_name() const throw()
{
  return rep->pkg_name.c_str();
}

const char* tcl::pkg::version() const throw()
{
  return rep->version.c_str();
}

const char* tcl::pkg::make_pkg_cmd_name(const char* cmd_name_cstr,
                                        int flags)
{
GVX_TRACE("tcl::pkg::make_pkg_cmd_name");
  string cmd_name(cmd_name_cstr);

  // Look for a namespace qualifier "::" -- if there is already one,
  // then we assume the caller has something special in mind -- if
  // there is not one, then we do the default thing and prepend the
  // package name as a namespace qualifier.
  if (cmd_name.find("::") != string::npos)
    {
      return cmd_name_cstr;
    }
  else
    {
      if (!(flags & NO_EXPORT))
        {
          const tcl::namesp ns(rep->interp, rep->namesp_name.c_str());

          ns.export_cmd(rep->interp, cmd_name_cstr);
        }

      static string name;
      name = namesp_name();
      name += "::";
      name += cmd_name;
      return name.c_str();
    }
}

void tcl::pkg::eval(const char* script)
{
GVX_TRACE("tcl::pkg::eval");
  rep->interp.eval(script);
}

void tcl::pkg::link_var(const char* var_name, int& var)
{
GVX_TRACE("tcl::pkg::link_var int");
  rep->interp.link_int(var_name, &var, false);
}

void tcl::pkg::link_var(const char* var_name, double& var)
{
GVX_TRACE("tcl::pkg::link_var double");
  rep->interp.link_double(var_name, &var, false);
}

void tcl::pkg::link_var_copy(const char* var_name, int var)
{
GVX_TRACE("tcl::pkg::link_var_copy int");
  shared_ptr<int> copy(new int(var));
  rep->owned_ints.push_back(copy);
  rep->interp.link_int(var_name, copy.get(), true);
}

void tcl::pkg::link_var_copy(const char* var_name, double var)
{
GVX_TRACE("tcl::pkg::link_var_copy double");
  shared_ptr<double> copy(new double(var));
  rep->owned_doubles.push_back(copy);
  rep->interp.link_double(var_name, copy.get(), true);
}

void tcl::pkg::link_var_const(const char* var_name, int& var)
{
GVX_TRACE("tcl::pkg::link_var_const int");
  rep->interp.link_int(var_name, &var, true);
}

void tcl::pkg::link_var_const(const char* var_name, double& var)
{
GVX_TRACE("tcl::pkg::link_var_const double");
  rep->interp.link_double(var_name, &var, true);
}

void tcl::pkg::set_init_status_error() throw()
{
GVX_TRACE("tcl::pkg::set_init_status_error");
  rep->init_status = TCL_ERROR;
}

void tcl::pkg::verbose_init(bool verbose) throw()
{
GVX_TRACE("tcl::pkg::verbose_init");

  VERBOSE_INIT = verbose;
}

int tcl::pkg::finish_init() throw()
{
GVX_TRACE("tcl::pkg::finish_init");

  --INIT_DEPTH;

  if (rep->init_status == TCL_OK)
    {
      if (VERBOSE_INIT)
        {
          for (int i = 0; i < INIT_DEPTH; ++i)
            std::cerr << "    ";
          std::cerr << pkg_name() << " initialized.\n";
        }

      if ( !rep->pkg_name.empty() && !rep->version.empty() )
        {
          Tcl_PkgProvideEx(rep->interp.intp(),
                           rep->pkg_name.c_str(), rep->version.c_str(),
                           static_cast<void*>(this));
        }

      Tcl_CreateExitHandler(&impl::c_exit_handler,
                            static_cast<void*>(this));

      return rep->init_status;
    }

  // else (rep->init_status != TCL_OK)

  delete this;
  return TCL_ERROR;
}

const char* const tcl::pkg::action_usage = "objref(s)";
const char* const tcl::pkg::getter_usage = "objref(s)";
const char* const tcl::pkg::setter_usage = "objref(s) new_value(s)";

static const char __attribute__((used)) vcid_groovx_tcl_pkg_cc_utc20050628162420[] = "$Id: pkg.cc 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/pkg.cc $";
#endif // !GROOVX_TCL_PKG_CC_UTC20050628162420_DEFINED
