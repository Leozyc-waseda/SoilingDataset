/** @file tcl/pkg.h tcl package class, holds a set of commands, wraps
    calls to Tcl_PkgProvide(), etc. */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 1999-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Tue Jun 15 12:33:59 1999
// commit: $Id: pkg.h 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/pkg.h $
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

#ifndef GROOVX_TCL_PKG_H_UTC20050628162421_DEFINED
#define GROOVX_TCL_PKG_H_UTC20050628162421_DEFINED

#include "tcl/makecmd.h"

#include "rutz/fileposition.h"

struct Tcl_Interp;

namespace rutz
{
  struct file_pos;
}

namespace tcl
{
  class command;
  class interpreter;
  class pkg;

  const int NO_EXPORT = 1 << 0;
}

///////////////////////////////////////////////////////////////////////
//
// tcl::pkg class definition
//
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/**

    \c tcl::pkg is a class more managing groups of related \c
    tcl::command's. It provides several facilities:

    -- stores a list of \c tcl::command's, and ensures that these are
       properly destroyed upon exit from Tcl

    -- ensures that the package is provided to Tcl so that other
       packages can query for its presence

    -- provides a set of functions to define Tcl commands from C++
       functors

 **/
///////////////////////////////////////////////////////////////////////

class tcl::pkg
{
private:
  /// Private constructor.
  /** Clients should use the PKG_CREATE macro instead. */
  pkg(Tcl_Interp* interp, const char* name, const char* version);

  /// Destructor destroys all \c tcl::command's owned by the package.
  ~pkg() throw();

public:
  static const int STATUS_OK;
  static const int STATUS_ERR;

  /// Don't call this directly! Use the PKG_CREATE macro instead.
  static pkg* create_in_macro(Tcl_Interp* interp,
                              const char* name, const char* version)
  {
    return new pkg(interp, name, version);
  }

  typedef void (exit_callback)();

  /// Specify a function to be called when the package is destroyed.
  /** (Package destruction typically occurs at application exit, when
      the Tcl interpreter and all associated objects are
      destroyed.) */
  void on_exit(exit_callback* callback);

  /// Looks up the tcl::pkg associated with pkgname, and destroys it.
  /** This is intended to be called from pkg_Unload procedures called
      by Tcl when a dynamic library is unloaded. The return value can
      be returned as the return value of the pkg_Unload procedure; it
      will be TCL_OK (1) if the tcl::pkg was successfully found and
      destroyed and TCL_ERROR (0) otherwise. */
  static int destroy_on_unload(Tcl_Interp* interp, const char* pkgname);

  /// Find a package given its name and version.
  /** If the package is not already loaded, this function will attempt
      to "require" the package. If a null pointer is passed to version
      (the default), then any version will be acceptable. If no
      suitable package cannot be found or loaded, a null pointer will
      be returned. */
  static pkg* lookup(tcl::interpreter& interp,
                     const char* name, const char* version = 0) throw();

  /** Returns a Tcl status code indicating whether the package
      initialization was successful. */
  int init_status() const throw();

  /// Mark the package as having failed its initialization.
  void set_init_status_error() throw();

  /// Returns the Tcl interpreter that was passed to the constructor.
  tcl::interpreter& interp() throw();

  /// Trap a live exception, and leave a message in the Tcl_Interp's result.
  void handle_live_exception(const rutz::file_pos& pos) throw();

  /// Returns the package's "namespace name".
  /** Note that the "namespace name" will be the same as the "package
      name" except possibly for capitalization. The "namespace name"
      is the name of the namespace that is used as the default prefix
      all commands contained in the package. */
  const char* namesp_name() throw();

  /// Return the package's "package name".
  /** Note that the "package name" will be the same as the "namespace
      name" except possibly for capitalization. The "package name" is
      the name that is passed to Tcl_PkgProvide() and
      Tcl_PkgProvide(), and has a well-defined capitalization scheme:
      first character uppercase, all remaining letters lowercase. */
  const char* pkg_name() const throw();

  /// Returns the package version string.
  const char* version() const throw();

  /// Export commands into a different namespace.
  /** Causes all of our package's currently defined commands and
      procedures to be imported into the specified other namespace. If
      pattern is different from the default value of "*", then only
      commands whose names match pattern according to glob rules will
      be aliased into the other namespace. */
  void namesp_alias(const char* namesp, const char* pattern = "*");

  /// Import commands from a different namespace.
  /** Import all of the commands and procedures defined in the
      specified namespace into our own package namespace. If pattern
      is different from the default value of "*", then only commands
      whose names match pattern according to glob rules will be
      imported into our own package namespace. */
  void inherit_namesp(const char* namesp, const char* pattern = "*");

  /// Import all commands and procedures defined in the named pkg.
  /** If the named pkg has not yet been loaded, this function will
      attempt to load it via loookup(). If a null pointer is passed to
      version (the default), then any version will be acceptable. */
  void inherit_pkg(const char* name, const char* version = 0);

  /// Evaluates \a script using the package's \c Tcl_Interp.
  void eval(const char* script);

  /// Links the \a var with the Tcl variable \a var_name.
  void link_var(const char* var_name, int& var);

  /// Links \a var with the Tcl variable \a var_name.
  void link_var(const char* var_name, double& var);

  /// Links a copy of \a var with the Tcl variable \a var_name.
  /** The Tcl variable will be read-only.*/
  void link_var_copy(const char* var_name, int var);

  /// Links a copy of \a var with the Tcl variable \a var_name.
  /** The Tcl variable will be read-only.*/
  void link_var_copy(const char* var_name, double var);

  /// Links \a var with the Tcl variable \a var_name.
  /** The Tcl variable will be read_only. */
  void link_var_const(const char* var_name, int& var);

  ///Links \a var with the Tcl variable \a var_name.
  /** The Tcl variable will be read_only. */
  void link_var_const(const char* var_name, double& var);


  template <class Func>
  inline void def(const char* cmd_name, const char* usage, Func f,
                  const rutz::file_pos& src_pos, int flags = 0)
  {
    make_command(interp(), f, make_pkg_cmd_name(cmd_name, flags),
                 usage, src_pos);
  }

  template <class Func>
  inline void def_vec(const char* cmd_name, const char* usage, Func f,
                      unsigned int keyarg /*default is 1*/,
                      const rutz::file_pos& src_pos, int flags = 0)
  {
    make_vec_command(interp(), f, make_pkg_cmd_name(cmd_name, flags),
                     usage, keyarg, src_pos);
  }

  template <class Func>
  inline void def_raw(const char* cmd_name, const arg_spec& spec,
                      const char* usage, Func f,
                      const rutz::file_pos& src_pos, int flags = 0)
  {
    make_generic_command(interp(), f, make_pkg_cmd_name(cmd_name, flags),
                         usage, spec, src_pos);
  }

  template <class Func>
  inline void def_vec_raw(const char* cmd_name, const arg_spec& spec,
                          const char* usage, Func f,
                          unsigned int keyarg /*default is 1*/,
                          const rutz::file_pos& src_pos, int flags = 0)
  {
    make_generic_vec_command(interp(), f, make_pkg_cmd_name(cmd_name, flags),
                             usage, spec, keyarg, src_pos);
  }

  template <class C>
  void def_action(const char* cmd_name, void (C::* action_func) (),
                  const rutz::file_pos& src_pos, int flags = 0)
  {
    def_vec( cmd_name, action_usage, action_func, 1, src_pos, flags );
  }

  template <class C>
  void def_action(const char* cmd_name, void (C::* action_func) () const,
                  const rutz::file_pos& src_pos, int flags = 0)
  {
    def_vec( cmd_name, action_usage, action_func, 1, src_pos, flags );
  }

  template <class C, class T>
  void def_getter(const char* cmd_name, T (C::* getter_func) () const,
                  const rutz::file_pos& src_pos, int flags = 0)
  {
    def_vec( cmd_name, getter_usage, getter_func, 1, src_pos, flags );
  }

  template <class C, class T>
  void def_setter(const char* cmd_name, void (C::* setter_func) (T),
                  const rutz::file_pos& src_pos, int flags = 0)
  {
    def_vec( cmd_name, setter_usage, setter_func, 1, src_pos, flags );
  }

  template <class C, class T>
  void def_get_set(const char* cmd_name,
                   T (C::* getter_func) () const,
                   void (C::* setter_func) (T),
                   const rutz::file_pos& src_pos, int flags = 0)
  {
    def_getter( cmd_name, getter_func, src_pos, flags );
    def_setter( cmd_name, setter_func, src_pos, flags );
  }

  /// Control whether packages should be verbose as they start up.
  static void verbose_init(bool verbose) throw();

  /// Called just prior to returning from the *_Init function.
  /** If the package's status is OK, then this does the relevant
      Tcl_PkgProvide and returns TCL_OK. Otherwise, it returns
      TCL_ERROR. */
  int finish_init() throw();

private:
  pkg(const pkg&); // not implemented
  pkg& operator=(const pkg&); // not implemented

  /** Returns a namespace'd command name in the form of
      pkg_name::cmd_name. The result of this function is valid only
      until the next time it is called, so callers should make a copy
      of the result. This function also has the side effect of setting
      up a Tcl namespace export pattern for the named command, if
      flags doesn't include NO_EXPORT. */
  const char* make_pkg_cmd_name(const char* cmd_name, int flags);

  static const char* const action_usage;
  static const char* const getter_usage;
  static const char* const setter_usage;

  struct impl;
  friend struct impl;
  impl* rep;
};

#include "rutz/debug.h"
GVX_DBG_REGISTER

/*
  These macros make it slightly more convenient to make sure that
  *_Init() package initialization functions don't leak any exceptions
  (as they are called directly from C code within the Tcl core).
 */

/// This macro should go at the top of each *_Init() function.
/** Constructs a \c tcl::pkg with a Tcl interpreter, package name, and
    package version. The version string should be in the form MM.mm
    where MM is major version, and mm is minor version. This
    constructor can also correctly parse a version string such as
    given by the RCS revision tag. If you're using svn, the suggested
    form is to choose a fixed major version number, and let the svn
    revision be the minor number, so you would pass a version string
    such as "4.$Revision: 11876 $". */
#define GVX_PKG_CREATE(pkg, interp, pkgname, pkgversion)              \
                                                                      \
int GVX_PKG_STATUS = tcl::pkg::STATUS_ERR;                            \
{                                                                     \
  tcl::pkg* pkg = 0;                                                  \
                                                                      \
  try                                                                 \
    { pkg = tcl::pkg::create_in_macro(interp, pkgname, pkgversion); } \
  catch (...)                                                         \
    { return 1; }                                                     \
                                                                      \
  static bool recursive_initialization = false;                       \
  GVX_ASSERT(!recursive_initialization);                              \
  recursive_initialization = true;                                    \
                                                                      \
  try                                                                 \
  {


/// This macro should go at the end of each *_Init() function.
#define GVX_PKG_RETURN(pkg)                     \
  }                                             \
  catch(...)                                    \
  {                                             \
    pkg->handle_live_exception(SRC_POS);        \
  }                                             \
  recursive_initialization = false;             \
  GVX_PKG_STATUS = pkg->finish_init();          \
}                                               \
return GVX_PKG_STATUS;

/// Use this instead of GVX_PKG_RETURN(pkg) if more work needs to be done after the package is initialized.
#define GVX_PKG_FINISH(pkg)                     \
  }                                             \
  catch(...)                                    \
  {                                             \
    pkg->handle_live_exception(SRC_POS);        \
  }                                             \
  recursive_initialization = false;             \
  GVX_PKG_STATUS = pkg->finish_init();          \
}


static const char __attribute__((used)) vcid_groovx_tcl_pkg_h_utc20050628162421[] = "$Id: pkg.h 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/pkg.h $";
#endif // !GROOVX_TCL_PKG_H_UTC20050628162421_DEFINED
