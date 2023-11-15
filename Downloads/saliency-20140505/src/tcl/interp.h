/** @file tcl/interp.h c++ wrapper for Tcl_Interp, translates between
    tcl error codes and c++ exceptions */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2000-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Wed Oct 11 10:25:36 2000
// commit: $Id: interp.h 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/interp.h $
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

#ifndef GROOVX_TCL_INTERP_H_UTC20050628162420_DEFINED
#define GROOVX_TCL_INTERP_H_UTC20050628162420_DEFINED

#include "tcl/conversions.h"
#include "tcl/obj.h"

#include "rutz/fileposition.h"
#include "rutz/shared_ptr.h"

struct Tcl_Interp;
typedef struct Tcl_Obj Tcl_Obj;

namespace rutz
{
  class fstring;
}

namespace tcl
{
  class interpreter;
  class list;

  /// Different error-handling strategies for tcl::interpreter::eval().
  enum error_strategy
    {
      THROW_ERROR,
      IGNORE_ERROR
    };
}


//  ########################################################
/// tcl::interpreter provides a wrapper around Tcl_Interp calls.
/** The advantage over using the raw Tcl C API is that certain error
    conditions are handled in a more C++-ish way, by throwing
    exceptions. */

class tcl::interpreter
{
  interpreter& operator=(const interpreter&);

public:
  interpreter(Tcl_Interp* interp);
  interpreter(const interpreter& other) throw();
  ~interpreter() throw();

  // Interpreter
  bool is_valid() const throw() { return m_interp != 0; }

  /// Get the interpreter (if valid), otherwise throw an exception.
  Tcl_Interp* intp() const;

  bool is_deleted() const throw();
  void forget_interp() throw();
  void destroy() throw();

  /// Wrapper around Tcl_PkgProvide().
  void pkg_provide(const char* name, const char* version);

  /// Evaluate the given expression, return its result as a bool.
  bool eval_boolean_expr(const tcl::obj& obj) const;

  /// Evaluates code.
  /** If strategy is THROW_ERROR, then an exception is thrown if the
      evaluation produces an error. If strategy is IGNORE_ERROR, then
      a return value of true indicates a successful evaluation, and a
      return value of false indicates an error during evaluation. */
  bool eval(const char* code, error_strategy strategy = THROW_ERROR);

  /// Evaluates code.
  /** If strategy is THROW_ERROR, then an exception is thrown if the
      evaluation produces an error. If strategy is IGNORE_ERROR, then
      a return value of true indicates a successful evaluation, and a
      return value of false indicates an error during evaluation. */
  bool eval(const rutz::fstring& code, error_strategy strategy = THROW_ERROR);

  /// Evaluates code.
  /** If strategy is THROW_ERROR, then an exception is thrown if the
      evaluation produces an error. If strategy is IGNORE_ERROR, then
      a return value of true indicates a successful evaluation, and a
      return value of false indicates an error during evaluation. */
  bool eval(const tcl::obj& code, error_strategy strategy = THROW_ERROR);

  /// Evaluates code using Tcl_EvalObjv(), exploiting the fact that the object is already a list.
  /** If strategy is THROW_ERROR, then an exception is thrown if the
      evaluation produces an error. If strategy is IGNORE_ERROR, then
      a return value of true indicates a successful evaluation, and a
      return value of false indicates an error during evaluation. */
  bool eval_objv(const tcl::list& objv, error_strategy strategy = THROW_ERROR);

  /// Evaluate the tcl code in the named file.
  /** Returns true on success, or false on failure. */
  bool eval_file(const char* fname);

  void source_rc_file();

  // Result
  void reset_result() const;
  void append_result(const char* msg) const;
  void append_result(const rutz::fstring& msg) const;

  template <class T>
  T get_result() const
  {
    return tcl::convert_to<T>(get_obj_result());
  }

  template <class T>
  void set_result(const T& x)
  {
    set_obj_result(tcl::convert_from(x).get());
  }

  // Variables
  void set_global_var(const char* var_name, const tcl::obj& var) const;
  void unset_global_var(const char* var_name) const;

  template <class T>
  T get_global_var(const char* name1, const char* name2=0) const
  {
    return tcl::convert_to<T>(get_obj_global_var(name1, name2));
  }

  void link_int(const char* var_name, int* addr, bool read_only);
  void link_double(const char* var_name, double* addr, bool read_only);
  void link_boolean(const char* var_name, int* addr, bool read_only);

  // Errors
  void handle_live_exception(const char* where,
                             const rutz::file_pos& pos) throw();
  void background_error() throw();

  void add_error_info(const char* info);

  // Events
  static void clear_event_queue();

  // Commands/procedures
  bool has_command(const char* cmd_name) const;
  void delete_command(const char* cmd_name);

  rutz::fstring get_proc_body(const char* proc_name);
  void create_proc(const char* namesp, const char* proc_name,
                   const char* args, const char* body);
  void delete_proc(const char* namesp, const char* proc_name);

private:
  Tcl_Obj* get_obj_result() const;
  Tcl_Obj* get_obj_global_var(const char* name1, const char* name2) const;
  void set_obj_result(Tcl_Obj* obj);

  Tcl_Interp* m_interp;
};

static const char __attribute__((used)) vcid_groovx_tcl_interp_h_utc20050628162420[] = "$Id: interp.h 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/interp.h $";
#endif // !GROOVX_TCL_INTERP_H_UTC20050628162420_DEFINED
