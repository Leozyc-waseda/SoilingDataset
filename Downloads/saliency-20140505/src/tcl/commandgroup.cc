/** @file tcl/commandgroup.cc represents a set of overloaded
    tcl::command objects */
///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2004-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Wed Jun  9 09:45:36 2004
// commit: $Id: commandgroup.cc 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/commandgroup.cc $
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

#ifndef GROOVX_TCL_COMMANDGROUP_CC_UTC20050628162421_DEFINED
#define GROOVX_TCL_COMMANDGROUP_CC_UTC20050628162421_DEFINED

#include "tcl/commandgroup.h"

#include "tcl/command.h"
#include "tcl/obj.h"
#include "tcl/interp.h"

#include "rutz/fileposition.h"
#include "rutz/fstring.h"
#include "rutz/sfmt.h"
#include "rutz/shared_ptr.h"

#include <list>
#include <sstream>
#include <tcl.h>
#ifdef HAVE_TCLINT_H
#include <tclInt.h> // for Tcl_GetFullCommandName()
#endif

#include "rutz/trace.h"
#include "rutz/debug.h"
GVX_DBG_REGISTER

using rutz::fstring;
using rutz::shared_ptr;

namespace
{
#ifdef HAVE_TCLINT_H
  fstring get_full_command_name(tcl::interpreter& interp,
                                Tcl_Command token)
  {
    tcl::obj result;
    // Note, this Tcl API call requires Tcl 8.4.6 or greater (or 8.5
    // or greater)
    Tcl_GetCommandFullName(interp.intp(), token, result.get());
    return tcl::convert_to<rutz::fstring>(result);
  }
#endif

  void append_usage(std::ostream& dest, const fstring& usage)
  {
    if (!usage.is_empty())
      dest << " " << usage;
  }
}

class tcl::command_group::impl
{
public:
  impl(tcl::interpreter& intp, const fstring& cmd_name,
       const rutz::file_pos& src_pos)
    :
    interp(intp),
    cmd_token(Tcl_CreateObjCommand(interp.intp(),
                                   cmd_name.c_str(),
                                   // see comment in command_group's
                                   // constructor for why we pass
                                   // zeros here
                                   static_cast<Tcl_ObjCmdProc*>(0),
                                   static_cast<ClientData>(0),
                                   static_cast<Tcl_CmdDeleteProc*>(0))),
#ifdef HAVE_TCLINT_H
    initial_cmd_name(get_full_command_name(intp, cmd_token)),
#else
    initial_cmd_name(cmd_name),
#endif
    cmd_list(),
    prof_name(rutz::sfmt("tcl/%s", cmd_name.c_str())),
    prof(prof_name.c_str(), src_pos.m_file_name, src_pos.m_line_no)
  {}

  ~impl() throw() {}

  typedef std::list<shared_ptr<tcl::command> > cmd_list_type;

  tcl::interpreter       interp;
  Tcl_Command      const cmd_token;
  fstring          const initial_cmd_name;
  cmd_list_type          cmd_list;
  fstring          const prof_name;
  rutz::prof             prof;

  fstring usage_warning(const fstring& argv0) const;

  static int c_invoke_callback(void* clientdata,
                               Tcl_Interp* interp,
                               int s_objc,
                               Tcl_Obj *const objv[]) throw();

  static void c_delete_callback(void* clientdata) throw();

  static void c_exit_callback(void* clientdata) throw();

  static tcl::command_group* lookup_helper(tcl::interpreter& interp,
                                           const char* name) throw();

private:
  impl(const impl&);
  impl& operator=(const impl&);
};

fstring tcl::command_group::impl::usage_warning(
                                      const fstring& argv0) const
{
GVX_TRACE("tcl::command_group::usage_warning");

  std::ostringstream warning;
  warning << "wrong # args: should be ";

  if (cmd_list.size() == 1)
    {
      warning << "\"" << argv0;
      append_usage(warning, cmd_list.front()->usage_string());
      warning << "\"";
    }
  else
    {
      warning << "one of:";
      for (impl::cmd_list_type::const_iterator
             itr = cmd_list.begin(),
             end = cmd_list.end();
           itr != end;
           ++itr)
        {
          warning << "\n\t\"" << argv0;
          append_usage(warning, (*itr)->usage_string());
          warning << "\"";
        }
    }

  warning << "\n(";

  if (argv0 != initial_cmd_name)
    warning << "resolves to " << initial_cmd_name << ", ";

  warning << "defined at "
          << prof.src_file_name() << ":"
          << prof.src_line_no() << ")";

  return fstring(warning.str().c_str());
}

#ifdef REAL_BACKTRACE
#include <execinfo.h>
#endif

int tcl::command_group::impl::c_invoke_callback(
    void* clientdata,
    Tcl_Interp* interp,
    int s_objc,
    Tcl_Obj *const objv[]) throw()
{
  command_group* c = static_cast<command_group*>(clientdata);

  GVX_ASSERT(c != 0);
  GVX_ASSERT(interp == c->rep->interp.intp());

#ifdef REAL_BACKTRACE
  typedef void* voidp;

  voidp addresses[64];

  const int n = backtrace(&addresses[0], 64);

  char** names = backtrace_symbols(&addresses[0], n);

  for (int i = 0; i < n; ++i)
    {
      dbg_print_nl(0, names[i]);
    }

  free(names);
#endif

  return c->invoke_raw(s_objc, objv);
}

void tcl::command_group::impl::c_delete_callback(
    void* clientdata) throw()
{
GVX_TRACE("tcl::command_group::impl::c_delete_callback");
  command_group* c = static_cast<command_group*>(clientdata);
  GVX_ASSERT(c != 0);
  delete c;
}

void tcl::command_group::impl::c_exit_callback(
    void* clientdata) throw()
{
GVX_TRACE("tcl::command_group::c_exit_callback");
  command_group* c = static_cast<command_group*>(clientdata);
  GVX_ASSERT(c != 0);
  Tcl_DeleteCommandFromToken(c->rep->interp.intp(), c->rep->cmd_token);
}

tcl::command_group* tcl::command_group::impl::lookup_helper(
    tcl::interpreter& interp,
    const char* name) throw()
{
GVX_TRACE("tcl::command_group::impl::lookup_helper");

  /*
    typedef struct Tcl_CmdInfo {
    int isNativeObjectProc;
    Tcl_ObjCmdProc *objProc;
    void* objClientData;
    Tcl_CmdProc *proc;
    void* clientdata;
    Tcl_CmdDeleteProc *deleteProc;
    void* deleteData;
    Tcl_Namespace *namespacePtr;
    } Tcl_CmdInfo;
  */
  Tcl_CmdInfo info;
  const int result = Tcl_GetCommandInfo(interp.intp(), name, &info);

  if (result == 1 &&
      info.isNativeObjectProc == 1 &&
      info.objProc == &impl::c_invoke_callback &&
      info.deleteProc == &impl::c_delete_callback)
    {
      return static_cast<command_group*>(info.objClientData);
    }
  return 0;
}

tcl::command_group::command_group(tcl::interpreter& interp,
                                const fstring& cmd_name,
                                const rutz::file_pos& src_pos)
  :
  rep(new impl(interp, cmd_name, src_pos))
{
GVX_TRACE("tcl::command_group::command_group");

  // Register the command procedure. We do a two-step
  // initialization. When we call Tcl_CreateObjCommand in impl's
  // constructor, we don't fill in the clientdata/objProc/deleteProc
  // values there, but instead wait to fill them in here. The reason
  // is that we don't want to set up any callbacks from Tcl until
  // after we're sure that everything else in the construction
  // sequence has succeeded. We want to ensure that we don't have
  // "dangling callbacks" in case an exception escapes from an earlier
  // part of impl's or CommandGroups's constructor.
  Tcl_CmdInfo info;
  const int result = Tcl_GetCommandInfoFromToken(rep->cmd_token, &info);
  GVX_ASSERT(result == 1);
  GVX_ASSERT(info.isNativeObjectProc == 1);
  info.objClientData = static_cast<void*>(this);
  info.objProc = &impl::c_invoke_callback;
  info.deleteData = static_cast<void*>(this);
  info.deleteProc = &impl::c_delete_callback;
  Tcl_SetCommandInfoFromToken(rep->cmd_token, &info);

  Tcl_CreateExitHandler(&impl::c_exit_callback,
                        static_cast<void*>(this));
}

// A destruction sequence can get triggered in a number of ways:
/*
   (1) application exit might trigger the c_exit_callback

   (2) the c_delete_callback might get triggered either by explicit
       deletion by the user (e.g. [rename]ing the command to the empty
       string "")

   General principles:

   (1) it is always "safe" to destroy the Tcl_Command, in the sense
       that it can't cause any crashes... in particular, it's OK to
       destroy the Tcl_Command even if rep->cmd_list is not empty; that
       would just mean that the remaining tcl::command objects in
       rep->cmd_list won't have any input sent their way
 */
tcl::command_group::~command_group() throw()
{
GVX_TRACE("tcl::command_group::~command_group");

  Tcl_DeleteExitHandler(&impl::c_exit_callback,
                        static_cast<void*>(this));

  delete rep;
}

tcl::command_group* tcl::command_group::lookup(tcl::interpreter& interp,
                                             const char* name) throw()
{
GVX_TRACE("tcl::command_group::lookup");

  return impl::lookup_helper(interp, name);
}

tcl::command_group* tcl::command_group::lookup_original(
                                            tcl::interpreter& interp,
                                            const char* name) throw()
{
GVX_TRACE("tcl::command_group::lookup_original");

  const fstring script = rutz::sfmt("namespace origin %s", name);
  if (interp.eval(script, tcl::IGNORE_ERROR) == false)
    {
      return 0;
    }

  // else...
  const fstring original = interp.get_result<fstring>();
  return impl::lookup_helper(interp, original.c_str());
}

shared_ptr<tcl::command>
tcl::command_group::make(tcl::interpreter& interp,
                         shared_ptr<tcl::function> callback,
                         const char* cmd_name,
                         const char* usage,
                         const arg_spec& spec,
                         const rutz::file_pos& src_pos)
{
GVX_TRACE("tcl::command_group::make");

  // Here we want to find the command_group that corresponds to the
  // given command name, creating it anew if necessary. Then, we
  // create a new tcl::command object and add it to the group. The
  // command_group object handles the actual interface with tcl, and
  // when the command_group gets callback from tcl, it selects among
  // its various tcl::command overloads by checking which one matches
  // the number of arguments in the current command invocation (see
  // invoke_raw()).

  command_group* group =
    tcl::command_group::lookup(interp, cmd_name);

  if (group == 0)
    group = new command_group(interp, cmd_name, src_pos);

  GVX_ASSERT(group != 0);

  shared_ptr<tcl::command> cmd(new tcl::command(callback, usage, spec));

  // We don't want to have to keep 'group' as a member of tcl::command
  // since it involves circular references -- tcl::command_group keeps
  // a list of tcl::command objects, so we'd prefer tcl::command to
  // not need a backreference to tcl::command_group. If it becomes
  // necessary to keep a back-reference, then there needs to be a way
  // for tcl::command_group to notify its tcl::command list that it is
  // destructing, so that the tcl::command objects can "forget" their
  // back-reference.
  group->add(cmd);

  return cmd;
}

void tcl::command_group::add(shared_ptr<tcl::command> p)
{
GVX_TRACE("tcl::command_group::add");
  rep->cmd_list.push_back(p);
}

fstring tcl::command_group::resolved_name() const
{
#ifdef HAVE_TCLINT_H
  return get_full_command_name(rep->interp, rep->cmd_token);
#else
  return rep->initial_cmd_name;
#endif
}

fstring tcl::command_group::usage() const
{
GVX_TRACE("tcl::command_group::usage");

  std::ostringstream result;

  impl::cmd_list_type::const_iterator
    itr = rep->cmd_list.begin(),
    end = rep->cmd_list.end();

  while (true)
    {
      result << "\t" << resolved_name();
      append_usage(result, (*itr)->usage_string());
      result << "\n";
      if (++itr == end)
        break;
    }

  result << "\t(defined at "
         << rep->prof.src_file_name() << ":"
         << rep->prof.src_line_no() << ")";

  return fstring(result.str().c_str());
}

int tcl::command_group::invoke_raw(int s_objc,
                                   Tcl_Obj *const objv[]) throw()
{
GVX_TRACE("tcl::command_group::invoke_raw");

  // This is to use the separate rutz::prof object that each
  // command_group has. This way we can trace the timing of individual
  // Tcl commands.
  rutz::trace tracer(rep->prof, GVX_TRACE_EXPR);

  // Should always be at least one since there is always the
  // command-name itself as the first argument.
  GVX_ASSERT(s_objc >= 1);

  if (GVX_DBG_LEVEL() > 1)
    {
      for (int argi = 0; argi < s_objc; ++argi)
        {
          const char* arg = Tcl_GetString(objv[argi]);
          dbg_print(1, argi);
          dbg_print(1, " argv = ");
          dbg_print_nl(1, arg);
        }
    }

  unsigned int objc = static_cast<unsigned int>(s_objc);

  // catch all possible exceptions since this is a callback from C
  try
    {
      for (impl::cmd_list_type::const_iterator
             itr = rep->cmd_list.begin(),
             end = rep->cmd_list.end();
           itr != end;
           ++itr)
        {
          if ((*itr)->rejects_argc(objc))
            continue;

          // Found a matching overload, so try it:
          (*itr)->call(rep->interp, objc, objv);

          if (GVX_DBG_LEVEL() > 1)
            {
              const char* result = rep->interp.get_result<const char*>();
              dbg_eval_nl(1, result);
            }
          return TCL_OK;
        }

      const fstring argv0(Tcl_GetString(objv[0]));

      // Here, we run out of potential overloads, so abort the command.
      rep->interp.reset_result();
      rep->interp.append_result(rep->usage_warning(argv0).c_str());
      return TCL_ERROR;
    }
  catch (...)
    {
      rep->interp.handle_live_exception(Tcl_GetString(objv[0]), SRC_POS);
    }

  return TCL_ERROR;
}

static const char __attribute__((used)) vcid_groovx_tcl_commandgroup_cc_utc20050628162421[] = "$Id: commandgroup.cc 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/commandgroup.cc $";
#endif // !GROOVX_TCL_COMMANDGROUP_CC_UTC20050628162421_DEFINED
