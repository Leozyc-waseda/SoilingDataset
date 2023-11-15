/** @file tcl/scriptapp.cc helper class used in main() to initialize
    and run a scripting application */
///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2005-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Mon Jun 27 13:34:19 2005
// commit: $Id: scriptapp.cc 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/scriptapp.cc $
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

#ifndef GROOVX_TCL_SCRIPTAPP_CC_UTC20050628162421_DEFINED
#define GROOVX_TCL_SCRIPTAPP_CC_UTC20050628162421_DEFINED

#include "tcl/scriptapp.h"

#include "nub/objfactory.h"

#include "rutz/sfmt.h"

#include "tcl/list.h"
#include "tcl/eventloop.h"
#include "tcl/pkg.h"
#include "tcl/interp.h"

#include <cstring>
#include <iostream>
#include <signal.h>
#include <sstream>
#include <tk.h>

#include "rutz/debug.h"
GVX_DBG_REGISTER
#include "rutz/trace.h"

namespace
{

  bool havearg(char** args, const char* arg)
  {
    for ( ; *args != 0; ++args)
      if (strcmp(*args, arg) == 0)
        return true;

    return false;
  }

  std::string centerline(unsigned int totallen,
                         const char* pfx, const char* sfx,
                         std::string txt)
  {
    if (strlen(pfx) + strlen(sfx) >= totallen)
      {
        // ok, the line's too long, so don't do any centering, just
        // keep it as short as possible:
        std::string out(pfx);
        out += txt;
        out += sfx;
        return out;
      }

    unsigned int midlen = totallen - strlen(pfx) - strlen(sfx);

    std::string out(pfx);

    if (txt.length() < midlen)
      {
        int c = midlen - txt.length();
        while (--c >= 0)
          { if (c % 2) txt += ' '; else out += ' '; }
      }

    out += txt;
    out += sfx;

    return out;
  }

  std::string wrapstring(unsigned int totallen,
                         const char* pfx, const char* sfx,
                         const std::string& s)
  {
    GVX_ASSERT(strlen(pfx) + strlen(sfx) < totallen);
    unsigned int len = totallen - strlen(pfx) - strlen(sfx);

    std::istringstream strm(s);
    std::string out;
    std::string line;
    std::string word;
    while (strm >> word)
      {
        if (word.length() + line.length() + 1 <= len)
          {
            if (line.length() > 0)
              line += ' ';
            line += word;
          }
        else
          {
            out += line;
            out += '\n';
            line = word;
          }
      }

    out += line;

    return out;
  }

  std::string centerlines(unsigned int totallen,
                          const char* pfx, const char* sfx,
                          std::string ss)
  {
    if (ss.length() == 0)
      {
        return centerline(totallen, pfx, sfx, ss);
      }

    std::istringstream strm(ss);
    std::string line;
    std::string out;
    bool first = true;
    while (getline(strm, line))
      {
        if (!first) out += '\n';
        first = false;
        out += centerline(totallen, pfx, sfx, line);
      }

    return out;
  }

  std::string wrapcenterlines(unsigned int totallen,
                              const char* pfx, const char* sfx,
                              std::string ss, std::string emptyline)
  {
    std::istringstream strm(ss);
    std::string line;
    std::string out;
    while (getline(strm, line))
      {
        if (line.length() == 0)
          {
            out += emptyline;
            out += '\n';
          }
        else
          {
            out += centerlines(totallen, pfx, sfx,
                               wrapstring(totallen, pfx, sfx, line));
            out += '\n';
          }
      }

    return out;
  }

  // This is a fallback function to be used by the object
  // factory... if the factory can't figure out how to create a given
  // type, it will call this fallback function first before giving up
  // for good. This callback function tries to load a Tcl package
  // named after the desired object type.
  void factory_pkg_loader(const rutz::fstring& type)
  {
    dbg_eval_nl(3, type);

    tcl::pkg::lookup(tcl::event_loop::interp(), type.c_str());
  }

  void sig_handler(int signum)
  {
    switch (signum)
      {
        case SIGSEGV: GVX_PANIC("Segmentation fault (SIGSEGV)");
        case SIGFPE:  GVX_PANIC("Floating point exception (SIGFPE)");
        case SIGBUS:  GVX_PANIC("Bus error (SIGBUS)");
      }
    GVX_ASSERT(0);
  }

}

void tcl::script_app::init_in_macro_only()
{
  signal(SIGSEGV, &sig_handler);
  signal(SIGFPE, &sig_handler);
  signal(SIGBUS, &sig_handler);
}

void tcl::script_app::handle_exception_in_macro_only
                                         (const std::exception* e)
{
  if (e != 0)
    std::cerr << "caught in main: ("
              << rutz::demangled_name(typeid(*e))
              << "): " << e->what() << '\n';
  else
    std::cerr << "caught in main: (an exception of unknown type)\n";
}

tcl::script_app::script_app(const char* appname,
                            int argc_, char** argv_) throw()
  :
  m_appname(appname),
  m_script_argc(0),
  m_script_argv(new char*[argc_+1]),
  m_minimal(false),
  m_nowindow(false),
  m_splashmsg(),
  m_pkgdir(),
  m_pkgs(0),
  m_exitcode(0)
{
  // We are going to take a quick pass over the command-line args here
  // to see if there are any we care about; if there are, then we will
  // cull those from the arg list that gets exposed to the script.

  // Quick check argv to optionally turn on global tracing and/or set
  // the global debug level. This method is particularly useful for
  // helping to diagnose problems that are occurring during
  // application startup, before we have a chance to get to the
  // command-line prompt and do a "::gtrace 1" or a "::dbgLevel 9".
  for (int i = 0; i < argc_; ++i)
    {
      if (strcmp(argv_[i], "-dbglevel") == 0)
        {
          ++i;
          if (argv_[i] != 0)
            rutz::debug::set_global_level( atoi(argv_[i]) );
        }
      else if (strcmp(argv_[i], "-gtrace") == 0)
        {
          rutz::trace::set_global_trace(true);
        }
      else if (strcmp(argv_[i], "-showinit") == 0)
        {
          tcl::pkg::verbose_init(true);
        }
      else if (strcmp(argv_[i], "-minimal") == 0)
        {
          this->m_minimal = true;
        }
      else if (strcmp(argv_[i], "-nw") == 0)
        {
          this->m_nowindow = true;
        }
      else
        {
          // ok, we didn't recognize this arg, so we'll pass it along
          // to the script:
          m_script_argv[m_script_argc++] = argv_[i];
        }
    }

  // now null-terminate the argv that will be passed to the script:
  m_script_argv[m_script_argc] = 0;
}

tcl::script_app::~script_app() throw()
{
  delete [] m_script_argv;
}

void tcl::script_app::run()
{
  tcl::event_loop tclmain(this->m_script_argc,
                          this->m_script_argv, this->m_nowindow);

  if (tcl::event_loop::is_interactive())
    {
      const char* const pfx = "###  ";
      const char* const sfx = "  ###";
      const unsigned int linelen = 75;
      std::string hashes(linelen, '#');

      std::cerr << hashes
                << '\n'
                << wrapcenterlines(linelen, pfx, sfx,
                                   m_splashmsg.c_str(), hashes)
                << hashes << '\n' << '\n';
    }

  tcl::interpreter& interp = tclmain.interp();

  nub::obj_factory::instance().set_fallback(&factory_pkg_loader);
  nub::set_default_ref_vis(nub::PUBLIC);

  const rutz::time ru1 = rutz::time::user_rusage();
  const rutz::time rs1 = rutz::time::sys_rusage();
  const rutz::time wc1 = rutz::time::wall_clock_now();

  package_info IMMEDIATE_PKGS[] =
    {
      { "Tcl",      Tcl_Init,  "", false },
      { "Tk",       Tk_Init,   "", true },
    };

  for (size_t i = 0; i < sizeof(IMMEDIATE_PKGS)/sizeof(package_info); ++i)
    {
      if (m_nowindow && IMMEDIATE_PKGS[i].requires_gui)
        continue;

      int result = IMMEDIATE_PKGS[i].init_proc(interp.intp());
      if (result != TCL_OK)
        {
          std::cerr << "fatal initialization error (package '"
                    << IMMEDIATE_PKGS[i].name << "'):\n";
          rutz::fstring msg = interp.get_result<const char*>();
          if ( !msg.is_empty() )
            std::cerr << '\t' << msg << '\n';
          interp.reset_result();

          this->m_exitcode = 2; return;
        }
    }

  if (tcl::event_loop::is_interactive())
    {
      const rutz::time ru = rutz::time::user_rusage() - ru1;
      const rutz::time rs = rutz::time::sys_rusage() - rs1;
      const rutz::time wc = rutz::time::wall_clock_now() - wc1;

      fprintf(stderr, "\tstartup time (%6s) "
              "%6.3fs (user) %6.3fs (sys) %6.3fs (wall)\n",
              "tcl+tk", ru.sec(), rs.sec(), wc.sec());
    }

  const rutz::time ru2 = rutz::time::user_rusage();
  const rutz::time rs2 = rutz::time::sys_rusage();
  const rutz::time wc2 = rutz::time::wall_clock_now();

  for (const package_info* pkg = m_pkgs; pkg->name != 0; ++pkg)
    {
      Tcl_StaticPackage(static_cast<Tcl_Interp*>(0),
                        // (Tcl_Interp*) 0 means this package
                        // hasn't yet been loaded into any
                        // interpreter
                        pkg->name,
                        pkg->init_proc,
                        0);

      const rutz::fstring ifneededcmd =
        rutz::sfmt("package ifneeded %s %s {load {} %s }",
                   pkg->name, pkg->version, pkg->name);

      interp.eval(ifneededcmd);
    }

  if (!m_minimal)
    {
      for (const package_info* pkg = m_pkgs; pkg->name != 0; ++pkg)
        {
          if (m_nowindow && pkg->requires_gui)
            continue;

          const char* ver =
            Tcl_PkgRequire(interp.intp(), pkg->name, pkg->version, 0);

          if (ver == 0)
            {
              std::cerr << "initialization error (package '"
                        << pkg->name << "'):\n";
              rutz::fstring msg = interp.get_result<const char*>();
              if ( !msg.is_empty() )
                std::cerr << '\t' << msg << '\n';
              interp.reset_result();
            }
        }
    }

  if (tcl::event_loop::is_interactive())
    {
      const rutz::time ru = rutz::time::user_rusage() - ru2;
      const rutz::time rs = rutz::time::sys_rusage() - rs2;
      const rutz::time wc = rutz::time::wall_clock_now() - wc2;

      fprintf(stderr, "\tstartup time (%6s) "
              "%6.3fs (user) %6.3fs (sys) %6.3fs (wall)\n",
              m_appname.c_str(), ru.sec(), rs.sec(), wc.sec());
    }

  tcl::list path = interp.get_global_var<tcl::list>("auto_path");

  if (m_pkgdir.length() > 0)
    path.append(m_pkgdir);

  interp.set_global_var("auto_path", path.as_obj());

  // specifies a file to be 'source'd upon startup
  interp.set_global_var("tcl_rcFileName",
                      tcl::convert_from("./groovx_startup.tcl"));

  tclmain.run();
}

static const char __attribute__((used)) vcid_groovx_tcl_scriptapp_cc_utc20050628162421[] = "$Id: scriptapp.cc 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/scriptapp.cc $";
#endif // !GROOVX_TCL_SCRIPTAPP_CC_UTC20050628162421_DEFINED
