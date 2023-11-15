/** @file tcl/eventloop.cc singleton class that operates the tcl main
    event loop, reading commands from a script file or from stdin,
    with readline-enabled command-line editing */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Mon Jul 22 16:34:05 2002
// commit: $Id: eventloop.cc 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/eventloop.cc $
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

#ifndef GROOVX_TCL_EVENTLOOP_CC_UTC20050628162420_DEFINED
#define GROOVX_TCL_EVENTLOOP_CC_UTC20050628162420_DEFINED

#include "tcl/eventloop.h"

#include "tcl/interp.h"

#include "rutz/backtrace.h"
#include "rutz/backtraceformat.h"
#include "rutz/error.h"
#include "rutz/fstring.h"
#include "rutz/sfmt.h"

#include <iostream>
#include <sstream>
#include <string>
#include <tk.h>
#include <unistd.h>

#ifndef GVX_NO_READLINE
#define GVX_WITH_READLINE
#endif

#ifdef GVX_WITH_READLINE
#  include <cstdlib> // for malloc/free
#  include <readline/readline.h>
#  include <readline/history.h>
#endif

#include "rutz/trace.h"
#include "rutz/debug.h"
GVX_DBG_REGISTER

namespace tcl
{
  class event_loop_impl;
}

namespace
{
  void c_exit_handler(void* /*clientdata*/) throw()
  {
#ifdef GVX_WITH_READLINE
    rl_callback_handler_remove();
#endif
  }
}

// Singleton implementation class for tcl::event_loop
class tcl::event_loop_impl
{
private:
  static event_loop_impl* s_event_loop_impl;

  // Data members

  int                 m_argc;
  const char**        m_argv;
  tcl::interpreter    m_interp;
  const char*         m_startup_filename;
  const char*         m_argv0;
  Tcl_Channel         m_stdin_chan;
  std::string         m_command; // Build lines of tty input into Tcl commands.
  bool                m_got_partial;
  bool                m_is_interactive; // True if input is a terminal-like device.
  rutz::fstring       m_command_line; // Entire command-line as a string
  bool                m_no_window; // whether this is a windowless environment

  // Function members

  event_loop_impl(int argc, char** argv, bool nowindow);

  int history_next();

  void do_prompt(const char* text, unsigned int length);

  enum prompt_type { FULL, PARTIAL };

  void prompt(prompt_type t);

  void grab_input();

  void handle_line(const char* line, int count);

  void eval_command();

  static void c_stdin_proc(void* /*clientdata*/, int /*mask*/);

public:
  static void create(int argc, char** argv, bool nowindow)
  {
    GVX_ASSERT(s_event_loop_impl == 0);

    s_event_loop_impl = new event_loop_impl(argc, argv, nowindow);
    Tcl_CreateExitHandler(c_exit_handler, static_cast<void*>(0));
  }

  static event_loop_impl* get()
  {
    if (s_event_loop_impl == 0)
      {
        throw rutz::error("no tcl::event_loop object has yet been created",
                          SRC_POS);
      }

    return s_event_loop_impl;
  }

  bool is_interactive() const { return m_is_interactive; }

  tcl::interpreter& interp() { return m_interp; }

  void run();

  int argc() const { return m_argc; }

  const char* const* argv() const { return m_argv; }

  rutz::fstring command_line() const { return m_command_line; }

#ifdef GVX_WITH_READLINE
  static void readline_line_complete(char* line);
#endif
};

tcl::event_loop_impl* tcl::event_loop_impl::s_event_loop_impl = 0;

//---------------------------------------------------------------------
//
// tcl::event_loop_impl::event_loop_impl()
//
//---------------------------------------------------------------------

tcl::event_loop_impl::event_loop_impl(int argc, char** argv, bool nowindow) :
  m_argc(argc),
  m_argv(const_cast<const char**>(argv)),
  m_interp(Tcl_CreateInterp()),
  m_startup_filename(0),
  m_argv0(0),
  m_stdin_chan(0),
  m_command(),
  m_got_partial(false),
  m_is_interactive(isatty(0)),
  m_command_line(),
  m_no_window(nowindow)
{
GVX_TRACE("tcl::event_loop_impl::event_loop_impl");

  Tcl_FindExecutable(argv[0]);

  {
    std::ostringstream buf;

    buf << argv[0];

    for (int i = 1; i < argc; ++i)
      {
        buf << " " << argv[i];
      }

    m_command_line = rutz::fstring(buf.str().c_str());
  }

  // Parse command-line arguments.  If the next argument doesn't start
  // with a "-" then strip it off and use it as the name of a script
  // file to process.

  if ((argc > 1) && (argv[1][0] != '-'))
    {
      m_argv0 = m_startup_filename = argv[1];
      --argc;
      ++argv;
      m_is_interactive = false;
    }
  else
    {
      m_argv0 = argv[0];
    }

  // Make command-line arguments available in the Tcl variables "argc"
  // and "argv".

  m_interp.set_global_var("argc", tcl::convert_from(argc-1));

  char* args = Tcl_Merge(argc-1,
                         const_cast<const char**>(argv+1));
  m_interp.set_global_var("argv", tcl::convert_from(args));
  Tcl_Free(args);

  m_interp.set_global_var("argv0", tcl::convert_from(m_argv0));

  m_interp.set_global_var("tcl_interactive",
                          tcl::convert_from(m_is_interactive ? 1 : 0));

#ifdef GVX_WITH_READLINE
  using_history();
#endif
}

//---------------------------------------------------------------------
//
// Get the next number in the history count.
//
//---------------------------------------------------------------------

int tcl::event_loop_impl::history_next()
{
GVX_TRACE("tcl::event_loop_impl::history_next");

#ifdef GVX_WITH_READLINE
  return history_length+1;
#else
  tcl::obj obj = m_interp.get_result<Tcl_Obj*>();

  m_interp.eval("history nextid", tcl::IGNORE_ERROR);

  int result = m_interp.get_result<int>();

  m_interp.set_result(obj);

  return result;
#endif
}

//---------------------------------------------------------------------
//
// Actually write the prompt characters to the terminal.
//
//---------------------------------------------------------------------

void tcl::event_loop_impl::do_prompt(const char* text,
#ifdef GVX_WITH_READLINE
                             unsigned int /*length*/
#else
                             unsigned int length
#endif
                             )
{
GVX_TRACE("tcl::event_loop_impl::do_prompt");

  rutz::fstring color_prompt = text;

  if (isatty(1))
    {
#if defined(GVX_WITH_READLINE) && defined(RL_PROMPT_START_IGNORE)
      color_prompt = rutz::sfmt("%c\033[1;32m%c%s%c\033[0m%c",
                                RL_PROMPT_START_IGNORE,
                                RL_PROMPT_END_IGNORE,
                                text,
                                RL_PROMPT_START_IGNORE,
                                RL_PROMPT_END_IGNORE);
#else
      color_prompt = rutz::sfmt("\033[1;32m%s\033[0m", text);
#endif
    }

#ifdef GVX_WITH_READLINE
  rl_callback_handler_install(color_prompt.c_str(),
                              readline_line_complete);
#else
  if (length > 0)
    {
      std::cout.write(color_prompt.c_str(), color_prompt.length());
      std::cout.flush();
    }
#endif
}

//---------------------------------------------------------------------
//
// tcl::event_loop_impl::prompt()
//
//---------------------------------------------------------------------

void tcl::event_loop_impl::prompt(tcl::event_loop_impl::prompt_type t)
{
GVX_TRACE("tcl::event_loop_impl::prompt");

  if (t == PARTIAL)
    {
      do_prompt("", 0);
    }
  else
    {
#ifdef GVX_WITH_READLINE
      const rutz::fstring text = rutz::sfmt("%s %d>>> ", m_argv0, history_next());
#else
      const rutz::fstring text = rutz::sfmt("%s %d> ", m_argv0, history_next());
#endif

      do_prompt(text.c_str(), text.length());
    }
}

//---------------------------------------------------------------------
//
// Callback triggered from the readline library when it has a full
// line of input for us to handle.
//
//---------------------------------------------------------------------

#ifdef GVX_WITH_READLINE

void tcl::event_loop_impl::readline_line_complete(char* line)
{
GVX_TRACE("tcl::event_loop_impl::readline_line_complete");

  dbg_eval_nl(3, line);

  rl_callback_handler_remove();

  get()->handle_line(line, line == 0 ? -1 : int(strlen(line)));
}

#endif

//---------------------------------------------------------------------
//
// Pull any new characters in from the input stream. Returns the
// number of characters read from the input stream.
//
//---------------------------------------------------------------------

void tcl::event_loop_impl::grab_input()
{
GVX_TRACE("tcl::event_loop_impl::grab_input");

#ifndef GVX_WITH_READLINE
  Tcl_DString line;

  Tcl_DStringInit(&line);

  int count = Tcl_Gets(m_stdin_chan, &line);

  handle_line(Tcl_DStringValue(&line), count);

  Tcl_DStringFree(&line);

#else // GVX_WITH_READLINE
  rl_callback_read_char();
#endif
}

//---------------------------------------------------------------------
//
// Handle a complete line of input (though not necessarily a complete
// Tcl command).
//
//---------------------------------------------------------------------

void tcl::event_loop_impl::handle_line(const char* line, int count)
{
GVX_TRACE("tcl::event_loop_impl::handle_line");

  if (count < 0)
    {
      if (!m_got_partial)
        {
          if (m_is_interactive)
            {
              // OK, here we're in interactive mode and we're going to
              // exit because stdin has been closed -- this means that
              // the user probably typed Ctrl-D, so let's send a
              // newline to stdout before exiting so that the user's
              // shell prompt starts on a fresh line rather than
              // overwriting the final tcl prompt
              Tcl_Channel out_chan = Tcl_GetStdChannel(TCL_STDOUT);
              if (out_chan)
                {
                  const char nl = '\n';
                  Tcl_WriteChars(out_chan, &nl, 1);
                }

              Tcl_Exit(0);
            }
          else
            {
              Tcl_DeleteChannelHandler(m_stdin_chan,
                                       &c_stdin_proc,
                                       static_cast<void*>(0));
            }
        }
      return;
    }

  GVX_ASSERT(line != 0);

  m_command += line;
  m_command += "\n";

  dbg_eval_nl(3, m_command.c_str());

  if (m_command.length() > 0 &&
      Tcl_CommandComplete(m_command.c_str()))
    {
      m_got_partial = false;
      eval_command();
    }
  else
    {
      m_got_partial = true;
    }

  if (m_is_interactive)
    {
      prompt(m_got_partial ? PARTIAL : FULL);
    }

  m_interp.reset_result();
}

//---------------------------------------------------------------------
//
// Executes a complete command string in the Tcl interpreter.
//
//---------------------------------------------------------------------

void tcl::event_loop_impl::eval_command()
{
GVX_TRACE("tcl::event_loop_impl::eval_command");

  // Disable the stdin channel handler while evaluating the command;
  // otherwise if the command re-enters the event loop we might
  // process commands from stdin before the current command is
  // finished.  Among other things, this will trash the text of the
  // command being evaluated.

  Tcl_CreateChannelHandler(m_stdin_chan, 0, &c_stdin_proc,
                           static_cast<void*>(0));

  bool should_display_result = false;

#ifdef GVX_WITH_READLINE
  char* expansion = 0;
  const int status =
    history_expand(const_cast<char*>(m_command.c_str()), &expansion);
#else
  const char* expansion = m_command.data();
  const int status = 0;
#endif

  dbg_eval_nl(3, m_command.c_str());
  dbg_eval_nl(3, expansion);
  dbg_eval_nl(3, status);

  // status: -1 --> error
  //          0 --> no expansions occurred
  //          1 --> expansions occurred
  //          2 --> display but don't execute

  if (status == -1 || status == 2) // display expansion?
    {
      m_interp.append_result(expansion);
      should_display_result = true;
    }

  if (status == 1)
    {
      Tcl_Channel out_chan = Tcl_GetStdChannel(TCL_STDOUT);
      if (out_chan)
        {
          Tcl_WriteChars(out_chan, expansion, -1);
          Tcl_Flush(out_chan);
        }
    }

  if (status == 0 || status == 1) // execute expansion?
    {
      // The idea here is that we want to keep the readline history
      // and the Tcl history in sync. Tcl's "history add" command will
      // skip adding the string if it is empty or has whitespace
      // only. So, we need to make that same check here before adding
      // to the readline history. In fact, if we find that the command
      // is empty, we can just skip executing it altogether.

      // Skip over leading whitespace
#ifdef GVX_WITH_READLINE
      char* trimmed = expansion;
#else
      const char* trimmed = expansion;
#endif

      while (isspace(trimmed[0]) && trimmed[0] != '\0')
        {
          ++trimmed;
        }

      size_t len = strlen(trimmed);

      if (len > 0)
        {
          int code = Tcl_RecordAndEval(m_interp.intp(),
                                       trimmed, TCL_EVAL_GLOBAL);

#ifdef GVX_WITH_READLINE
          char c = trimmed[len-1];

          if (c == '\n')
            trimmed[len-1] = '\0';

          add_history(trimmed);

          trimmed[len-1] = c;
#endif

          dbg_eval_nl(3, m_interp.get_result<const char*>());

          should_display_result =
            ((m_interp.get_result<const char*>())[0] != '\0') &&
            ((code != TCL_OK) || m_is_interactive);
        }
    }

  if (should_display_result)
    {
      Tcl_Channel out_chan = Tcl_GetStdChannel(TCL_STDOUT);
      if (out_chan)
        {
          Tcl_WriteObj(out_chan, m_interp.get_result<Tcl_Obj*>());
          Tcl_WriteChars(out_chan, "\n", 1);
        }
    }

  m_stdin_chan = Tcl_GetStdChannel(TCL_STDIN);

  if (m_stdin_chan)
    {
      Tcl_CreateChannelHandler(m_stdin_chan, TCL_READABLE,
                               &c_stdin_proc,
                               static_cast<void*>(0));
    }

  m_command.clear();

#ifdef GVX_WITH_READLINE
  free(expansion);
#endif
}

//---------------------------------------------------------------------
//
// This procedure is invoked by the event dispatcher whenever standard
// input becomes readable.  It grabs the next line of input
// characters, adds them to a command being assembled, and executes
// the command if it's complete.
//
//---------------------------------------------------------------------

void tcl::event_loop_impl::c_stdin_proc(void* /*clientdata*/, int /*mask*/)
{
GVX_TRACE("tcl::event_loop_impl::c_stdin_proc");

  tcl::event_loop_impl::get()->grab_input();
}

//---------------------------------------------------------------------
//
// tcl::event_loop_impl::run()
//
//---------------------------------------------------------------------

void tcl::event_loop_impl::run()
{
GVX_TRACE("tcl::event_loop_impl::run");

  /*
   * Invoke the script specified on the command line, if any.
   */

  if (m_startup_filename != NULL)
    {
      m_interp.reset_result();
      bool success = m_interp.eval_file(m_startup_filename);
      if (!success)
        {
          // ensure errorInfo is set properly:
          m_interp.add_error_info("");

          rutz::backtrace b;
          rutz::error::get_last_backtrace(b);
          const rutz::fstring bt = rutz::format(b);

          std::cerr << m_interp.get_global_var<const char*>("errorInfo")
                    << "\n" << bt << "\nError in startup script\n";
          m_interp.destroy();
          Tcl_Exit(1);
        }
    }
  else
    {
      // Evaluate the .rc file, if one has been specified.
      m_interp.source_rc_file();

      // Set up a stdin channel handler.
      m_stdin_chan = Tcl_GetStdChannel(TCL_STDIN);
      if (m_stdin_chan)
        {
          Tcl_CreateChannelHandler(m_stdin_chan, TCL_READABLE,
                                   &event_loop_impl::c_stdin_proc,
                                   static_cast<void*>(0));
        }
      if (m_is_interactive)
        {
          this->prompt(FULL);
        }
    }

  Tcl_Channel out_channel = Tcl_GetStdChannel(TCL_STDOUT);
  if (out_channel)
    {
      Tcl_Flush(out_channel);
    }
  m_interp.reset_result();

  // Loop indefinitely, waiting for commands to execute, until there
  // are no main windows left, then exit.

  while ((m_is_interactive && m_no_window)
         || Tk_GetNumMainWindows() > 0)
    {
      Tcl_DoOneEvent(0);
    }
  m_interp.destroy();
  Tcl_Exit(0);
}

///////////////////////////////////////////////////////////////////////
//
// tcl::event_loop functions delegate to tcl::event_loop_impl
//
///////////////////////////////////////////////////////////////////////

tcl::event_loop::event_loop(int argc, char** argv, bool nowindow)
{
  tcl::event_loop_impl::create(argc, argv, nowindow);
}

tcl::event_loop::~event_loop()
{}

bool tcl::event_loop::is_interactive()
{
GVX_TRACE("tcl::event_loop::is_interactive");
  return tcl::event_loop_impl::get()->is_interactive();
}

tcl::interpreter& tcl::event_loop::interp()
{
GVX_TRACE("tcl::event_loop::interp");
  return tcl::event_loop_impl::get()->interp();
}

void tcl::event_loop::run()
{
GVX_TRACE("tcl::event_loop::run");
  tcl::event_loop_impl::get()->run();
}

int tcl::event_loop::argc()
{
GVX_TRACE("tcl::event_loop::argc");
  return tcl::event_loop_impl::get()->argc();
}

const char* const* tcl::event_loop::argv()
{
GVX_TRACE("tcl::event_loop::argv");
  return tcl::event_loop_impl::get()->argv();
}

rutz::fstring tcl::event_loop::command_line()
{
GVX_TRACE("tcl::event_loop::command_line");
  return tcl::event_loop_impl::get()->command_line();
}

static const char __attribute__((used)) vcid_groovx_tcl_eventloop_cc_utc20050628162420[] = "$Id: eventloop.cc 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/eventloop.cc $";
#endif // !GROOVX_TCL_EVENTLOOP_CC_UTC20050628162420_DEFINED
