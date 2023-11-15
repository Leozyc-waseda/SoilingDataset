/** @file tcl/argspec.h specify min/max number of arguments that a tcl
    command can accept */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2005-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Mon Jun 27 16:47:04 2005
// commit: $Id: argspec.h 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/argspec.h $
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

#ifndef GROOVX_TCL_ARGSPEC_H_UTC20050628064704_DEFINED
#define GROOVX_TCL_ARGSPEC_H_UTC20050628064704_DEFINED

#include <limits>

namespace tcl
{
  /// Specify how many args a command can take.
  /** By convention, argc_min() and argc_max() INCLUDE the zero'th
      argument (i.e. the command name) in the arg count. Thus a
      command that takes no parameters would have an arg count of
      1. If is_exact() is true, then the argc of a command invocation
      is required to be exactly equal either argc_min() or argc_max();
      if it is false, then argc must be between argc_min() and
      argc_max(), inclusive. */
  class arg_spec
  {
  public:
    arg_spec()
      :
      m_argc_min(0),
      m_argc_max(0),
      m_is_exact(false)
    {}

    /// Construct with initial values for m_argc_min/m_argc_max/m_is_exact.
    /** If the value given for nmax is negative, then m_argc_max will
        be set to the same value as nmin. */
    explicit arg_spec(int nmin, int nmax = -1, bool ex = false)
      :
      m_argc_min(nmin < 0
                 ? 0
                 : static_cast<unsigned int>(nmin)),
      m_argc_max(nmax == -1
                 ? m_argc_min
                 : static_cast<unsigned int>(nmax)),
      m_is_exact(ex)
    {}

    arg_spec& min(int nmin) { m_argc_min = nmin; return *this; }
    arg_spec& max(int nmax) { m_argc_max = nmax; return *this; }
    arg_spec& exact(bool ex) { m_is_exact = ex; return *this; }

    arg_spec& nolimit()
    {
      m_argc_max = std::numeric_limits<unsigned int>::max();
      m_is_exact = false;
      return *this;
    }

    bool allows_argc(unsigned int objc) const
    {
      if (this->m_is_exact)
        {
          return (objc == this->m_argc_min ||
                  objc == this->m_argc_max);
        }
      // else...
      return (objc >= this->m_argc_min &&
              objc <= this->m_argc_max);
    }

    unsigned int argc_min() const { return m_argc_min; }
    unsigned int argc_max() const { return m_argc_max; }
    bool         is_exact() const { return m_is_exact; }

  private:
    unsigned int m_argc_min;
    unsigned int m_argc_max;
    bool         m_is_exact;
  };
}

static const char __attribute__((used)) vcid_groovx_tcl_argspec_h_utc20050628064704[] = "$Id: argspec.h 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/argspec.h $";
#endif // !GROOVX_TCL_ARGSPEC_H_UTC20050628064704DEFINED
