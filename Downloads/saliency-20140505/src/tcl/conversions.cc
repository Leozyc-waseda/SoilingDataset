/** @file tcl/conversions.cc tcl conversion functions for basic types */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2001-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Wed Jul 11 08:58:53 2001
// commit: $Id: conversions.cc 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/conversions.cc $
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

#ifndef GROOVX_TCL_CONVERSIONS_CC_UTC20050628162420_DEFINED
#define GROOVX_TCL_CONVERSIONS_CC_UTC20050628162420_DEFINED

#include "tcl/conversions.h"

#include "rutz/error.h"
#include "rutz/fstring.h"
#include "rutz/sfmt.h"
#include "rutz/value.h"

#include <limits>
#include <tcl.h>

#include "rutz/trace.h"
#include "rutz/debug.h"
GVX_DBG_REGISTER

using rutz::fstring;

///////////////////////////////////////////////////////////////////////
//
// File scope functions
//
///////////////////////////////////////////////////////////////////////

namespace
{
  class safe_unshared_obj
  {
  private:
    Tcl_Obj* m_obj;
    bool m_is_owning;

    safe_unshared_obj(const safe_unshared_obj&);
    safe_unshared_obj& operator=(const safe_unshared_obj&);

  public:
    safe_unshared_obj(Tcl_Obj* obj, const Tcl_ObjType* target_type) :
      m_obj(obj), m_is_owning(false)
    {
      if ( (m_obj->typePtr != target_type) && Tcl_IsShared(m_obj) )
        {
          m_is_owning = true;
          m_obj = Tcl_DuplicateObj(m_obj);
          Tcl_IncrRefCount(m_obj);
        }
    }

    Tcl_Obj* get() const { return m_obj; }

    ~safe_unshared_obj()
    {
      if (m_is_owning)
        Tcl_DecrRefCount(m_obj);
    }
  };
}

///////////////////////////////////////////////////////////////////////
//
// (Tcl --> C++) aux_convert_to specializations
//
///////////////////////////////////////////////////////////////////////

int tcl::aux_convert_to(Tcl_Obj* obj, int*)
{
GVX_TRACE("tcl::aux_convert_to(int*)");

  int val;

  static const Tcl_ObjType* const int_type = Tcl_GetObjType("int");

  GVX_ASSERT(int_type != 0);

  safe_unshared_obj safeobj(obj, int_type);

  if ( Tcl_GetIntFromObj(0, safeobj.get(), &val) != TCL_OK )
    {
      throw rutz::error(rutz::sfmt("expected integer but got \"%s\"",
                                   Tcl_GetString(obj)), SRC_POS);
    }

  return val;
}

unsigned int tcl::aux_convert_to(Tcl_Obj* obj, unsigned int*)
{
GVX_TRACE("tcl::aux_convert_to(unsigned int*)");

  int sval = aux_convert_to(obj, static_cast<int*>(0));

  if (sval < 0)
    {
      throw rutz::error(rutz::sfmt("expected integer but got \"%s\" "
                                   "(value was negative)",
                                   Tcl_GetString(obj)), SRC_POS);
    }

  return static_cast<unsigned int>(sval);
}

long tcl::aux_convert_to(Tcl_Obj* obj, long*)
{
GVX_TRACE("tcl::aux_convert_to(long*)");

  Tcl_WideInt wideval;

  static const Tcl_ObjType* const wide_int_type = Tcl_GetObjType("wideInt");

  GVX_ASSERT(wide_int_type != 0);

  safe_unshared_obj safeobj(obj, wide_int_type);

  const long longmax = std::numeric_limits<long>::max();
  const long longmin = std::numeric_limits<long>::min();

  if ( Tcl_GetWideIntFromObj(0, safeobj.get(), &wideval) != TCL_OK )
    {
      throw rutz::error(rutz::sfmt("expected long value but got \"%s\"",
                                   Tcl_GetString(obj)), SRC_POS);
    }
  else if (wideval > static_cast<Tcl_WideInt>(longmax))
    {
      throw rutz::error(rutz::sfmt("expected long value but got \"%s\" "
                                   "(value too large, max is %ld)",
                                   Tcl_GetString(obj),
                                   longmax), SRC_POS);
    }
  else if (wideval < static_cast<Tcl_WideInt>(longmin))
    {
      throw rutz::error(rutz::sfmt("expected long value but got \"%s\" "
                                   "(value too small, min is %ld)",
                                   Tcl_GetString(obj),
                                   longmin), SRC_POS);
    }

  return static_cast<long>(wideval);
}

unsigned long tcl::aux_convert_to(Tcl_Obj* obj, unsigned long*)
{
GVX_TRACE("tcl::aux_convert_to(unsigned long*)");

  Tcl_WideInt wideval;

  static const Tcl_ObjType* const wide_int_type = Tcl_GetObjType("int"); /// wideInt??

  GVX_ASSERT(wide_int_type != 0);

  safe_unshared_obj safeobj(obj, wide_int_type);

  const unsigned long ulongmax = std::numeric_limits<unsigned long>::max();

  if ( Tcl_GetWideIntFromObj(0, safeobj.get(), &wideval) != TCL_OK )
    {
      throw rutz::error(rutz::sfmt("expected unsigned long value "
                                   "but got \"%s\"", Tcl_GetString(obj)),
                        SRC_POS);
    }
  else if (wideval < 0)
    {
      throw rutz::error(rutz::sfmt("expected unsigned long value "
                                   "but got \"%s\" (value was negative)",
                                   Tcl_GetString(obj)), SRC_POS);
    }
  // OK, now we know our wideval is non-negative, so we can safely
  // cast it to an unsigned type (Tcl_WideUInt) for comparison against
  // ulongmax (note: don't try to do this comparison by casting
  // ulongmax to a signed type like Tcl_WideInt, since the result of
  // the cast will be a negative number, leading to a bogus
  // comparison)
  else if (static_cast<Tcl_WideUInt>(wideval) > ulongmax)
    {
      throw rutz::error(rutz::sfmt("expected unsigned long value "
                                   "but got \"%s\" "
                                   "(value too large, max is %lu)",
                                   Tcl_GetString(obj), ulongmax),
                        SRC_POS);
    }

  return static_cast<unsigned long>(wideval);
}

long long tcl::aux_convert_to(Tcl_Obj* obj, long long*)
{
GVX_TRACE("tcl::aux_convert_to(long long*)");

  static const Tcl_ObjType* const wide_int_type = Tcl_GetObjType("wideInt");

  GVX_ASSERT(wide_int_type != 0);

  safe_unshared_obj safeobj(obj, wide_int_type);

  long long wideval;

  if ( Tcl_GetWideIntFromObj(0, safeobj.get(), &wideval) != TCL_OK )
    {
      throw rutz::error(rutz::sfmt("expected long value but got \"%s\"",
                                   Tcl_GetString(obj)), SRC_POS);
    }

  return wideval;
}

bool tcl::aux_convert_to(Tcl_Obj* obj, bool*)
{
GVX_TRACE("tcl::aux_convert_to(bool*)");

  int int_val;

  static const Tcl_ObjType* const boolean_type = Tcl_GetObjType("boolean");

  GVX_ASSERT(boolean_type != 0);

  safe_unshared_obj safeobj(obj, boolean_type);

  if ( Tcl_GetBooleanFromObj(0, safeobj.get(), &int_val) != TCL_OK )
    {
      throw rutz::error(rutz::sfmt("expected boolean value but got \"%s\"",
                                   Tcl_GetString(obj)), SRC_POS);
    }
  return bool(int_val);
}

double tcl::aux_convert_to(Tcl_Obj* obj, double*)
{
GVX_TRACE("tcl::aux_convert_to(double*)");

  double val;

  static const Tcl_ObjType* const double_type = Tcl_GetObjType("double");

  GVX_ASSERT(double_type != 0);

  safe_unshared_obj safeobj(obj, double_type);

  if ( Tcl_GetDoubleFromObj(0, safeobj.get(), &val) != TCL_OK )
    {
      throw rutz::error(rutz::sfmt("expected floating-point number "
                                   "but got \"%s\"",
                                   Tcl_GetString(obj)), SRC_POS);
    }
  return val;
}

float tcl::aux_convert_to(Tcl_Obj* obj, float*)
{
GVX_TRACE("tcl::aux_convert_to(float*)");

  return float(aux_convert_to(obj, static_cast<double*>(0)));
}

const char* tcl::aux_convert_to(Tcl_Obj* obj, const char**)
{
GVX_TRACE("tcl::aux_convert_to(const char**)");

  return Tcl_GetString(obj);
}

fstring tcl::aux_convert_to(Tcl_Obj* obj, fstring*)
{
GVX_TRACE("tcl::aux_convert_to(fstring*)");

  int length;

  char* text = Tcl_GetStringFromObj(obj, &length);

  GVX_ASSERT(length >= 0);

  return fstring(rutz::char_range(text, static_cast<unsigned int>(length)));
}


///////////////////////////////////////////////////////////////////////
//
// (C++ --> Tcl) aux_convert_from specializations
//
///////////////////////////////////////////////////////////////////////

tcl::obj tcl::aux_convert_from(long long val)
{
GVX_TRACE("tcl::aux_convert_from(long long)");

  return Tcl_NewWideIntObj(val);
}

tcl::obj tcl::aux_convert_from(long val)
{
GVX_TRACE("tcl::aux_convert_from(long)");

  return Tcl_NewLongObj(val);
}

tcl::obj tcl::aux_convert_from(unsigned long val)
{
GVX_TRACE("tcl::aux_convert_from(unsigned long)");

  long sval(val);

  if (sval < 0)
    throw rutz::error("signed/unsigned conversion failed", SRC_POS);

  return Tcl_NewLongObj(sval);
}

tcl::obj tcl::aux_convert_from(int val)
{
GVX_TRACE("tcl::aux_convert_from(int)");

  return Tcl_NewIntObj(val);
}

tcl::obj tcl::aux_convert_from(unsigned int val)
{
GVX_TRACE("tcl::aux_convert_from(unsigned int)");

  int sval(val);

  if (sval < 0)
    throw rutz::error("signed/unsigned conversion failed", SRC_POS);

  return Tcl_NewIntObj(sval);
}

tcl::obj tcl::aux_convert_from(unsigned char val)
{
GVX_TRACE("tcl::aux_convert_from(unsigne char)");

  return Tcl_NewIntObj(val);
}

tcl::obj tcl::aux_convert_from(bool val)
{
GVX_TRACE("tcl::aux_convert_from(bool)");

  return Tcl_NewBooleanObj(val);
}

tcl::obj tcl::aux_convert_from(double val)
{
GVX_TRACE("tcl::aux_convert_from(double)");

  return Tcl_NewDoubleObj(val);
}

tcl::obj tcl::aux_convert_from(float val)
{
GVX_TRACE("tcl::aux_convert_from(float)");

  return Tcl_NewDoubleObj(val);
}

tcl::obj tcl::aux_convert_from(const char* val)
{
GVX_TRACE("tcl::aux_convert_from(const char*)");

  return Tcl_NewStringObj(val, -1);
}

tcl::obj tcl::aux_convert_from(const fstring& val)
{
GVX_TRACE("tcl::aux_convert_from(const fstring&)");

  return Tcl_NewStringObj(val.c_str(), val.length());
}

tcl::obj tcl::aux_convert_from(const rutz::value& val)
{
GVX_TRACE("tcl::aux_convert_from(const rutz::value&)");

  return Tcl_NewStringObj(val.get_string().c_str(), -1);
}

static const char __attribute__((used)) vcid_groovx_tcl_conversions_cc_utc20050628162420[] = "$Id: conversions.cc 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/conversions.cc $";
#endif // !GROOVX_TCL_CONVERSIONS_CC_UTC20050628162420_DEFINED
