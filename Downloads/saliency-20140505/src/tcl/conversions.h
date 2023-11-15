/** @file tcl/conversions.h tcl conversion functions for basic types */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2001-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Wed Jul 11 08:57:31 2001
// commit: $Id: conversions.h 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/conversions.h $
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

#ifndef GROOVX_TCL_CONVERSIONS_H_UTC20050628162420_DEFINED
#define GROOVX_TCL_CONVERSIONS_H_UTC20050628162420_DEFINED

#include "rutz/traits.h"
#include "tcl/obj.h"

typedef struct Tcl_Obj Tcl_Obj;

namespace rutz
{
  class fstring;
  class value;
  template <class T> class fwd_iter;
}

namespace nub
{
  template <class T> class ref;
  template <class T> class soft_ref;
}

namespace tcl
{
  class obj;

  /// Trait class for extracting an appropriate return-type from T.
  template <class T>
  struct returnable
  {
    // This machinery is simple to set up the rule that we want to
    // convert all rutz::value subclasses via strings. All other types
    // are converted directly.
    typedef typename rutz::select_if<
      rutz::is_sub_super<T, rutz::value>::result,
      rutz::fstring, T>::result_t
    type;
  };

  /// Specialization of tcl::returnable for const T.
  template <class T>
  struct returnable<const T>
  {
    typedef typename rutz::select_if<
      rutz::is_sub_super<T, rutz::value>::result,
      rutz::fstring, T>::result_t
    type;
  };

  /// Specialization of tcl::returnable for const T&.
  template <class T>
  struct returnable<const T&>
  {
    typedef typename rutz::select_if<
      rutz::is_sub_super<T, rutz::value>::result,
      rutz::fstring, T>::result_t
    type;
  };

  //
  // Functions for converting from Tcl objects to C++ types.
  //

  /// Convert a tcl::obj to a native c++ object.
  /** Will select a matching aux_convert_to() overload. NOTE! Due to
      two-phase name lookup, this convert_to() definition must occur
      before ALL aux_convert_to() overloads. */
  template <class T>
  inline typename returnable<T>::type convert_to( const tcl::obj& obj )
  {
    return aux_convert_to(obj.get(), static_cast<typename returnable<T>::type*>(0));
  }

  /// Convert a tcl::obj to a native c++ object.
  /** Will select a matching aux_convert_to() overload. NOTE! Due to
      two-phase name lookup, this convert_to() definition must occur
      before ALL aux_convert_to() overloads. */
  template <class T>
  inline typename returnable<T>::type convert_to( Tcl_Obj* obj )
  {
    return aux_convert_to(obj, static_cast<typename returnable<T>::type*>(0));
  }

  // Note that the trailing pointer params are not actually used, they
  // are simply present to allow overload selection. See e.g. how
  // aux_convert_to() is called below in the implementation of
  // convert_to().

  int           aux_convert_to(Tcl_Obj* obj, int*);
  unsigned int  aux_convert_to(Tcl_Obj* obj, unsigned int*);
  long          aux_convert_to(Tcl_Obj* obj, long*);
  unsigned long aux_convert_to(Tcl_Obj* obj, unsigned long*);
  long long     aux_convert_to(Tcl_Obj* obj, long long*);
  bool          aux_convert_to(Tcl_Obj* obj, bool*);
  double        aux_convert_to(Tcl_Obj* obj, double*);
  float         aux_convert_to(Tcl_Obj* obj, float*);
  const char*   aux_convert_to(Tcl_Obj* obj, const char**);
  rutz::fstring aux_convert_to(Tcl_Obj* obj, rutz::fstring*);

  inline
  Tcl_Obj*      aux_convert_to(Tcl_Obj* obj, Tcl_Obj**)
  { return obj; }

  inline
  tcl::obj      aux_convert_to(Tcl_Obj* obj, tcl::obj*)
  { return tcl::obj(obj); }

  //
  // Functions for converting from C++ types to Tcl objects.
  //

  /// Convert a native c++ object to a tcl::obj.
  /** Will select a matching aux_convert_from() overload. NOTE! Due to
      two-phase name lookup, this convert_from() definition must occur
      before ALL aux_convert_from() overloads. */
  template <class T>
  inline tcl::obj convert_from(const T& val)
  {
    return aux_convert_from(val);
  }

  tcl::obj aux_convert_from(long long val);
  tcl::obj aux_convert_from(long val);
  tcl::obj aux_convert_from(unsigned long val);
  tcl::obj aux_convert_from(int val);
  tcl::obj aux_convert_from(unsigned int val);
  tcl::obj aux_convert_from(unsigned char val);
  tcl::obj aux_convert_from(bool val);
  tcl::obj aux_convert_from(double val);
  tcl::obj aux_convert_from(float val);
  tcl::obj aux_convert_from(const char* val);
  tcl::obj aux_convert_from(const rutz::fstring& val);
  tcl::obj aux_convert_from(const rutz::value& v);

  inline
  tcl::obj aux_convert_from(Tcl_Obj* val)
  { return tcl::obj(val); }

  inline
  tcl::obj aux_convert_from(tcl::obj val)
  { return val; }
}

static const char __attribute__((used)) vcid_groovx_tcl_conversions_h_utc20050628162420[] = "$Id: conversions.h 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/conversions.h $";
#endif // !GROOVX_TCL_CONVERSIONS_H_UTC20050628162420_DEFINED
