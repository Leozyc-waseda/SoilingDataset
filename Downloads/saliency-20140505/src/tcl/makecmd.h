/** @file tcl/makecmd.h construct tcl commands from c++ functions via
    templatized argument deduction and conversion between tcl and
    c++ */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2001-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Fri Jun 22 09:07:27 2001
// commit: $Id: makecmd.h 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/makecmd.h $
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

#ifndef GROOVX_TCL_MAKECMD_H_UTC20050628162421_DEFINED
#define GROOVX_TCL_MAKECMD_H_UTC20050628162421_DEFINED

#include "nub/ref.h"

#include "tcl/argspec.h"
#include "tcl/conversions.h"
#include "tcl/command.h"
#include "tcl/commandgroup.h"
#include "tcl/vecdispatch.h"

#include "rutz/functors.h"
#include "rutz/shared_ptr.h"

namespace rutz
{
  struct file_pos;

  /// Specialization of func_traits for mem_functor.
  template <class MF>
  struct func_traits<mem_functor<MF> > : public func_traits<MF>
  {
    typedef nub::soft_ref<typename mem_functor<MF>::C> arg1_t;
  };
}

namespace tcl
{
  /// Overload of aux_convert_to for nub::ref.
  /** This allows us to receive nub::ref objects from Tcl via the
      nub::uid's of the referred-to objects. */
  template <class T>
  inline nub::ref<T> aux_convert_to(Tcl_Obj* obj, nub::ref<T>*)
  {
    nub::uid uid = tcl::convert_to<nub::uid>(obj);
    return nub::ref<T>(uid);
  }

  /// Overload of aux_convert_from for nub::ref.
  /** This allows us to pass nub::ref objects to Tcl via the
      nub::uid's of the referred-to objects. */
  template <class T>
  inline tcl::obj aux_convert_from(nub::ref<T> obj)
  {
    return convert_from<nub::uid>(obj.id());
  }

  /// Overload of aux_convert_to for nub::soft_ref.
  /** This allows us to receive nub::soft_ref objects from Tcl via the
      nub::uid's of the referred-to objects. */
  template <class T>
  inline nub::soft_ref<T> aux_convert_to(Tcl_Obj* obj, nub::soft_ref<T>*)
  {
    nub::uid uid = tcl::convert_to<nub::uid>(obj);
    return nub::soft_ref<T>(uid);
  }

  /// Overload of aux_convert_from for nub::soft_ref.
  /** This allows us to pass nub::soft_ref objects to Tcl via the
      nub::uid's of the referred-to objects. */
  template <class T>
  inline tcl::obj aux_convert_from(nub::soft_ref<T> obj)
  {
    return convert_from<nub::uid>(obj.id());
  }


///////////////////////////////////////////////////////////////////////
//
// func_wrapper<> template definitions. Each specialization takes a
// C++-style functor (could be a free function, or struct with
// operator()), and transforms it into a functor with an
// operator()(tcl::call_context&) which can be called from a
// tcl::command. This transformation requires extracting the
// appropriate parameters from the tcl::call_context, passing them to
// the C++ functor, and returning the result back to the
// tcl::call_context.
//
///////////////////////////////////////////////////////////////////////

#ifdef EXTRACT_PARAM
#  error EXTRACT_PARAM macro already defined
#endif

#define EXTRACT_PARAM(N) \
  typename rutz::func_traits<Func>::arg##N##_t p##N = \
  ctx.template get_arg<typename rutz::func_traits<Func>::arg##N##_t>(N);

  /// Generic tcl::func_wrapper definition.
  template <unsigned int N, class R, class Func>
  class func_wrapper
  {};
}

namespace rutz
{
  /// Specialization of func_traits for tcl::func_wrapper.
  template <unsigned int N, class F, class Func>
  struct func_traits<tcl::func_wrapper<N, F, Func> >
  {
    typedef typename rutz::func_traits<Func>::retn_t retn_t;
  };
}

namespace tcl
{


// ########################################################
/// tcl::func_wrapper<0> -- zero arguments

  template <class R, class Func>
  struct func_wrapper<0, R, Func>
  {
  private:
    Func m_held_func;

  public:
    func_wrapper<0, R, Func>(Func f) : m_held_func(f) {}

    ~func_wrapper() throw() {}

    R operator()(tcl::call_context& /*ctx*/)
    {
      return m_held_func();
    }
  };


// ########################################################
/// tcl::func_wrapper<1> -- one argument

  template <class R, class Func>
  struct func_wrapper<1, R, Func>
  {
  private:
    Func m_held_func;

  public:
    func_wrapper<1, R, Func>(Func f) : m_held_func(f) {}

    ~func_wrapper() throw() {}

    R operator()(tcl::call_context& ctx)
    {
      EXTRACT_PARAM(1);
      return m_held_func(p1);
    }
  };


// ########################################################
/// tcl::func_wrapper<2> -- two arguments

  template <class R, class Func>
  struct func_wrapper<2, R, Func>
  {
  private:
    Func m_held_func;

  public:
    func_wrapper<2, R, Func>(Func f) : m_held_func(f) {}

    ~func_wrapper() throw() {}

    R operator()(tcl::call_context& ctx)
    {
      EXTRACT_PARAM(1); EXTRACT_PARAM(2);
      return m_held_func(p1, p2);
    }
  };


// ########################################################
/// tcl::func_wrapper<3> -- three arguments

  template <class R, class Func>
  struct func_wrapper<3, R, Func>
  {
  private:
    Func m_held_func;

  public:
    func_wrapper<3, R, Func>(Func f) : m_held_func(f) {}

    ~func_wrapper() throw() {}

    R operator()(tcl::call_context& ctx)
    {
      EXTRACT_PARAM(1); EXTRACT_PARAM(2); EXTRACT_PARAM(3);
      return m_held_func(p1, p2, p3);
    }
  };


// ########################################################
/// tcl::func_wrapper<4> -- four arguments

  template <class R, class Func>
  struct func_wrapper<4, R, Func>
  {
  private:
    Func m_held_func;

  public:
    func_wrapper<4, R, Func>(Func f) : m_held_func(f) {}

    ~func_wrapper() throw() {}

    R operator()(tcl::call_context& ctx)
    {
      EXTRACT_PARAM(1); EXTRACT_PARAM(2); EXTRACT_PARAM(3);
      EXTRACT_PARAM(4);
      return m_held_func(p1, p2, p3, p4);
    }
  };


// ########################################################
/// tcl::func_wrapper<5> -- five arguments

  template <class R, class Func>
  struct func_wrapper<5, R, Func>
  {
  private:
    Func m_held_func;

  public:
    func_wrapper<5, R, Func>(Func f) : m_held_func(f) {}

    ~func_wrapper() throw() {}

    R operator()(tcl::call_context& ctx)
    {
      EXTRACT_PARAM(1); EXTRACT_PARAM(2); EXTRACT_PARAM(3);
      EXTRACT_PARAM(4); EXTRACT_PARAM(5);
      return m_held_func(p1, p2, p3, p4, p5);
    }
  };


// ########################################################
/// tcl::func_wrapper<6> -- six arguments

  template <class R, class Func>
  struct func_wrapper<6, R, Func>
  {
  private:
    Func m_held_func;

  public:
    func_wrapper<6, R, Func>(Func f) : m_held_func(f) {}

    ~func_wrapper() throw() {}

    R operator()(tcl::call_context& ctx)
    {
      EXTRACT_PARAM(1); EXTRACT_PARAM(2); EXTRACT_PARAM(3);
      EXTRACT_PARAM(4); EXTRACT_PARAM(5); EXTRACT_PARAM(6);
      return m_held_func(p1, p2, p3, p4, p5, p6);
    }
  };

// ########################################################
/// tcl::func_wrapper<7> -- seven arguments

  template <class R, class Func>
  struct func_wrapper<7, R, Func>
  {
  private:
    Func m_held_func;

  public:
    func_wrapper<7, R, Func>(Func f) : m_held_func(f) {}

    ~func_wrapper() throw() {}

    R operator()(tcl::call_context& ctx)
    {
      EXTRACT_PARAM(1); EXTRACT_PARAM(2); EXTRACT_PARAM(3);
      EXTRACT_PARAM(4); EXTRACT_PARAM(5); EXTRACT_PARAM(6);
      EXTRACT_PARAM(7);
      return m_held_func(p1, p2, p3, p4, p5, p6, p7);
    }
  };

// ########################################################
/// tcl::func_wrapper<8> -- eight arguments

  template <class R, class Func>
  struct func_wrapper<8, R, Func>
  {
  private:
    Func m_held_func;

  public:
    func_wrapper<8, R, Func>(Func f) : m_held_func(f) {}

    ~func_wrapper() throw() {}

    R operator()(tcl::call_context& ctx)
    {
      EXTRACT_PARAM(1); EXTRACT_PARAM(2); EXTRACT_PARAM(3);
      EXTRACT_PARAM(4); EXTRACT_PARAM(5); EXTRACT_PARAM(6);
      EXTRACT_PARAM(7); EXTRACT_PARAM(8);
      return m_held_func(p1, p2, p3, p4, p5, p6, p7, p8);
    }
  };

#undef EXTRACT_PARAM

// ########################################################
/// Factory function to make tcl::func_wrapper's from any functor or function ptr.

  template <class Fptr>
  inline func_wrapper<rutz::func_traits<Fptr>::num_args,
                      typename rutz::func_traits<Fptr>::retn_t,
                      typename rutz::functor_of<Fptr>::type>
  build_func_wrapper(Fptr f)
  {
    return rutz::build_functor(f);
  }


// ########################################################
/// generic_function implements tcl::command using a held functor.

  template <class R, class func_wrapper>
  class generic_function : public tcl::function
  {
  protected:
    generic_function<R, func_wrapper>(func_wrapper f) : m_held_func(f) {}

  public:
    static rutz::shared_ptr<tcl::function> make(func_wrapper f)
    {
      return rutz::shared_ptr<tcl::function>(new generic_function(f));
    }

    virtual ~generic_function() throw() {}

  protected:
    virtual void invoke(tcl::call_context& ctx)
    {
      R res(m_held_func(ctx)); ctx.set_result(res);
    }

  private:
    func_wrapper m_held_func;
  };

// ########################################################
/// Specialization for functors with void return types.

  template <class func_wrapper>
  class generic_function<void, func_wrapper> : public tcl::function
  {
  protected:
    generic_function<void, func_wrapper>(func_wrapper f) : m_held_func(f) {}

  public:
    static rutz::shared_ptr<tcl::function> make(func_wrapper f)
    {
      return rutz::shared_ptr<tcl::function>(new generic_function(f));
    }

    virtual ~generic_function() throw() {}

  protected:
    virtual void invoke(tcl::call_context& ctx)
    {
      m_held_func(ctx);
    }

  private:
    func_wrapper m_held_func;
  };


// ########################################################
/// Factory function for tcl::command's from functors.

  template <class func_wrapper>
  inline void
  make_generic_command(tcl::interpreter& interp,
                       func_wrapper f,
                       const char* cmd_name,
                       const char* usage,
                       const arg_spec& spec,
                       const rutz::file_pos& src_pos)
  {
    typedef typename rutz::func_traits<func_wrapper>::retn_t retn_t;
    tcl::command_group::make(interp,
                       generic_function<retn_t, func_wrapper>::make(f),
                       cmd_name, usage, spec, src_pos);
  }


// ########################################################
/// Factory function for vectorized tcl::command's from functors.

  template <class func_wrapper>
  inline void
  make_generic_vec_command(tcl::interpreter& interp,
                           func_wrapper f,
                           const char* cmd_name,
                           const char* usage,
                           const arg_spec& spec,
                           unsigned int keyarg,
                           const rutz::file_pos& src_pos)
  {
    typedef typename rutz::func_traits<func_wrapper>::retn_t retn_t;
    rutz::shared_ptr<tcl::command> cmd =
      tcl::command_group::make(interp,
                         generic_function<retn_t, func_wrapper>::make(f),
                         cmd_name, usage, spec, src_pos);
    tcl::use_vec_dispatch(*cmd, keyarg);
  }

///////////////////////////////////////////////////////////////////////
//
// And finally... make_command
//
///////////////////////////////////////////////////////////////////////

// ########################################################
/// Factory function for tcl::command's from function pointers.

  template <class Func>
  inline void
  make_command(tcl::interpreter& interp,
          Func f,
          const char* cmd_name,
          const char* usage,
          const rutz::file_pos& src_pos)
  {
    make_generic_command
      (interp, build_func_wrapper(f), cmd_name, usage,
       arg_spec(rutz::func_traits<Func>::num_args + 1, -1, true),
       src_pos);
  }

// ########################################################
/// Factory function for vectorized tcl::command's from function pointers.

  template <class Func>
  inline void
  make_vec_command(tcl::interpreter& interp,
                   Func f,
                   const char* cmd_name,
                   const char* usage,
                   unsigned int keyarg /*default is 1*/,
                   const rutz::file_pos& src_pos)
  {
    make_generic_vec_command
      (interp, build_func_wrapper(f), cmd_name, usage,
       arg_spec(rutz::func_traits<Func>::num_args + 1, -1, true),
       keyarg, src_pos);
  }

} // end namespace tcl

static const char __attribute__((used)) vcid_groovx_tcl_makecmd_h_utc20050628162421[] = "$Id: makecmd.h 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/makecmd.h $";
#endif // !GROOVX_TCL_MAKECMD_H_UTC20050628162421_DEFINED
