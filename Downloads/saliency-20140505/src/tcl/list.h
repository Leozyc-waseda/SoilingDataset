/** @file tcl/list.h c++ wrapper of tcl list objects; handles ref
    counting and c++/tcl type conversion, offers c++-style list
    iterators */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2001-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Wed Jul 11 12:00:17 2001
// commit: $Id: list.h 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/list.h $
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

#ifndef GROOVX_TCL_LIST_H_UTC20050628162420_DEFINED
#define GROOVX_TCL_LIST_H_UTC20050628162420_DEFINED

#include "tcl/conversions.h"
#include "tcl/obj.h"

#include "rutz/shared_ptr.h"

namespace tcl
{
  class list;

  tcl::list aux_convert_to(Tcl_Obj* obj, tcl::list*);
  tcl::obj aux_convert_from(tcl::list list_value);
}

///////////////////////////////////////////////////////////////////////
/**
 *
 * tcl::list class definition
 *
 **/
///////////////////////////////////////////////////////////////////////

class tcl::list
{
public:
  /// Default constructor makes an empty list
  list();

  list(const tcl::obj& x);

  list(const list& other) :
    m_list_obj(other.m_list_obj),
    m_elements(other.m_elements),
    m_length(other.m_length)
  {}

  list& operator=(const list& other)
  {
    m_list_obj = other.m_list_obj;
    m_elements = other.m_elements;
    m_length = other.m_length;
    return *this;
  }

  tcl::obj as_obj() const { return m_list_obj; }

  /// Checked access to element at \a index.
  Tcl_Obj* at(unsigned int index) const;

  /// Unchecked access to element at \a index.
  Tcl_Obj* operator[](unsigned int index) const
    {
      update(); return m_elements[index];
    }

  Tcl_Obj* const* elements() const
    {
      update(); return m_elements;
    }

  template <class T>
  inline T get(unsigned int index, T* /*dummy*/=0) const;

  unsigned int size() const { update(); return m_length; }
  unsigned int length() const { update(); return m_length; }

  template <class T>
  void append(T t) { do_append(tcl::convert_from(t), 1); }

  template <class T>
  void append(T t, unsigned int times)
    {
      do_append(tcl::convert_from(t), times);
    }

  template <class Itr>
  void append_range(Itr itr, Itr end)
    {
      while (itr != end)
        {
          append(*itr);
          ++itr;
        }
    }

  class iterator_base;
  template <class T> class iterator;

  template <class T>
  iterator<T> begin(T* /*dummy*/=0);

  template <class T>
  iterator<T> end(T* /*dummy*/=0);

  /// A back-insert iterator for tcl::list.
  class appender
  {
    tcl::list& m_list_obj;
  public:
    appender(tcl::list& x) : m_list_obj(x) {}

    template <class T>
    appender& operator=(const T& val)
    { m_list_obj.append(val); return *this; }

    appender& operator*() { return *this; }
    appender& operator++() { return *this; }
    appender operator++(int) { return *this; }
  };

  appender back_appender() { return appender(*this); }

  /// Utility function to return the list length of a Tcl object
  static unsigned int get_obj_list_length(Tcl_Obj* obj);

private:
  void do_append(const tcl::obj& obj, unsigned int times);

  void update() const
    {
      if (m_elements==0)
        split();
    }

  void split() const;

  void invalidate() { m_elements = 0; m_length = 0; }

  mutable tcl::obj      m_list_obj;
  mutable Tcl_Obj**     m_elements;
  mutable unsigned int  m_length;
};


///////////////////////////////////////////////////////////////////////
/**
 *
 * tcl::list::iterator_base class definition
 *
 **/
///////////////////////////////////////////////////////////////////////

class tcl::list::iterator_base
{
protected:
  // Make protected to prevent people from instantiating iterator_base directly.
  ~iterator_base() {}

public:
  typedef int difference_type;

  enum position { BEGIN, END };

  iterator_base(const list& owner, position start_pos = BEGIN) :
    m_list_obj(owner),
    m_index(start_pos == BEGIN ? 0 : owner.length())
  {}

  iterator_base(Tcl_Obj* x, position start_pos = BEGIN) :
    m_list_obj(x),
    m_index(start_pos == BEGIN ? 0 : m_list_obj.length())
  {}

  // default copy-constructor, assignment operator OK

  iterator_base& operator++()
    { ++m_index; return *this; }

  iterator_base operator++(int)
    { iterator_base temp(*this); ++m_index; return temp; }

  difference_type operator-(const iterator_base& other) const
    {
      if (this->m_index > other.m_index)
        return int(this->m_index - other.m_index);
      else
        return -(int(other.m_index - this->m_index));
    }

  bool operator==(const iterator_base& other) const
    { return m_index == other.m_index; }

  bool operator!=(const iterator_base& other) const
    { return !operator==(other); }

  bool is_valid() const
    { return m_index < m_list_obj.length(); }

  bool has_more() const
    { return m_index < (m_list_obj.length()-1); }

  bool nelems() const
    { return m_list_obj.length(); }

protected:
  Tcl_Obj* current() const
    { return m_list_obj.at(m_index); }

private:
  list m_list_obj;
  unsigned int m_index;

};


///////////////////////////////////////////////////////////////////////
/**
 *
 * \c tcl::list::iterator is an adapter that provides an STL-style
 * iterator interface to Tcl list objects. \c tcl::list::iterator is a
 * model of \c input \c iterator.
 *
 **/
///////////////////////////////////////////////////////////////////////

template <class T>
class tcl::list::iterator : public tcl::list::iterator_base
{
  // Keep a copy of the current value here so that operator*() can
  // return a reference rather than by value.
  mutable rutz::shared_ptr<const T> m_current;

public:
  iterator(const list& owner, position start_pos = BEGIN) :
    iterator_base(owner, start_pos), m_current() {}

  iterator(Tcl_Obj* x, position start_pos = BEGIN) :
    iterator_base(x, start_pos), m_current() {}

  typedef T value_type;

  const T& operator*() const
  {
    m_current.reset(new T(tcl::convert_to<T>(current())));
    return *m_current;
  }
};


///////////////////////////////////////////////////////////////////////
//
// Inline member definitions
//
///////////////////////////////////////////////////////////////////////

#include "tcl/conversions.h"

template <class T>
inline T tcl::list::get(unsigned int index, T* /*dummy*/) const
{
  return tcl::convert_to<T>(at(index));
}

template <class T>
inline tcl::list::iterator<T> tcl::list::begin(T* /*dummy*/)
{
  return iterator<T>(*this, iterator_base::BEGIN);
}

template <class T>
inline tcl::list::iterator<T> tcl::list::end(T* /*dummy*/)
{
  return iterator<T>(*this, iterator_base::END);
}

static const char __attribute__((used)) vcid_groovx_tcl_list_h_utc20050628162420[] = "$Id: list.h 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/list.h $";
#endif // !GROOVX_TCL_LIST_H_UTC20050628162420_DEFINED
