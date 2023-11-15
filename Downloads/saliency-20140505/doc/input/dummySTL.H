/*!@file dummySTL.H A dummy STL definition for doxygen
  <B>DO NOT INCLUDE THIS FILE!!!</B> */

//! The Standard Template Library (STL)
/*! This is not the real STL documentation, you can find that at the
  <A HREF="http://www.sgi.com/tech/stl/">Silicon Graphics STL site</A>. */
namespace std
{
  //! The STL vector container class. Use with <CODE>\#include \<vector\></CODE>
  /*! See <A HREF="http://www.sgi.com/tech/stl/Vector.html">SGI's
    documentation</A> of std::vector. */
  template<class T, class Alloc>
  class vector { public: T data; Alloc allocator; };

  //! The STL deque container class. Use with <CODE>\#include \<deque\></CODE>
  /*! See <A HREF="http://www.sgi.com/tech/stl/Deque.html">SGI's
    documentation</A> of std::deque. */
  template<class T, class Alloc>
  class deque { public: T data; Alloc allocator; };

  //! The STL list container class. Use with <CODE>\#include \<list\></CODE>
  /*! See <A HREF="http://www.sgi.com/tech/stl/List.html">SGI's
    documentation</A> of std::list. */
  template<class T, class Alloc>
  class list { public: T data; Alloc allocator; };

  //! The STL slist container class. Use with <CODE>\#include \<slist\></CODE>
  /*! See <A HREF="http://www.sgi.com/tech/stl/Slist.html">SGI's
    documentation</A> of std::slist. */
  template<class T, class Alloc>
  class slist { public: T data; Alloc allocator; };

  //! The STL set container class. Use with <CODE>\#include \<set\></CODE>
  /*! See <A HREF="http://www.sgi.com/tech/stl/set.html">SGI's
    documentation</A> of std::set. */
  template<class Key, class Compare, class Alloc>
  class set { public: Key keys; Compare comparator; Alloc allocator; };

  //! The STL map container class. Use with <CODE>\#include \<map\></CODE>
  /*! See <A HREF="http://www.sgi.com/tech/stl/Map.html">SGI's
    documentation</A> of std::map. */
  template<class Key, class Data, class Compare, class Alloc>
  class map { public: Key keys; Data data; Compare comparator;
    Alloc allocator; };

  //! The STL multiset container class. Use with <CODE>\#include \<set\></CODE>
  /*! See <A HREF="http://www.sgi.com/tech/stl/multiset.html">SGI's
    documentation</A> of std::multiset. */
  template<class Key, class Compare, class Alloc>
  class multiset { public: Key keys; Compare comparator; Alloc allocator; };

  //! The STL multimap container class. Use with <CODE>\#include \<map\></CODE>
  /*! See <A HREF="http://www.sgi.com/tech/stl/Multimap.html">SGI's
    documentation</A> of std::multimap. */
  template<class Key, class Data, class Compare, class Alloc>
  class multimap { public: Key keys; Data data; Compare comparator;
    Alloc allocator; };

  //! The STL string class. Use with <CODE>\#include \<string\></CODE>
  /*! std::string is actually std::basic_string<char>.
    See <A HREF="http://www.sgi.com/tech/stl/basic_string.html">SGI's
    documentation</A> of std::basic_string. */
  class string { };
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
