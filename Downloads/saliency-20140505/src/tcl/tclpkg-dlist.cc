/** @file tcl/tclpkg-dlist.cc tcl interface package for extended
    list-manipulation functions */
///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 1998-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Tue Dec  1 08:00:00 1998
// commit: $Id: tclpkg-dlist.cc 11876 2009-10-22 15:53:06Z icore $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/tclpkg-dlist.cc $
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

#ifndef GROOVX_TCL_TCLPKG_DLIST_CC_UTC20050628161246_DEFINED
#define GROOVX_TCL_TCLPKG_DLIST_CC_UTC20050628161246_DEFINED

#include "tcl/tclpkg-dlist.h"

#include "tcl/tclpkg-dlist.h"

// This file implements additional Tcl list manipulation functions

#include "tcl/list.h"
#include "tcl/pkg.h"

#include "rutz/algo.h"
#include "rutz/arrays.h"
#include "rutz/error.h"
#include "rutz/rand.h"

#include "rutz/trace.h"
#include "rutz/debug.h"
GVX_DBG_REGISTER

#include <algorithm> // for std::random_shuffle
#include <cmath>

// Helper functions
namespace
{
  template <class Itr>
  double perm_distance_aux(Itr itr, Itr end)
  {
    int c = 0;
    double result = 0.0;
    while (itr != end)
      {
        result += fabs(double(*itr) - double(c));
        ++itr;
        ++c;
      }

    return result / double(c);
  }

  template <class Itr>
  double perm_distance2_aux(Itr itr, Itr end, double power)
  {
    int c = 0;
    double result = 0.0;
    while (itr != end)
      {
        result += pow(fabs(double(*itr) - double(c)), power);
        ++itr;
        ++c;
      }

    return pow(result, 1.0/power) / double(c);
  }
}

namespace Dlist
{

  //---------------------------------------------------------
  //
  // This command takes two lists as arguments, and uses the integers in
  // the second (index) list to return a permutation of the elements in
  // the first (source) list
  //
  // Example:
  //      dlist_choose { 3 5 7 } { 2 0 1 }
  // returns
  //      7 3 5
  //
  //---------------------------------------------------------

  tcl::list choose(tcl::list source_list, tcl::list index_list)
  {
    tcl::list result;

    for (unsigned int i = 0; i < index_list.length(); ++i)
      {
        unsigned int index = index_list.get<unsigned int>(i);

        // use that int as an index into source list, getting the
        // corresponding list element and appending it to the output list
        result.append(source_list.at(index));
      }

    GVX_ASSERT(result.length() == index_list.length());

    return result;
  }

  //---------------------------------------------------------
  //
  // Cyclically shift the elements of the list leftward by n steps.
  //
  //---------------------------------------------------------

  tcl::list cycle_left(tcl::list source_list, unsigned int n)
  {
    n = n % source_list.length();

    if (n == 0)
      return source_list;

    tcl::list result;

    for (unsigned int i = n; i < source_list.length(); ++i)
      {
        result.append(source_list.at(i));
      }

    for (unsigned int i = 0; i < n; ++i)
      {
        result.append(source_list.at(i));
      }

    return result;
  }

  //---------------------------------------------------------
  //
  // Cyclically shift the elements of the list rightward by n steps.
  //
  //---------------------------------------------------------

  tcl::list cycle_right(tcl::list source_list, unsigned int n)
  {
    n = n % source_list.length();

    if (n == 0)
      return source_list;

    return cycle_left(source_list, source_list.length() - n);
  }

  //---------------------------------------------------------
  //
  // Returns the n'th element of the list; generates an error if n is
  // out of range.
  //
  //---------------------------------------------------------

  tcl::obj index(tcl::list source_list, unsigned int n)
  {
    return source_list.at(n);
  }

  //---------------------------------------------------------
  //
  // This command takes as its argument a single list containing only
  // integers, and returns a list in which each element is the logical
  // negation of its corresponding element in the source list.
  //
  //---------------------------------------------------------

  tcl::list not_(tcl::list source_list)
  {
    tcl::obj one = tcl::convert_from<int>(1);
    tcl::obj zero = tcl::convert_from<int>(0);
    tcl::list result;

    for (unsigned int i = 0; i < source_list.length(); ++i)
      {
        if ( source_list.get<int>(i) == 0 )
          result.append(one);
        else
          result.append(zero);
      }

    GVX_ASSERT(result.length() == source_list.length());

    return result;
  }

  //---------------------------------------------------------
  //
  // this command produces a list of ones of the length specified by its
  // lone argument
  //
  //---------------------------------------------------------

  tcl::list ones(unsigned int num_ones)
  {
    tcl::list result;
    result.append(1, num_ones);

    return result;
  }

  //---------------------------------------------------------
  //
  // This commmand returns a single element chosen at random
  // from the source list
  //
  //---------------------------------------------------------

  tcl::obj pickone(tcl::list source_list)
  {
    if (source_list.length() == 0)
      {
        throw rutz::error("source_list is empty", SRC_POS);
      }

    return source_list.at(rutz::rand_range(0u, source_list.length()));
  }

  //---------------------------------------------------------
  //
  // this command produces an ordered list of all integers between begin
  // and end, inclusive.
  //
  //---------------------------------------------------------

  tcl::list range(int begin, int end, int step)
  {
    tcl::list result;

    if (step == 0)
      {
        // special case: if step is 0, we return an empty list
      }
    else if (step > 0)
      {
        for (int i = begin; i <= end; i += step)
          {
            result.append(i);
          }
      }
    else // step < 0
      {
        for (int i = begin; i >= end; i += step)
          {
            result.append(i);
          }
      }

    return result;
  }

  //---------------------------------------------------------
  //
  // Make a series of linearly spaced values between (and including) two
  // endpoints
  //
  //---------------------------------------------------------

  tcl::list linspace(double begin, double end, unsigned int npts)
  {
    tcl::list result;

    if (npts < 2)
      {
        throw rutz::error("npts must be at least 2", SRC_POS);
      }

    const double skip = (end - begin) / (npts - 1);

    bool integer_mode = (skip == int(skip) && begin == int(begin));

    if (integer_mode)
      for (unsigned int i = 0; i < npts; ++i)
        {
          result.append(int(begin + i*skip));
        }
    else
      for (unsigned int i = 0; i < npts; ++i)
        {
          result.append(begin + i*skip);
        }

    return result;
  }

  double perm_distance(tcl::list src)
  {
    return perm_distance_aux(src.begin<unsigned int>(),
                             src.end<unsigned int>());
  }

  double perm_distance2(tcl::list src, double power)
  {
    return perm_distance2_aux(src.begin<unsigned int>(),
                              src.end<unsigned int>(),
                              power);
  }

  //---------------------------------------------------------
  //
  // generate a complete/pure permutation of the numbers 0..N-1
  // the result is such that:
  //   result[i] != i         for all i
  //   sum(abs(result[i]-i))  is maximal
  //
  // WARNING: At first glance this might sound like it yields a nice random
  // list, but in fact simply reversing the order of elements gives a
  // result that satisfies the constraints of this algorithm, without being
  // random at all!
  //
  //---------------------------------------------------------

  tcl::list permute_maximal(unsigned int N)
  {
    if (N < 2)
      throw rutz::error("N must be at least 2 to make a permutation",
                        SRC_POS);

    double maxdist = double(N)/2.0;

    if (N%2)
      {
        const double half = double(N)/2.0;
        maxdist = half + 1.0/(2.0 + 1.0/half);
      }

    maxdist -= 0.0001;

    rutz::fixed_block<unsigned int> slots(N);

    for (unsigned int i = 0; i < slots.size()-1; ++i)
      slots[i] = i+1;

    slots[slots.size()-1] = 0;

    double dist = perm_distance_aux(slots.begin(), slots.end());

    for (int c = 0; c < 100000; ++c)
      {
        unsigned int i = rutz::rand_range(0u, N);
        unsigned int j = i;
        while (j == i)
          {
            j = rutz::rand_range(0u, N);
          }

        if (slots[j] != i && slots[i] != j)
          {
            const double origdist =
              fabs(double(i)-double(slots[i])) +
              fabs(double(j)-double(slots[j]));

            const double newdist =
              fabs(double(j)-double(slots[i])) +
              fabs(double(i)-double(slots[j]));

            if (newdist > origdist)
              {
                rutz::swap2(slots[i], slots[j]);
                dist += (newdist-origdist)/double(N);
              }
          }

        if (dist >= maxdist)
          {
            double distcheck = perm_distance_aux(slots.begin(), slots.end());
            if (distcheck < maxdist)
              {
                throw rutz::error("snafu in permutation "
                                  "distance computation", SRC_POS);
              }

            dbg_eval_nl(3, c);

            tcl::list result;

            for (unsigned int i = 0; i < slots.size(); ++i)
              {
                result.append(slots[i]);
              }

            return result;
          }
      }

    throw rutz::error("permutation algorithm failed to converge",
                      SRC_POS);
    return tcl::list(); // can't happen, but placate compiler
  }

  //---------------------------------------------------------
  //
  // generate a random permutation of the numbers 0..N-1 such that:
  //   result[i] != i         for all i
  //
  //---------------------------------------------------------

  tcl::list permute_moveall(unsigned int N)
  {
    if (N < 2)
      throw rutz::error("N must be at least 2 to make a permutation", SRC_POS);

    rutz::fixed_block<bool> used(N);
    for (unsigned int i = 0; i < N; ++i)
      used[i] = false;

    rutz::fixed_block<unsigned int> slots(N);

    // fill slots[0] ... slots[N-2]
    for (unsigned int i = 0; i < N-1; ++i)
      {
        unsigned int v = i;
        while (v == i || used[v])
          v = rutz::rand_range(0u, N);

        GVX_ASSERT(v < N);

        used[v] = true;
        slots[i] = v;
      }

    // figure out which is the last available slot
    unsigned int lastslot = N;
    for (unsigned int i = 0; i < N; ++i)
      if (!used[i])
        {
          lastslot = i;
          break;
        }

    GVX_ASSERT(lastslot != N);

    if (lastslot == N)
      {
        slots[N-1] = slots[N-2];
        slots[N-2] = lastslot;
      }
    else
      {
        slots[N-1] = lastslot;
      }

    tcl::list result;

    for (unsigned int i = 0; i < slots.size(); ++i)
      {
        result.append(slots[i]);
      }

    return result;
  }

  //---------------------------------------------------------
  //
  // this command produces a list of random numbers each between min and
  // max, and of the given
  //
  //---------------------------------------------------------

  tcl::list rand(double min, double max, unsigned int N)
  {
    tcl::list result;

    static rutz::urand generator;

    for (unsigned int i = 0; i < N; ++i)
      {
        result.append(generator.fdraw_range(min, max));
      }

    return result;
  }

  //---------------------------------------------------------
  //
  // This command taks two lists as arguments. Each element from the
  // first (source) list is appended to the result multiple times; the
  // number of times is determined by the corresponding integer found in
  // the second (times) list.
  //
  // For example:
  //      dlist_repeat { 4 5 6 } { 1 2 3 }
  // returns
  //      4 5 5 6 6 6
  //
  //---------------------------------------------------------

  tcl::list repeat(tcl::list source_list, tcl::list times_list)
  {
    // find the minimum of the two lists' lengths
    unsigned int min_len = rutz::min(source_list.length(), times_list.length());

    tcl::list result;

    for (unsigned int t = 0; t < min_len; ++t)
      {
        result.append(source_list.at(t),
                      times_list.get<unsigned int>(t));
      }

    return result;
  }

  //---------------------------------------------------------
  //
  // Return a new list containing the elements of the source list in
  // reverse order.
  //
  //---------------------------------------------------------

  tcl::list reverse(tcl::list src)
  {
    if (src.length() < 2)
      return src;

    tcl::list result;
    for (unsigned int i = 0; i < src.length(); ++i)
      result.append(src.at(src.length()-i-1));
    return result;
  }

  //---------------------------------------------------------
  //
  // This command takes two lists as arguments, using the binary flags
  // in the second (flags) list to choose which elements from the first
  // (source) list should be appended to the output list
  //
  //---------------------------------------------------------

  tcl::list select(tcl::list source_list, tcl::list flags_list)
  {
    unsigned int src_len = source_list.length();
    unsigned int flg_len = flags_list.length();

    if (flg_len < src_len)
      {
        throw rutz::error("flags list must be as long as source_list",
                          SRC_POS);
      }

    tcl::list result;

    for (unsigned int i = 0; i < src_len; ++i)
      {
        // if the flag is true, add the corresponding source_list
        // element to the result list
        if ( flags_list.get<int>(i) )
          {
            result.append(source_list.at(i));
          }
      }

    return result;
  }

  //---------------------------------------------------------
  //
  // dlist::shuffle
  //
  //---------------------------------------------------------

  tcl::list shuffle(tcl::list src, int seed)
  {
    rutz::fixed_block<tcl::obj> objs(src.begin<tcl::obj>(),
                                        src.end<tcl::obj>());

    rutz::urand generator(seed);

    std::random_shuffle(objs.begin(), objs.end(), generator);

    tcl::list result;

    for (unsigned int i = 0; i < objs.size(); ++i)
      {
        result.append(objs[i]);
      }

    return result;
  }

  //---------------------------------------------------------
  //
  // Shuffle an input list through a random permutation such that no
  // element remains in its initial position.
  //
  //---------------------------------------------------------

  tcl::list shuffle_moveall(tcl::list src)
  {
    tcl::list permutation = permute_moveall(src.length());
    return Dlist::choose(src, permutation);
  }

  //---------------------------------------------------------
  //
  // Shuffle an input list through a maximal permutation.
  //
  //---------------------------------------------------------

  tcl::list shuffle_maximal(tcl::list src)
  {
    tcl::list permutation = permute_maximal(src.length());
    return Dlist::choose(src, permutation);
  }

  //---------------------------------------------------------
  //
  // dlist::slice
  //
  //---------------------------------------------------------

  tcl::list slice(tcl::list src, unsigned int slice)
  {
    tcl::list result;

    for (unsigned int i = 0, end = src.length(); i < end; ++i)
      {
        tcl::list sub(src.at(i));
        result.append(sub.at(slice));
      }

    GVX_ASSERT(result.length() == src.length());

    return result;
  }

  //---------------------------------------------------------
  //
  // this command sums the numbers in a list, trying to return an int
  // result if possible, but returning a double result if any doubles
  // are found in the source list
  //
  //---------------------------------------------------------

  tcl::obj sum(tcl::list source_list)
  {
    int isum=0;
    double dsum=0.0;
    bool seen_double=false;

    for (unsigned int i = 0; i < source_list.length(); /* incr in loop body*/)
      {
        if ( !seen_double )
          {
            try
              {
                isum += source_list.get<int>(i);
              }
            catch(rutz::error&)
              {
                seen_double = true;
                dsum = isum;
                continue; // skip the increment
              }
          }
        else
          {
            dsum += source_list.get<double>(i);
          }

        ++i; // here's the increment
      }

    if ( !seen_double )
      return tcl::convert_from<int>(isum);
    else
      return tcl::convert_from<double>(dsum);
  }

  //---------------------------------------------------------
  //
  // this command produces a list of zeros of the length specified by its
  // lone argument
  //
  //---------------------------------------------------------

  tcl::list zeros(unsigned int num_zeros)
  {
    tcl::list result;
    result.append(0, num_zeros);

    return result;
  }

} // end namespace Dlist


extern "C"
int Dlist_Init(Tcl_Interp* interp)
{
GVX_TRACE("Dlist_Init");

  GVX_PKG_CREATE(pkg, interp, "Dlist", "4.$Revision: 11876 $");

  pkg->def( "choose", "source_list index_list", &Dlist::choose, SRC_POS );
  pkg->def( "cycle_left", "list n", &Dlist::cycle_left, SRC_POS );
  pkg->def( "cycle_right", "list n", &Dlist::cycle_right, SRC_POS );
  pkg->def( "index", "list index", &Dlist::index, SRC_POS );
  pkg->def( "not", "list", &Dlist::not_, SRC_POS );
  pkg->def( "ones", "num_ones", &Dlist::ones, SRC_POS );
  pkg->def( "linspace", "begin end npts", &Dlist::linspace, SRC_POS );
  pkg->def( "perm_distance", "list", &Dlist::perm_distance, SRC_POS );
  pkg->def( "perm_distance2", "list power", &Dlist::perm_distance2, SRC_POS );
  pkg->def( "permute_maximal", "N", &Dlist::permute_maximal, SRC_POS );
  pkg->def( "permute_moveall", "N", &Dlist::permute_moveall, SRC_POS );
  pkg->def( "pickone", "list", &Dlist::pickone, SRC_POS );
  pkg->def( "rand", "min max N", &Dlist::rand, SRC_POS );
  pkg->def( "range", "begin end ?step=1?", &Dlist::range, SRC_POS );
  pkg->def( "range", "begin end", rutz::bind_last(&Dlist::range, 1), SRC_POS );
  pkg->def( "repeat", "source_list times_list", &Dlist::repeat, SRC_POS );
  pkg->def( "reverse", "list", &Dlist::reverse, SRC_POS );
  pkg->def( "select", "source_list flags_list", &Dlist::select, SRC_POS );
  pkg->def( "shuffle", "list ?seed=0?", &Dlist::shuffle, SRC_POS );
  pkg->def( "shuffle", "list", rutz::bind_last(&Dlist::shuffle, 0), SRC_POS );
  pkg->def( "shuffle_maximal", "list", &Dlist::shuffle_maximal, SRC_POS );
  pkg->def( "shuffle_moveall", "list", &Dlist::shuffle_moveall, SRC_POS );
  pkg->def( "slice", "list n", &Dlist::slice, SRC_POS );
  pkg->def( "sum", "list", &Dlist::sum, SRC_POS );
  pkg->def( "zeros", "num_zeros", &Dlist::zeros, SRC_POS );

  GVX_PKG_RETURN(pkg);
}

static const char __attribute__((used)) vcid_groovx_tcl_tclpkg_dlist_cc_utc20050628161246[] = "$Id: tclpkg-dlist.cc 11876 2009-10-22 15:53:06Z icore $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/tcl/tclpkg-dlist.cc $";
#endif // !GROOVX_TCL_TCLPKG_DLIST_CC_UTC20050628161246_DEFINED
