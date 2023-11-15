/* This is the input for a precompiled header file.

   If precompiled headers are enabled by the configure script, then
   the Makefile will create a precompiled c++stdlib.h.gch file from
   this file, and then will insert a "-include c++stdlib.h" in each
   command-line. When g++ sees an "-include <stem>.h" option, it first
   looks to see if there is a precompiled header named "<stem>.h.gch",
   and if so, it uses that file instead of the uncompiled header.

   Precompiled headers can speed compilation times dramatically. On a
   2.8GHz P4, it takes about 0.55s to compile a trivial .C file that
   includes the c++ std headers listed below, but takes only about
   0.05s to compile the same file when the c++ std headers are
   included via a precompiled header. So, in essence, using the
   precompiled header could knock 0.5s off the compile time of every
   source file. At the time of this writing, 'make all' involved
   compiling about 800 source files, for a theoretical gain of
   800*0.5s=400s; and in practice the actual gain is slightly less
   (closer to 350s) since not all of the source files necessarily
   relied on all of these std headers to begin with. Nevertheless the
   300s improvement represents a 12% decrease in the total cpu time
   needed for a 'make all'.

   For more information on using precompiled headers with gcc, see
   http://gcc.gnu.org/onlinedocs/gcc/Precompiled-Headers.html
*/

// $Id: c++stdlib.h 14376 2011-01-11 02:44:34Z pez $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/c++stdlib.h $

#ifndef CXXSTDLIB_H_UTC20051130204159_DEFINED
#define CXXSTDLIB_H_UTC20051130204159_DEFINED

// Make sure that "config.h" gets included here, since some of the
// definitions there will cause conditional compilation of various
// things that will appear in the C and C++ std library headers that
// we include below
#include "config.h"

// Check for __cplusplus here so that we can use a single -include
// c++stdlib.h command-line option that will work for compilations of
// both C and C++ source files
#ifdef __cplusplus

// NOTE! Don't include any INVT headers here -- only c++ std library
// headers. Q: Why? A: Because if we included any local headers, then
// we would need to regenerate the precompiled header when any of
// those headers changes, and then we would need to recompile the
// ENTIRE project because every object file depends on the precompiled
// header. So we only include std library headers here, since those
// never change.

#include <cstddef>
#include <algorithm>
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <vector>

#endif // __cplusplus

#endif // !CXXSTDLIB_H_UTC20051130204159DEFINED
