/** @file rutz/freelist.h memory allocation via a free-list pool */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2001-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Fri Jul 20 07:54:29 2001
// commit: $Id: cudafreelist.H 12962 2010-03-06 02:13:53Z irock $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/cudafreelist.H $
//
// --------------------------------------------------------------------
//
// This file is part of GroovX.
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

#ifndef RUTZ_CUDAFREELIST_H_DEFINED
#define RUTZ_FREELIST_H_DEFINED

#include <cstddef>
#include <map>
#include "CUDA/cudadefs.h"

namespace rutz
{
  class cuda_free_list_base;
  template <class T> class cuda_free_list;
}


/// Un-typesafe base class for maintaining a free-list memory pool.
class rutz::cuda_free_list_base
{
private:
  /// Free-node class for free-list memory pools.
  struct node
  {
    node* next;
    void *mem;
  };

  node* m_node_list[MAX_CUDA_DEVICES];
  std::size_t m_num_allocations[MAX_CUDA_DEVICES];
  std::map<int,int> devices;
  const std::size_t m_size_check;
  // Return -1 if device not found
  int get_index_from_device(int dev);
  // Add device if not found
  int find_index_from_device(int dev);
  cuda_free_list_base(const cuda_free_list_base&);
  cuda_free_list_base& operator=(const cuda_free_list_base&);

public:
  /// Construct an (empty) free list.
  /** All objects from this list must be of size \a size_check. */
  cuda_free_list_base(std::size_t size_check);

  /// Allocate space for a new object.
  /** If there are chunks available in the free list, one of those is
      returned; otherwise new memory is allocated with malloc() or
      equivalent. */
  void* allocate(std::size_t bytes, int dev);

  /// Return an object to the free list.
  void deallocate(void* space, int dev);

  /// Release all nodes currently on the free list (e.g. to conserve memory).
  void release_free_nodes();

  // Get the number of nodes currently on this list
  int get_num_nodes(int dev);

  std::map<int,int>::const_iterator getDevicesBegin() { return devices.begin(); }
  std::map<int,int>::const_iterator getDevicesEnd() { return devices.end(); }

  /// Query the chunk size that this freelist is for.
  std::size_t alloc_size() const { return m_size_check; }

  /// Query how many allocations have been made.
  std::size_t num_allocations(int dev) { int i= get_index_from_device(dev); return (i<0) ? std::size_t(0) : m_num_allocations[i]; }
};

/// Typesafe wrapper of free_list_base for maintaining free-list memory pools.
template <class T>
class rutz::cuda_free_list
{
private:
  rutz::cuda_free_list_base m_base;

public:
  /// Construct an (empty) free list.
  /** All objects allocated from this list must be of size sizeof(T). */
  cuda_free_list() : m_base(sizeof(T)) {}

  void* allocate(std::size_t bytes, int dev)
  {
    return m_base.allocate(bytes,dev);
  }

  void deallocate(void* space, int dev)
  {
    m_base.deallocate(space,dev);
  }

  /// Release all nodes currently on the free list (e.g. to conserve memory).
  void release_free_nodes() { m_base.release_free_nodes(); }

  // Get the number of nodes currently on this list
  int get_num_nodes(int dev) { return m_base.get_num_nodes(dev);}

};

static const char __attribute__((used)) vcid_rutz_cudafreelist_h[] = "$Id: cudafreelist.H 12962 2010-03-06 02:13:53Z irock $ $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/cudafreelist.H $";
#endif // !RUTZ_CUDAFREELIST_H_DEFINED
