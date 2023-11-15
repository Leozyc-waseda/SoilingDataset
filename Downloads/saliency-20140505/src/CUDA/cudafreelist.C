/** @file rutz/freelist.cc memory allocation via a free-list pool */

///////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2001-2004 California Institute of Technology
// Copyright (c) 2004-2007 University of Southern California
// Rob Peters <rjpeters at usc dot edu>
//
// created: Fri Jul 20 08:00:31 2001
// commit: $Id: cudafreelist.C 12962 2010-03-06 02:13:53Z irock $
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/cudafreelist.C $
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

#ifndef RUTZ_CUDAFREELIST_CC_DEFINED
#define RUTZ_CUDAFREELIST_CC_DEFINED

#include "CUDA/cudafreelist.H"
#include "CUDA/CudaDevices.H"
#include "Util/Assert.H"
#include <map>

rutz::cuda_free_list_base::cuda_free_list_base(std::size_t size_check) :
  m_size_check(size_check)
{
  for(int d=0;d<MAX_CUDA_DEVICES;d++)
    {
      m_node_list[d] = 0;
      m_num_allocations[d] = 0;
    }
}

void* rutz::cuda_free_list_base::allocate(std::size_t bytes, int dev)
{
  ASSERT(bytes == m_size_check);
  // Check if the device has been used
  int index = find_index_from_device(dev);

  if (m_node_list[index] == 0)
    {
      ++m_num_allocations[index];
      void *d;
      CudaDevices::malloc(&d,bytes,dev);
      return d;
    }
  node* n = m_node_list[index];
  m_node_list[index] = m_node_list[index]->next;
  void *mem = n->mem;
  // Free the node
  delete n;
  return mem;
}

void rutz::cuda_free_list_base::deallocate(void* space, int dev)
{
  int index = find_index_from_device(dev);

  node* n = new node();
  n->mem = space;
  n->next = m_node_list[index];
  m_node_list[index] = n;
}

int rutz::cuda_free_list_base::get_num_nodes(int dev)
{
  int index = find_index_from_device(dev);
  node *n = m_node_list[index];
  int nodes=0;
  while(n != NULL)
    {
      n = n->next;
      nodes++;
    }
  return nodes;
}

int rutz::cuda_free_list_base::get_index_from_device(int dev)
{
 if( devices.find(dev) == devices.end ())
    {
      return -1;
    }
  else
    {
      return devices[dev];
    }
}

int rutz::cuda_free_list_base::find_index_from_device(int dev)
{
  int index = get_index_from_device(dev);
  if(index == -1)
    {
      index = devices.size();
      devices.insert(std::pair<int,int>(dev,index));
    }
  return index;
}

void rutz::cuda_free_list_base::release_free_nodes()
{
  std::map<int,int>::iterator it;
  for ( it=devices.begin() ; it != devices.end(); it++ )
    {
      // Key is device number, value is the index
      int dev = (*it).first;
      int index = (*it).second;
      while (m_node_list[index] != 0)
        {
          void* mem = m_node_list[index]->mem;
          node* n = m_node_list[index];
          m_node_list[index] = m_node_list[index]->next;
          delete n;
          CudaDevices::free(mem,dev);
          --m_num_allocations[index];
        }
    }
}

#endif // !RUTZ_CUDAFREELIST_CC_DEFINED
