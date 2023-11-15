/*!@file Util/AllocAux.C memory allocation routines for 16-byte alignment */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaAlloc.C $
// $Id: CudaAlloc.C 15310 2012-06-01 02:29:24Z itti $
//

#ifndef CUDAALLOC_C_DEFINED
#define CUDAALLOC_C_DEFINED

#include "Util/Assert.H"
#include "Util/log.H"
#include "Util/sformat.H"
#include "CUDA/cudafreelist.H"
#include "rutz/mutex.h"
#include "rutz/trace.h"
#include "CUDA/CudaDevices.H"
#include <map>
#include <pthread.h>

namespace
{
  //! Trivial allocator that just calls operator new() and operator delete()
  struct cuda_trivial_alloc
  {
    void set_debug(bool /*do_debug*/)
    {
      // no debug settings for this allocator type
    }

    void set_allow_caching(bool /*on*/)
    {
      // no caching here in any case
    }

    void show_stats(int /*verbosity*/, const char* /*pfx*/,
                    const size_t block_size, const size_t overhead) const
    {
      // nothing to do here
    }

    void* allocate(size_t nbytes, rutz::cuda_free_list_base** source, int dev)
    {
      if (source != 0)
        *source = 0;
      void *ret;
      CudaDevices::malloc(&ret,nbytes,dev);
      return ret;
    }

    void deallocate(void* space, rutz::cuda_free_list_base* source, int dev)
    {
      ASSERT(source == 0);
      CudaDevices::free(space,dev);
    }

    void release_free_mem()
    {
      // nothing to do here
    }
  };

  //! Caching allocator with free lists for common allocation sizes
  template <size_t cache_size>
  struct cuda_fastcache_alloc
  {
    rutz::cuda_free_list_base* cache[cache_size];
    mutable size_t num_alloc[cache_size][MAX_CUDA_DEVICES];
    bool allow_caching;

    cuda_fastcache_alloc()
      :
      allow_caching(true)
    {
      for (size_t i = 0; i < cache_size; ++i)
        {
          this->cache[i] = 0;
          for(int d=0;d<MAX_CUDA_DEVICES;d++)
            this->num_alloc[i][d] = 0;
        }
    }

    void set_debug(bool /*do_debug*/)
    {
      // no debug settings for this allocator type
    }

    void set_allow_caching(bool on)
    {
      if (!on && this->allow_caching)
        {
          // if we are currently caching but are being asked to turn
          // off caching, then let's first free any existing caches

          this->release_free_mem();
        }
      this->allow_caching = on;
    }

    void show_stats(int verbosity, const char* pfx,
                    const size_t block_size, const size_t overhead) const
    {
      size_t nused = 0;

      std::map<size_t, std::string> msgs;
      size_t bytes_allocated = 0;
      for (size_t i = 0; i < cache_size; ++i)
        {

          if (this->cache[i] != 0)
            {
              std::map<int,int>::const_iterator devAt = this->cache[i]->getDevicesBegin();
              std::map<int,int>::const_iterator devStop = this->cache[i]->getDevicesEnd();
              for(;devAt!=devStop;devAt++)
                {

                  // TODO: go through all of the devices
                  int dev = (*devAt).first;
                  int dev_index = (*devAt).second;

                  ++nused;
                  const size_t nb = (this->cache[i]->num_allocations(dev)
                                     * this->cache[i]->alloc_size());

                  const size_t extra = (this->cache[i]->num_allocations(dev)
                                        - this->num_alloc[i][dev_index]);

                  this->num_alloc[i][dev_index] = this->cache[i]->num_allocations(dev);

                  bytes_allocated += nb;

                  if (verbosity <= 0)
                    continue;

                  std::string msg =
                    sformat("%s%sfastcache[%02" ZU "/%02" ZU "]: CUDA device %d, "
                            "%10.4fMB in %4" ZU " allocations of %10.4fkB",
                            pfx ? pfx : "", pfx ? ": " : "",
                            i, cache_size, dev, nb / (1024.0*1024.0),
                            this->cache[i]->num_allocations(dev),
                            this->cache[i]->alloc_size() / 1024.0);

                  if (block_size > 0)
                    {
                      if (this->cache[i]->alloc_size() - overhead >= block_size
                          || this->cache[i]->alloc_size() - overhead <= 1)
                        msg += sformat(" (%.2fkB * %7.1f + %" ZU "B)",
                                       block_size / 1024.0,
                                       (double(this->cache[i]->alloc_size() - overhead)
                                        / double(block_size)),
                                       overhead);
                      else
                        msg += sformat(" (%.2fkB / %7.1f + %" ZU "B)",
                                       block_size / 1024.0,
                                       (double(block_size)
                                        / double(this->cache[i]->alloc_size() - overhead)),
                                       overhead);
                    }

                  if (extra > 0)
                    msg += sformat(" (+%" ZU " new)", extra);

                  msgs[this->cache[i]->alloc_size()] = msg;


                  msg =
                    sformat("%s%sfastcache_alloc<%" ZU ">: %" ZU "/%" ZU " cache table "
                            "entries in use, %fMB total allocated",
                            pfx ? pfx : "", pfx ? ": " : "",
                            cache_size, nused, cache_size,
                            bytes_allocated / (1024.0*1024.0));

                  if (block_size > 0)
                    msg += sformat(" (%.2fkB * %7.1f)",
                                   block_size / 1024.0,
                                   double(bytes_allocated) / double(block_size));

                  LINFO("%s", msg.c_str());
                }
            }
        }
      for (std::map<size_t, std::string>::const_iterator
             itr = msgs.begin(), stop = msgs.end();
           itr != stop; ++itr)
        LINFO("%s", (*itr).second.c_str());
    }

    // allocate memory block of size nbytes; also return the address
    // of the rutz::cuda_free_list_base, if any, that was used for
    // allocation
    void* allocate(size_t nbytes, int dev)
    {
      if (this->allow_caching)
        for (size_t i = 0; i < cache_size; ++i)
          {
            if (this->cache[i] != 0)
              {
                // we found a filled slot, let's see if it matches our
                // requested size
                if (this->cache[i]->alloc_size() == nbytes)
                  {
                    return this->cache[i]->allocate(nbytes,dev);
                  }
                // else, continue
              }
            else // this->cache[i] == 0
              {
                // we found an empty slot, let's set up a new free
                // list for our requested size:
                this->cache[i] = new rutz::cuda_free_list_base(nbytes);
                return this->cache[i]->allocate(nbytes,dev);
              }
          }
      void *ret;
      CudaDevices::malloc(&ret,nbytes,dev);//::operator new(nbytes);
      return ret;
    }

    // deallocate memory from the given rutz::cuda_free_list_base,
    // otherwise free it globally
    void deallocate(void* space, int dev, size_t nbytes)
    {
      if (this->allow_caching)
        {
        for (size_t i = 0; i < cache_size; ++i)
          {
            if (this->cache[i] != 0)
              {
                // we found a filled slot, let's see if it matches our
                // requested size
                if (this->cache[i]->alloc_size() == nbytes)
                  {
                    this->cache[i]->deallocate(space,dev);
                    return;
                  }
                // else, continue
              }
            else // this->cache[i] == 0
              {
                // we found an empty slot, let's set up a new free
                // list to store our deallocated size:
                this->cache[i] = new rutz::cuda_free_list_base(nbytes);
                this->cache[i]->deallocate(space,dev);
              }
          }
        }
      else
        {
          CudaDevices::free(space,dev);
        }
    }

    void release_free_mem()
    {
      for (size_t i = 0; i < cache_size; ++i)
        if (this->cache[i] != 0)
          this->cache[i]->release_free_nodes();
    }
  };







  /* Here are the various macros that you can twiddle if you need to
     change the allocation strategy. Basically you can have aligned
     allocation (DO_ALIGN) at an arbitrary N-byte boundary (NALIGN),
     with optional freelist caching (DO_FASTCACHE) of (NCACHE)
     commonly-requested memory sizes.

     If you turn off both DO_ALIGN and DO_FASTCACHE, you will end up
     using trivial_alloc, which is just a bare wrapper around operator
     new() and operator delete(). By default, malloc() returns 8-byte
     aligned memory on gnu/linux/x86 machines.

     Note that certain libraries (fftw [see FourierEngine] in
     particular) require greater than 8-byte alignment, so if you are
     going to be using those parts of the code, then you'll need to
     leave DO_ALIGN set, with NALIGN>=16. Also note that NALIGN must
     be at least 4*sizeof(void*) -- in particular, 16 will be too
     small on 64-bit systems for which sizeof(void*) is 8; for those
     systems we'll need NALIGN>=32.

     DO_FASTCACHE is here primarily for performance; since our memory
     usage pattern tends to involve many many allocations of Image
     objects with only a few different Dims shapes, it helps to cache
     those memory allocations in a freelist. Profiling tests showed
     that this can give a 15-20% speedup.
  */

#define DO_FASTCACHE
#define NCACHE        64

#  ifdef DO_FASTCACHE
  typedef cuda_fastcache_alloc<NCACHE> cuda_alloc_type;
#  else
  typedef cuda_trivial_alloc      cuda_alloc_type;
#endif

  // Here is our global allocator object, whose type is determined by
  // the various macro settings abovve, and a corresponding mutex. For
  // now, we use a heavy-handed approach and just use the mutex to
  // lock the entire structure during each call to any of the public
  // functions. If this turns out to be a performance problem, we
  // could turn to finer-grained locking within the various allocator
  // classes themselves.
  cuda_alloc_type    cuda_alloc;
  pthread_mutex_t cuda_alloc_mutex = PTHREAD_MUTEX_INITIALIZER;

  size_t          cuda_stats_units = 0;
}

void* cuda_invt_allocate(size_t user_nbytes, int dev)
{
  GVX_MUTEX_LOCK(&cuda_alloc_mutex);
  return cuda_alloc.allocate(user_nbytes,dev);
}

void cuda_invt_deallocate(void* mem, int dev, size_t nbytes)
{
  GVX_MUTEX_LOCK(&cuda_alloc_mutex);
  cuda_alloc.deallocate(mem,dev,nbytes);
}

void cuda_invt_allocation_release_free_mem()
{
  GVX_MUTEX_LOCK(&cuda_alloc_mutex);
  cuda_alloc.release_free_mem();
}

void cuda_invt_allocation_allow_caching(bool on)
{
  GVX_MUTEX_LOCK(&cuda_alloc_mutex);
  cuda_alloc.set_allow_caching(on);
}

void cuda_invt_allocation_debug_print(bool do_debug)
{
  GVX_MUTEX_LOCK(&cuda_alloc_mutex);
  cuda_alloc.set_debug(do_debug);
}

void cuda_invt_allocation_show_stats(int verbosity, const char* pfx,
                                const size_t block_size)
{
  GVX_MUTEX_LOCK(&cuda_alloc_mutex);
  cuda_alloc.show_stats(verbosity, pfx,
                     block_size ? block_size : cuda_stats_units, 0);
}

void cuda_invt_allocation_set_stats_units(const size_t units)
{
  cuda_stats_units = units;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // CUDAALLOC_C_DEFINED
