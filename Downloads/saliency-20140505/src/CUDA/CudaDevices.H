/*!@file CUDA/CudaDevices.H Static class to keep track of status and properties of CUDA devices */

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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaDevices.H $
// $Id: CudaDevices.H 12962 2010-03-06 02:13:53Z irock $
//


#ifndef CUDADEVICES_H_DEFINED
#define CUDADEVICES_H_DEFINED

#include <map>
#include <list>
#include <cmath>
#include "Image/Dims.H"
#include "CUDA/cutil.h"
#include <cuda_runtime_api.h>
#include "Util/log.H"
#include <stdlib.h>
#include <stdio.h>

using std::map;
using std::list;

class CudaDevices
{
 public:
  static void initDevice(const int deviceNum)
  {
    // Optimization, avoid repeated initializations
    if(isValid(deviceNum))
      return;
    // Initializing CUDA devices should not happen often, we can be thorough
    int deviceCount;
    CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
      LFATAL("No devices supporting CUDA");

    if (deviceNum > deviceCount-1)
      LFATAL("Device [%d] out of valid device range [0-%d]",deviceNum,deviceCount-1);

    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, deviceNum));

    if (deviceProp.major < 1)
      LFATAL("Device [%d] does not support CUDA",deviceNum);

    // Determine the optimal tile size for this device
    // Take the square root of the maximum num of threads per block and then find
    // the next smallest power of 2
    int sqSize = nextpow2down((int) sqrt(deviceProp.maxThreadsPerBlock));
    Dims tileSize = Dims(sqSize,sqSize);
    int smSize = deviceProp.sharedMemPerBlock;
    setDeviceTileSize(deviceNum,tileSize);
    setDeviceSharedMemorySize(deviceNum,smSize);
  }

  static inline int nextpow2down (int x)
  {
    if (x < 0)
      return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x = x>>1;
    return x+1;
  }

  static void memcpyHostToDevice(void *trg, const void *src, const int len, const int deviceNum)
  {
    setCurrentDevice(deviceNum);
    CUDA_SAFE_CALL( cudaMemcpy(trg, src, len, cudaMemcpyHostToDevice));
  }

  static void memcpyDeviceToHost(void *trg, const void *src, const int len, const int deviceNum)
  {
    setCurrentDevice(deviceNum);
    CUDA_SAFE_CALL( cudaMemcpy(trg, src, len, cudaMemcpyDeviceToHost));
  }

  static void memcpyDeviceToDevice(void *trg, const void *src, const int len, const int trgDev, const int srcDev)
  {
    if(trgDev == srcDev)
    {
      setCurrentDevice(srcDev);
      CUDA_SAFE_CALL( cudaMemcpy(trg, src, len, cudaMemcpyDeviceToDevice));
    }
    else
    {
      char *tmp;
      tmp = (char *) std::malloc(len);
      memcpyDeviceToHost(tmp,src,len,srcDev);
      memcpyHostToDevice(trg,tmp,len,trgDev);
      std::free(tmp);
    }
  }

  static void malloc(void **ret, const int sz, const int deviceNum)
  {
    setCurrentDevice(deviceNum);
    cudaError err = cudaMalloc(ret,sz);
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",
                __FILE__, __LINE__, cudaGetErrorString( err) );
        printf("Attempted to malloc %d bytes on cuda device\n",sz);
        abort();
    }
  //CUDA_SAFE_CALL(cudaMalloc(ret, sz) );
    if(*ret == NULL && sz > 0)
      {
        LFATAL("CUDA Device [%d] failed to allocate %d bytes of memory",deviceNum,sz);
      }
  }

  static void free(void *mem, const int deviceNum)
  {
    setCurrentDevice(deviceNum);
    cudaError err = cudaFree(mem);
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",
                __FILE__, __LINE__, cudaGetErrorString( err) );
        printf("Attempted to free %p address on cuda device\n",mem);
        abort();
    }

  //CUDA_SAFE_CALL(cudaFree(mem) );
  }

  static void memset(void *mem, const char val, const int sz, const int deviceNum)
  {
    setCurrentDevice(deviceNum);
    CUDA_SAFE_CALL(cudaMemset(mem, val, sz) );
  }

  static inline Dims getDeviceTileSize(const int deviceNum)
  {
    return deviceTileSizes[deviceNum];
  }

  static inline int getDeviceSharedMemorySize(const int deviceNum)
  {
    return deviceSharedMemorySizes[deviceNum];
  }


  static inline Dims getDeviceTileSize1D(const int deviceNum)
  {
    Dims tile1D = Dims(deviceTileSizes[deviceNum].sz(),1);
    return tile1D;
  }

  static inline bool isValid(const int deviceNum)
  {
    if(deviceTileSizes[deviceNum].isEmpty())
      return false;
    return true;
  }

  static void displayProperties( const int deviceNum )
  {
    setCurrentDevice(deviceNum);
    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, deviceNum));
    if (deviceProp.major < 1)
    {
      LINFO("Video Device [%d] does not support CUDA",deviceNum);
      return;
    }
    printf( "\nCUDA Device Name \t - %s ", deviceProp.name );
    printf( "\n**************************************");
    printf( "\nTotal Global Memory\t\t -%lu KB", deviceProp.totalGlobalMem/1024 );
    printf( "\nShared memory available per block \t - %lu KB", deviceProp.sharedMemPerBlock/1024 );
    printf( "\nNumber of registers per thread block \t - %d", deviceProp.regsPerBlock );
    printf( "\nWarp size in threads \t - %d", deviceProp.warpSize );
    printf( "\nMemory Pitch \t - %d bytes", (int) deviceProp.memPitch );
    printf( "\nMaximum threads per block \t - %d",  deviceProp.maxThreadsPerBlock );
    printf( "\nMaximum Thread Dimension (block) \t - %d %d %d", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2] );
    printf( "\nMaximum Thread Dimension (grid) \t - %d %d %d", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2] );
    printf( "\nTotal constant memory \t - %d bytes", (int) deviceProp.totalConstMem );
    printf( "\nCUDA ver \t - %d.%d", deviceProp.major, deviceProp.minor );
    printf( "\nClock rate \t - %d KHz", deviceProp.clockRate );
    printf( "\nTexture Alignment \t - %d bytes", (int) deviceProp.textureAlignment );
    printf( "\nDevice Overlap \t - %s", deviceProp. deviceOverlap?"Allowed":"Not Allowed" );
    printf( "\nNumber of Multi processors \t - %d\n", deviceProp.multiProcessorCount );
  }


  static inline void setCurrentDevice(const int deviceNum)
  {
    if(!isValid(deviceNum)) initDevice(deviceNum);
    _setCurrentDevice(deviceNum);
  }


 private:
  // Don't allow objects of CudaDevices
  CudaDevices();

  // Don't check if our setup is valid
  static inline void _setCurrentDevice(const int deviceNum)
  {
    // Only change if it's different
    if(deviceNum != currentDevice)
      {
        if(deviceNum < 0)
          {
            fprintf(stderr,"Attempting to set device to %d from %d in file '%s' in line %i.\n",deviceNum,currentDevice,
                    __FILE__, __LINE__ );
            abort();
          }

        CUDA_SAFE_CALL(cudaSetDevice(deviceNum));
        currentDevice = deviceNum;
      }
  }

  static inline void setDeviceTileSize(const int deviceNum, const Dims tileSize)
  {
    deviceTileSizes[deviceNum] = tileSize;
  }

  static inline void setDeviceSharedMemorySize(const int deviceNum, const int smSize)
  {
    deviceSharedMemorySizes[deviceNum] = smSize;
  }

  static map<int,Dims> deviceTileSizes;
  static map<int,int> deviceSharedMemorySizes;
  static int currentDevice;
};

#endif
