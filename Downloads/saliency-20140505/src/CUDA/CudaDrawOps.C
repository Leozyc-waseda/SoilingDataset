/*!@file Image/DrawOps.C functions for drawing on images
 */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   //
// by the University of Southern California (USC) and the iLab at USC.  //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; filed July 23, 2001, following provisional applications     //
// No. 60/274,674 filed March 8, 2001 and 60/288,724 filed May 4, 2001).//
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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaDrawOps.C $
// $Id: CudaDrawOps.C 12962 2010-03-06 02:13:53Z irock $
//

#include "CUDA/CudaDrawOps.H"
#include "CUDA/CudaImage.H"
#include "Image/Rectangle.H"
#include "CUDA/cudadefs.h"
#include "Util/Assert.H"
#include "Util/sformat.H"
#include "rutz/trace.h"
#include <cmath>


void cudaDrawFilledRect(CudaImage<float> &dst, const Rectangle& r, const float color)
{
  ASSERT(dst.initialized());
  const MemoryPolicy mp = dst.getMemoryPolicy();
  int dev = dst.getMemoryDevice();
  const int w=dst.getWidth();
  const int h=dst.getHeight();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  int tile_width, tile_height;
  tile_width = tile_height = 1;
  if(r.width() > r.height())
    tile_width = tile.sz();
  else
    tile_height = tile.sz();
  cuda_c_drawFilledRect(dst.getCudaArrayPtr(),r.top(),r.left(),r.bottomI(),r.rightI(),color,w,h,tile_width,tile_height);
}

void cudaDrawFilledRect(CudaImage<PixRGB<float> > &dst, const Rectangle& r, const PixRGB<float> color)
{
  ASSERT(dst.initialized());

  const MemoryPolicy mp = dst.getMemoryPolicy();
  int dev = dst.getMemoryDevice();
  const int w=dst.getWidth();
  const int h=dst.getHeight();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize1D(dev);
  int tile_width, tile_height;
  tile_width = tile_height = 1;
  if(r.width() > r.height())
    tile_width = tile.sz();
  else
    tile_height = tile.sz();
  cuda_c_drawFilledRectRGB((float3_t *) dst.getCudaArrayPtr(),r.top(),r.left(),r.bottomI(),r.rightI(),(float3_t *)&color,w,h,tile_width,tile_height);

}



// ######################################################################
template <class T> void cudaDrawRect(CudaImage<T>& dst, const Rectangle& r, const T color, const int rad)
{
  ASSERT(dst.initialized());
  ASSERT(r.isValid());
  ASSERT(dst.rectangleOk(r));

  int w=dst.getWidth();
  int h=dst.getHeight();
  int topB,topE,botB,botE,leftB,leftE,rightB,rightE;

  topB = std::max(r.top()-rad,0);
  topE = std::min(r.top()+rad,h-1);
  botB = std::max(r.bottomI()-rad,0);
  botE = std::min(r.bottomI()+rad,h-1);
  leftB = std::max(r.left()-rad,0);
  leftE = std::min(r.left()+rad,w-1);
  rightB = std::max(r.rightI()-rad,0);
  rightE = std::min(r.rightI()+rad,w-1);
  // Top Line
  cudaDrawFilledRect(dst,Rectangle::tlbrI(topB,r.left(),topE,r.rightI()),color);
  // Bottom Line
  cudaDrawFilledRect(dst,Rectangle::tlbrI(botB,r.left(),botE,r.rightI()),color);
  // Left Line
  cudaDrawFilledRect(dst,Rectangle::tlbrI(topB,leftB,botE,leftE),color);
  // Right Line
  cudaDrawFilledRect(dst,Rectangle::tlbrI(topB,rightB,botE,rightE),color);

}

template void cudaDrawRect(CudaImage<PixRGB<float> >& dst, const Rectangle& r, const PixRGB<float> color, const int rad);
template void cudaDrawRect(CudaImage<float>& dst, const Rectangle& r, const float color, const int rad);

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
