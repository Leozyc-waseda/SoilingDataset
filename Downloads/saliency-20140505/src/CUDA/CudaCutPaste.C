/*!@file CUDA/CudaCutPaste.C Cut+paste operations from/to CudaImage subregions */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
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
// Primary maintainer for this file: Rob Peters <rjpeters@klab.caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaCutPaste.C $
// $Id: CudaCutPaste.C 12962 2010-03-06 02:13:53Z irock $
//

#include "CUDA/CudaCutPaste.H"
#include "CUDA/CudaMathOps.H"
#include "CUDA/CudaImage.H"
#include "Image/Pixels.H"
#include "Util/Assert.H"
#include <algorithm>

// ######################################################################
CudaImage<float> cudaCrop(const CudaImage<float>& src, const Point2D<int>& pt, const Dims& dims,
              const bool zerofill)
{

  if (pt == Point2D<int>(0,0) && dims == src.getDims())
    return src;

  const int dev = src.getMemoryDevice();
  const MemoryPolicy mp = src.getMemoryPolicy();
  ASSERT(mp != HOST_MEMORY);
  Dims tile = CudaDevices::getDeviceTileSize(dev);

  CudaImage<float> result(dims, NO_INIT,mp,dev);

  if(!zerofill)
    {
      ASSERT(src.coordsOk(pt));
      ASSERT(src.coordsOk(pt.i + dims.w() - 1, pt.j + dims.h() - 1));
    }

  int endX = pt.i+dims.w(); int maxX = std::min(endX,src.getWidth());
  int endY = pt.j+dims.h(); int maxY = std::min(endY,src.getHeight());
  cuda_c_crop(src.getCudaArrayPtr(),result.getCudaArrayPtr(),src.getWidth(),src.getHeight(),pt.i,pt.j,endX,endY,maxX,maxY,tile.w(),tile.h());
  return result;
}

// ######################################################################
CudaImage<float> cudaCrop(const CudaImage<float>& src, const Rectangle& rect, const bool zerofill)
{
  return cudaCrop(src,
              Point2D<int>(rect.left(), rect.top()),
              Dims(rect.width(), rect.height()),
              zerofill);
}

// ######################################################################
CudaImage<float> cudaShiftImage(const CudaImage<float>& src, const float dx, const float dy)
{
  // make sure the source image is valid
  ASSERT(src.initialized());

  // create and clear the return image
  Dims dim(src.getDims());
  const int dev = src.getMemoryDevice();
  const MemoryPolicy mp = src.getMemoryPolicy();
  CudaImage<float> result(dim, ZEROS,mp,dev);
  Dims tile = CudaDevices::getDeviceTileSize(dev);
  cuda_c_shiftImage(src.getCudaArrayPtr(),result.getCudaArrayPtr(),dim.w(),dim.h(),dx,dy,tile.w(),tile.h());
  return result;
}



// ######################################################################
CudaImage<float> cudaShiftClean(const CudaImage<float>& src, const int dx, const int dy,
                    const float bgval)
{
  // make sure the source image is valid
  ASSERT(src.initialized());

  // create and clear the return image
  int w = src.getWidth(), h = src.getHeight();
  const int dev = src.getMemoryDevice();
  const MemoryPolicy mp = src.getMemoryPolicy();
  Dims tile = CudaDevices::getDeviceTileSize(dev);

  CudaImage<float> result(w, h, NO_INIT, mp, dev); cudaClear(result,bgval);

  return result;

  // // find range of pixels to copy:
  // int startx = std::max(0, -dx), endx = std::min(w - 1, w - 1 - dx);
  // if (startx >= w || endx < 0) return retImg; // empty result
  // int starty = std::max(0, -dy), endy = std::min(h - 1, h - 1 - dy);
  // if (starty >= h || endy < 0) return retImg; // empty result

  // int dstx = std::max(0, std::min(w - 1, dx));
  // int dsty = std::max(0, std::min(h - 1, dy));

  // src += startx + starty * w;
  // dst += dstx + dsty * w;

  // int skip = w - endx + startx - 1;

  // // do the copy:
  // for (int j = starty; j <= endy; j ++)
  //   {
  //     for (int i = startx; i <= endx; i ++) *dst++ = *src++;

  //     // ready for next row of pixels:
  //     src += skip; dst += skip;
  //   }

}



void cudaInplacePaste(CudaImage<float>& dst,
                  const CudaImage<float>& img, const Point2D<int>& pos)
{
  int w = dst.getWidth(), h = dst.getHeight();
  int iw = img.getWidth(), ih=img.getHeight();

  ASSERT(pos.i + iw <= w && pos.j + ih <= h);
  const int dev = dst.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize(dev);
  cuda_c_inplacePaste(dst.getCudaArrayPtr(),img.getCudaArrayPtr(),w,h,iw,ih,pos.i,pos.j,tile.w(),tile.h());
}

void cudaInplacePaste(CudaImage<PixRGB<float> >& dst,
                      const CudaImage<PixRGB<float> >& img, const Point2D<int>& pos)
{
  int w = dst.getWidth(), h = dst.getHeight();
  int iw = img.getWidth(), ih=img.getHeight();

  ASSERT(pos.i + iw <= w && pos.j + ih <= h);
  const int dev = dst.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize(dev);
  cuda_c_inplacePasteRGB((float3_t *)dst.getCudaArrayPtr(),(float3_t *)img.getCudaArrayPtr(),w,h,iw,ih,pos.i,pos.j,tile.w(),tile.h());
}


void cudaInplaceOverlay(CudaImage<float>& dst, const CudaImage<float>& img, const Point2D<int>& pos)
{
  int w = dst.getWidth(), h = dst.getHeight();
  int iw = img.getWidth(), ih=img.getHeight();

  ASSERT(pos.i + iw <= w && pos.j + ih <= h);
  const int dev = dst.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize(dev);
  cuda_c_inplaceOverlay(dst.getCudaArrayPtr(),img.getCudaArrayPtr(),w,h,iw,ih,pos.i,pos.j,tile.w(),tile.h());
}


void cudaInplaceOverlay(CudaImage<PixRGB<float> >&dst, const CudaImage<PixRGB<float> >&img, const Point2D<int>& pos)
{
  int w = dst.getWidth(), h = dst.getHeight();
  int iw = img.getWidth(), ih=img.getHeight();

  ASSERT(pos.i + iw <= w && pos.j + ih <= h);
  const int dev = dst.getMemoryDevice();
  Dims tile = CudaDevices::getDeviceTileSize(dev);
  cuda_c_inplaceOverlayRGB((float3_t *)dst.getCudaArrayPtr(),(float3_t *)img.getCudaArrayPtr(),w,h,iw,ih,pos.i,pos.j,tile.w(),tile.h());
}

