/*!@file CUDA/Debayer.C C++ wrapper for CUDA Debayer operations */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CUDAdebayer.C $
// $Id: CUDAdebayer.C 12962 2010-03-06 02:13:53Z irock $
//

#include "CUDA/CudaImage.H"
#include "Util/Assert.H"
#include "CUDA/cudadefs.h"
#include "CUDAdebayer.H"
#include "CudaDevices.H"
#include "wrap_c_cuda.h"

// Our CUDA library only supports float implementation, no use pretending to support others with template style
CudaImage<PixRGB<float> > cuda_1_debayer(const CudaImage<float>& src)
{
  // Ensure that the data is valid
  ASSERT(src.initialized());

  // Ensure that we are on a CUDA device
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  const int dev = src.getMemoryDevice();
  // Output is the same size as the input for this filter

  //Allocating the memory on output

  CudaImage<PixRGB<float> > result(src.getDims(), NO_INIT, src.getMemoryPolicy(), dev);

  const Dims tile = CudaDevices::getDeviceTileSize(dev);
  // printf("Tile dimension : %d %d",tile.w(),tile.h());
  // Now call the CUDA implementation
  cuda_2_debayer((float *)src.getCudaArrayPtr(),
                 (float3_t *)result.getCudaArrayPtr(),result.getWidth(),result.getHeight(),
                 tile.w(),tile.h());
  // We have to do the following at some point
  // CUDA_Bind2TextureArray(tex1,result,FloatKind);
  return result;
}

