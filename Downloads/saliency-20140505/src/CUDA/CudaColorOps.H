/*!@file CUDA/CudaColorOps.H C++ wrapper for CUDA Color operations */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaColorOps.H $
// $Id: CudaColorOps.H 12962 2010-03-06 02:13:53Z irock $
//



#ifndef CUDACOLOROPS_H_DEFINED
#define CUDACOLOROPS_H_DEFINED

#include "CUDA/CudaImage.H"
#include "Image/PixelsTypes.H"

// NOTE: Our CUDA library only supports float implementation, no use pretending to support others with template style
void cudaGetRGBY(const CudaImage<PixRGB<float> >& src, CudaImage<float>& rg, CudaImage<float>& by,
                 const float thresh, const float min_range=0.0f, const float max_range=255.0f);
CudaImage<PixRGB<float> >  cudaToRGB(const CudaImage<float>& src);
void cudaGetComponents(const CudaImage<PixRGB<float> >& src, CudaImage<float>& red, CudaImage<float>& green, CudaImage<float>& blue);
CudaImage<float> cudaLuminance(const CudaImage<PixRGB<float> >& src);
CudaImage<float> cudaLuminanceNTSC(const CudaImage<PixRGB<float> >& src);

#endif
