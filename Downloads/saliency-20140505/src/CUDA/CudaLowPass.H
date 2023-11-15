/*!@file CUDA/CudaLowPass.H C++ wrapper for CUDA Low pass operations */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaLowPass.H $
// $Id: CudaLowPass.H 12962 2010-03-06 02:13:53Z irock $
//

#ifndef CUDALOWPASS_H_DEFINED
#define CUDALOWPASS_H_DEFINED

CudaImage<float> cudaLowPass5xDec(const CudaImage<float>&src);
CudaImage<float> cudaLowPass5yDec(const CudaImage<float>&src);
CudaImage<float> cudaLowPass5Dec(const CudaImage<float>& src, const bool go_x=true, const bool go_y=true);
CudaImage<float> cudaLowPass9xDec(const CudaImage<float>&src);
CudaImage<float> cudaLowPass9yDec(const CudaImage<float>&src);
CudaImage<float> cudaLowPass9Dec(const CudaImage<float>& src, const bool go_x=true, const bool go_y=true);
CudaImage<float> cudaLowPass9xyDec(const CudaImage<float>&src);
CudaImage<float> cudaLowPass3x(const CudaImage<float>&src);
CudaImage<float> cudaLowPass3y(const CudaImage<float>&src);
CudaImage<float> cudaLowPass5x(const CudaImage<float>&src);
CudaImage<float> cudaLowPass5y(const CudaImage<float>&src);
CudaImage<float> cudaLowPass9x(const CudaImage<float>&src);
CudaImage<float> cudaLowPass9y(const CudaImage<float>&src);
CudaImage<float> cudaLowPass9(const CudaImage<float>& src, const bool go_x=true, const bool go_y=true);
CudaImage<float> cudaLowPass(const int N, const CudaImage<float>& src, const bool go_x = true, const bool go_y = true);
CudaImage<float> cudaLowPassX(const int N, const CudaImage<float>& src);
CudaImage<float> cudaLowPassY(const int N, const CudaImage<float>& src);
#endif
