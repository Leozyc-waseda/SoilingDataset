/*!@file CUDA/cudadefs.h CUDA/GPU definitions */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/cudadefs.h $
// $Id: cudadefs.h 12962 2010-03-06 02:13:53Z irock $
//

#ifndef CUDADEFS_H_DEFINED
#define CUDADEFS_H_DEFINED

#include <math.h>

// Saliency C++ Toolkit implements Pixels as a template
// that takes the dimensionality as a template argument
// That definition will not work in CUDA C, so we are left
// with this ugliness

// 24-bit multiplication is faster on G80, but we must be sure to
// multiply integers only within [-8M, 8M - 1] range
#define IMUL(a, b) __mul24(a, b)

//! Structure to handle 3 dimensional float
typedef struct
{
  float p[3];
} float3_t;

//! Structure to handle 4 dimensional float
typedef struct
{
  float p[4];
} float4_t;

// The tile size should be the same across all CUDA functions within a particular device and run, since one function might call another
#define MAX_CUDA_DEVICES 4
#define CUDA_TILE_W 16
#define CUDA_TILE_H 16
#define CUDA_1D_TILE_W 256
#define PI 3.14159265358979f

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

//Round a / b to nearest higher integer value in CUDA code
#define IDIVUP(a, b) ((a % b != 0) ? (a / b + 1) : (a / b))

// Find min/max
#define MIN(a,b) ((a<b) ? a : b)
#define MAX(a,b) ((a>b) ? a : b)

#endif
