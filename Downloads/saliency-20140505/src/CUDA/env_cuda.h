/*!@file CUDA/env_cuda.h CUDA ops for Envision */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/env_cuda.h $
// $Id: env_cuda.h 12962 2010-03-06 02:13:53Z irock $
//

#ifndef CUDA_ENV_CUDA_H_DEFINED
#define CUDA_ENV_CUDA_H_DEFINED

#include "Envision/env_image_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

  //! Direct CUDA-accelerated replacement for env_pyr_build_lowpass_5
  void env_pyr_build_lowpass_5_cuda(const struct env_image* image, env_size_t firstlevel,
                                    const struct env_math* imath, struct env_pyr* result);

  //! Core lowpass5 pyramid computation
  /*! this runs fully in device memory and assumes you have allocated
      it. Note that level 0 is also not computed here. See
      env_pyr_build_lowpass_5_cuda() for what you have to do to
      prepare the work and collect the results. */
  void cudacore_pyr_build_lowpass_5(const int *dsrc, int *dtmp, int *dres, const int depth,
                                    const env_size_t w, const env_size_t h,
                                    env_size_t *outw, env_size_t *outh);

#ifdef __cplusplus
}
#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // CUDA_ENV_CUDA_H_DEFINED
