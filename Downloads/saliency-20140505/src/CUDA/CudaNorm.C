/*!@file CUDA/CudaNorm.C C++ wrapper for CUDA normalization methods */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaNorm.C $
// $Id: CudaNorm.C 12962 2010-03-06 02:13:53Z irock $
//

#include "CUDA/CudaImage.H"
#include "Util/Assert.H"
#include "CUDA/cudadefs.h"
#include "CUDA/CudaConvolutions.H"
#include "CUDA/CudaMathOps.H"
#include "CUDA/CudaShapeOps.H"
#include "CUDA/CudaKernels.H"
#include "Image/fancynorm.H"
#include "CUDA/CudaNorm.H"
#include "CudaDevices.H"
#include "wrap_c_cuda.h"


// ######################################################################
CudaImage<float> cudaMaxNormalize(const CudaImage<float>& src,
                      const float mi, const float ma, const MaxNormType normtyp,
                      int nbiter, const CudaImage<float> *lrexcit)
{

  // do normalization depending on desired type:
  switch(normtyp)
    {
    case VCXNORM_FANCY:
      return cudaMaxNormalizeFancy(src, mi, ma, nbiter, 1.0, lrexcit);
      break;
    default:
      LFATAL("Unhandled normalization type: %d", int(normtyp));
    }
  return CudaImage<float>();
}

// ######################################################################
// ##### fancyNorm from Itti et al, JEI, 2001 -- FULL implementation:
CudaImage<float> cudaMaxNormalizeFancy(const CudaImage<float>& src, const float mi, const float ma,
                           const int nbiter, const float weakness,
                           const CudaImage<float>* lrexcit)
{
  // Normalize using fancy normalization: multiple iterations of
  // filtering by a DoG
  ASSERT(src.initialized());
  ASSERT(src.getMemoryPolicy() != HOST_MEMORY);

  MemoryPolicy mp = src.getMemoryPolicy();
  int dev = src.getMemoryDevice();
  const int w = src.getWidth();
  const int h = src.getHeight();
  int siz = std::max(w, h);
  int maxhw = std::max(0, std::min(w, h) / 2 - 1);

  // Allocate max buffer so that it can be reused
  CudaImage<float> maxim = CudaImage<float>(1,1,NO_INIT,mp,dev);
  CudaImage<float> result = src;
  // first clamp negative values to zero
  cudaInplaceRectify(result);

  // then, normalize between mi and ma if not zero
  if (mi != 0.0F || ma != 0.0F) cudaInplaceNormalize(result, mi, ma);


  // build separable Gaussians for DoG computation:
  float esig = (float(siz) * FANCYESIG) * 0.01F;
  float isig = (float(siz) * FANCYISIG) * 0.01F;
  CudaImage<float> gExc = cudaGaussian(mp,dev,FANCYCOEX/(esig*sqrt(2.0*M_PI)) * weakness, esig, maxhw);
  CudaImage<float> gInh = cudaGaussian(mp,dev,FANCYCOIN/(isig*sqrt(2.0*M_PI)) * weakness, isig, maxhw);

  for (int i = 0; i < nbiter; ++i)
    {
      if (lrexcit)           // tuned long-range excitation
        {

          CudaImage<float> tmp = cudaDownSize(result, w / (1<<LRLEVEL), h / (1<<LRLEVEL));
          tmp = cudaConvolve(tmp, *lrexcit, CONV_BOUNDARY_ZERO);  // full influence
          tmp += 1.0F;     // will be used as multiplicative excitation
          cudaInplaceClamp(tmp, 1.0F, 1.25F);  // avoid crazyness
          tmp = cudaRescale(tmp, w, h);
          result *= tmp;
        }

      CudaImage<float> excit = cudaSepFilter(result, gExc, gExc, CONV_BOUNDARY_CLEAN); // excitatory part
      CudaImage<float> inhib = cudaSepFilter(result, gInh, gInh, CONV_BOUNDARY_CLEAN); // inhibitory part
      excit -= inhib;
      cudaGetMax(result, maxim);

      result += excit;        // we get broad inhibition from everywhere

      result += cudaGetScalar(maxim)*(-0.01F * FANCYINHI) ;    // we get fixed global inhibition

      cudaInplaceRectify(result);

      //      sigmoid(FANCYG, FANCYH, FANCYS);
    }
  return result;
}
