/*!@file CUDA/CudaHmaxFL.C Riesenhuber & Poggio's HMAX model for object recognition */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
//
//
#include "CUDA/CudaSaliency.H"
#include "CUDA/CudaHmaxFLSal.H"
#include "CUDA/CudaHmaxFL.H"
#include "CUDA/CudaHmax.H"
#include "CUDA/CudaFilterOps.H"
#include "CUDA/CudaImage.H"
//#include "CUDA/CudaKernels.H"   // for cudaDogFilterHmax()
#include "Image/Kernels.H"   // for dogFilterHmax()
#include "CUDA/CudaMathOps.H"
#include "CUDA/CudaConvolutions.H" // for cudaConvolve() etc.
#include "Image/Normalize.H"
#include "CUDA/CudaCutPaste.H"
#include "Raster/Raster.H"
#include "Raster/RasterFileFormat.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Util/safecopy.H"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <limits>
#include <sstream>
#include <stdexcept>

// #########################################################################

CudaHmaxFLSal::CudaHmaxFLSal(MemoryPolicy mp, int dev, nub::ref<CudaSaliency> sal_in) :
  sal(sal_in)
{
  CudaHmaxFL(mp,dev,0,std::vector<int>(),std::vector<int>());
}

CudaHmaxFLSal::~CudaHmaxFLSal()
{
}

// ######################################################################
void CudaHmaxFLSal::extractRandC1Patches(Image<float> *&  posTrainingImages, int numPosTrainImages, std::vector<int> patchSizes, int numPatchesPerSize, int no)
{
  // Create a temp CudaHmaxFL object to extract C1Patches
  std::vector<int> c1ScaleSS(2);
  c1ScaleSS[0] = 1; c1ScaleSS[1] = 3;
  std::vector<int> c1SpaceSS(2);
  c1SpaceSS[0] = 10; c1SpaceSS[1] = 11;
  // desired frame sizes [11 and 13]
  CudaHmaxFL hmax(itsMp,itsDev,no,c1SpaceSS,c1ScaleSS,2,true,1.0F,1.0F,0.3F,4.05F,-0.05F,11,2);


  CudaImage<float> ***patches = new CudaImage<float>**[patchSizes.size()];
  for(unsigned int i=0;i<patchSizes.size();i++){
    patches[i] = new CudaImage<float>*[numPatchesPerSize];
    for(int j=0;j<numPatchesPerSize;j++) {
      patches[i][j] = new CudaImage<float>[no];
      for(int k=0;k<no;k++) {
        // Set patches to be all zeros
        patches[i][j][k].resize(patchSizes[i],patchSizes[i],ZEROS);
      }
    }
  }

  CudaImage<float> **c1Res;
  CudaImage<float> stim;
  std::srand(time(0));
  int sb = 0; // Only one scale band

  for(int i=0;i<numPatchesPerSize;i++){
    // Randomly select an image from the list
    unsigned int imInd = static_cast<unsigned int>(floor((rand()-1.0F)/RAND_MAX*numPosTrainImages));
    stim = CudaImage<float>(posTrainingImages[imInd],itsMp,itsDev);


    //hmax.initC1(c1Res);
    hmax.getC1(stim,c1Res);

    int bsize1 = c1Res[sb][0].getWidth();
    int bsize2 = c1Res[sb][0].getHeight();
    hmax.printCorners("input",stim,i<5);
    for(unsigned int j=0;j<patchSizes.size();j++) {
      int xy1 = int(floor((rand()-1.0F)/RAND_MAX*(bsize1-patchSizes[j])));
      int xy2 = int(floor((rand()-1.0F)/RAND_MAX*(bsize2-patchSizes[j])));
      Rectangle r = Rectangle::tlbrI(xy2,xy1,xy2+patchSizes[j]-1,xy1+patchSizes[j]-1);
      for(int k=0;k<no;k++) {
        patches[j][i][k] = cudaCrop(c1Res[sb][k],r);
        patches[j][i][k] *= 255*10;
      }
    }
    hmax.printCorners("patch[3][i][0]",patches[3][i][0],i<5);
    hmax.clearC1(c1Res);

  }
setC1Patches(patches,patchSizes,numPatchesPerSize);
}

// ######################################################################
void CudaHmaxFLSal::getC1(CudaImage<float>**& c1Res)
{
  //Combination 1
  c1Res[0][0] = sal->getIMap();
  c1Res[0][1] = sal->getCMap();
  c1Res[0][2] = sal->getOMap();
  c1Res[0][3] = sal->getFMap();
  c1Res[0][4] = sal->getMMap();



}

// ######################################################################
void CudaHmaxFLSal::getC1(const CudaImage<float>& input, CudaImage<float>**& c1Res)
{
  sal->doInput(input.exportToImage());
  //Combination 1
  c1Res[0][0] = sal->getIMap();
  c1Res[0][1] = sal->getCMap();
  c1Res[0][2] = sal->getOMap();
  c1Res[0][3] = sal->getFMap();
  c1Res[0][4] = sal->getMMap();

}
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
