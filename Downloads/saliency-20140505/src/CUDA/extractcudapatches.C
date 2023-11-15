/*!@file HMAX/test-hmax5.C Test Hmax class and compare to original code */

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
// Primary maintainer for this file: Dan Parks <danielfp@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/extractcudapatches.C $
// $Id: extractcudapatches.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Component/ModelManager.H"
#include "GUI/XWindow.H"
#include "CUDA/CudaHmaxFL.H"
#include "CUDA/CudaHmax.H"
#include "CUDA/CudaImage.H"
#include "CUDA/CudaCutPaste.H"
#include "Image/Rectangle.H"
#include "CUDA/CudaMathOps.H"
#include "Image/Normalize.H"
#include "Image/Transforms.H"
#include "CUDA/CudaConvolutions.H"
#include "Learn/svm.h"
#include "Media/FrameSeries.H"
#include "nub/ref.h"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <cstdlib>


// number of orientations to use in HmaxFL
#define NORI 4
#define NUM_PATCHES_PER_SIZE 250


int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager *mgr = new ModelManager("Extract Patches for Hmax with Feature Learning");

  mgr->exportOptions(MC_RECURSE);

  // required arguments
  // <c1patchesDir> <trainPosDir>

  if (mgr->parseCommandLine(
                            (const int)argc, (const char**)argv, "<cudadev> <c1patchesDir> <trainPosDir>", 3, 3) == false)
    return 1;

  std::vector<int> scss(9);
  scss[0] = 1; scss[1] = 3; scss[2] = 5; scss[3] = 7; scss[4] = 9;
  scss[5] = 11; scss[6] = 13; scss[7] = 15; scss[8] = 17;
  std::vector<int> spss(8);
  spss[0] = 8; spss[1] = 10; spss[2] = 12; spss[3] = 14;
  spss[4] = 16; spss[5] = 18; spss[6] = 20; spss[7] = 22;
  // std::vector<int> scss(4);
  // scss[0] = 3; scss[1] = 7; scss[2] = 11; scss[3] = 15;
  // std::vector<int> spss(4);
  // spss[0] = 10; spss[1] = 14; spss[2] = 18; spss[3] = 22;

  int dev = strtol(mgr->getExtraArg(0).c_str(),NULL,0);
  MemoryPolicy mp = GLOBAL_DEVICE_MEMORY;
  std::string c1PatchesBaseDir;
  std::string trainPosName; // Directory where positive images are
  c1PatchesBaseDir = mgr->getExtraArg(1);
  trainPosName = mgr->getExtraArg(2);

  CudaHmaxFL hmax(mp,dev,NORI, spss, scss);

  // Extract random patches from a set of images in a positive training directory
  std::vector<std::string> trainPos = hmax.readDir(trainPosName);
  int posTrainSize = trainPos.size();

  //Image<byte> inputb;

  Image<float> *trainPosImages = new Image<float>[posTrainSize];

  std::cout << "Scanned training and testing images" << std::endl;

  for(int imgInd = 0; imgInd < posTrainSize; imgInd++) {
    //inputb = Raster::ReadGray(trainPos[imgInd]);
    //trainPosImages[imgInd] = Image<float>(inputb);
    trainPosImages[imgInd] = Raster::ReadGrayNTSC(trainPos[imgInd]);
  }

  std::vector<int> pS(4);
  pS[0] = 4; pS[1] = 8, pS[2] = 12; pS[3] = 16;

  // Learn the appropriate simple S2 patches from the C1 results
  hmax.extractRandC1Patches(trainPosImages,posTrainSize,pS,NUM_PATCHES_PER_SIZE,NORI);
  std::cout << "Completed extraction of C1 Patches" << std::endl;


  delete [] trainPosImages;

  hmax.writeOutC1Patches(c1PatchesBaseDir);
  return 0;
}




// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
