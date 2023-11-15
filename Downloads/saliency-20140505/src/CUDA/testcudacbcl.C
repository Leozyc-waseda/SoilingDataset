/*!@file HMAX/testcudacbcl.C Test Hmax CBCL model */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/testcudacbcl.C $
// $Id: testcudacbcl.C 13354 2010-05-04 00:59:00Z dparks $
//

#include "Component/ModelManager.H"
#include "GUI/XWindow.H"
#include "Learn/SVMClassifier.H"
#include "CUDA/CudaHmaxCBCL.H"
#include "Image/Image.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/Rectangle.H"
#include "Image/MathOps.H"
#include "Image/MatrixOps.H"
#include "Image/Transforms.H"
#include "Image/Convolutions.H"
#include "Learn/svm.h"
#include "Media/FrameSeries.H"
#include "nub/ref.h"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Util/Timer.H"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <unistd.h>
#include <cstdlib>


// number of orientations to use in HmaxFL
#define NORI 4
#define NUM_PATCHES_PER_SIZE 250

int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;

  ModelManager *mgr = new ModelManager("Test Hmax with Feature Learning");

  mgr->exportOptions(MC_RECURSE);

  // required arguments
  // <dev> <c0patches> <c1patches> <dir|list> <svmModel> <svmRange> <outputfile>
  //

  if (mgr->parseCommandLine(
                            (const int)argc, (const char**)argv, "<cudadev> <c0patches> <c1patches> <dir|list:images> <svmModel> <svmRange> <noprob|withprob:outputfile>", 7, 7) == false)
    return 1;

  std::string c0Patches;
  std::string c1Patches;
  std::string images,svmModel,svmRange,devArg;
  std::string answerFileName;

  devArg = mgr->getExtraArg(0);
  c0Patches = mgr->getExtraArg(1);
  c1Patches = mgr->getExtraArg(2);
  images = mgr->getExtraArg(3);
  svmModel = mgr->getExtraArg(4);
  svmRange = mgr->getExtraArg(5);
  answerFileName = mgr->getExtraArg(6);

  int dev = strtol(devArg.c_str(),NULL,0);
  CudaDevices::setCurrentDevice(dev);
  std::string::size_type dirArg=images.find("dir:",0);
  std::string::size_type listArg=images.find("list:",0);
  if((dirArg == std::string::npos &&
      listArg == std::string::npos) ||
     (dirArg != 0 && listArg != 0)){
    LFATAL("images argument is in one of the following formats -  dir:<DIRNAME>  or  list:<LISTOFIMAGEPATHSFILE>");
    return EXIT_FAILURE;
  }
  if(dirArg == 0)
    images = images.substr(4);
  else
    images = images.substr(5);

  std::string::size_type noProbArg=answerFileName.find("noprob:",0);
  std::string::size_type withProbArg=answerFileName.find("withprob:",0);

  if((noProbArg == std::string::npos &&
      withProbArg == std::string::npos) ||
     (noProbArg != 0 && withProbArg != 0)){
    LFATAL("output file argument is in one of the following formats -  noprob:<FILENAME>  or  withprob:<FILENAME>");
    return EXIT_FAILURE;
  }
  if(noProbArg == 0)
    answerFileName = answerFileName.substr(7);
  else
    answerFileName = answerFileName.substr(9);



  // Now we run if needed
  mgr->start();

  CudaHmaxCBCL hmax(c0Patches,c1Patches);

  // Load the SVM Classifier Model and Range in
  SVMClassifier svm;
  svm.readModel(svmModel);
  svm.readRange(svmRange);

  std::vector<std::string> imageNames;
  if(dirArg == 0)
    imageNames = hmax.readDir(images);
  else
    imageNames = hmax.readList(images);

  std::ofstream answerFile;
  answerFile.open(answerFileName.c_str(),std::ios::out);

  for(unsigned int imgInd=0;imgInd<imageNames.size();imgInd++){

    Image<float> inputf = Raster::ReadGrayNTSC(imageNames[imgInd]);
    Timer tim;
    tim.reset();
    hmax.getC2(inputf.getArrayPtr(),inputf.getWidth(),inputf.getHeight());
    printf("CUDA CBCL HMAX %f secs\n",tim.getSecs());
    int numC2 = hmax.numC2Features();
    float *c2 = hmax.getC2Features();
    double prob;
    double pred = svm.predict(c2,numC2,&prob);
    printf("Prediction is %f, prob is %f\n",pred,prob);
    int predId = (int) pred;
    answerFile << predId;
    // Add probabilities if desired
    if(withProbArg == 0)
      answerFile << "\t" << prob;
    answerFile << std::endl;
    hmax.clearC2();
  }

  answerFile.close();
  return 0;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
