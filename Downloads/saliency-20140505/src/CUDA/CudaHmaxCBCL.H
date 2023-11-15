/*!@file CUDA/CudaHmaxCBCL.H Wrapper class for CBCL CUDA model */

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
// Primary maintainer for this file: Daniel Parks <danielfp@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaHmaxCBCL.H $
// $Id: CudaHmaxCBCL.H 14154 2010-10-21 05:07:25Z dparks $
//

#ifndef CUDAHMAXCBCL_H_DEFINED
#define CUDAHMAXCBCL_H_DEFINED

#include "cbcl_model.h"
#include "Image/Image.H"

class CudaHmaxCBCL
{
public:
  CudaHmaxCBCL();
  CudaHmaxCBCL(std::string c0patchFile, std::string c1patchFile);
  ~CudaHmaxCBCL();
  void loadC0(std::string c0patchFile);
  void loadC1(std::string c1patchFile);
  std::vector<std::string> readList(std::string inName);
  std::vector<std::string> readDir(std::string inName);
  void extractC1Patch(Image<float> &stim, int index, int patchSize);
  void initC1Patches(int numPatches, int patchSize, int patchDepth);
  void extractC1Patches(std::vector<std::string> files, int numPatches, int patchSize, int patchDepth);
  void writeOutFeatures(std::string c2FileName, int id, bool clearFile=false);
  void writeOutC1Patches(std::string filename);
  void readFilters(std::string filename,band_info** ppfilt,int* pnfilts);
  void writeFilters(band_info* pfilt,int nfilts, std::string filename);
  void extractPatches(std::vector<std::string> files, int numPatches);
  float *getC2Features();
  int numC2Features();
  void getC1(const float *pimg, int width, int height, bool copy=true);
  void getC2(const float *pimg, int width, int height);
  void clearC1();
  void clearC2();
private:
    band_info *c0;
    band_info *s1;
    band_info* c1;
    band_info* s2;
    float*     c2b;

    band_info *c0patches;
    band_info *c1patches;
    int     nc0patches;
    int     nc1patches;
    int     nc0bands;
    int     ns1bands;
    int     nc1bands;
    int     ns2bands;
    int     nc2units;
};

#endif
