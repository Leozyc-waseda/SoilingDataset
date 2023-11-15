/*!@file CUDA/CudaHmaxCBCL.C Wrapper class for CBCL CUDA model */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaHmaxCBCL.C $
// $Id: CudaHmaxCBCL.C 15310 2012-06-01 02:29:24Z itti $
//
#include "CUDA/CudaHmaxCBCL.H"
#include "Image/Image.H"
#include "Raster/Raster.H"
#include "Util/log.H"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <unistd.h>
#include <cstdlib>
#include <vector>
#include <dirent.h>



CudaHmaxCBCL::CudaHmaxCBCL()
{
  nc0patches = 0;
  nc1patches = 0;
}

CudaHmaxCBCL::CudaHmaxCBCL(std::string c0patchFile, std::string c1patchFile)
{
  loadC0(c0patchFile);
  loadC1(c1patchFile);
}

CudaHmaxCBCL::~CudaHmaxCBCL()
{
  if(nc0patches > 0)
    cpu_release_images(&c0patches,nc0patches);
  if(nc1patches > 0)
    cpu_release_images(&c1patches,nc1patches);
}

void CudaHmaxCBCL::loadC0(std::string c0patchFile)
{
  readFilters(c0patchFile,&c0patches,&nc0patches);
}

void CudaHmaxCBCL::loadC1(std::string c1patchFile)
{
  readFilters(c1patchFile,&c1patches,&nc1patches);
}

float *CudaHmaxCBCL::getC2Features()
{
  return c2b;
}

int CudaHmaxCBCL::numC2Features()
{
  return nc2units;
}

void CudaHmaxCBCL::writeOutFeatures(std::string c2FileName, int id, bool clearFile)
{
  std::ofstream c2File;
  if(clearFile)
    c2File.open(c2FileName.c_str(),std::ios::out);
  else
    c2File.open(c2FileName.c_str(),std::ios::app);
  if (c2File.is_open()) {
    c2File << id << " ";
    for(int i=0;i<nc2units;i++) {
      c2File << std::setiosflags(std::ios::fixed) << std::setprecision(10) <<
        (i+1) << ":" << c2b[i] << " ";
    }
    c2File << std::endl;
  }
  c2File.close();
}

void CudaHmaxCBCL::extractC1Patch(Image<float> &stim, int index, int patchSize)
{
  getC1(stim.getArrayPtr(),stim.getWidth(),stim.getHeight(),true);
  // Determine the width, height and depth of your c1 bandinfo
  int targC1Band = nc1bands/2;
  int w = cpu_get_width(&c1,nc1bands,targC1Band);
  int h = cpu_get_height(&c1,nc1bands,targC1Band);
  int wStart = floor((rand()-1.0F)/RAND_MAX*(w-patchSize));
  int hStart = floor((rand()-1.0F)/RAND_MAX*(h-patchSize));
  assert(wStart>=0);
  assert(hStart>=0);
  // Copy c1 output
  int xPatch=0;
  for(int x=wStart;x<wStart+patchSize;x++)
    {
      int yPatch=0;
      for(int y=hStart;y<hStart+patchSize;y++)
	{
	  cpu_copy_filter(&c1patches,nc1patches,index,xPatch,yPatch,&c1,targC1Band,x,y);
	  yPatch++;
	}
      xPatch++;
    }
  clearC1();
  printf("Loaded patch %d\n",index);

}

void CudaHmaxCBCL::initC1Patches(int numPatches, int patchSize, int patchDepth)
{
 cpu_create_filters(&c1patches,numPatches,patchSize,patchSize,patchDepth);
 nc1patches = numPatches;
}

void CudaHmaxCBCL::extractC1Patches(std::vector<std::string> files, int numPatches, int patchSize, int patchDepth)
{
  int numPosTrainImages = files.size();
  Image<float> stim;

  std::srand(time(0));

  initC1Patches(numPatches,patchSize,patchDepth);

  for(int i=0;i<numPatches;i++){
    // Randomly select an image from the list
    unsigned int imInd = static_cast<unsigned int>(floor((rand()-1.0F)/RAND_MAX*numPosTrainImages));
    stim = Raster::ReadGrayNTSC(files[imInd].c_str());
    extractC1Patch(stim,i,patchSize);
  }

}

void CudaHmaxCBCL::writeOutC1Patches(std::string filename)
{
  writeFilters(c1patches,nc1patches,filename.c_str());
}

void CudaHmaxCBCL::readFilters(std::string filename,band_info** ppfilt,int* pnfilts)
{
  cpu_read_filters(filename.c_str(),ppfilt,pnfilts);
}

void CudaHmaxCBCL::writeFilters(band_info* pfilt,int nfilts, std::string filename)
{
  cpu_write_filters(pfilt,nfilts,filename.c_str());
}


void CudaHmaxCBCL::getC1(const float *pimg, int width, int height, bool copy)
{
  cpu_create_c0(pimg,width,height,&c0,&nc0bands,1.113,12);
  gpu_s_norm_filter(c0,nc0bands,c0patches,nc0patches,&s1,&ns1bands,false);
  gpu_c_local(s1,ns1bands,8,3,2,2,&c1,&nc1bands,copy);
}

void CudaHmaxCBCL::clearC1()
{
  cpu_release_images(&c0,nc0bands);
  cpu_release_images(&s1,ns1bands);
  cpu_release_images(&c1,nc1bands);
}

void CudaHmaxCBCL::getC2(const float *pimg, int width, int height)
{
  getC1(pimg,width,height,false);
  gpu_s_rbf(c1,nc1bands,c1patches,nc1patches,sqrtf(0.5),&s2,&ns2bands);
  cpu_c_global(s2,ns2bands,&c2b,&nc2units);
  clearC1();
  cpu_release_images(&s2,ns2bands);
  // printf("S1bands %d\nC1bands %d\nC1patches %d\nS2bands %d\nC2units %d\n",
  //          ns1bands,nc1bands,nc1patches,ns2bands,nc2units);
}

void CudaHmaxCBCL::clearC2()
{
  delete [] c2b;
}


std::vector<std::string> CudaHmaxCBCL::readDir(std::string inName)
{
        DIR *dp = opendir(inName.c_str());
        if(dp == NULL)
        {
          LFATAL("Directory does not exist %s",inName.c_str());
        }
        dirent *dirp;
        std::vector<std::string> fList;
        while ((dirp = readdir(dp)) != NULL ) {
                if (dirp->d_name[0] != '.')
                        fList.push_back(inName + '/' + std::string(dirp->d_name));
        }
        LINFO("%" ZU " files in the directory\n", fList.size());
        LINFO("file list : \n");
        for (unsigned int i=0; i<fList.size(); i++)
                LINFO("\t%s", fList[i].c_str());
        std::sort(fList.begin(),fList.end());
        return fList;
}

// ######################################################################
std::vector<std::string> CudaHmaxCBCL::readList(std::string inName)
{
  std::ifstream inFile;
  inFile.open(inName.c_str(),std::ios::in);
  if(!inFile){
    LFATAL("Unable to open image path list file: %s",inName.c_str());
  }
  std::string sLine;
  std::vector<std::string> fList;
  while (std::getline(inFile, sLine)) {
      std::cout << sLine << std::endl;
      fList.push_back(sLine);
  }
  LINFO("%" ZU " paths in the file\n", fList.size());
  LINFO("file list : \n");
  for (unsigned int i=0; i<fList.size(); i++)
    LINFO("\t%s", fList[i].c_str());
  inFile.close();
  return fList;
}


