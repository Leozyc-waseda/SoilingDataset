/*!@file CUDA/CudaHmax.C Riesenhuber & Poggio's HMAX model for object recognition */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaHmax.C $
// $Id: CudaHmax.C 15310 2012-06-01 02:29:24Z itti $
//

#include "CUDA/CudaHmax.H"

#include "CUDA/CudaFilterOps.H"
#include "CUDA/CudaImage.H"
//#include "CUDA/CudaKernels.H"   // for cudaDogFilter()
#include "Image/Image.H"
#include "Image/Kernels.H"   // for dogFilter()
#include "CUDA/CudaMathOps.H"
#include "CUDA/CudaConvolutions.H" // for cudaConvolve() etc.
#include "Image/MathOps.H"

#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Util/safecopy.H"

#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <limits>

// ######################################################################
CudaHmax::CudaHmax()
{ initialized = false; }

// ######################################################################
CudaHmax::CudaHmax(MemoryPolicy mp, int dev, const int nori, const std::vector<int>& spacess,
           const std::vector<int>& scaless, const int c1spaceol,
           const bool angleflag, const float s2t, const float s2s,
           const float stdmin, const float stdstep,
           const int fsmin, const int fsstep)
{
  initialized = false;
  init(mp,dev,nori, spacess, scaless, c1spaceol, angleflag, s2t, s2s);
  initFilters(stdmin,stdstep,fsmin,fsstep);
}

// ######################################################################
void CudaHmax::init(MemoryPolicy mp, int dev, const int nori, const std::vector<int>& spacess,
                const std::vector<int>& scaless, const int c1spaceol,
                const bool angleflag, const float s2t, const float s2s)
{
  itsMp = mp;
  itsDev = dev;
  freeMem(); initialized = true; ns = scaless[scaless.size() - 1]; no = nori;
  c1SpaceOL = c1spaceol; angleFlag = angleflag; s2Target = s2t; s2Sigma = s2s;
  spaceSS.resize(spacess.size()); scaleSS.resize(scaless.size());

 // detrmine number of scale bands from length of vector scaleSS:
  nsb = scaleSS.size() - 1;

  for (unsigned int i = 0; i < spacess.size(); i ++) spaceSS[i] = spacess[i];
  for (unsigned int i = 0; i < scaless.size(); i ++) scaleSS[i] = scaless[i];

}

void CudaHmax::initFilters(const float stdmin, const float stdstep, const int fsmin, const int fsstep)
{
  // create the filters:
  typedef CudaImage<float>* FloatImagePtr;
  filter = new FloatImagePtr[ns];
  for(int s = 0; s < ns; s++)
    {
      filter[s] = new CudaImage<float>[no];
      for(int o = 0; o < no; o ++)
        {
          // create DoG filter:
          Image<float> tmp = dogFilter<float>(stdmin + stdstep * s,
                                          (float)o * 180.0F / (float)no,
                                          fsmin + fsstep * s);
          filter[s][o] = CudaImage<float>(tmp,itsMp,itsDev);
          // filter[s][o] = cudaDogFilter(itsMp,itsDev,stdmin + stdstep * s,
          //                                 (float)o * 180.0F / (float)no,
          //                                 fsmin + fsstep * s);
          // normalize to zero mean:
          filter[s][o] -= cudaGetAvg(filter[s][o]);

          // normalize to unit of sum-of-squares:
          filter[s][o] /= cudaGetSum(cudaSquared(filter[s][o]));
        }
    }
}

// ######################################################################
void CudaHmax::freeMem()
{
  if (initialized)
    {
      for(int s = 0; s < ns; s++) delete [] filter[s];
      delete [] filter;
      initialized = false;
    }
}

// ######################################################################
CudaHmax::~CudaHmax()
{ freeMem(); }

// ######################################################################
std::vector<std::string> CudaHmax::readDir(std::string inName)
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
std::vector<std::string> CudaHmax::readList(std::string inName)
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




// ######################################################################
void CudaHmax::getC1(const CudaImage<float>& input, CudaImage<float>** &c1Res)
{
  CudaImage<float> *c1Buff = new CudaImage<float>[no];
  CudaImage<float> s1Res;
  // loop over scale bands:


  for(int sb = 0; sb < nsb; sb ++) {

    // clear our buffers:
    for(int i = 0; i < no; i ++)
      c1Buff[i] = CudaImage<float>(input.getWidth(), input.getHeight(), ZEROS, itsMp, itsDev);;

    // loop over scales within current scale band:
    for(int s = scaleSS[sb]; s < scaleSS[sb + 1]; s++) {
      // loop over orientations:
      for(int o = 0; o < no; o ++) {
        // convolve image by filter at current orient & scale:
        if (angleFlag) {
          s1Res = cudaConvolveHmax(input, filter[s][o]); // normalize by image energy
          //printCorners("s1Res",s1Res,s==scaleSS[0]&&sb==0&&o==0);
        }
        else {
          s1Res = cudaConvolve(input, filter[s][o], CONV_BOUNDARY_CLEAN); // no normalization
          // take absolute value of the convolution result:
          cudaAbs(s1Res);
        }

        // take max between this convolution and previous ones for
        // that orientation but other scales within current scale band:
        c1Buff[o] = cudaTakeMax(c1Buff[o], s1Res);
      }
    }

    // compute RF spacing (c1R) and pool range (c1PR):
    int c1R = (int)ceil((float)spaceSS[sb] / (float)c1SpaceOL);
    int c1PR = spaceSS[sb];

    // pool over space for each orientation (and scale band):
    for(int o = 0; o < no; o ++){
      c1Res[sb][o] = cudaSpatialPoolMax(c1Buff[o], c1R, c1R, c1PR, c1PR);
      //printCorners("c1Res",c1Res[sb][o],sb==0&&o==0);
    }

  }

  delete [] c1Buff;
}

void CudaHmax::initC1(CudaImage<float> **&c1Res)
{
  c1Res = new CudaImage<float>*[nsb];
  for (int sb = 0; sb < nsb; sb ++) c1Res[sb] = new CudaImage<float>[no];
}

void CudaHmax::clearC1(CudaImage<float> **&c1Res)
{
   for (int sb = 0; sb < nsb; sb ++) delete [] c1Res[sb];
   delete [] c1Res;
}

void CudaHmax::printCorners(const char name[], const CudaImage<float>& cim, bool cond)
{
  Image<float> im = cim.exportToImage();
  if(cond) {
    printf("%s\n",name);
    int w = im.getWidth();
    int h = im.getHeight();
    std::string s;
    if(w>2 && h>2) {
      printf("%f\t%f\t%f\t%f\t%f\n",im.getVal(0,0),im.getVal(1,0),im.getVal(2,0),im.getVal(w-2,0),im.getVal(w-1,0));
      printf("%f\t%f\t\t\t%f\t%f\n\n", im.getVal(0,1),im.getVal(1,1),im.getVal(w-2,1),im.getVal(w-1,1));
      printf("%f\t%f\t\t\t%f\t%f\n", im.getVal(0,h-2),im.getVal(1,h-2),im.getVal(w-2,h-2),im.getVal(w-1,h-2));
      printf("%f\t%f\t%f\t%f\t%f\n",im.getVal(0,h-1),im.getVal(1,h-1),im.getVal(2,h-1),im.getVal(w-2,h-1),im.getVal(w-1,h-1));
    }
    else if(w>1 && h>1) {
      printf("%f\t%f\n",im.getVal(0,0),im.getVal(1,0));
      printf("%f\t%f\n", im.getVal(0,1),im.getVal(1,1));
    }
    else if(w>0 && h>0){
      printf("%f\n",im.getVal(0,0));
    }
    std::cout << "Mean of " << name << " " << mean(im) << std::endl;
    std::cout << "Var of " << name << " " << (stdev(im)*stdev(im)) << std::endl;
    std::cout << "Width of " << w << " and height of " << h << std::endl;
    //float mi,ma; getMinMax(input,mi,ma);
    //writeOutImage(inputf,name);
    //std::cout << "Min " << mi << " Max " << ma << std::endl;
  }
}

void CudaHmax::writeOutImage(const CudaImage<float>& cim,std::string & fName)
{
  std::ofstream oFile;
  Image<float> d;
  oFile.open(fName.c_str(),std::ios::out);
  d = cim.exportToImage();
  int w,h;
  w = d.getWidth();
  h = d.getHeight();
  for(int i=0;i<w;i++){
    for(int j=0;j<h;j++){
      oFile << d.getVal(i,j) <<" ";
    }
    if(i!=w-1)
      oFile << std::endl;
  }
  oFile.close();

}


// ######################################################################
CudaImage<float> CudaHmax::getC2(const CudaImage<float>& input)
{

  // allocate buffers for intermediary results:
  CudaImage<float> **c1Res;
  initC1(c1Res);

  // ******************************
  // *** Compute S1/C1 output:
  // ******************************
  getC1(input, c1Res);

  // ******************************
  // *** Compute S2/C2 output:
  // ******************************
  CudaImage<float> c2Res(no * no, no * no, NO_INIT,itsMp,itsDev);
  cudaClear(c2Res,-1.0E10F);

  // // detrmine number of scale bands from length of vector scaleSS:
  // int nsb = scaleSS.size() - 1;
  // int idx = 0;
  // // loop over four filters giving inputs to an S2 cell:
  // for (int f1 = 0; f1 < no; f1++)
  //   for (int f2 = 0; f2 < no; f2++)
  //     for (int f3 = 0; f3 < no; f3++)
  //       for (int f4 = 0; f4 < no; f4++) {

  //         float c2r = -1.0E10;
  //         // loop over scale band:
  //         for (int sb = 0; sb < nsb; sb ++) {

  //           float s2r = featurePoolHmax(c1Res[sb][f1], c1Res[sb][f2],
  //                                       c1Res[sb][f3], c1Res[sb][f4],
  //                                       c1SpaceOL, c1SpaceOL, s2Target);
  //           if (s2r > c2r) c2r = s2r;
  //         }
  //         c2Res.setVal(idx, c2r); idx ++;
  //       }

  // // free allocated temporary images:
  // clearC1(c1Res);

  // return hmaxActivation(c2Res, s2Sigma);

  LFATAL("Generic HMAX C2 Calculation not supported");
  return c2Res;
}

void CudaHmax::sumFilter(const CudaImage<float>& image, const float radius, CudaImage<float>& newImage)
{
  Rectangle sumSupport = Rectangle::tlbrI(0,0,int(radius*2.0F),int(radius*2.0F));
  sumFilter(image,sumSupport,newImage);
}

void CudaHmax::sumFilter(const CudaImage<float>& image, const Rectangle& support, CudaImage<float>& newImage)
{

  Dims d(support.top()+support.bottomI()+1,support.left()+support.rightI()+1);
  //Dims d(support.bottomI()-support.top()+1,support.rightI()-support.left()+1);
  CudaImage<float> a(d,NO_INIT,itsMp,itsDev);
  cudaClear(a,1.0F);

  //convolve the image with a matrix of 1's that is as big as the rectangle
  // This two step process is doing effectively the same thing by taking the center part of the convolution
  //I2 = conv2(ones(1,radius(2)+radius(4)+1), ones(radius(1)+radius(3)+1,1), I);
  //I3 = I2((radius(4)+1:radius(4)+size(I,1)), (radius(3)+1:radius(3)+size(I,2)));
  //CudaImage<float> i;
  //i = convolution(image,a,MATLAB_STYLE_CONVOLUTION);
  //Rectangle r = Rectangle::tlbrI(support.bottomI()+1,support.rightI()+1,support.bottomI()+image.getHeight(),support.rightI()+image.getWidth());
  //newImage = crop(i,r);
  // Can be done in one step
  newImage = cudaConvolve(image,a,CONV_BOUNDARY_ZERO);
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
