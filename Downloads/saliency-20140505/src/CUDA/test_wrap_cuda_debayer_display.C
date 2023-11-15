/*!@file CUDA/test_wrap_cuda_debayer_display.C Testing CudaImageDisplayGL.C */

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



#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/ColorOps.H"
#include "Raster/Raster.H"
#include "CUDA/CudaImage.H"
#include "CUDA/wrap_c_cuda.h"
#include "CUDA/cutil.h"
#include "CUDA/CudaColorOps.H"
#include "CUDA/CudaDevices.H"
#include "CUDA/CudaImageSet.H"
#include "CUDA/CudaImageSetOps.H"
#include "CUDA/CUDAdebayer.H"
#include "Util/fpu.H"
#include <cmath>
#include <fstream>
#include <sys/time.h>
#include "CUDA/CudaImageDisplayGL.H"
// ######################################################################


void bayer_test(int argc,char **argv)
{
  //Variable allocations
  Image<float> i = Raster::ReadFloat(argv[1]);
  int deviceNum = 0 ;
  CudaDevices::displayProperties(deviceNum);
  CudaImage<PixRGB<float> > res_cuda;
  CudaImage<float> f;
  CudaImage<float> res_cuda_r_only;
  CudaImage<float> res_cuda_g_only;
  CudaImage<float> res_cuda_b_only;
  Image<PixRGB<float> > res_cuda_r;
  Image<PixRGB<float> > res_cuda_g;
  Image<PixRGB<float> > res_cuda_b;
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  //Getting the Debayerd Image
  f =        CudaImage<float>(i,GLOBAL_DEVICE_MEMORY,deviceNum);
  res_cuda = cuda_1_debayer(f);
  // cuda_init_layout(nDesired,layoutW,layoutH);
  // for(int i=0;i<nDesired;i++)
  //   {
  //     cuda_generate_layout(f[i],i,startX,startY,sizeX,sizeY);
  //   }


  //iDispGL(res_cuda,argc,argv,0,0,1920,1080,1.0);

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime,start,stop);
  printf("Elasped Time GPU in ms for 1 iteration : %f\n",elapsedTime);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //Printing the images

  //Get color components
  cudaGetComponents(res_cuda,res_cuda_r_only,res_cuda_g_only,res_cuda_b_only);

  //Make R,0,0 from r only
  Image<float> zero(res_cuda.getDims(),ZEROS);

  res_cuda_r = makeRGB(res_cuda_r_only.exportToImage(),zero,zero);
  res_cuda_g = makeRGB(zero,res_cuda_g_only.exportToImage(),zero);
  res_cuda_b = makeRGB(zero,zero,res_cuda_b_only.exportToImage());

  Raster::WriteRGB(res_cuda_r,"test_gpu_r.ppm");
  Raster::WriteRGB(res_cuda_g,"test_gpu_g.ppm");
  Raster::WriteRGB(res_cuda_b,"test_gpu_b.ppm");
  //##############################################################
  Image<PixRGB<float> > res_copy_cuda = res_cuda.exportToImage();
  Raster::WriteRGB(res_copy_cuda,"test_gpu.ppm");
  //imageDisplayGL(res_cuda.getAryPtr(),argc,argv);
}



int main(int argc, char **argv)
{
  bayer_test(argc,argv);
}
