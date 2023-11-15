/*!@file CUDA/test-cuda-lowpass.C test CUDA/GPU optimized lowpass filtering routines */


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


#include "Envision/env_c_math_ops.h" // reference implementation
#include "Envision/env_image_ops.h" // reference implementation
#include "Envision/env_pyr.h"
#include "CUDA/cuda-lowpass.h"
#include "CUDA/cutil.h"
#include "CUDA/env_cuda.h"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Raster/Raster.H"
#include "Util/log.H"
#include "Util/Timer.H"
#include <cuda_runtime_api.h>
#include <stdio.h>

#include "Envision/env_alloc.h"
#include "Envision/env_c_math_ops.h"
#include "Envision/env_image.h"
#include "Envision/env_image_ops.h"
#include "Envision/env_log.h"
#include "Envision/env_mt_visual_cortex.h"
#include "Envision/env_params.h"
#include "Envision/env_stdio_interface.h"

////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

//Round a / b to nearest lower integer value
inline int iDivDown(int a, int b) { return a / b; }

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b) { return (a % b != 0) ?  (a - a % b + b) : a; }

//Align a to nearest lower multiple of b
inline int iAlignDown(int a, int b) { return a - a % b; }

#define NREP 20

// ######################################################################
// Thunk to convert from env_size_t to size_t
static void* malloc_thunk(env_size_t n)
{
        return malloc(n);
}

// ######################################################################
void compareregions(Image<int> &c, Image<int> &g, const uint rowStart, const uint rowStop, const uint colStart, const uint colStop)
{
  uint w,h;
  w = c.getWidth();
  h = c.getHeight();
  if(w != (uint) g.getWidth() || h != (uint) g.getHeight())
  {
    LINFO("Images are not the same size");
    return;
  }
  if(rowStart > rowStop || colStart > colStop || rowStop > h || colStop > w)
  {
    LINFO("Invalid regions to compare");
    return;
  }
  for(uint i=colStart;i<colStop;i++)
  {
    printf("\nC[%d]: ",i);
    for(uint j=rowStart;j<rowStop;j++)
    {
      printf("%d ",c.getVal(i,j));
    }
    printf("\nG[%d]: ",i);
    for(uint j=rowStart;j<rowStop;j++)
    {
      printf("%d ",g.getVal(i,j));
    }
  }
  printf("\n");

}

// ######################################################################
void imgcompare(const int* cpu, const int* gpu, const uint w, const uint h)
{
  Image<int> c(cpu, w, h), g(gpu, w, h);
  Image<float> diff = g - c;
  float mi, ma, av; getMinMaxAvg(diff, mi, ma, av);
  LINFO("%s: %ux%u image, GPU - CPU: avg=%f, diff = [%f .. %f]",
        mi == ma && ma == 0.0F ? "PASS" : "FAIL", w, h, av, mi, ma);
  //compareregions(c,g,0,30,575,600);
}

// ######################################################################
void imgcompare(const float* cpu, const float* gpu, const uint w, const uint h)
{
  Image<float> c(cpu, w, h), g(gpu, w, h);
  Image<float> diff = g - c;
  float mi, ma, av; getMinMaxAvg(diff, mi, ma, av);
  LINFO("%s: %ux%u image, GPU - CPU: avg=%f, diff = [%f .. %f]",
        mi == ma && ma == 0.0F ? "PASS" : "FAIL", w, h, av, mi, ma);
  //compareregions(c,g,0,30,575,600);
}

void test_lowpass5(Image<int> &iimg, char *cpu_file, char *gpu_int_file);
void test_lowpass9(Image<int> &iimg, char *cpu_file, char *gpu_int_file);

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  if (argc != 4) LFATAL("USAGE: %s <input.pgm> <outCPU.pgm> <outGPU.pgm>", argv[0]);

  CUT_DEVICE_INIT(0);
  LINFO("Reading: %s", argv[1]);
  Image<byte> img = Raster::ReadGray(argv[1]);
  Image<int> iimg = img; // convert to ints
  test_lowpass5(iimg,argv[2],argv[3]);
  //test_lowpass9(iimg,argv[2],argv[3]);

}

void test_lowpass5(Image<int> &iimg, char *cpu_file, char *gpu_int_file)
{
  int *dsrc, *ddst, *ddst2;
  const uint w = iimg.getWidth(), h = iimg.getHeight();
  const uint siz = w * h * sizeof(int);
  LINFO("Processing %ux%u image on lowpass5...", w, h);
  CUDA_SAFE_CALL( cudaMalloc( (void **)(void *)&dsrc, siz) ); // note: (void*) to avoid compiler warn
  CUDA_SAFE_CALL( cudaMalloc( (void **)(void *)&ddst, siz/2) );
  CUDA_SAFE_CALL( cudaMalloc( (void **)(void *)&ddst2, siz/4) );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  CUDA_SAFE_CALL( cudaMemcpy(dsrc, iimg.getArrayPtr(), siz, cudaMemcpyHostToDevice) );
  Timer tim;

  LINFO("GPU go!"); tim.reset();
  for (uint ii = 0; ii < NREP; ++ii)
    {
      cuda_lowpass_5_x_dec_x_fewbits_optim(dsrc, w, h, ddst);
      CUT_CHECK_ERROR("convolutionRowGPU() execution failed\n");
      cuda_lowpass_5_y_dec_y_fewbits_optim(ddst, w/2, h, ddst2);
      CUT_CHECK_ERROR("convolutionColumnGPU() execution failed\n");
    }
  LINFO("GPU done! %fms", tim.getSecs() * 1000.0F);

  LINFO("Reading back GPU results... siz/4 %d",siz/4);
  Image<int> ires(iDivUp(w,2), iDivUp(h,2), ZEROS);

  CUDA_SAFE_CALL( cudaMemcpy(ires.getArrayPtr(), ddst2, siz/4, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  Image<byte> bres = ires; // will convert and clamp as necessary

  Raster::WriteGray(bres, gpu_int_file);

  // compare with CPU:
  Image<int> ires2(iDivUp(w,2), h, NO_INIT), ires3(iDivUp(w,2), iDivUp(h,2), NO_INIT);
  LINFO("CPU go!"); tim.reset();
  for (uint ii = 0; ii < NREP; ++ii)
    {
      env_c_lowpass_5_x_dec_x_fewbits_optim((const intg32*)iimg.getArrayPtr(), w, h,
                                            (intg32*)ires2.getArrayPtr(), w/2);
      env_c_lowpass_5_y_dec_y_fewbits_optim((const intg32*)ires2.getArrayPtr(), w/2, h,
                                            (intg32*)ires3.getArrayPtr(), h/2);
    }
  LINFO("CPU done! %fms", tim.getSecs() * 1000.0F);

  Raster::WriteGray(Image<byte>(ires3), cpu_file);

  imgcompare(ires3.getArrayPtr(), ires.getArrayPtr(), w/2, h/2);

  CUDA_SAFE_CALL( cudaFree(ddst2) );
  CUDA_SAFE_CALL( cudaFree(ddst) );
  CUDA_SAFE_CALL( cudaFree(dsrc) );

  // ######################################################################
  LINFO("Moving on to pyramid test...");

  struct env_params envp;
  env_params_set_defaults(&envp);

  envp.maxnorm_type = ENV_VCXNORM_MAXNORM;
  envp.scale_bits = 16;

  env_assert_set_handler(&env_stdio_assert_handler);
  env_allocation_init(&malloc_thunk, &free);

  env_params_validate(&envp);
  struct env_math imath;
  env_init_integer_math(&imath, &envp);

  struct env_image einput; const struct env_dims di = { iimg.getWidth(), iimg.getHeight() };
  env_img_init(&einput, di);

  const int d = 8, firstlevel = 0;
  memcpy((void*)env_img_pixelsw(&einput), iimg.getArrayPtr(), iimg.getSize() * sizeof(int));
  struct env_pyr gpyr; env_pyr_init(&gpyr, d);

  // if firstlevel is zero, copy source image into level 0 of the pyramid:
  if (firstlevel == 0) env_img_copy_src_dst(&einput, env_pyr_imgw(&gpyr, 0));

  // allocate device memory:
  const env_size_t depth = env_pyr_depth(&gpyr);
  const env_size_t wh = w * h;
  const env_size_t rsiz = siz / 2; // siz/3 would be enough except for a bunch of small 1x1 levels
  int *dres, *dtmp;
  CUDA_SAFE_CALL(cudaMalloc((void **)(void *)&dsrc, siz + siz/2 + rsiz)); // (void*) to avoid compiler warn
  dtmp = dsrc + wh; dres = dtmp + wh/2;

  // copy source image to device memory:
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(dsrc, env_img_pixels(&einput), siz, cudaMemcpyHostToDevice));

  // run the pyramid in DEVICE memory:
  env_size_t outw[depth], outh[depth];
  LINFO("GPU pyramid go!"); tim.reset();
  for (uint ii = 0; ii < NREP; ++ii)
    cudacore_pyr_build_lowpass_5(dsrc, dtmp, dres, depth, w, h, outw, outh);
  LINFO("GPU pyramid done! %fms", tim.getSecs() * 1000.0F);

  // collect the results, starting at firstlevel and ignoring previous
  // levels; level 0 (if desired) has been handled already on the CPU
  // (simple memcpy) and is not in dres, which starts at level 1:
  int *dresptr = dres;
  for (env_size_t lev = 1; lev < depth; ++lev) {
    // get a pointer to image at that level:
    struct env_image *res = env_pyr_imgw(&gpyr, lev);
    const env_size_t ww = outw[lev], hh = outh[lev];

    if (lev < firstlevel)
      env_img_make_empty(res); // kill that level
    else {
      const struct env_dims di = { ww, hh };
      env_img_resize_dims(res, di);
      CUDA_SAFE_CALL(cudaMemcpy((void *)env_img_pixels(res), dresptr,
                                ww * hh * sizeof(int), cudaMemcpyDeviceToHost));
    }

    // ready for next level:
    dresptr += ww * hh;
  }

  // free allocated memory:
  CUDA_SAFE_CALL( cudaFree(dsrc) );

  // compute on CPU:
  struct env_pyr cpyr; env_pyr_init(&cpyr, d);
  LINFO("CPU pyramid go!"); tim.reset();
  for (uint ii = 0; ii < NREP; ++ii)
    env_pyr_build_lowpass_5_cpu(&einput, firstlevel, &imath, &cpyr);
  LINFO("CPU pyramid done! %fms", tim.getSecs() * 1000.0F);

  // compare results:
  outw[0] = w; outh[0] = h;
  for (uint ii = firstlevel; ii < depth; ++ii)
    imgcompare((const int*)env_img_pixels(env_pyr_img(&cpyr, ii)),
               (const int*)env_img_pixels(env_pyr_img(&gpyr, ii)), outw[ii], outh[ii]);
}




void test_lowpass9(Image<int> &iimg, char *cpu_file, char *gpu_int_file)
{
  int *dsrc, *ddst, *ddst2;
  const uint w = iimg.getWidth(), h = iimg.getHeight();
  const uint siz = w * h * sizeof(int);

  LINFO("Processing %ux%u image on lowpass9...", w, h);
  CUDA_SAFE_CALL( cudaMalloc( (void **)(void *)&dsrc, siz) ); // note: (void*) to avoid compiler warn
  CUDA_SAFE_CALL( cudaMalloc( (void **)(void *)&ddst, siz) );
  CUDA_SAFE_CALL( cudaMalloc( (void **)(void *)&ddst2, siz) );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  CUDA_SAFE_CALL( cudaMemcpy(dsrc, iimg.getArrayPtr(), siz, cudaMemcpyHostToDevice) );
  Timer tim;

  LINFO("GPU int go!"); tim.reset();
  for (uint ii = 0; ii < NREP; ++ii)
    {
      cuda_lowpass_9_x_fewbits_optim(dsrc, w, h, ddst);
      CUT_CHECK_ERROR("convolutionRowGPU() execution failed\n");
      cuda_lowpass_9_y_fewbits_optim(ddst, w, h, ddst2);
      CUT_CHECK_ERROR("convolutionColumnGPU() execution failed\n");
    }
  LINFO("GPU int done! %fms", tim.getSecs() * 1000.0F);

  LINFO("Reading back GPU int results... siz %d",siz);
  Image<int> ires(w, h, ZEROS);

  CUDA_SAFE_CALL( cudaMemcpy(ires.getArrayPtr(), ddst2, siz, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaFree(ddst2) );
  CUDA_SAFE_CALL( cudaFree(ddst) );
  CUDA_SAFE_CALL( cudaFree(dsrc) );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );

  Image<byte> bres = ires; // will convert and clamp as necessary
  Raster::WriteGray(bres, gpu_int_file);

  // compare with CPU:
  Image<int> ires2(w, h, NO_INIT), ires3(w, h, NO_INIT);
  LINFO("CPU go!"); tim.reset();
  for (uint ii = 0; ii < NREP; ++ii)
    {
      env_c_lowpass_9_x_fewbits_optim((const intg32*)iimg.getArrayPtr(), w, h,
                                            (intg32*)ires2.getArrayPtr());
      env_c_lowpass_9_y_fewbits_optim((const intg32*)ires2.getArrayPtr(), w, h,
                                            (intg32*)ires3.getArrayPtr());
    }
  LINFO("CPU done! %fms", tim.getSecs() * 1000.0F);

  Raster::WriteGray(Image<byte>(ires3), cpu_file);

  imgcompare((Image<float>(ires3)).getArrayPtr(), (Image<float>(ires)).getArrayPtr(), w, h);

//   // ######################################################################
//   LINFO("Moving on to pyramid test...");

//   struct env_params envp;
//   env_params_set_defaults(&envp);

//   envp.maxnorm_type = ENV_VCXNORM_MAXNORM;
//   envp.scale_bits = 16;

//   env_assert_set_handler(&env_stdio_assert_handler);
//   env_allocation_init(&malloc_thunk, &free);

//   env_params_validate(&envp);
//   struct env_math imath;
//   env_init_integer_math(&imath, &envp);

//   struct env_image einput; const struct env_dims di = { iimg.getWidth(), iimg.getHeight() };
//   env_img_init(&einput, di);

//   const int d = 8, firstlevel = 0;
//   memcpy((void*)env_img_pixelsw(&einput), iimg.getArrayPtr(), iimg.getSize() * sizeof(int));
//   struct env_pyr gpyr; env_pyr_init(&gpyr, d);

//   // if firstlevel is zero, copy source image into level 0 of the pyramid:
//   if (firstlevel == 0) env_img_copy_src_dst(&einput, env_pyr_imgw(&gpyr, 0));

//   // allocate device memory:
//   const env_size_t depth = env_pyr_depth(&gpyr);
//   const env_size_t wh = w * h;
//   const env_size_t rsiz = siz / 2; // siz/3 would be enough except for a bunch of small 1x1 levels
//   int *dres, *dtmp;
//   CUDA_SAFE_CALL(cudaMalloc((void **)(void *)&dsrc, siz + siz/2 + rsiz)); // (void*) to avoid compiler warn
//   dtmp = dsrc + wh; dres = dtmp + wh/2;

//   // copy source image to device memory:
//   CUDA_SAFE_CALL(cudaThreadSynchronize());
//   CUDA_SAFE_CALL(cudaMemcpy(dsrc, env_img_pixels(&einput), siz, cudaMemcpyHostToDevice));

//   // run the pyramid in DEVICE memory:
//   env_size_t outw[depth], outh[depth];
//   LINFO("GPU pyramid go!"); tim.reset();
//   for (uint ii = 0; ii < NREP; ++ii)
//     cudacore_pyr_build_lowpass_5(dsrc, dtmp, dres, depth, w, h, outw, outh);
//   LINFO("GPU pyramid done! %fms", tim.getSecs() * 1000.0F);

//   // collect the results, starting at firstlevel and ignoring previous
//   // levels; level 0 (if desired) has been handled already on the CPU
//   // (simple memcpy) and is not in dres, which starts at level 1:
//   int *dresptr = dres;
//   for (env_size_t lev = 1; lev < depth; ++lev) {
//     // get a pointer to image at that level:
//     struct env_image *res = env_pyr_imgw(&gpyr, lev);
//     const env_size_t ww = outw[lev], hh = outh[lev];

//     if (lev < firstlevel)
//       env_img_make_empty(res); // kill that level
//     else {
//       const struct env_dims di = { ww, hh };
//       env_img_resize_dims(res, di);
//       CUDA_SAFE_CALL(cudaMemcpy((void *)env_img_pixels(res), dresptr,
//                                 ww * hh * sizeof(int), cudaMemcpyDeviceToHost));
//     }

//     // ready for next level:
//     dresptr += ww * hh;
//   }

//   // free allocated memory:
//   CUDA_SAFE_CALL( cudaFree(dsrc) );

//   // compute on CPU:
//   struct env_pyr cpyr; env_pyr_init(&cpyr, d);
//   LINFO("CPU pyramid go!"); tim.reset();
//   for (uint ii = 0; ii < NREP; ++ii)
//     env_pyr_build_lowpass_5_cpu(&einput, firstlevel, &imath, &cpyr);
//   LINFO("CPU pyramid done! %fms", tim.getSecs() * 1000.0F);

//   // compare results:
//   outw[0] = w; outh[0] = h;
//   for (uint ii = firstlevel; ii < depth; ++ii)
//     imgcompare((const int*)env_img_pixels(env_pyr_img(&cpyr, ii)),
//                (const int*)env_img_pixels(env_pyr_img(&gpyr, ii)), outw[ii], outh[ii]);
}








// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */
