
/*!@file CUDA/wrap_c_cuda.h CUDA/GPU optimized saliency code */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/wrap_c_cuda.h $
// $Id: wrap_c_cuda.h 12962 2010-03-06 02:13:53Z irock $
//

#ifndef CUDA_WRAP_C_CUDA_H_DEFINED
#define CUDA_WRAP_C_CUDA_H_DEFINED
#ifdef __cplusplus
extern "C"
{
#endif

#include "cudadefs.h"

/*! Note that src and dst should have been allocated already by the
    caller in DEVICE memory, and source data should have been copied
    to src. The caller may have to copy the result back to host memory
    if no further GPU processing is needed. */

// Declare all host callable functions here

//! Get double color opponency maps
void cuda_c_getRGBY(const float3_t *src, float *rgptr, float *byptr, const float thresh,
                    const float min_range, const float max_range,
                    const int w, const int h, const int tile_width, const int tile_height);

//! Convert luminance to an RGB formatted image (which will still be grayscale)
void cuda_c_toRGB(float3_t *dst, const float *src,int sz, const int tile_len);

//! Get color components of an RGB image
void cuda_c_getComponents(const float3_t *srcptr, float *rptr, float *gptr, float *bptr, int w, int h, int tile_width, int tile_height);

//! Get luminance using (r+g+b)/3 calculation
void cuda_c_luminance(float3_t *aptr, float *dptr, int w, int h, int tile_width, int tile_height);

//! Get luminance using Matlab's NTSC calculation
void cuda_c_luminanceNTSC(float3_t *aptr, float *dptr, int w, int h, int tile_width, int tile_height);

//! Draw a filled in rectangle on top of an image with a particular intensity
void cuda_c_drawFilledRect(float *dst, int top, int left, int bottom, int right, const float intensity, const int w, const int h, const int tile_width, const int tile_height);

//! Draw a filled in rectangle on top of an image with a particular intensity
void cuda_c_drawFilledRectRGB(float3_t *dst, int top, int left, int bottom, int right, const float3_t *color, const int w, const int h, const int tile_width, const int tile_height);


//! Convolve and decimate in X direction with 5-tap lowpass filter
void cuda_c_lowpass_5_x_dec_x(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width);

//! Convolve and decimate in Y direction with 5-tap lowpass filter
void cuda_c_lowpass_5_y_dec_y(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width, int tile_height);

//! Convolve in X direction with 9-tap lowpass filter
void cuda_c_lowpass_9_x(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width);

//! Convolve in Y direction with 9-tap lowpass filter
void cuda_c_lowpass_9_y(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width, int tile_height);

//! Convolve and decimate in X direction with 9-tap lowpass filter
void cuda_c_lowpass_9_x_dec_x(const float* src, const unsigned int w, const unsigned int h, float* dst, const int dw, const int dh, int tile_width);

//! Convolve and decimate in Y direction with 9-tap lowpass filter
void cuda_c_lowpass_9_y_dec_y(const float* src, const unsigned int w, const unsigned int h, float* dst, const int dw, const int dh, int tile_width, int tile_height);

//! Convolve and decimate in X direction with 9-tap lowpass filter using texture memory
void cuda_c_lowpass_texture_9_x_dec_x(const float *src, int w, int h, float *dst, int dw, int dh, int tile_width, int tile_height);

//! Convolve and decimate in Y direction with 9-tap lowpass filter using texture memory
void cuda_c_lowpass_texture_9_y_dec_y(const float *src, int w, int h, float *dst, int dw, int dh, int tile_width, int tile_height);

//! Convolve in X direction with 5-tap lowpass filter
void cuda_c_lowpass_5_x(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width);

//! Convolve in Y direction with 5-tap lowpass filter
void cuda_c_lowpass_5_y(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width, int tile_height);

//! Convolve in X direction with 3-tap lowpass filter
void cuda_c_lowpass_3_x(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width);

//! Convolve in Y direction with 3-tap lowpass filter
void cuda_c_lowpass_3_y(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width, int tile_height);

//! Decimate image in the X and Y directions by factors
void cuda_c_dec_xy(const float *src,  float* dst, const int x_factor, const int y_factor, const unsigned int w, const unsigned int h, int tile_width);

//! Decimate image in X direction
void cuda_c_dec_x(const float *src,  float* dst, const int x_factor, const unsigned int w, const unsigned int h, int tile_width);

//! Decimate image in Y direction
void cuda_c_dec_y(const float *src,  float* dst, const int y_factor, const unsigned int w, const unsigned int h, int tile_width);

//! Take a local average of size scalex by scaley
void cuda_c_quickLocalAvg(const float *in, float *res, float fac, int lw, int lh, int sw, int sh, int tile_width, int tile_height);

//! Take a local average of 2x2 blocks
void cuda_c_quickLocalAvg2x2(const float *in, float *res, int lw, int lh, int sw, int sh, int tile_width, int tile_height);

//! Take a local max of size scalex by scaley
void cuda_c_quickLocalMax(const float *in, float *res, int lw, int lh, int sw, int sh, int tile_width, int tile_height);

//! Rescale image using bilinear interpolation
void cuda_c_rescaleBilinear(const float *src, float *res, float sw, float sh, int orig_w, int orig_h, int new_w, int new_h, int tile_width, int tile_height);

//! Rescale RGB image using bilinear interpolation
void cuda_c_rescaleBilinearRGB(const float3_t *src, float3_t *res, float sw, float sh, int orig_w, int orig_h, int new_w, int new_h, int tile_width, int tile_height);

//! Rectify image
void cuda_c_inplaceRectify(float *ptr, const int tile_len, const int sz);

//! Clamp image
void cuda_c_inplaceClamp(float *ptr, const float cmin, const float cmax, const int tile_len, const int sz);

//! Normalize an image inplace given the old min/max and the new min max
void cuda_c_inplaceNormalize(float *src, const float *omin, const float *omax, const float nmin, const float nmax, const int tile_len, const int sz);

//! Take absolute value of the image
void cuda_c_abs(float *src,const int tile_len, const int sz);

//! Set the value of an entire image to a particular value
void cuda_c_clear(float *src, const float val, const int tile_len, const int sz);

//! Add device scalar inplace
void cuda_c_inplaceAddScalar(float *ptr, const float *offset, const int tile_len, const int sz);

//! Subtract device scalar inplace
void cuda_c_inplaceSubtractScalar(float *ptr, const float *offset, const int tile_len, const int sz);

//! Multiply device scalar inplace
void cuda_c_inplaceMultiplyScalar(float *ptr, const float *offset, const int tile_len, const int sz);

//! Divide device scalar inplace
void cuda_c_inplaceDivideScalar(float *ptr, const float *offset, const int tile_len, const int sz);

//! Add host scalar
void cuda_c_inplaceAddHostScalar(float *ptr, const float val, const int tile_len, const int sz);

//! Subtract host scalar
void cuda_c_inplaceSubtractHostScalar(float *ptr, const float val, const int tile_len, const int sz);

//! Multiply host scalar
void cuda_c_inplaceMultiplyHostScalar(float *ptr, const float val, const int tile_len, const int sz);

//! Divide host scalar
void cuda_c_inplaceDivideHostScalar(float *ptr, const float val, const int tile_len, const int sz);

//! Add images inplace
void cuda_c_inplaceAddImages(float *im1, const float *im2, const int tile_len, const int sz);

//! Subtract images inplace
void cuda_c_inplaceSubtractImages(float *im1, const float *im2, const int tile_len, const int sz);

//! Multiply images inplace
void cuda_c_inplaceMultiplyImages(float *im1, const float *im2, const int tile_len, const int sz);

//! Divide images inplace
void cuda_c_inplaceDivideImages(float *im1, const float *im2, const int tile_len, const int sz);

//! Add images
void cuda_c_addImages(const float *im1, const float *im2, float *res, const int tile_len, const int sz);

//! Subtract images
void cuda_c_subtractImages(const float *im1, const float *im2, float *res, const int tile_len, const int sz);

//! Multiply images
void cuda_c_multiplyImages(const float *im1, const float *im2, float *res, const int tile_len, const int sz);

//! Divide images
void cuda_c_divideImages(const float *im1, const float *im2, float *res, const int tile_len, const int sz);

//! Take the max of each pixel from the two input images as the output
void cuda_c_takeMax(const float *im1, const float *im2, float *res, const int tile_len, const int sz);

//! Add device scalar
void cuda_c_addScalar(const float *im1, const float *im2, float *res, const int tile_len, const int sz);

//! Subtract device scalar
void cuda_c_subtractScalar(const float *im1, const float *im2, float *res, const int tile_len, const int sz);

//! Multiply device scalar
void cuda_c_multiplyScalar(const float *im1, const float *im2, float *res, const int tile_len, const int sz);

//! Divide device scalar
void cuda_c_divideScalar(const float *im1, const float *im2, float *res, const int tile_len, const int sz);

//! Add host scalar
void cuda_c_addHostScalar(const float *im1, const float val, float *res, const int tile_len, const int sz);

//! Subtract host scalar
void cuda_c_subtractHostScalar(const float *im1, const float val, float *res, const int tile_len, const int sz);

//! Multiply host scalar
void cuda_c_multiplyHostScalar(const float *im1, const float val, float *res, const int tile_len, const int sz);

//! Divide host scalar
void cuda_c_divideHostScalar(const float *im1, const float val, float *res, const int tile_len, const int sz);

//! Get the global min of an image
void cuda_c_getMin(const float *src, float *dest, float *buf, const int tile_len, const int sz);

//! Get the global max of an image
void cuda_c_getMax(const float *src, float *dest, float *buf, const int tile_len, const int sz);

//! Get the global avg of an image
void cuda_c_getAvg(const float *src, float *dest, float *buf, const int tile_len, const int sz);

//! Get the sum of all of the pixels of the image
void cuda_c_getSum(const float *src, float *dest, float *buf, const int tile_len, const int sz);

//! Square each pixel
void cuda_c_squared(const float *im, float *res, const int tile_len, const int sz);

//! Take square root of each pixel
void cuda_c_sqrt(const float *im, float *res, const int tile_len, const int sz);

//! Get the quad energy
void cuda_c_quadEnergy(const float *real, const float *imag, float *out, int tile_len, int sz);

//! Progressive attenuation of the border of an image
void cuda_c_inplaceAttenuateBorders(float *im, int borderSize, int tile_len, int w, int h);

//! Find the index of the largest value of an image
void cuda_c_findMax(const float *src, float *buf, int *loc, const int tile_len, const int sz);

//! Find the index of the lowest value of an image
void cuda_c_findMin(const float *src, float *buf, int *loc, const int tile_len, const int sz);

//! Generate a difference of Gaussian filter as parameterized by HMAX
void cuda_c_dogFilterHmax(float *dest, const float theta, const float gamma, const int size, const float div, const int tile_width, const int tile_height);

//! Generate a difference of Gaussian filter
void cuda_c_dogFilter(float *dest, float theta, float stddev, int half_size, int size, int tile_width, int tile_height);

//! Generate gabor kernel
void cuda_c_gaborFilter3(float *kern, const float major_stddev, const float minor_stddev,
                         const float period, const float phase,
                         const float theta, const int size, const int tile_len, const int sz);

//! Generate a 1D Gaussian kernel
void cuda_c_gaussian(float *res, float c, float sig22, int hw, int tile_len, int sz);

//! Run 2d oriented filter over image
void cuda_c_orientedFilter(const float *src, float *re, float *im, const float kx, const float ky, const float intensity, const int w, const int h, const int tile_width);

//! Compute abs(Center-Surround)
void cuda_c_centerSurroundAbs(const float *center, const float *surround, float *res, int lw, int lh, int sw, int sh, int tile_width );

//! Compute rectified(Center-Surround)
void cuda_c_centerSurroundClamped(const float *center, const float *surround, float *res, int lw, int lh, int sw, int sh, int tile_width );

//! Compute pos(Center-Surround) and neg(Center-Surround) separately to maintain direction
void cuda_c_centerSurroundDirectional(const float *center, const float *surround, float *pos, float *neg, int lw, int lh, int sw, int sh, int tile_width );

//! Compute abs(Center-Surround) with an attenuated border
void cuda_c_centerSurroundAbsAttenuate(const float *center, const float *surround, float *res, int lw, int lh, int sw, int sh, int attBorder, int tile_width, int tile_height);

//! Do local max over a window of activation
void cuda_c_spatialPoolMax(const float *src, float *res, float *buf1, float *buf2, const int src_w_in, const int src_h_in, const int skip_w_in, const int skip_h_in,
                           const int reg_w_in, const int reg_h_in, int tile_width_in, int tile_height_in);

//! Seed the random number generator -- MANDATORY!
void cuda_c_seedMT(unsigned int seed);

//! Get a bank of random numbers
void cuda_c_randomMT(float *d_Random, int numVals, int tile_len);

//! Add background noise to the image
void cuda_c_inplaceAddBGnoise2(float *in, float *rnd, const int brd_siz, const float range, int w, int h, int tile_len);

//! Hmax image energy normalized convolution
void cuda_c_convolveHmaxHelper(float *res, const float *src, const int src_w, const int src_h,
                               const float *f, const int Nx, const int Ny, const int tile_width, const int tile_height);

//! Zero boundary convolution
void cuda_c_convolveZeroHelper(float *res, const float *src, const int src_w, const int src_h,
                               const float *f, const int Nx, const int Ny, const int tile_width, const int tile_height);

//! Clean boundary convolution
void cuda_c_convolveCleanHelper(float *res, const float *src, const int src_w, const int src_h,
                                const float *f, const int Nx, const int Ny, const int tile_width, const int tile_height);

//! Optimized Hmax image energy normalized convolution
void cuda_c_convolveHmaxHelperOptimized(float *res, const float *src, const int src_w, const int src_h,
                                        const float *f, const int Nx, const int Ny, const int tile_width, const int tile_height);

//! Optimized zero boundary convolution
void cuda_c_convolveZeroHelperOptimized(float *res, const float *src, const int src_w, const int src_h,
                                        const float *f, const int Nx, const int Ny, const int tile_width, const int tile_height);

//! Optimized convolution
void cuda_c_optConvolve(float *res, const float *src, const int src_w, const int src_h,
                        const float *f, const int fil_w, const int fil_h, const int tile_width, const int tile_height);

//! Zero boundary X dimension separable filter convolution
void cuda_c_xFilterZero(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int share_len, const int tile_len);

//! Clean boundary X dimension separable filter convolution
void cuda_c_xFilterClean(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int share_len, const int tile_len);

//! Replicated boundary X dimension separable filter convolution
void cuda_c_xFilterReplicate(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int share_len, const int tile_len);

//! Zero boundary Y dimension separable filter convolution
void cuda_c_yFilterZero(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int share_len, const int tile_len);

//! Clean boundary Y dimension separable filter convolution
void cuda_c_yFilterClean(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int share_len, const int tile_len);

//! Replicated boundary Y dimension separable filter convolution
void cuda_c_yFilterReplicate(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int share_len, const int tile_len);

//! Optimized version of zero boundary X dimension separable filter, limited to only a certain size
void cuda_c_optXFilterZero(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len);

//! Optimized version of zero boundary Y dimension separable filter, limited to only a certain size
void cuda_c_optYFilterZero(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len);

//! Optimized version of clean boundary X dimension separable filter, limited to only a certain size
void cuda_c_optXFilterClean(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len);

//! Optimized version of clean boundary Y dimension separable filter, limited to only a certain size
void cuda_c_optYFilterClean(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len);

//! Optimized version of replicate boundary X dimension separable filter, limited to only a certain size
void cuda_c_optXFilterReplicate(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len);

//! Optimized version of replicate boundary Y dimension separable filter, limited to only a certain size
void cuda_c_optYFilterReplicate(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len);


//! Debayer the image
void cuda_2_debayer(float *src,float3_t *dptr,int w, int h, int tile_width, int tile_height);

//! Crop the image
void cuda_c_crop(const float *src, float *res, int srcw, int srch, int startx, int starty, int endx, int endy, int maxx,int maxy, int tile_width, int tile_height);

//! Translate image by deltax, deltay
void cuda_c_shiftImage(const float *src, float *dst, int w, int h, float deltax, float deltay, int tile_width, int tile_height);

// Paste an image onto an existing image
void cuda_c_inplacePaste(float *dst, const float *img, int w, int h, int iw, int ih, int dx, int dy, int tile_width, int tile_height);

// Paste an RGB image onto an existing image
void cuda_c_inplacePasteRGB(float3_t *dst, const float3_t *img, int w, int h, int iw, int ih, int dx, int dy, int tile_width, int tile_height);

// Overlay an image onto an existing image, only overwite pixels if overlaid image pixel is nonzero
void cuda_c_inplaceOverlay(float *dst, const float *img, int w, int h, int iw, int ih, int dx, int dy, int tile_width, int tile_height);

// Overlay an RGB image onto an existing image, only overwite pixels if overlaid image pixel is nonzero
void cuda_c_inplaceOverlayRGB(float3_t *dst, const float3_t *img, int w, int h, int iw, int ih, int dx, int dy, int tile_width, int tile_height);

// Calculate an inertia map
void cuda_c_inertiaMap(float_t *dst, float s, float r_inv, int px, int py, int tile_width, int tile_height, int w, int h);

// Calculate an inhibition map taking into account existing inhibition
void cuda_c_inhibitionMap(float *dst, float factorOld, float factorNew, float radius, int px, int py, int tile_width, int tile_height, int w, int h);

#ifdef __cplusplus
}
#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */


#endif
