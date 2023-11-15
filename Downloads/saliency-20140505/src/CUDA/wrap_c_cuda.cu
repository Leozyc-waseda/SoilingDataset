// Serves as a link to shared memory location on CUDA device
extern __shared__ float shared_data[];

#include "wrap_c_cuda.h"
#include <cuda_runtime.h>
#include "cuda_mersennetwisterkernel.h"
#include "cuda_colorops.h"
#include "cuda_lowpass.h"
#include "cuda_mathops.h"
#include "cuda_kernels.h"
#include "cuda_shapeops.h"
#include "cuda_drawops.h"
#include "cuda_filterops.h"
#include "cuda_transforms.h"
#include "cuda_convolutions.h"
#include "cuda_saliencyops.h"
#include <stdio.h>
#include "cuda_debayer.h"
#include "cuda_cutpaste.h"
// Externally callable host functions that instructs the device to call the appropriate CUDA function

// ######################################################################
void cuda_c_getRGBY(const float3_t *src, float *rgptr, float *byptr, const float thresh, 
		    const float min_range, const float max_range,
		    const int w, const int h, const int tile_width, const int tile_height)
{
  dim3 blockGridRows(iDivUp(w,tile_width),iDivUp(h,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_global_getRGBY<<<blockGridRows,threadBlockRows>>>(src,rgptr,byptr,thresh,min_range,max_range,w,h,tile_width,tile_height);  
}

void cuda_c_toRGB(float3_t *dst, const float *src,int sz, const int tile_len)
{
  dim3 blockGridRows(iDivUp(sz,tile_len),1);
  dim3 threadBlockRows(tile_len,1);
  cuda_global_toRGB<<<blockGridRows,threadBlockRows>>>(dst,src,sz,tile_len);
}

// ######################################################################
void cuda_c_getComponents(const float3_t *srcptr, float *rptr, float *gptr, float *bptr, int w, int h, int tile_width, int tile_height)
{
  dim3 blockGridRows(iDivUp(w,tile_width),iDivUp(h,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_global_getComponents<<<blockGridRows,threadBlockRows>>>(srcptr,rptr,gptr,bptr,w,h,tile_width,tile_height);  
}

// ######################################################################
void cuda_c_luminance(float3_t *aptr, float *dptr, int w, int h, int tile_width, int tile_height)
{
  dim3 blockGridRows(iDivUp(w,tile_width),iDivUp(h,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_global_luminance<<<blockGridRows,threadBlockRows>>>(aptr,dptr,w,h,tile_width,tile_height);
}

// ######################################################################
void cuda_c_luminanceNTSC(float3_t *aptr, float *dptr, int w, int h, int tile_width, int tile_height)
{
  dim3 blockGridRows(iDivUp(w,tile_width),iDivUp(h,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);

  cuda_global_luminanceNTSC<<<blockGridRows,threadBlockRows>>>(aptr,dptr,w,h,tile_width,tile_height);
}

// ######################################################################
void cuda_c_drawFilledRect(float *dst, int top, int left, int bottom, int right, const float intensity, const int w, const int h, const int tile_width, const int tile_height)
{
  int rect_width = right-left+1;
  int rect_height = bottom-top+1;
  dim3 blockGridRows(iDivUp(rect_width,tile_width),iDivUp(rect_height,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);

  cuda_global_drawFilledRect<<<blockGridRows,threadBlockRows>>>(dst,top,left,bottom,right,intensity,w,h,tile_width,tile_height);
}

// ######################################################################
void cuda_c_drawFilledRectRGB(float3_t *dst, int top, int left, int bottom, int right, const float3_t *color, const int w, const int h, const int tile_width, const int tile_height)
{
  int rect_width = right-left+1;
  int rect_height = bottom-top+1;
  dim3 blockGridRows(iDivUp(rect_width,tile_width),iDivUp(rect_height,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_global_drawFilledRectRGB<<<blockGridRows,threadBlockRows>>>(dst,top,left,bottom,right,color->p[0],color->p[1],color->p[2],w,h,tile_width,tile_height);
}

// ######################################################################
void cuda_c_lowpass_5_x_dec_x(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width)
{
  dim3 blockGridRows(iDivUp(w, tile_width), h);
  dim3 threadBlockRows(tile_width);

  cuda_global_lowpass_5_x_dec_x<<<blockGridRows, threadBlockRows,(tile_width+4)*sizeof(float)>>>
    (src, w, h, dst, tile_width);
}

// ######################################################################
void cuda_c_lowpass_5_y_dec_y(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width, int tile_height)
{
  dim3 blockGridColumns(iDivUp(w, tile_width), iDivUp(h, tile_height));
  dim3 threadBlockColumns(tile_width, tile_height);

  cuda_global_lowpass_5_y_dec_y<<<blockGridColumns, threadBlockColumns,tile_width*(tile_height+4)*sizeof(float)>>>
    (src, w, h, dst, tile_width, tile_height); 
}

// #####################################################################
void cuda_c_lowpass_9_x(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width)
{
  dim3 blockGridRows(iDivUp(w, tile_width), h);
  dim3 threadBlockRows(tile_width);
  cuda_global_lowpass_9_x<<<blockGridRows, threadBlockRows,(tile_width+8)*sizeof(float)>>>(src, w, h, dst, tile_width);
}

// #####################################################################
void cuda_c_lowpass_9_y(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width, int tile_height)
{
  dim3 blockGridColumns(iDivUp(w, tile_width), iDivUp(h, tile_height));
  dim3 threadBlockColumns(tile_width, tile_height);

  cuda_global_lowpass_9_y<<<blockGridColumns, threadBlockColumns,tile_width*(tile_height+8)*sizeof(float)>>>(src, w, h, dst, tile_width, tile_height);
}

// #####################################################################
void cuda_c_lowpass_9_x_dec_x(const float* src, const unsigned int w, const unsigned int h, float* dst, const int dw, const int dh, int tile_width)
{
  const int dest_tile_width = tile_width>>1; // Need to reduce, so source can be stored
  dim3 blockGridRows(iDivUp(dw, dest_tile_width), dh);
  dim3 threadBlockRows(dest_tile_width);
  cuda_global_lowpass_9_x_dec_x<<<blockGridRows, threadBlockRows,(tile_width+8)*sizeof(float)>>>(src, w, h, dst, dw, dh, dest_tile_width);
}

// #####################################################################
void cuda_c_lowpass_9_y_dec_y(const float* src, const unsigned int w, const unsigned int h, float* dst, const int dw, const int dh, int tile_width, int tile_height)
{
  const int dest_tile_height = tile_height>>1; // Need to reduce, so source can be stored
  dim3 blockGridColumns(iDivUp(dw, tile_width), iDivUp(dh, dest_tile_height));
  dim3 threadBlockColumns(tile_width, dest_tile_height);
  cuda_global_lowpass_9_y_dec_y<<<blockGridColumns, threadBlockColumns,tile_width*(tile_height+8)*sizeof(float)>>>(src, w, h, dst, dw, dh, tile_width, dest_tile_height);
}

// ######################################################################
void cuda_c_lowpass_5_x(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width)
{
  dim3 blockGridRows(iDivUp(w, tile_width), h);
  dim3 threadBlockRows(tile_width);

  cuda_global_lowpass_5_x<<<blockGridRows, threadBlockRows,(tile_width+4)*sizeof(float)>>>
    (src, w, h, dst, tile_width);
}

// ######################################################################
void cuda_c_lowpass_5_y(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width, int tile_height)
{
  dim3 blockGridColumns(iDivUp(w, tile_width), iDivUp(h, tile_height));
  dim3 threadBlockColumns(tile_width, tile_height);

  cuda_global_lowpass_5_y<<<blockGridColumns, threadBlockColumns,tile_width*(tile_height+4)*sizeof(float)>>>
    (src, w, h, dst, tile_width, tile_height); 
}

// ######################################################################
void cuda_c_lowpass_3_x(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width)
{
  dim3 blockGridRows(iDivUp(w, tile_width), h);
  dim3 threadBlockRows(tile_width);

  cuda_global_lowpass_3_x<<<blockGridRows, threadBlockRows,(tile_width+4)*sizeof(float)>>>
    (src, w, h, dst, tile_width);
}

// ######################################################################
void cuda_c_lowpass_3_y(const float* src, const unsigned int w, const unsigned int h, float* dst, int tile_width, int tile_height)
{
  dim3 blockGridColumns(iDivUp(w, tile_width), iDivUp(h, tile_height));
  dim3 threadBlockColumns(tile_width, tile_height);

  cuda_global_lowpass_3_y<<<blockGridColumns, threadBlockColumns,tile_width*(tile_height+4)*sizeof(float)>>>
    (src, w, h, dst, tile_width, tile_height); 
}

// ######################################################################
void cuda_c_dec_xy(const float *src,  float* dst, const int x_factor, const int y_factor, const unsigned int w, const unsigned int h, int tile_width)
{
  dim3 blockGridRows(iDivUp(w/x_factor, tile_width), h/y_factor);
  dim3 threadBlockRows(tile_width);
  cuda_global_dec_xy<<<blockGridRows, threadBlockRows>>>(src,dst,x_factor,y_factor,w,h,tile_width);
}

// ######################################################################
void cuda_c_dec_x(const float *src,  float* dst, const int x_factor, const unsigned int w, const unsigned int h, int tile_width)
{
  dim3 blockGridRows(iDivUp(w/x_factor, tile_width), h);
  dim3 threadBlockRows(tile_width);
  cuda_global_dec_x<<<blockGridRows, threadBlockRows>>>(src,dst,x_factor,w,h,tile_width);
}

// ######################################################################
void cuda_c_dec_y(const float *src,  float* dst, const int y_factor, const unsigned int w, const unsigned int h, int tile_width)
{
  dim3 blockGridRows(iDivUp(w, tile_width), h/y_factor);
  dim3 threadBlockRows(tile_width);
  cuda_global_dec_y<<<blockGridRows, threadBlockRows>>>(src,dst,y_factor,w,h,tile_width);
}

void cuda_c_quickLocalAvg(const float *in, float *res, float fac, int lw, int lh, int sw, int sh, int tile_width, int tile_height)
{
  int scalex = lw / sw, scaley = lh / sh; // in case the image is very small
  // IMPORTANT!  remx is JUST the remainder, unlike the CPU version, which is remx=lw-1-(lh%sh)
  int remx = (lw % sw), remy = (lh % sh);

  dim3 blockGridColumns(iDivUp(sw, tile_width), iDivUp(sh, tile_height));
  dim3 threadBlockColumns(tile_width, tile_height);  
  cuda_global_quickLocalAvg<<<blockGridColumns,threadBlockColumns>>>(in,res,fac,scalex,scaley,remx,remy,lw,lh,sw,sh,tile_width,tile_height);
}

void cuda_c_quickLocalAvg2x2(const float *in, float *res, int lw, int lh, int sw, int sh, int tile_width, int tile_height)
{
  dim3 blockGridColumns(iDivUp(sw, tile_width), iDivUp(sh, tile_height));
  dim3 threadBlockColumns(tile_width, tile_height);  
  cuda_global_quickLocalAvg2x2<<<blockGridColumns,threadBlockColumns>>>(in,res,lw,lh,sw,sh,tile_width,tile_height);
}

void cuda_c_quickLocalMax(const float *in, float *res, int lw, int lh, int sw, int sh, int tile_width, int tile_height)
{
  int scalex = lw / sw, scaley = lh / sh;  // in case the image is very small
  int remx = lw - 1 - (lw % sw), remy = lh - 1 - (lh % sh);

  dim3 blockGridColumns(iDivUp(sw, tile_width), iDivUp(sh, tile_height));
  dim3 threadBlockColumns(tile_width, tile_height);  
  cuda_global_quickLocalMax<<<blockGridColumns,threadBlockColumns>>>(in,res,scalex,scaley,remx,remy,lw,lh,sw,sh,tile_width,tile_height);
}


void cuda_c_rescaleBilinear(const float *src, float *res, float sw, float sh, int orig_w, int orig_h, int new_w, int new_h, int tile_width, int tile_height)
{
  dim3 blockGridColumns(iDivUp(new_w, tile_width), iDivUp(new_h, tile_height));
  dim3 threadBlockColumns(tile_width, tile_height);  
  cuda_global_rescaleBilinear<<<blockGridColumns,threadBlockColumns>>>(src,res,sw,sh,orig_w,orig_h,new_w,new_h,tile_width,tile_height);
}

void cuda_c_rescaleBilinearRGB(const float3_t *src, float3_t *res, float sw, float sh, int orig_w, int orig_h, int new_w, int new_h, int tile_width, int tile_height)
{
  dim3 blockGridColumns(iDivUp(new_w, tile_width), iDivUp(new_h, tile_height));
  dim3 threadBlockColumns(tile_width, tile_height);  
  cuda_global_rescaleBilinearRGB<<<blockGridColumns,threadBlockColumns>>>(src,res,sw,sh,orig_w,orig_h,new_w,new_h,tile_width,tile_height);
}

void cuda_c_clear(float *src, const float val, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_clear<<<blockGridRows,threadBlockRows>>>(src,val,tile_len,sz);  
}

void cuda_c_abs(float *src,const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_abs<<<blockGridRows,threadBlockRows>>>(src,tile_len,sz);  
}

void cuda_c_inplaceAddScalar(float *ptr, const float *offset, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_inplaceAddScalar<<<blockGridRows,threadBlockRows>>>(ptr,offset,tile_len,sz);  
}

void cuda_c_inplaceSubtractScalar(float *ptr, const float *offset, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_inplaceSubtractScalar<<<blockGridRows,threadBlockRows>>>(ptr,offset,tile_len,sz);  
}

void cuda_c_inplaceMultiplyScalar(float *ptr, const float *offset, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_inplaceMultiplyScalar<<<blockGridRows,threadBlockRows>>>(ptr,offset,tile_len,sz);  
}

void cuda_c_inplaceDivideScalar(float *ptr, const float *offset, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_inplaceDivideScalar<<<blockGridRows,threadBlockRows>>>(ptr,offset,tile_len,sz);  
}

void cuda_c_inplaceAddHostScalar(float *ptr, const float val, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_inplaceAddHostScalar<<<blockGridRows,threadBlockRows>>>(ptr,val,tile_len,sz);  
}

void cuda_c_inplaceSubtractHostScalar(float *ptr, const float val, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_inplaceSubtractHostScalar<<<blockGridRows,threadBlockRows>>>(ptr,val,tile_len,sz);  
}

void cuda_c_inplaceMultiplyHostScalar(float *ptr, const float val, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_inplaceMultiplyHostScalar<<<blockGridRows,threadBlockRows>>>(ptr,val,tile_len,sz);  
}

void cuda_c_inplaceDivideHostScalar(float *ptr, const float val, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_inplaceDivideHostScalar<<<blockGridRows,threadBlockRows>>>(ptr,val,tile_len,sz);  
}


void cuda_c_inplaceAddImages(float *im1, const float *im2, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_inplaceAddImages<<<blockGridRows,threadBlockRows>>>(im1,im2,tile_len,sz);  
}

void cuda_c_inplaceSubtractImages(float *im1, const float *im2, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_inplaceSubtractImages<<<blockGridRows,threadBlockRows>>>(im1,im2,tile_len,sz);  
}

void cuda_c_inplaceMultiplyImages(float *im1, const float *im2, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_inplaceMultiplyImages<<<blockGridRows,threadBlockRows>>>(im1,im2,tile_len,sz);  
}

void cuda_c_inplaceDivideImages(float *im1, const float *im2, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_inplaceDivideImages<<<blockGridRows,threadBlockRows>>>(im1,im2,tile_len,sz);  
}

void cuda_c_addImages(const float *im1, const float *im2, float *res, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_addImages<<<blockGridRows,threadBlockRows>>>(im1,im2,res,tile_len,sz);  
}

void cuda_c_subtractImages(const float *im1, const float *im2, float *res, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_subtractImages<<<blockGridRows,threadBlockRows>>>(im1,im2,res,tile_len,sz);  
}

void cuda_c_multiplyImages(const float *im1, const float *im2, float *res, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_multiplyImages<<<blockGridRows,threadBlockRows>>>(im1,im2,res,tile_len,sz);  
}

void cuda_c_divideImages(const float *im1, const float *im2, float *res, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_divideImages<<<blockGridRows,threadBlockRows>>>(im1,im2,res,tile_len,sz);  
}

void cuda_c_takeMax(const float *im1, const float *im2, float *res, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_takeMax<<<blockGridRows,threadBlockRows>>>(im1,im2,res,tile_len,sz);  
}

void cuda_c_addScalar(const float *im1, const float *im2, float *res, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_addScalar<<<blockGridRows,threadBlockRows>>>(im1,im2,res,tile_len,sz);  
}

void cuda_c_subtractScalar(const float *im1, const float *im2, float *res, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_subtractScalar<<<blockGridRows,threadBlockRows>>>(im1,im2,res,tile_len,sz);  
}

void cuda_c_multiplyScalar(const float *im1, const float *im2, float *res, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_multiplyScalar<<<blockGridRows,threadBlockRows>>>(im1,im2,res,tile_len,sz);  
}

void cuda_c_divideScalar(const float *im1, const float *im2, float *res, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_divideScalar<<<blockGridRows,threadBlockRows>>>(im1,im2,res,tile_len,sz);  
}

void cuda_c_addHostScalar(const float *im1, const float val, float *res, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_addHostScalar<<<blockGridRows,threadBlockRows>>>(im1,val,res,tile_len,sz);  
}

void cuda_c_subtractHostScalar(const float *im1, const float val, float *res, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_subtractHostScalar<<<blockGridRows,threadBlockRows>>>(im1,val,res,tile_len,sz);  
}

void cuda_c_multiplyHostScalar(const float *im1, const float val, float *res, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_multiplyHostScalar<<<blockGridRows,threadBlockRows>>>(im1,val,res,tile_len,sz);  
}

void cuda_c_divideHostScalar(const float *im1, const float val, float *res, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_divideHostScalar<<<blockGridRows,threadBlockRows>>>(im1,val,res,tile_len,sz);  
}

void cuda_c_inplaceRectify(float *ptr, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_inplaceRectify<<<blockGridRows,threadBlockRows>>>(ptr,tile_len,sz);
}

void cuda_c_inplaceClamp(float *ptr, const float cmin, const float cmax, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);

  cuda_global_inplaceClamp<<<blockGridRows,threadBlockRows>>>(ptr,cmin,cmax,tile_len,sz);
}

void cuda_c_inplaceNormalize(float *src, const float *omin, const float *omax, const float nmin, const float nmax, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_inplaceNormalize<<<blockGridRows,threadBlockRows>>>(src,omin,omax,nmin,nmax,tile_len,sz);
}

void cuda_c_getMin(const float *src, float *dest, float *buf, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  // This must be iteratively called, because we need to synchronize the whole grid (and that is guaranteed between kernel calls)
  const float *in = src;
  for(int cur_sz=sz;   cur_sz>1; cur_sz=IDIVUP(cur_sz,tile_len))
  {
    cuda_global_getMin<<<blockGridRows,threadBlockRows,tile_len*sizeof(float)>>>(in,dest,buf,tile_len,cur_sz);
    in=buf;
  }
}


void cuda_c_getMax(const float *src, float *dest, float *buf, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  // This must be iteratively called, because we need to synchronize the whole grid (and that is guaranteed between kernel calls)
  const float *in = src;
  for(int cur_sz=sz;   cur_sz>1; cur_sz=IDIVUP(cur_sz,tile_len))
  {
    cuda_global_getMax<<<blockGridRows,threadBlockRows,tile_len*sizeof(float)>>>(in,dest,buf,tile_len,cur_sz);
    in=buf;
  }
}

void cuda_c_getAvg(const float *src, float *dest, float *buf, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  // This must be iteratively called, because we need to synchronize the whole grid (and that is guaranteed between kernel calls)
  const float *in = src;
  for(int cur_sz=sz;   cur_sz>1; cur_sz=IDIVUP(cur_sz,tile_len))
  {
    cuda_global_getAvg<<<blockGridRows,threadBlockRows,tile_len*sizeof(float)>>>(in,dest,buf,tile_len,cur_sz,sz);
    in=buf;
  }
}

void cuda_c_getSum(const float *src, float *dest, float *buf, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  // This must be iteratively called, because we need to synchronize the whole grid (and that is guaranteed between kernel calls)
  const float *in = src;
  for(int cur_sz=sz;   cur_sz>1; cur_sz=IDIVUP(cur_sz,tile_len))
  {
    cuda_global_getSum<<<blockGridRows,threadBlockRows,tile_len*sizeof(float)>>>(in,dest,buf,tile_len,cur_sz,sz);
    in=buf;
  }
}

void cuda_c_squared(const float *im, float *res, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_squared<<<blockGridRows,threadBlockRows>>>(im,res,tile_len,sz);
}

void cuda_c_sqrt(const float *im, float *res, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_sqrt<<<blockGridRows,threadBlockRows>>>(im,res,tile_len,sz);
}


void cuda_c_quadEnergy(const float *real, const float *imag, float *out, int tile_len, int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_quadEnergy<<<blockGridRows,threadBlockRows>>>(real,imag,out,tile_len,sz);
}

void cuda_c_inplaceAttenuateBorders(float *im, int borderSize, int tile_len, int w, int h)
{
  dim3 xBlock(iDivUp(w,tile_len),borderSize*2);
  dim3 xThread(tile_len);
  cuda_global_inplaceAttenuateBorders_x<<<xBlock,xThread>>>(im,borderSize,tile_len,w,h);
  dim3 yBlock(borderSize*2,iDivUp(h,tile_len));
  dim3 yThread(1,tile_len);
  cuda_global_inplaceAttenuateBorders_y<<<yBlock,yThread>>>(im,borderSize,tile_len,w,h);
}

void cuda_c_findMax(const float *src, float *buf, int *loc, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  // This must be iteratively called, because we need to synchronize the whole grid (and that is guaranteed between kernel calls)
  const float *in = src;
  const int *inLoc = NULL;
  for(int cur_sz=sz;   cur_sz>1; cur_sz=IDIVUP(cur_sz,tile_len))
  {
    cuda_global_findMax<<<blockGridRows,threadBlockRows,2*tile_len*sizeof(float)>>>(in,inLoc,buf,loc,tile_len,cur_sz);
    in=buf;
    inLoc=loc;
  }
}

void cuda_c_findMin(const float *src, float *buf, int *loc, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  // This must be iteratively called, because we need to synchronize the whole grid (and that is guaranteed between kernel calls)
  const float *in = src;
  const int *inLoc = NULL;
  for(int cur_sz=sz;   cur_sz>1; cur_sz=IDIVUP(cur_sz,tile_len))
  {
    cuda_global_findMin<<<blockGridRows,threadBlockRows,2*tile_len*sizeof(float)>>>(in,inLoc,buf,loc,tile_len,cur_sz);
    in=buf;
    inLoc=loc;
  }
}



void cuda_c_dogFilterHmax(float *dest, const float theta, const float gamma, const int size, const float div, const int tile_width, const int tile_height)
{
  dim3 blockGridRows(iDivUp(size,tile_width),iDivUp(size,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_global_dogFilterHmax<<<blockGridRows,threadBlockRows>>>(dest,theta,gamma,size,div,tile_width,tile_height);
}

void cuda_c_dogFilter(float *dest, float stddev, float theta, int half_size, int size, int tile_width, int tile_height)
{
  dim3 blockGridRows(iDivUp(size,tile_width),iDivUp(size,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_global_dogFilter<<<blockGridRows,threadBlockRows>>>(dest,stddev,theta,half_size,size,tile_width,tile_height);
}

void cuda_c_gaborFilter3(float *kern, const float major_stddev, const float minor_stddev,
                          const float period, const float phase,
			 const float theta, const int size, const int tile_len, const int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_gaborFilter3<<<blockGridRows,threadBlockRows>>>(kern,major_stddev,minor_stddev,period,phase,theta,size,tile_len,sz);
}

void cuda_c_gaussian(float *res, float c, float sig22, int hw, int tile_len, int sz)
{
  dim3 blockGridRows(iDivUp(sz,tile_len));
  dim3 threadBlockRows(tile_len);
  cuda_global_gaussian<<<blockGridRows,threadBlockRows>>>(res,c,sig22,hw,tile_len,sz);
}

void cuda_c_orientedFilter(const float *src, float *re, float *im, const float kx, const float ky, const float intensity, const int w, const int h, const int tile_width)
{
  dim3 blockGridRows(iDivUp(w, tile_width), h);
  dim3 threadBlockRows(tile_width);
  cuda_global_orientedFilter<<<blockGridRows,threadBlockRows>>>(src,re,im,kx,ky,intensity,w,h,tile_width);
}

void cuda_c_centerSurroundAbs(const float *center, const float *surround, float *res, int lw, int lh, int sw, int sh, int tile_width )
{
  int scalex = lw / sw, remx = lw - 1 - (lw % sw);
  int scaley = lh / sh, remy = lh - 1 - (lh % sh);
  dim3 blockGridRows(iDivUp(lw, tile_width), lh);
  dim3 threadBlockRows(tile_width);
  cuda_global_centerSurroundAbs<<<blockGridRows,threadBlockRows>>>(center,surround,res,lw,lh,sw,sh,scalex,scaley,remx,remy,tile_width);
}

void cuda_c_centerSurroundClamped(const float *center, const float *surround, float *res, int lw, int lh, int sw, int sh, int tile_width )
{
  int scalex = lw / sw, remx = lw - 1 - (lw % sw);
  int scaley = lh / sh, remy = lh - 1 - (lh % sh);
  dim3 blockGridRows(iDivUp(lw, tile_width), lh);
  dim3 threadBlockRows(tile_width);
  cuda_global_centerSurroundClamped<<<blockGridRows,threadBlockRows>>>(center,surround,res,lw,lh,sw,sh,scalex,scaley,remx,remy,tile_width);
}

void cuda_c_centerSurroundDirectional(const float *center, const float *surround, float *pos, float *neg, int lw, int lh, int sw, int sh, int tile_width )
{
  int scalex = lw / sw, remx = lw - 1 - (lw % sw);
  int scaley = lh / sh, remy = lh - 1 - (lh % sh);
  dim3 blockGridRows(iDivUp(lw, tile_width), lh);
  dim3 threadBlockRows(tile_width);
  cuda_global_centerSurroundDirectional<<<blockGridRows,threadBlockRows>>>(center,surround,pos,neg,lw,lh,sw,sh,scalex,scaley,remx,remy,tile_width);
}

void cuda_c_centerSurroundAbsAttenuate(const float *center, const float *surround, float *res, int lw, int lh, int sw, int sh, int attBorder, int tile_width, int tile_height)
{
  int scalex = lw / sw, remx = lw - 1 - (lw % sw);
  int scaley = lh / sh, remy = lh - 1 - (lh % sh);
  dim3 blockGridRows(iDivUp(lw, tile_width), iDivUp(lh,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_global_centerSurroundAbsAttenuate<<<blockGridRows,threadBlockRows>>>(center,surround,res,lw,lh,sw,sh,attBorder,scalex,scaley,remx,remy,tile_width,tile_height);
}

void cuda_c_spatialPoolMax(const float *src, float *res, float *buf1, float *buf2, const int src_w_in, const int src_h_in, const int skip_w_in, const int skip_h_in,  
			   const int reg_w_in, const int reg_h_in, int tile_width_in, int tile_height_in)
{
  int reg_w = reg_w_in;  int reg_h = reg_h_in;
  int skip_w = skip_w_in;  int skip_h = skip_h_in;
  int src_w = src_w_in; int src_h = src_h_in;
  int tilesperregion_w, tilesperregion_h;
  int tile_width, tile_height;
  const int orig_tile_size = tile_width_in*tile_height_in;
  const float *in = src;
  float *out = buf1;
  float *next = buf2;
  // Run the kernel recursively until the region is smaller than the tile
  do
    {
      // Modify the tile size to optimize based on the region
      if(reg_w < orig_tile_size)
	{
	  tile_width = reg_w;
	  // Minimum of the region width or the remaining tile dimension
	  tile_height = MIN(orig_tile_size/tile_width,reg_h);
	}
      else if(reg_h < orig_tile_size)
	{
	  tile_height = reg_h;
	  // Minimum of the region width or the remaining tile dimension
	  tile_width = MIN(orig_tile_size/tile_height,reg_w);
	}
      else
	{
	  tile_width = tile_width_in;
	  tile_height = tile_height_in;
	}
      tilesperregion_w = iDivUp(reg_w,tile_width);
      tilesperregion_h = iDivUp(reg_h,tile_height);
      // If this is the last time through, set the output to be the result
      if(tilesperregion_w == 1 && tilesperregion_h == 1)
	out = res;
      int num_blocks_w = iDivUp(src_w, skip_w)*tilesperregion_w;
      int num_blocks_h = iDivUp(src_h, skip_h)*tilesperregion_h;
      dim3 blockGridRows(num_blocks_w,num_blocks_h);
      dim3 threadBlockRows(tile_width,tile_height);
      //printf("tile_w:%d _h:%d, reg_w:%d _h:%d, skip_w:%d _h:%d, src_w:%d _h:%d\n",
      //     tile_width,tile_height,reg_w,reg_h,skip_w,skip_h,src_w,src_h);
      cuda_global_spatialPoolMax<<<blockGridRows,threadBlockRows,tile_width*tile_height*sizeof(float)>>>(in,out,src_w,src_h,skip_w,skip_h,reg_w,reg_h,tile_width,tile_height);
      src_w = num_blocks_w;
      src_h = num_blocks_h;
      reg_w = tilesperregion_w;
      reg_h = tilesperregion_h;
      skip_w = reg_w;
      skip_h = reg_h;
      // Update the location of the input
      in = out;
      out = next;
      next = (buf1 == in) ? buf1 : buf2;
    } while(tilesperregion_w > 1 || tilesperregion_h > 1);

}

// Number of values requested must be divisible by MT_RNG_COUNT

void cuda_c_randomMT(float *d_Random, int numVals, int tile_len)
{
  if(numVals % MT_RNG_COUNT != 0)
  {
    printf("ERROR!!!: Cannot request a set of random numbers that is not in MT_RNG_COUNT [%d]units\n",MT_RNG_COUNT);
    return;
  }
  int NPerRng = numVals/MT_RNG_COUNT;
  dim3 blockGridColumns(NPerRng,1);
  dim3 threadBlockColumns(tile_len,1);
  cuda_global_randomMT<<<blockGridColumns,threadBlockColumns>>>(d_Random,NPerRng);
}

void cuda_c_inplaceAddBGnoise2(float *in, float *rnd, const int brd_siz, const float range, int w, int h, int tile_len)
{
  dim3 blockGridColumns(iDivUp(w,tile_len),h);
  dim3 threadBlockColumns(tile_len,1);
  cuda_global_inplaceAddBGnoise2<<<blockGridColumns,threadBlockColumns>>>(in,rnd,brd_siz,range,w,h,tile_len);
}

void cuda_c_convolveHmaxHelper(float *res, const float *src, const int src_w, const int src_h, 
			     const float *f, const int Nx, const int Ny, const int tile_width, const int tile_height)
{
  dim3 blockGridColumns(iDivUp(src_w,tile_width),iDivUp(src_h,tile_height));
  dim3 threadBlockColumns(tile_width,tile_height);
  cuda_global_convolveHmaxHelper<<<blockGridColumns,threadBlockColumns,Nx*Ny*sizeof(float)>>>(res,src,src_w,src_h,f,Nx,Ny,tile_width,tile_height);
}

void cuda_c_convolveZeroHelper(float *res, const float *src, const int src_w, const int src_h, 
			     const float *f, const int Nx, const int Ny, const int tile_width, const int tile_height)
{
  dim3 blockGridColumns(iDivUp(src_w,tile_width),iDivUp(src_h,tile_height));
  dim3 threadBlockColumns(tile_width,tile_height);
  cuda_global_convolveZeroHelper<<<blockGridColumns,threadBlockColumns,Nx*Ny*sizeof(float)>>>(res,src,src_w,src_h,f,Nx,Ny,tile_width,tile_height);
}

void cuda_c_convolveCleanHelper(float *res, const float *src, const int src_w, const int src_h, 
			     const float *f, const int Nx, const int Ny, const int tile_width, const int tile_height)
{
  dim3 blockGridColumns(iDivUp(src_w,tile_width),iDivUp(src_h,tile_height));
  dim3 threadBlockColumns(tile_width,tile_height);
  cuda_global_convolveCleanHelper<<<blockGridColumns,threadBlockColumns,Nx*Ny*sizeof(float)>>>(res,src,src_w,src_h,f,Nx,Ny,tile_width,tile_height);
}

void cuda_c_convolveHmaxHelperOptimized(float *res, const float *src, const int src_w, const int src_h, 
			     const float *f, const int Nx, const int Ny, const int tile_width, const int tile_height)
{
  dim3 blockGridColumns(iDivUp(src_w,tile_width),iDivUp(src_h,tile_height));
  dim3 threadBlockColumns(tile_width,tile_height);
  cuda_global_convolveHmaxHelperOptimized<<<blockGridColumns,threadBlockColumns,(Nx*Ny+(Nx+tile_width)*(Ny+tile_height))*sizeof(float)>>>(res,src,src_w,src_h,f,Nx,Ny,tile_width,tile_height);
}

void cuda_c_convolveZeroHelperOptimized(float *res, const float *src, const int src_w, const int src_h, 
			     const float *f, const int Nx, const int Ny, const int tile_width, const int tile_height)
{
  dim3 blockGridColumns(iDivUp(src_w,tile_width),iDivUp(src_h,tile_height));
  dim3 threadBlockColumns(tile_width,tile_height);
  cuda_global_convolveZeroHelperOptimized<<<blockGridColumns,threadBlockColumns,(Nx*Ny+(Nx+tile_width)*(Ny+tile_height))*sizeof(float)>>>(res,src,src_w,src_h,f,Nx,Ny,tile_width,tile_height);
}

void cuda_c_optConvolve(float *res, const float *src, const int src_w, const int src_h, 
			     const float *f, const int fil_w, const int fil_h, const int tile_width, const int tile_height)
{
  dim3 blockGridColumns(iDivUp(src_w,tile_width),iDivUp(src_h,tile_height));
  dim3 threadBlockColumns(tile_width,tile_height);
  cuda_global_optConvolve<<<blockGridColumns,threadBlockColumns,fil_w*fil_h*sizeof(float)>>>(res,src,src_w,src_h,f,fil_w,fil_h,tile_width,tile_height);
}

void cuda_c_xFilterZero(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int share_len, const int tile_len)
{
  dim3 blockGridColumns(iDivUp(src_w,tile_len),src_h);
  dim3 threadBlockColumns(tile_len,1);
  cuda_global_xFilterZero<<<blockGridColumns,threadBlockColumns,share_len*sizeof(float)>>>(res,src,src_w,src_h,f,hfs,share_len,tile_len);
}

void cuda_c_xFilterClean(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int share_len, const int tile_len)
{
  dim3 blockGridColumns(iDivUp(src_w,tile_len),src_h);
  dim3 threadBlockColumns(tile_len,1);
  cuda_global_xFilterClean<<<blockGridColumns,threadBlockColumns,share_len*sizeof(float)>>>(res,src,src_w,src_h,f,hfs,share_len,tile_len);
}

void cuda_c_xFilterReplicate(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int share_len, const int tile_len)
{
  dim3 blockGridColumns(iDivUp(src_w,tile_len),src_h);
  dim3 threadBlockColumns(tile_len,1);
  cuda_global_xFilterReplicate<<<blockGridColumns,threadBlockColumns,share_len*sizeof(float)>>>(res,src,src_w,src_h,f,hfs,share_len,tile_len);
}

void cuda_c_yFilterZero(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int share_len, const int tile_len)
{
  dim3 blockGridColumns(src_w,iDivUp(src_h,tile_len));
  dim3 threadBlockColumns(1,tile_len);
  cuda_global_yFilterZero<<<blockGridColumns,threadBlockColumns,share_len*sizeof(float)>>>(res,src,src_w,src_h,f,hfs,share_len,tile_len);
}

void cuda_c_yFilterClean(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int share_len, const int tile_len)
{
  dim3 blockGridColumns(src_w,iDivUp(src_h,tile_len));
  dim3 threadBlockColumns(1,tile_len);
  cuda_global_yFilterClean<<<blockGridColumns,threadBlockColumns,share_len*sizeof(float)>>>(res,src,src_w,src_h,f,hfs,share_len,tile_len);
}

void cuda_c_yFilterReplicate(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int share_len, const int tile_len)
{
  dim3 blockGridColumns(src_w,iDivUp(src_h,tile_len));
  dim3 threadBlockColumns(1,tile_len);
  cuda_global_yFilterReplicate<<<blockGridColumns,threadBlockColumns,share_len*sizeof(float)>>>(res,src,src_w,src_h,f,hfs,share_len,tile_len);
}

void cuda_c_optXFilterZero(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len)
{
  dim3 blockGridColumns(iDivUp(src_w,tile_len),src_h);
  dim3 threadBlockColumns(tile_len,1);
  cuda_global_optXFilterZero<<<blockGridColumns,threadBlockColumns,(tile_len+hfs*2)*sizeof(float)>>>(res,src,src_w,src_h,f,hfs,tile_len);
}

void cuda_c_optYFilterZero(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len)
{
  dim3 blockGridColumns(src_w,iDivUp(src_h,tile_len));
  dim3 threadBlockColumns(1,tile_len);
  cuda_global_optYFilterZero<<<blockGridColumns,threadBlockColumns,(tile_len+hfs*2)*sizeof(float)>>>(res,src,src_w,src_h,f,hfs,tile_len);
}


void cuda_c_optXFilterClean(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len)
{
  dim3 blockGridColumns(iDivUp(src_w,tile_len),src_h);
  dim3 threadBlockColumns(tile_len,1);
  cuda_global_optXFilterClean<<<blockGridColumns,threadBlockColumns,(tile_len+hfs*2)*sizeof(float)>>>(res,src,src_w,src_h,f,hfs,tile_len);
}

void cuda_c_optYFilterClean(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len)
{
  dim3 blockGridColumns(src_w,iDivUp(src_h,tile_len));
  dim3 threadBlockColumns(1,tile_len);
  cuda_global_optYFilterClean<<<blockGridColumns,threadBlockColumns,(tile_len+hfs*2)*sizeof(float)>>>(res,src,src_w,src_h,f,hfs,tile_len);
}

void cuda_c_optXFilterReplicate(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len)
{
  dim3 blockGridColumns(iDivUp(src_w,tile_len),src_h);
  dim3 threadBlockColumns(tile_len,1);
  cuda_global_optXFilterReplicate<<<blockGridColumns,threadBlockColumns,(tile_len+hfs*2)*sizeof(float)>>>(res,src,src_w,src_h,f,hfs,tile_len);
}

void cuda_c_optYFilterReplicate(float *res, const float *src, const int src_w, const int src_h, const float *f, const int hfs, const int tile_len)
{
  dim3 blockGridColumns(src_w,iDivUp(src_h,tile_len));
  dim3 threadBlockColumns(1,tile_len);
  cuda_global_optYFilterReplicate<<<blockGridColumns,threadBlockColumns,(tile_len+hfs*2)*sizeof(float)>>>(res,src,src_w,src_h,f,hfs,tile_len);
}


void cuda_2_debayer(float *src,float3_t *dptr,int w, int h, int tile_width, int tile_height)
{
  dim3 blockGridRows(iDivUp(w,tile_width),iDivUp(h,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_kernel_debayer<<<blockGridRows,threadBlockRows>>>(src,dptr,w,h,tile_width,tile_height);
}

void cuda_c_crop(const float *src, float *res, int srcw, int srch, int startx, int starty, int endx, int endy, int maxx,int maxy, int tile_width, int tile_height)
{
  dim3 blockGridRows(iDivUp(srcw,tile_width),iDivUp(srch,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_global_crop<<<blockGridRows,threadBlockRows>>>(src,res,srcw,srch,startx,starty,endx,endy,maxx,maxy,tile_width,tile_height);
}

void cuda_c_shiftImage(const float *src, float *dst, int w, int h, float deltax, float deltay, int tile_width, int tile_height)
{
  dim3 blockGridRows(iDivUp(w,tile_width),iDivUp(h,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_global_shiftImage<<<blockGridRows,threadBlockRows,(tile_width*tile_height+tile_height+tile_width+1)*sizeof(float)>>>(src,dst,w,h,deltax,deltay,tile_width,tile_height);
}

void cuda_c_inplacePaste(float *dst, const float *img, int w, int h, int iw, int ih, int dx, int dy, int tile_width, int tile_height)
{
  dim3 blockGridRows(iDivUp(iw,tile_width),iDivUp(ih,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_global_inplacePaste<<<blockGridRows,threadBlockRows>>>(dst,img, w,h,iw,ih,dx,dy,tile_width,tile_height);
}

void cuda_c_inplacePasteRGB(float3_t *dst, const float3_t *img, int w, int h, int iw, int ih, int dx, int dy, int tile_width, int tile_height)
{
  dim3 blockGridRows(iDivUp(iw,tile_width),iDivUp(ih,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_global_inplacePasteRGB<<<blockGridRows,threadBlockRows>>>(dst,img, w,h,iw,ih,dx,dy,tile_width,tile_height);
}

void cuda_c_inplaceOverlay(float *dst, const float *img, int w, int h, int iw, int ih, int dx, int dy, int tile_width, int tile_height)
{
  dim3 blockGridRows(iDivUp(iw,tile_width),iDivUp(ih,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_global_inplaceOverlay<<<blockGridRows,threadBlockRows>>>(dst,img, w,h,iw,ih,dx,dy,tile_width,tile_height);
}

void cuda_c_inplaceOverlayRGB(float3_t *dst, const float3_t *img, int w, int h, int iw, int ih, int dx, int dy, int tile_width, int tile_height)
{
  dim3 blockGridRows(iDivUp(iw,tile_width),iDivUp(ih,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_global_inplaceOverlayRGB<<<blockGridRows,threadBlockRows>>>(dst,img, w,h,iw,ih,dx,dy,tile_width,tile_height);
}

void cuda_c_inertiaMap(float_t *dst, float s, float r_inv, int px, int py, int tile_width, int tile_height, int w, int h)
{
  dim3 blockGridRows(iDivUp(w,tile_width),iDivUp(h,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_global_inertiaMap<<<blockGridRows,threadBlockRows>>>(dst,s,r_inv,px,py,tile_width,tile_height,w,h);
}

void cuda_c_inhibitionMap(float *dst, float factorOld, float factorNew, float radius, int px, int py, int tile_width, int tile_height, int w, int h)
{
  dim3 blockGridRows(iDivUp(w,tile_width),iDivUp(h,tile_height));
  dim3 threadBlockRows(tile_width,tile_height);
  cuda_global_inhibitionMap<<<blockGridRows,threadBlockRows>>>(dst,factorOld,factorNew,radius,px,py,tile_width,tile_height,w,h);
}



