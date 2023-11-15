//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include <stdio.h>
#include "CUDA/cutil.h"
#include "cudaImage.h"


int iDivUp(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

int iDivDown(int a, int b) {
  return a / b;
}

int iAlignUp(int a, int b) {
  return (a % b != 0) ?  (a - a % b + b) : a;
}

int iAlignDown(int a, int b) {
  return a - a % b;
}

double AllocCudaImage(CudaImage *img, int w, int h, int p, bool host, bool dev)
{
  int sz = sizeof(float)*p*h;
  img->width = w;
  img->height = h;
  img->pitch = p;
  img->h_data = NULL;
  if (host) {
#ifdef VERBOSE
    printf("Allocating host data...\n");
#endif
    //img->h_data = (float *)malloc(sz);
    CUDA_SAFE_CALL(cudaMallocHost((void **)&img->h_data, sz));
  }
  img->d_data = NULL;
  if (dev) {
#ifdef VERBOSE
    printf("Allocating device data...\n");
#endif
    CUDA_SAFE_CALL(cudaMalloc((void **)&img->d_data, sz));
    if (img->d_data==NULL) 
      printf("Failed to allocate device data\n");
  }
  img->t_data = NULL;
  return 0.0;
}

double FreeCudaImage(CudaImage *img)
{
  if (img->d_data!=NULL) {
#ifdef VERBOSE
    printf("Freeing device data...\n");
#endif
    CUDA_SAFE_CALL(cudaFree(img->d_data));
  }
  img->d_data = NULL;
  if (img->h_data!=NULL) {
#ifdef VERBOSE
    printf("Freeing host data...\n");
#endif
    //free(img->h_data);
    CUDA_SAFE_CALL(cudaFreeHost(img->h_data));
  }
  img->h_data = NULL;
  if (img->t_data!=NULL) {
#ifdef VERBOSE
    printf("Freeing texture data...\n");
#endif
    CUDA_SAFE_CALL(cudaFreeArray((cudaArray *)img->t_data));
  }
  img->t_data = NULL;
  return 0;
}

double Download(CudaImage *img)
{
  if (img->d_data!=NULL && img->h_data!=NULL) 
    CUDA_SAFE_CALL(cudaMemcpy(img->d_data, img->h_data, 
			      sizeof(float)*img->pitch*img->height, cudaMemcpyHostToDevice));
  return 0;
}

double Readback(CudaImage *img, int w, int h)
{
  int p = sizeof(float)*img->pitch;
  w = sizeof(float)*(w<0 ? img->width : w);
  h = (h<0 ? img->height : h); 
  CUDA_SAFE_CALL(cudaMemcpy2D(img->h_data, p, img->d_data, p, 
			      w, h, cudaMemcpyDeviceToHost));
  //CUDA_SAFE_CALL(cudaMemcpy(img->h_data, img->d_data, 
  //  sizeof(float)*img->pitch*img->height, cudaMemcpyDeviceToHost));
  return 0;
}

double InitTexture(CudaImage *img)
{
  cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>(); 
  CUDA_SAFE_CALL(cudaMallocArray((cudaArray **)&img->t_data, &t_desc, 
				 img->pitch, img->height)); 
#ifdef VERBOSE
  printf("InitTexture(%d, %d)\n", img->pitch, img->height); 
#endif
  if (img->t_data==NULL)
    printf("Failed to allocated texture data\n");
  return 0;
}
 
double CopyToTexture(CudaImage *src, CudaImage *dst, bool host)
{
  if (dst->t_data==NULL) {
    printf("Error CopyToTexture: No texture data\n");
    return 0.0;
  }
  if ((!host || src->h_data==NULL) && (host || src->d_data==NULL)) {
    printf("Error CopyToTexture: No source data\n");
    return 0.0;
  }
  if (host)
    {CUDA_SAFE_CALL(cudaMemcpyToArray((cudaArray *)dst->t_data, 0, 0,
				      src->h_data, sizeof(float)*src->pitch*dst->height, 
				      cudaMemcpyHostToDevice));}
  else
    {CUDA_SAFE_CALL(cudaMemcpyToArray((cudaArray *)dst->t_data, 0, 0, 
				      src->d_data, sizeof(float)*src->pitch*dst->height, 
				      cudaMemcpyDeviceToDevice));}
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  return 0;
}
