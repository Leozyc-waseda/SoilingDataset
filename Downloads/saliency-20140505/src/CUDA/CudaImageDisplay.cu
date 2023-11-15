/*
Simple image display kernel that uses CUDA and OpenGL
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "CudaImageDisplay.h"
//#include "CUDA/CudaImage.H"
////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
float Max(float x, float y){
  return (x > y) ? x : y;
}

float Min(float x, float y){
  return (x < y) ? x : y;
}


__device__ float lerpf(float a, float b, float c){
  return a + (b - a) * c;
}

__device__ float vecLen(float4 a, float4 b){
  return (
	  (b.x - a.x) * (b.x - a.x) +
	  (b.y - a.y) * (b.y - a.y) +
	  (b.z - a.z) * (b.z - a.z)
	  );
}


__device__ unsigned int make_color(unsigned char r, unsigned char g, unsigned char b, unsigned char a){
  return (unsigned int) (
    ((int)(a) << 24) |
    ((int)(b) << 16) |
    ((int)(g) <<  8) |
    ((int)(r) <<  0));
}

__device__ unsigned int make_color(float r, float g, float b, float a){
  return (unsigned int) (
    ((int)(a) << 24) |
    ((int)(b) << 16) |
    ((int)(g) <<  8) |
    ((int)(r) <<  0));
}



////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//Texture reference and channel descriptor for image texture
texture<uchar4, 2, cudaReadModeElementType> texImage;
//texture<float4, 2, cudaReadModeElementType> copyImage;
cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();
//cudaChannelFormatDesc float3tex = cudaCreateChannelDesc(32,32,32,0,cudaChannelFormatKindFloat);

//CUDA array descriptor
cudaArray *a_Src1;
cudaArray *a_Src2;
// We have dptr in cuda memory and we need a way to map dptr data to a_Src
// Need to convert memory to array and then map to texture or directly 
////////////////////////////////////////////////////////////////////////////////
// Filtering kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void Copy(unsigned int *dst, int imageW, int imageH){
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  //Add half of a texel to always address exact texel centers
  const float x = (float)ix + 0.5f;
  const float y = (float)iy + 0.5f;
 
  if(ix < imageW && iy < imageH){
    
    uchar4 result;
    result = tex2D(texImage,x,y);
    dst[imageW * iy + ix] = make_color(result.x, result.y, result.z, 255.0);
  }
}


extern "C" 
void cuda_Copy(unsigned int *d_dst, int imageW, int imageH)
{
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
  dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

  Copy<<<grid, threads>>>(d_dst, imageW, imageH);
}



extern "C"
cudaError_t CUDA_Bind2TextureArray(int index)
{ 
  cudaError_t error;
  if(index==0)
  error = cudaBindTextureToArray(texImage,a_Src1);
  if(index==1)
  error = cudaBindTextureToArray(texImage,a_Src2);
  return error;
}

extern "C"
cudaError_t CUDA_UnbindTexture(int index)
{
  cudaError_t error;
  if(index==0)
  error =  cudaUnbindTexture(texImage);
  if(index==1)
  error =  cudaUnbindTexture(texImage);
  return error;
}

extern "C"
cudaError_t CUDA_MallocArray(unsigned int * src, int imageW, int imageH,int index)
{  
  cudaError_t error;
   if(index==0)
    { 
   error = cudaMallocArray(&a_Src1, &uchar4tex, imageW, imageH);

   error = cudaMemcpyToArray(a_Src1,0,0,
	  		    src, imageW * imageH * sizeof(unsigned int), cudaMemcpyDeviceToDevice
			    );
    }  
   if(index==1)
    {
      error = cudaMallocArray(&a_Src2, &uchar4tex, imageW, imageH);

      error = cudaMemcpyToArray(a_Src2,0,0,
	  		    src, imageW * imageH * sizeof(unsigned int), cudaMemcpyDeviceToDevice
			    );
    } 
 return error;
}


extern "C"
cudaError_t CUDA_UpdateArray(unsigned int * src, int imageW, int imageH,int index)
{  
  cudaError_t error;
   if(index==0)
    { 
      error = cudaMemcpyToArray(a_Src1,0,0,
	  		    src, imageW * imageH * sizeof(unsigned int), cudaMemcpyDeviceToDevice
			    );
    }  
   if(index==1)
    {

      error = cudaMemcpyToArray(a_Src2,0,0,
	  		    src, imageW * imageH * sizeof(unsigned int), cudaMemcpyDeviceToDevice
			    );
    } 
 return error;
}


__global__ void change_float_uint(float3_t* src,unsigned int *dst,int tile_length,int size)
{
  
  
   const int ix =  blockIdx.x * tile_length + threadIdx.x;
   if(ix<size)
     {
        dst[ix] = (unsigned int) make_color(src[ix].p[0],src[ix].p[1],src[ix].p[2],255.0F);
     }
}

extern "C"
void CUDA_convert_float_uint(float3_t* src,unsigned int *dst,int tile_length,int size)
{
  dim3 threads(tile_length);
  dim3 grid(iDivUp(size,tile_length));
  change_float_uint<<<grid, threads>>>(src,dst,tile_length,size);
}


extern "C"
cudaError_t CUDA_FreeArray()
{ 
  cudaError_t error;
  error = cudaFreeArray(a_Src1);
  error = cudaFreeArray(a_Src2);
  return error;     
}

