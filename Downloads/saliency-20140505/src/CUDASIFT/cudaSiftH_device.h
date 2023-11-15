#ifndef CUDASIFTH_DEVICE_H
#define CUDASIFTH_DEVICE_H

#include "cudaSiftD.h"
#include "cudaImage.h"

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

double Find3DMinMax(CudaArray *minmax, CudaImage *data1, CudaImage *data2,
                    CudaImage *data3, float thresh, int maxPts);
double UnpackPointers(CudaArray *minmax, int maxPts, int *ptrs, int *numPts);
double ComputePositions(CudaImage *data1, CudaImage *data2, CudaImage *data3,
                        int *h_ptrs, CudaArray *sift, int numPts, int maxPts, float scale,
                        float factor);
double RemoveEdgePoints(CudaArray *sift, int *initNumPts, int maxPts,
                        float edgeLimit);
double ComputeOrientations(CudaImage *img, int *h_ptrs, CudaArray *sift,
                           int numPts, int maxPts);
double SecondOrientations(CudaArray *sift, int *initNumPts, int maxPts);
double ExtractSiftDescriptors(CudaImage *img, CudaArray *sift,
                              CudaArray *desc, int numPts, int maxPts);
double AddSiftData(SiftData *data, float *d_sift, float *d_desc,
                   int numPts, int maxPts, float subsampling);

////////////////////////////////////////////////////////////////////
// Templated filter funtions
////////////////////////////////////////////////////////////////////
template<int RADIUS>
double SeparableFilter(CudaImage *dataA, CudaImage *dataB,
                       CudaImage *temp, float *h_Kernel)
{
  unsigned int width = dataA->width;
  unsigned int height = dataA->height;
  float *d_DataA = dataA->d_data;
  float *d_DataB = dataB->d_data;
  float *d_Temp = temp->d_data;
  if (d_DataA==NULL || d_DataB==NULL || d_Temp==NULL) {
    printf("SeparableFilter: missing data\n");
    return 0.0;
  }
  /* unsigned int hTimer; */
  /* CUT_SAFE_CALL(cutCreateTimer(&hTimer)); */
  const unsigned int kernelSize = (2*RADIUS+1)*sizeof(float);
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Kernel, h_Kernel, kernelSize));
  /* CUT_SAFE_CALL(cutResetTimer(hTimer)); */
  /* CUT_SAFE_CALL(cutStartTimer(hTimer)); */

#if 1
  dim3 blockGridRows(iDivUp(width, ROW_TILE_W), height);
  dim3 threadBlockRows(WARP_SIZE + ROW_TILE_W + RADIUS);
  ConvRowGPU<RADIUS><<<blockGridRows, threadBlockRows>>>(d_Temp,
                                                         d_DataA, width, height); //%%%%
  CUT_CHECK_ERROR("ConvRowGPU() execution failed\n");
  CUDA_SAFE_CALL(cudaThreadSynchronize());
#endif
#if 1
  dim3 blockGridColumns(iDivUp(width, COLUMN_TILE_W),
                        iDivUp(height, COLUMN_TILE_H));
  dim3 threadBlockColumns(COLUMN_TILE_W, 8);
  ConvColGPU<RADIUS><<<blockGridColumns, threadBlockColumns>>>(d_DataB,
                                                               d_Temp, width, height, COLUMN_TILE_W*8, width*8);
  CUT_CHECK_ERROR("ConvColGPU() execution failed\n");
  CUDA_SAFE_CALL(cudaThreadSynchronize());
#endif
/* CUT_SAFE_CALL(cutStopTimer(hTimer)); */
/*   double gpuTime = cutGetTimerValue(hTimer); */
/* #ifdef VERBOSE */
/*   printf("SeparableFilter time = %.2f msec\n", gpuTime); */
/* #endif */
/*   CUT_SAFE_CALL(cutDeleteTimer(hTimer)); */
/*   return gpuTime; */
  return 0.0;
}

template<int RADIUS>
double LowPass(CudaImage *dataB, CudaImage *dataA, CudaImage *temp, double var)
{
  float kernel[2*RADIUS+1];
  float kernelSum = 0.0f;
  for (int j=-RADIUS;j<=RADIUS;j++) {
    kernel[j+RADIUS] = (float)expf(-(double)j*j/2.0/var);
    kernelSum += kernel[j+RADIUS];
  }
  for (int j=-RADIUS;j<=RADIUS;j++)
    kernel[j+RADIUS] /= kernelSum;
  return SeparableFilter<RADIUS>(dataA, dataB, temp, kernel);
}

#endif
