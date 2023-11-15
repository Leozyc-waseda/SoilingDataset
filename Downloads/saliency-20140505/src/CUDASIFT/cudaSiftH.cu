//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include <stdio.h>
#include <string.h>
#include "CUDA/cutil.h"

#undef VERBOSE

#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftD_device.h"
#include "cudaSiftH.h"
#include "cudaSiftH_device.h"

#include "cudaSiftDcu"

void InitCuda(int argc, char **argv)
{
  //CUT_DEVICE_INIT(argc, argv); //Doesn't like iLab command line
  CUDA_SAFE_CALL(cudaSetDevice(3)); //Use non-display gpu
}

void ExtractSift(SiftData *siftData, CudaImage *img, int numLayers, 
		 int numOctaves, double initBlur, float thresh, float subsampling) 
{
  int w = img->width;
  int h = img->height;
  int p = img->pitch;
  if (numOctaves>1) {
    CudaImage subImg;
    AllocCudaImage(&subImg, w/2, h/2, p/2, false, true);
    ScaleDown(&subImg, img, 0.5f);
    float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
    ExtractSift(siftData, &subImg, numLayers, numOctaves-1, totInitBlur, 
		thresh, subsampling*2.0f);
    FreeCudaImage(&subImg);
  }
  ExtractSiftOctave(siftData, img, numLayers, initBlur, thresh, subsampling);
}

void ExtractSiftOctave(SiftData *siftData, CudaImage *img, int numLayers, 
		       double initBlur, float thresh, float subsampling)
{
  const double baseBlur = 1.0;
  const int maxPts = 512;
  int w = img->width; 
  int h = img->height;
  int p = img->pitch;
  // Images and arrays
  CudaImage blurImg[2];
  CudaImage diffImg[3];
  CudaImage tempImg;
  CudaImage textImg;
  CudaArray sift; // { xpos, ypos, scale, strength, edge, orient1, orient2 };
  CudaArray desc;
  CudaArray minmax;

  AllocCudaImage(&blurImg[0], w, h, p, false, true);   
  AllocCudaImage(&blurImg[1], w, h, p, false, true); 
  AllocCudaImage(&diffImg[0], w, h, p, false, true);
  AllocCudaImage(&diffImg[1], w, h, p, false, true);
  AllocCudaImage(&diffImg[2], w, h, p, false, true);
  AllocCudaImage(&tempImg, w, h, p, false, true);
  AllocCudaImage(&minmax, w, iDivUp(h,32), w, true, true);
  AllocCudaImage(&sift, maxPts, 7, maxPts, true, true);
  AllocCudaImage(&desc, 128, maxPts, 128, true, true);
  int *ptrs = 0; 
  //ptrs = (int *)malloc(sizeof(int)*maxPts);
  //Now, allocates page locked memory versus above.
  CUDA_SAFE_CALL(cudaMallocHost((void **)&ptrs, sizeof(int)*maxPts));
  AllocCudaImage(&textImg, w, h, p, false, false);     
  InitTexture(&textImg);
  CUT_CHECK_ERROR("Memory allocation failed\n");
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  double var = baseBlur*baseBlur - initBlur*initBlur;
  double gpuTime = 0.0, gpuTimeDoG = 0.0, gpuTimeSift = 0.0;
  int totPts = 0;

  if (var>0.0)
    gpuTime += LowPass<2>(&blurImg[0], img, &tempImg, var);
  for (int i=0;i<numLayers+2;i++) {
    double pSigma = baseBlur*pow(2.0, (double)(i-1)/numLayers);
    double oSigma = baseBlur*pow(2.0, (double)i/numLayers);
    double nSigma = baseBlur*pow(2.0, (double)(i+1)/numLayers);
    double var = nSigma*nSigma - oSigma*oSigma;
    if (i<=1) { 
      gpuTime += LowPass<7>(&blurImg[(i+1)%2], &blurImg[i%2], &tempImg, var);
      gpuTime += Subtract(&diffImg[(i+2)%3], &blurImg[i%2], &blurImg[(i+1)%2]);
    } else {
      gpuTimeDoG += LowPass<7>(&blurImg[(i+1)%2], &blurImg[i%2], &tempImg,var);
      gpuTimeDoG += Subtract(&diffImg[(i+2)%3], &blurImg[i%2], 
			     &blurImg[(i+1)%2]);
      gpuTimeDoG += Find3DMinMax(&minmax, &diffImg[(i+2)%3], &diffImg[(i+1)%3],
				 &diffImg[i%3], thresh, maxPts);
      int numPts = 0;
      gpuTimeSift += UnpackPointers(&minmax, maxPts, ptrs, &numPts);
      if (numPts>0) {
	gpuTimeDoG += CopyToTexture(&blurImg[i%2], &textImg, false);
	gpuTimeSift += ComputePositions(&diffImg[(i+2)%3], &diffImg[(i+1)%3], 
					&diffImg[i%3], ptrs, &sift, numPts, maxPts,
					pSigma, 1.0f/numLayers);
	gpuTimeSift += RemoveEdgePoints(&sift, &numPts, maxPts, 10.0f);
	if(numPts>0){ // Need to check again?
	  gpuTimeSift += ComputeOrientations(&blurImg[i%2], ptrs, &sift, 
					   numPts, maxPts);
	  gpuTimeSift += SecondOrientations(&sift, &numPts, maxPts);
	  gpuTimeSift += ExtractSiftDescriptors(&textImg, &sift, &desc, 
					      numPts, maxPts);
	  gpuTimeSift += AddSiftData(siftData, sift.d_data, desc.d_data, 
				   numPts, maxPts, subsampling);
	}
      }
      totPts += numPts; 
    }
  }

#if 0
  Readback(&diffImg[(2+2)%3]);
  float *imd = diffImg[(2+2)%3].h_data;
  for (int i=0;i<w*h;i++) imd[i] = 8*imd[i] + 128;
  CUT_SAFE_CALL(cutSavePGMf("data/limg_test2.pgm", 
			    diffImg[(2+2)%3].h_data, w, h));
  Readback(&diffImg[(2+1)%3]);
  imd = diffImg[(2+1)%3].h_data;
  for (int i=0;i<w*h;i++) imd[i] = 8*imd[i] + 128;
  CUT_SAFE_CALL(cutSavePGMf("data/limg_test1.pgm", 
			    diffImg[(2+1)%3].h_data, w, h));
  Readback(&diffImg[(2+0)%3]);
  imd = diffImg[(2+0)%3].h_data;
  for (int i=0;i<w*h;i++) imd[i] = 8*imd[i] + 128;
  CUT_SAFE_CALL(cutSavePGMf("data/limg_test0.pgm", 
			    diffImg[(2+0)%3].h_data, w, h));
#endif

  CUDA_SAFE_CALL(cudaThreadSynchronize());
  FreeCudaImage(&textImg);
  //free(ptrs);
  CUDA_SAFE_CALL(cudaFreeHost(ptrs));
  FreeCudaImage(&desc);
  FreeCudaImage(&sift);
  FreeCudaImage(&minmax);
  FreeCudaImage(&tempImg);
  FreeCudaImage(&diffImg[2]);
  FreeCudaImage(&diffImg[1]);
  FreeCudaImage(&diffImg[0]);
  FreeCudaImage(&blurImg[1]);
  FreeCudaImage(&blurImg[0]);

}

void InitSiftData(SiftData *data, int num, bool host, bool dev)
{
  data->numPts = 0;
  data->maxPts = num;
  int sz = sizeof(SiftPoint)*num;
  data->h_data = NULL;
  if (host)
    //data->h_data = (SiftPoint *)malloc(sz);
    CUDA_SAFE_CALL(cudaMallocHost((void **)&data->h_data, sz));
  data->d_data = NULL;
  if (dev)
    CUDA_SAFE_CALL(cudaMalloc((void **)&data->d_data, sz));
}

void FreeSiftData(SiftData *data)
{
  if (data->d_data!=NULL)
    CUDA_SAFE_CALL(cudaFree(data->d_data));
  data->d_data = NULL;
  if (data->h_data!=NULL)
    //free(data->h_data);
    CUDA_SAFE_CALL(cudaFreeHost(data->h_data));
  data->numPts = 0;
  data->maxPts = 0;
}

double AddSiftData(SiftData *data, float *d_sift, float *d_desc, 
		   int numPts, int maxPts, float subsampling)
{
  int newNum = data->numPts + numPts;
  if (data->maxPts<(data->numPts+numPts)) {
    int newMaxNum = 2*data->maxPts;
    while (newNum>newMaxNum)
      newMaxNum *= 2;
    if (data->h_data!=NULL) {
      SiftPoint *h_data = 0;
      //h_data = (SiftPoint *)malloc(sizeof(SiftPoint)*newMaxNum);
      CUDA_SAFE_CALL(cudaMallocHost((void**)&h_data, sizeof(SiftPoint)*newMaxNum));
      memcpy(h_data, data->h_data, sizeof(SiftPoint)*data->numPts);
      //free(data->h_data);
      //CUDA_SAFE_CALL(cudaFree(data->h_data)); //Was incorrect, should be cudaFreeHost.
      CUDA_SAFE_CALL(cudaFreeHost(data->h_data));
      data->h_data = h_data;
    }
    if (data->d_data!=NULL) {
      SiftPoint *d_data = NULL;
      CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, sizeof(SiftPoint)*newMaxNum));
      CUDA_SAFE_CALL(cudaMemcpy(d_data, data->d_data, 
				sizeof(SiftPoint)*data->numPts, cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaFree(data->d_data));
      data->d_data = d_data;
    }
    data->maxPts = newMaxNum;
  }
  int pitch = sizeof(SiftPoint);
  float *buffer;
  //buffer = (float *)malloc(sizeof(float)*3*numPts);
  CUDA_SAFE_CALL(cudaMallocHost((void**)&buffer, sizeof(float)*3*numPts));
  int bwidth = sizeof(float)*numPts;
  CUDA_SAFE_CALL(cudaMemcpy2D(buffer, bwidth, d_sift, sizeof(float)*maxPts, 
			      bwidth, 3, cudaMemcpyDeviceToHost));
  for (int i=0;i<3*numPts;i++) 
    buffer[i] *= subsampling;
  CUDA_SAFE_CALL(cudaMemcpy2D(d_sift, sizeof(float)*maxPts, buffer, bwidth, 
			      bwidth, 3, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  if (data->h_data!=NULL) {
    float *ptr = (float*)&data->h_data[data->numPts];
    for (int i=0;i<6;i++)
      CUDA_SAFE_CALL(cudaMemcpy2D(&ptr[i], pitch, &d_sift[i*maxPts], 4, 4, 
				  numPts, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy2D(&ptr[16], pitch, d_desc, sizeof(float)*128, 
				sizeof(float)*128, numPts, cudaMemcpyDeviceToHost));
  }
  if (data->d_data!=NULL) {
    float *ptr = (float*)&data->d_data[data->numPts];
    for (int i=0;i<6;i++)
      CUDA_SAFE_CALL(cudaMemcpy2D(&ptr[i], pitch, &d_sift[i*maxPts], 4, 4, 
				  numPts, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy2D(&ptr[16], pitch, d_desc, sizeof(float)*128, 
				sizeof(float)*128, numPts, cudaMemcpyDeviceToDevice));
  }
  data->numPts = newNum;
  CUDA_SAFE_CALL(cudaFreeHost(buffer));
  //free(buffer);
  return 0;
}

void PrintSiftData(SiftData *data)
{
  SiftPoint *h_data = data->h_data;
  if (data->h_data==NULL) {
    //h_data = (SiftPoint *)malloc(sizeof(SiftPoint)*data->maxPts);
    CUDA_SAFE_CALL(cudaMallocHost((void **)&h_data, sizeof(SiftPoint)*data->maxPts));
    CUDA_SAFE_CALL(cudaMemcpy(h_data, data->d_data, 
			      sizeof(SiftPoint)*data->numPts, cudaMemcpyDeviceToHost));
    data->h_data = h_data;
  }
  for (int i=0;i<data->numPts;i++) {
    printf("xpos         = %.2f\n", h_data[i].xpos);
    printf("ypos         = %.2f\n", h_data[i].ypos);
    printf("scale        = %.2f\n", h_data[i].scale);
    printf("sharpness    = %.2f\n", h_data[i].sharpness);
    printf("edgeness     = %.2f\n", h_data[i].edgeness);
    printf("orientation  = %.2f\n", h_data[i].orientation);
    float *siftData = (float*)&h_data[i].data;
    for (int j=0;j<8;j++) {
      if (j==0) printf("data = ");
      else printf("       ");
      for (int k=0;k<16;k++)
	if (siftData[j+8*k]<0.05)
	  printf(" .   ", siftData[j+8*k]);
	else
	  printf("%.2f ", siftData[j+8*k]);
      printf("\n");
    }
  }
  printf("Number of available points: %d\n", data->numPts);
  printf("Number of allocated points: %d\n", data->maxPts);
}

double MatchSiftData(SiftData *data1, SiftData *data2)
{
  if (data1->d_data==NULL || data2->d_data==NULL)
    return 0.0f;
  int numPts1 = data1->numPts;
  int numPts2 = data2->numPts;
  SiftPoint *sift1 = data1->d_data;
  SiftPoint *sift2 = data2->d_data;
  
  float *d_corrData; 
  int corrWidth = iDivUp(numPts2, 16)*16;
  int corrSize = sizeof(float)*numPts1*corrWidth;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_corrData, corrSize));
  dim3 blocks(numPts1, iDivUp(numPts2, 16));
  dim3 threads(16, 16); // each block: 1 points x 16 points
  MatchSiftPoints<<<blocks, threads>>>(sift1, sift2, 
				       d_corrData, numPts1, numPts2);
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  dim3 blocksMax(iDivUp(numPts1, 16));
  dim3 threadsMax(16, 16);
  FindMaxCorr<<<blocksMax, threadsMax>>>(d_corrData, sift1, sift2, 
					 numPts1, corrWidth, sizeof(SiftPoint));
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  CUT_CHECK_ERROR("MatchSiftPoints() execution failed\n");
  CUDA_SAFE_CALL(cudaFree(d_corrData));
  if (data1->h_data!=NULL) {
    float *h_ptr = &data1->h_data[0].score;
    float *d_ptr = &data1->d_data[0].score;
    CUDA_SAFE_CALL(cudaMemcpy2D(h_ptr, sizeof(SiftPoint), d_ptr,
				sizeof(SiftPoint), 5*sizeof(float), data1->numPts, 
				cudaMemcpyDeviceToHost));
  }

  return 0;
}		 
  
double FindHomography(SiftData *data, float *homography, int *numMatches, 
		      int numLoops, float minScore, float maxAmbiguity, float thresh)
{
  *numMatches = 0;
  homography[0] = homography[4] = homography[8] = 1.0f;
  homography[1] = homography[2] = homography[3] = 0.0f;
  homography[5] = homography[6] = homography[7] = 0.0f;
  if (data->d_data==NULL)
    return 0.0f;
  SiftPoint *d_sift = data->d_data;
  numLoops = iDivUp(numLoops,16)*16;
  int numPts = data->numPts;
  if (numPts<8)
    return 0.0f;
  int numPtsUp = iDivUp(numPts, 16)*16;
  float *d_coord, *d_homo;
  int *d_randPts, *h_randPts;
  int randSize = 4*sizeof(int)*numLoops;
  int szFl = sizeof(float);
  int szPt = sizeof(SiftPoint);
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_coord, 4*sizeof(float)*numPtsUp));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_randPts, randSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_homo, 8*sizeof(float)*numLoops));
  h_randPts = (int*)malloc(randSize);
  float *h_scores = (float *)malloc(sizeof(float)*numPtsUp);
  float *h_ambiguities = (float *)malloc(sizeof(float)*numPtsUp);
  CUDA_SAFE_CALL(cudaMemcpy2D(h_scores, szFl, &d_sift[0].score, szPt, 
			      szFl, numPts, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy2D(h_ambiguities, szFl, &d_sift[0].ambiguity, szPt,
			      szFl, numPts, cudaMemcpyDeviceToHost));
  int *validPts = (int *)malloc(sizeof(int)*numPts);
  int numValid = 0;
  for (int i=0;i<numPts;i++) {
    if (h_scores[i]>minScore && h_ambiguities[i]<maxAmbiguity)
      validPts[numValid++] = i;
  }
  free(h_scores);
  free(h_ambiguities);
  if (numValid>=8) {
    for (int i=0;i<numLoops;i++) {
      int p1 = rand() % numValid;
      int p2 = rand() % numValid;
      int p3 = rand() % numValid;
      int p4 = rand() % numValid;
      while (p2==p1) p2 = rand() % numValid;
      while (p3==p1 || p3==p2) p3 = rand() % numValid;
      while (p4==p1 || p4==p2 || p4==p3) p4 = rand() % numValid;
      h_randPts[i+0*numLoops] = validPts[p1];
      h_randPts[i+1*numLoops] = validPts[p2];
      h_randPts[i+2*numLoops] = validPts[p3];
      h_randPts[i+3*numLoops] = validPts[p4];
    }
    CUDA_SAFE_CALL(cudaMemcpy(d_randPts, h_randPts, randSize, 
			      cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy2D(&d_coord[0*numPtsUp], szFl, &d_sift[0].xpos, 
				szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy2D(&d_coord[1*numPtsUp], szFl, &d_sift[0].ypos, 
				szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy2D(&d_coord[2*numPtsUp], szFl, 
				&d_sift[0].match_xpos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy2D(&d_coord[3*numPtsUp], szFl, 
				&d_sift[0].match_ypos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice));
    ComputeHomographies<<<numLoops/16, 16>>>(d_coord, d_randPts, 
					     d_homo, numPtsUp);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("ComputeHomographies() execution failed\n");
    dim3 blocks(1, numLoops/TESTHOMO_LOOPS);
    dim3 threads(TESTHOMO_TESTS, TESTHOMO_LOOPS);
    TestHomographies<<<blocks, threads>>>(d_coord, d_homo, 
					  d_randPts, numPtsUp, thresh*thresh);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("TestHomographies() execution failed\n");
    CUDA_SAFE_CALL(cudaMemcpy(h_randPts, d_randPts, sizeof(int)*numLoops, 
			      cudaMemcpyDeviceToHost));
    int maxIndex = -1, maxCount = -1;
    for (int i=0;i<numLoops;i++) 
      if (h_randPts[i]>maxCount) {
	maxCount = h_randPts[i];
	maxIndex = i;
      }
    CUDA_SAFE_CALL(cudaMemcpy2D(homography, szFl, &d_homo[maxIndex], 
				sizeof(float)*numLoops, szFl, 8, cudaMemcpyDeviceToHost));
  }
  free(validPts);
  free(h_randPts);
  CUDA_SAFE_CALL(cudaFree(d_homo));
  CUDA_SAFE_CALL(cudaFree(d_randPts));
  CUDA_SAFE_CALL(cudaFree(d_coord));

  return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Host side master functions
///////////////////////////////////////////////////////////////////////////////

double LowPass5(CudaImage *res, CudaImage *data, float variance)
{
  int w = res->width;
  int h = res->height;
  if (res->d_data==NULL || data->d_data==NULL) {
    printf("LowPass5: missing data\n");
    return 0.0;
  }
  float h_Kernel[5];
  float kernelSum = 0.0f;
  for (int j=0;j<5;j++) {
    h_Kernel[j] = (float)expf(-(double)(j-2)*(j-2)/2.0/variance);      
    kernelSum += h_Kernel[j];
  }
  for (int j=0;j<5;j++)
    h_Kernel[j] /= kernelSum;  
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Kernel, h_Kernel, 5*sizeof(float)));
  dim3 blocks(iDivUp(w, LOWPASS5_DX), iDivUp(h, LOWPASS5_DY));
  dim3 threads(LOWPASS5_DX + WARP_SIZE + 2);
  LowPass5<<<blocks, threads>>>(res->d_data, data->d_data, w, h); 
  CUT_CHECK_ERROR("LowPass5() execution failed\n");
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  return 0;
}

double ScaleDown(CudaImage *res, CudaImage *data, float variance)
{
  int w = data->width;
  int h = data->height;
  if (res->d_data==NULL || data->d_data==NULL) {
    printf("ScaleDown: missing data\n");
    return 0.0;
  }
  float h_Kernel[5];
  float kernelSum = 0.0f;
  for (int j=0;j<5;j++) {
    h_Kernel[j] = (float)expf(-(double)(j-2)*(j-2)/2.0/variance);      
    kernelSum += h_Kernel[j];
  }
  for (int j=0;j<5;j++)
    h_Kernel[j] /= kernelSum;  
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Kernel, h_Kernel, 5*sizeof(float)));
  dim3 blocks(iDivUp(w, LOWPASS5_DX), iDivUp(h, LOWPASS5_DY));
  dim3 threads(LOWPASS5_DX + WARP_SIZE + 2);
  //printf("File %s, Number %d\n",__FILE__,__LINE__);
  //printf("blocks.x=%d,blocks.y=%d,threads.x=%d,res->d_data=%x, data->d_data=%x, w=%d, h=%d\n",
  //blocks.x,blocks.y, threads.x, res->d_data, data->d_data, w, h);
  ScaleDown<<<blocks, threads>>>(res->d_data, data->d_data, w, h); 
  CUT_CHECK_ERROR("ScaleDown() execution failed\n");
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  return 0;
}

double Subtract(CudaImage *res, CudaImage *dataA, CudaImage *dataB)
{    
  int w = res->width;
  int h = res->height;
  if (res->d_data==NULL || dataA->d_data==NULL || dataB->d_data==NULL) {
    printf("Subtract: missing data\n");
    return 0.0;
  }
  dim3 blocks(iDivUp(w, 16), iDivUp(h, 16));
  dim3 threads(16, 16);
  Subtract<<<blocks, threads>>>(res->d_data, dataA->d_data, dataB->d_data, 
				w, h);
  CUT_CHECK_ERROR("Subtract() execution failed\n");
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  return 0;
}

double MultiplyAdd(CudaImage *res, CudaImage *data, float constA, float constB)
{
  int w = res->width;
  int h = res->height;
  if (res->d_data==NULL || data->d_data==NULL) {
    printf("MultiplyAdd: missing data\n");
    return 0.0;
  }
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_ConstantA, &constA, sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_ConstantB, &constB, sizeof(float)));

  dim3 blocks(iDivUp(w, 16), iDivUp(h, 16));
  dim3 threads(16, 16);
  MultiplyAdd<<<blocks, threads>>>(res->d_data, data->d_data, w, h);
  CUT_CHECK_ERROR("MultiplyAdd() execution failed\n");
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  return 0;
}

double FindMinMax(CudaImage *img, float *minval, float *maxval)
{
  int w = img->width;
  int h = img->height;

  int dx = iDivUp(w, 128);
  int dy = iDivUp(h, 16);
  int sz = 2*dx*dy*sizeof(float);
  float *d_minmax = 0; 
  float *h_minmax = 0; 
  //h_minmax = (float *)malloc(sz);
  CUDA_SAFE_CALL(cudaMallocHost((void **)&h_minmax, sz)); 
  for (int i=0;i<2*dx*dy;i+=2) {
    h_minmax[i+0] = 1e6;
    h_minmax[i+1] = -1e6;
  }
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_minmax, sz));
  CUDA_SAFE_CALL(cudaMemcpy(d_minmax, h_minmax, sz, cudaMemcpyHostToDevice));

  dim3 blocks(dx, dy);
  dim3 threads(128, 1);
  FindMinMax<<<blocks, threads>>>(d_minmax, img->d_data, w, h);
  CUT_CHECK_ERROR("FindMinMax() execution failed\n");
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  CUDA_SAFE_CALL(cudaMemcpy(h_minmax, d_minmax, sz, cudaMemcpyDeviceToHost));
  *minval = 1e6;
  *maxval = -1e6;
  for (int i=0;i<2*dx*dy;i+=2) {
    if (h_minmax[i+0]<*minval) 
      *minval = h_minmax[i];
    if (h_minmax[i+1]>*maxval)  
      *maxval = h_minmax[i+1];
  }

  CUDA_SAFE_CALL(cudaFree(d_minmax));
  CUDA_SAFE_CALL(cudaFreeHost(h_minmax));
  return 0;
}
 
double Find3DMinMax(CudaArray *minmax, CudaImage *data1, CudaImage *data2, 
		    CudaImage *data3, float thresh, int maxPts)
{
  int *h_res = (int *)minmax->h_data;
  int *d_res = (int *)minmax->d_data;
  if (data1->d_data==NULL || data2->d_data==NULL || data3->d_data==NULL ||
      h_res==NULL || d_res==NULL) {
    printf("Find3DMinMax: missing data %p %p %p %p %p\n", 
	   data1->d_data, data2->d_data, data3->d_data, h_res, d_res);
    return 0.0;
  }
  int w = data1->width;
  int h = data1->height;
  float threshs[2] = { thresh, -thresh };
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_ConstantA, &threshs, 2*sizeof(float)));

  dim3 blocks(iDivUp(w, MINMAX_SIZE), iDivUp(h,32));
  dim3 threads(WARP_SIZE + MINMAX_SIZE + 1);
  Find3DMinMax<<<blocks, threads>>>(d_res, data1->d_data, data2->d_data, 
				    data3->d_data, w, h); 
  CUT_CHECK_ERROR("Find3DMinMax() execution failed\n");
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(h_res, d_res, 
			    sizeof(int)*minmax->pitch*minmax->height, cudaMemcpyDeviceToHost));
  return 0;
}

double UnpackPointers(CudaArray *minmax, int maxPts, int *ptrs, int *numPts)
{
  unsigned int *minmax_data = (unsigned int *)minmax->h_data;
  if (minmax_data==NULL || ptrs==NULL) {
    printf("UnpackPointers: missing data %p %p\n", minmax_data, ptrs);
    return 0.0;
  }
  int w = minmax->pitch;
  int h = 32*minmax->height;
  int num = 0;
  for (int y=0;y<h/32;y++) {
    for (int x=0;x<w;x++) {
      unsigned int val = minmax_data[y*w+x];
      if (val) {
	//printf("%d %d %08x\n", x, y, val);
	for (int k=0;k<32;k++) {
	  if (val&0x1 && num<maxPts)
	    ptrs[num++] = (y*32+k)*w + x;
	  val >>= 1;
	}
      }
    }
  }
  *numPts = num;
  return 0;
}

double ComputePositions(CudaImage *data1, CudaImage *data2, CudaImage *data3, 
			int *h_ptrs, CudaArray *sift, int numPts, int maxPts, float scale, 
			float factor)
{
  int w = data1->width;
  int h = data1->height;
  int *d_ptrs = 0;
  float *d_sift = sift->d_data;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_ptrs, sizeof(int)*numPts));
  CUDA_SAFE_CALL(cudaMemcpy(d_ptrs, h_ptrs, sizeof(int)*numPts, 
			    cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_ConstantA, &scale, sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_ConstantB, &factor, sizeof(float)));

  dim3 blocks(iDivUp(numPts, POSBLK_SIZE));
  dim3 threads(POSBLK_SIZE);
  ComputePositions<<<blocks, threads>>>(data1->d_data, data2->d_data, 
					data3->d_data, d_ptrs, d_sift, numPts, maxPts, w, h);
  CUT_CHECK_ERROR("ComputePositions() execution failed\n");
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  return 0;
}

double RemoveEdgePoints(CudaArray *sift, int *initNumPts, int maxPts, 
			float edgeLimit) 
{
  int numPts = *initNumPts;
  float *d_sift = sift->d_data;
  int bw = sizeof(float)*numPts;
  float *h_sift = 0; 
  //h_sift = (float *)malloc(5*bw);
  CUDA_SAFE_CALL(cudaMallocHost((void **)&h_sift, 5*bw));
  CUDA_SAFE_CALL(cudaMemcpy2D(h_sift, bw, d_sift, sizeof(float)*maxPts,  
			      bw, 5, cudaMemcpyDeviceToHost));
  int num = 0;
  for (int i=0;i<numPts;i++) 
    if (h_sift[4*numPts+i]<edgeLimit) {
      for (int j=0;j<5;j++) 
	h_sift[j*numPts+num] = h_sift[j*numPts+i];
      num ++;
    }
  CUDA_SAFE_CALL(cudaMemcpy2D(d_sift, sizeof(float)*maxPts, h_sift, bw,  
			      bw, 5, cudaMemcpyHostToDevice));
  //free(h_sift);
  CUDA_SAFE_CALL(cudaFreeHost(h_sift)); 
  *initNumPts = num;
  return 0;
}

double ComputeOrientations(CudaImage *img, int *h_ptrs, CudaArray *sift, 
			   int numPts, int maxPts)
{
  int w = img->pitch;
  int h = img->height;
  int *d_ptrs = 0;
  float *d_orient = &sift->d_data[5*maxPts];
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_ptrs, sizeof(int)*numPts));
  CUDA_SAFE_CALL(cudaMemcpy(d_ptrs, h_ptrs, sizeof(int)*numPts, 
			    cudaMemcpyHostToDevice));

  dim3 blocks(numPts);
  dim3 threads(32);
  ComputeOrientations<<<blocks, threads>>>(img->d_data, 
					   d_ptrs, d_orient, maxPts, w, h);
  CUT_CHECK_ERROR("ComputeOrientations() execution failed\n");
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  return 0;
}

double SecondOrientations(CudaArray *sift, int *initNumPts, int maxPts) 
{
  int numPts = *initNumPts;
  int numPts2 = 2*numPts;
  float *d_sift = sift->d_data;
  int bw = sizeof(float)*numPts2;
  float *h_sift = 0; 
  h_sift = (float *)malloc(7*bw);
  //CUDA_SAFE_CALL(cudaMallocHost((void **)&h_sift, 7*bw));
  CUDA_SAFE_CALL(cudaMemcpy2D(h_sift, bw, d_sift, sizeof(float)*maxPts,  
			      sizeof(float)*numPts, 7, cudaMemcpyDeviceToHost));
  int num = numPts;
  for (int i=0;i<numPts;i++) 
    if (h_sift[6*numPts2+i]>=0.0f && num<maxPts) {
      for (int j=0;j<5;j++) 
	h_sift[j*numPts2+num] = h_sift[j*numPts2+i];
      h_sift[5*numPts2+num] = h_sift[6*numPts2+i];
      h_sift[6*numPts2+num] = -1.0f;
      num ++;
    }
  CUDA_SAFE_CALL(cudaMemcpy2D(&d_sift[numPts], sizeof(float)*maxPts, 
			      &h_sift[numPts], bw, sizeof(float)*(num-numPts), 7, 
			      cudaMemcpyHostToDevice));
  free(h_sift);
  //CUDA_SAFE_CALL(cudaFreeHost(h_sift)); 
  *initNumPts = num;
  return 0;
}

double ExtractSiftDescriptors(CudaImage *img, CudaArray *sift, 
			      CudaArray *desc, int numPts, int maxPts)
{
  float *d_sift = sift->d_data, *d_desc = desc->d_data;

  tex.addressMode[0] = cudaAddressModeClamp;
  tex.addressMode[1] = cudaAddressModeClamp;
  tex.filterMode = cudaFilterModeLinear; 
  tex.normalized = false;
  CUDA_SAFE_CALL(cudaBindTextureToArray(tex, (cudaArray *)img->t_data));
   
  dim3 blocks(numPts); 
  dim3 threads(16);
  ExtractSiftDescriptors<<<blocks, threads>>>(img->d_data, 
					      d_sift, d_desc, maxPts);
  CUT_CHECK_ERROR("ExtractSiftDescriptors() execution failed\n");
  CUDA_SAFE_CALL(cudaThreadSynchronize());
  CUDA_SAFE_CALL(cudaUnbindTexture(tex));

  return 0;
}
