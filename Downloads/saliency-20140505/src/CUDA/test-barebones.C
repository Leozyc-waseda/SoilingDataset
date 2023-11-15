#include "Raster/Raster.H"
#include "wrap_c_cuda.h"
#include "CUDA/cutil.h"
#include "CUDA/CudaColorOps.H"
#include "CUDA/CudaLowPass.H"
#include "CUDA/CudaDevices.H"
#include "CUDA/CudaImageSet.H"
#include "CUDA/CudaImageSetOps.H"
#include "CUDA/CudaPyramidOps.H"
#include "CUDA/CudaMathOps.H"
#include "CUDA/CudaFilterOps.H"
#include "CUDA/CudaKernels.H"
#include "CUDA/CudaShapeOps.H"
#include "CUDA/CudaRandom.H"
#include "CUDA/CudaTransforms.H"
#include "CUDA/CudaConvolutions.H"
#include "CUDA/CudaNorm.H"

// relative feature weights:
#define IWEIGHT 0.7
#define CWEIGHT 1.0
#define OWEIGHT 1.0
#define FWEIGHT 1.0
#define SWEIGHT 0.7
#define COLOR_THRESH 0.1F

#define delta_min  3
#define delta_max  4
#define level_min  0
#define level_max  2
#define maxdepth   (level_max + delta_max + 1)
#define sml        3
#define normtyp    VCXNORM_FANCY

void postChannel(CudaImage<float> curImage, PyramidType ptyp, float orientation, float weight, CudaImage<float>& outmap)
{
  // compute pyramid:
  CudaImageSet<float> pyr =
    cudaBuildPyrGeneric(curImage, 0, maxdepth, ptyp, orientation);

  int dev = curImage.getMemoryDevice();
  MemoryPolicy mp = curImage.getMemoryPolicy();
  CudaImage<float> randBuf;
  // alloc conspicuity map and clear it:
  CudaImage<float> cmap(pyr[sml].getDims(), ZEROS, mp,dev);

  // intensities is the max-normalized weighted sum of IntensCS:
  for (int delta = delta_min; delta <= delta_max; delta ++)
    for (int lev = level_min; lev <= level_max; lev ++)
      {
        CudaImage<float> tmp = cudaCenterSurround(pyr, lev, lev + delta, true);
        //CudaImage<float> tmp = CudaImage<float>(cmap.getWidth(),cmap.getHeight(),ZEROS,mp,dev);
        tmp = cudaDownSize(tmp, cmap.getWidth(), cmap.getHeight());
        tmp = cudaMaxNormalize(tmp, MAXNORMMIN, MAXNORMMAX, normtyp);
        cmap += tmp;
      }

  cudaInplaceAddBGnoise(cmap, 25.0F,randBuf);

  if (normtyp == VCXNORM_MAXNORM)
    cmap = cudaMaxNormalize(cmap, MAXNORMMIN, MAXNORMMAX, normtyp);
  else
    cmap = cudaMaxNormalize(cmap, 0.0f, 0.0f, normtyp);

  // multiply by conspicuity coefficient:
  if (weight != 1.0F) cmap *= weight;

  // Add to saliency map:
  if (outmap.initialized()) outmap += cmap;
  else outmap = cmap;
}


void calcSaliency(const CudaImage<PixRGB<float> > & colImage, CudaImage<float>& outmap)
{
  int dev = colImage.getMemoryDevice();
  MemoryPolicy mp = colImage.getMemoryPolicy();
  CudaImage<float> rg,by,curImage,curLum,prevLum;
  // Calc intensity
  curLum = cudaLuminance(colImage);
  postChannel(curLum,Gaussian5,0.0F,IWEIGHT,outmap);
  // Calc rg and by opponent
  cudaGetRGBY(colImage,rg,by,COLOR_THRESH);
  postChannel(rg,Gaussian5,0.0F,CWEIGHT,outmap);
  postChannel(by,Gaussian5,0.0F,CWEIGHT,outmap);
  // Calc orientation from intensity
  for(float orientation=0.0F;orientation < 180.0F;orientation+=45.0F)
    postChannel(curLum,Oriented5,orientation,OWEIGHT,outmap);
  // Calc flicker
  if (prevLum.initialized() == false)
    {
      prevLum = curLum;
      curImage = CudaImage<float>(curLum.getDims(), ZEROS, mp, dev);
    }
  else
    {
      curImage = curLum - prevLum;
      prevLum = curLum;
    }
  postChannel(curImage,Gaussian5,0.0F,FWEIGHT,outmap);
}

int main(int argc, char **argv)
{
  if (argc != 3) LFATAL("USAGE: %s <input.pgm> <number of calculations>", argv[0]);
  LINFO("Reading: %s", argv[1]);
  Image<PixRGB<float> > img = Raster::ReadRGB(argv[1]);
  int numIter = atoi(argv[2]);
  if(numIter <0) LFATAL("Must have positive number, not [%d]",numIter);
  CudaImage<PixRGB<float> > cimg = CudaImage<PixRGB<float> >(img,GLOBAL_DEVICE_MEMORY,0);
   CudaImage<float> outmap;
   for(int i=0;i<numIter;i++)
    {
      calcSaliency(cimg,outmap);
      outmap.clear();
    }
  return 0;
}
