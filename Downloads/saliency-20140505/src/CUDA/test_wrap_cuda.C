#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/PyramidOps.H"
#include "Image/LowPass.H"
#include "Image/Pixels.H"
#include "Image/ColorOps.H"
#include "Image/MathOps.H"
#include "Image/FilterOps.H"
#include "Image/Kernels.H"
#include "Image/ShapeOps.H"
#include "Image/DrawOps.H"
#include "Image/Transforms.H"
#include "Image/Convolutions.H"
#include "Image/fancynorm.H"
#include "Image/CutPaste.H"
#include "Image/PyrBuilder.H"
#include "Raster/Raster.H"
#include "Raster/PngWriter.H"
#include "CUDA/CudaImage.H"

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
#include "CUDA/CudaDrawOps.H"
#include "CUDA/CudaRandom.H"
#include "CUDA/CudaTransforms.H"
#include "CUDA/CudaConvolutions.H"
#include "CUDA/CudaNorm.H"
#include "CUDA/CudaHmaxFL.H"
#include "CUDA/CudaCutPaste.H"
#include "CUDA/CudaPyrBuilder.H"
#include "Util/fpu.H"
#include "Util/Timer.H"
#include <cmath>

// Allowance for one particular pixel
#define ROUNDOFF_MARGIN 0.000115//0.00005
// Allow a greater tolerance for more complex operations
#define HIGHER_ROUNDOFF_MARGIN 0.0025
// Allowance for image wide bias
#define BIAS_MARGIN     0.000195//0.000001

// ######################################################################
void compareregions(const Image<float> &c, const Image<float> &g, const uint rowStart, const uint rowStop, const uint colStart, const uint colStop)
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
  for(uint j=rowStart;j<rowStop;j++)
  {
    printf("\nC[%d]: ",j);
    for(uint i=colStart;i<colStop;i++)
    {
      printf("%.3f ",c.getVal(i,j));
    }
    printf("\nG[%d]: ",j);
    for(uint i=colStart;i<colStop;i++)
    {
      printf("%.3f ",g.getVal(i,j));
    }
  }
  printf("\n");

}

void calculateCudaSaliency(const CudaImage<PixRGB<float> > &input, const int cudaDeviceNum)
{
  // Copy the input image to the CUDA device
  CudaImage<PixRGB<float> > cudainput = CudaImage<PixRGB<float> >(input,GLOBAL_DEVICE_MEMORY,cudaDeviceNum);
  CudaImage<float> red, green, blue;
  // Get the components
  cudaGetComponents(cudainput,red,green,blue);
  // Get the luminance
  CudaImage<float> lumin = cudaLuminance(cudainput);
}


void printImage(const Image<float> &in)
{
  int cnt=0;
  printf("******* IMAGE *********\n");
  for(int i=0;i<in.getWidth();i++)
  {
    for(int j=0;j<in.getHeight();j++)
    {
      printf("%g ",in.getVal(i,j));
      cnt++;
      if(cnt==30)
      {
        printf("\n");
        cnt=0;
      }
    }
  }
  printf("\n");
}

void writeImage(const Image<float> &in)
{
  PngWriter::writeGray(in,std::string("tmp.png"));
}

void printImages(const Image<float> &in1, const Image<float> &in2)
{
  int w,h;
  w = in1.getWidth(); h = in2.getHeight();
  if(in2.getWidth() != w || in2.getHeight() != h)
    LFATAL("Cannot compare two different image sizes im1[%dx%d] im2[%dx%d]",
           in1.getWidth(),in1.getHeight(),in2.getWidth(),in2.getHeight());
  for(int i=0;i<in1.getWidth();i++)
  {
    for(int j=0;j<in1.getHeight();j++)
    {
      printf("At [%d,%d] Im1 = %f Im2 = %f\n",i,j,in1.getVal(i,j),in2.getVal(i,j));
    }
  }
}

void testDiff(const float in1, const float in2, float rounderr=ROUNDOFF_MARGIN)
{
  float diff = in1-in2;
  bool acceptable = fabs(diff) < rounderr;
  LINFO("%s: %ux%u floats, #1 - #2: diff = [%f]",
        in1 == in2 ? "MATCH" : (acceptable ? "ACCEPT" : "FAIL"), 1, 1, diff);
}

void testDiff(const Image<float> in1, const Image<float> in2, float rounderr=ROUNDOFF_MARGIN)
{
  Image<float> diff = in1-in2;
  Point2D<int> p;
  float mi, ma, av;
  int w,h;
  w = in1.getWidth(); h = in2.getHeight();
  if(in2.getWidth() != w || in2.getHeight() != h)
    LFATAL("Cannot compare two different image sizes im1[%dx%d] im2[%dx%d]",
           in1.getWidth(),in1.getHeight(),in2.getWidth(),in2.getHeight());
  getMinMaxAvg(diff,mi,ma,av);
  bool acceptable = mi> -rounderr && ma< rounderr && std::abs(av) < BIAS_MARGIN;
  LINFO("%s: %ux%u image, #1 - #2: avg=%f, diff = [%f .. %f]",
        mi == ma && ma == 0.0F ? "MATCH" : (acceptable ? "ACCEPT" : "FAIL"), w, h, av, mi, ma);
  if(!acceptable)//(mi != ma || ma != 0.0F)
  {
    if(fabs(ma) > fabs(mi))
      {
        findMax(diff,p,ma);
        LINFO("Maximum difference %f is located at %dx%d",ma,p.i,p.j);
      }
    else
      {
        findMin(diff,p,mi);
        LINFO("Minimum difference %f is located at %dx%d",mi,p.i,p.j);
      }
    getMinMaxAvg(in1,mi,ma,av);
    LINFO("Image 1 [%dx%d]: avg=%f, diff = [%f .. %f]",
          w, h, av, mi, ma);
    getMinMaxAvg(in2,mi,ma,av);
    LINFO("Image 2 [%dx%d]: avg=%f, diff = [%f .. %f]",
          w, h, av, mi, ma);
    compareregions(in1,in2,std::max(0,p.j-5),std::min(h,p.j+5),std::max(0,p.i-5),std::min(w,p.i+5));
    writeImage(diff);
  }
  //printImage(diff);

}

void cmpNoise(const Image<float> in1, const Image<float> in2)
{
  Image<float> diff = in1-in2;
  float mi, ma, av;
  int w,h;
  w = in1.getWidth(); h = in2.getHeight();

  if(in2.getWidth() != w || in2.getHeight() != h)
    LFATAL("Cannot compare two different image sizes im1[%dx%d] im2[%dx%d]",
           in1.getWidth(),in1.getHeight(),in2.getWidth(),in2.getHeight());
  getMinMaxAvg(diff,mi,ma,av);
  LINFO("NOISE CHECK: %ux%u image, #1 - #2: avg=%f, diff = [%f .. %f]", w, h, av, mi, ma);
  //writeImage(diff);
}

void test_minmaxavgnorm(Image<float> im, MemoryPolicy mp, int dev)
{
  float mi,ma,av;
  Timer tim;
  printf("Testing global min/max/avg\n");
  tim.reset();
  getMinMaxAvg(im,mi,ma,av);
  LINFO("CPU done! %fms", tim.getSecs() * 1000.0F);
  CudaImage<float> cmin, cmax,cavg,cbuf;
  CudaImage<float> cim = CudaImage<float>(im,mp,dev);
  cbuf = CudaImage<float>(im.getWidth()/2,im.getHeight()/2,NO_INIT,mp,dev);
  cmin =  CudaImage<float>(1,1,NO_INIT,mp,dev);
  cmax =  CudaImage<float>(1,1,NO_INIT,mp,dev);
  cavg =  CudaImage<float>(1,1,NO_INIT,mp,dev);
  // The lazy allocation involved in the following line KILLS performance, do NOT use if preallocating memory for performance gain!!
  //cmin = cmax = cavg = CudaImage<float>(1,1,NO_INIT,mp,dev);
  tim.reset();
  cudaGetMinMaxAvg(cim,cmin,cmax,cavg,&cbuf);
  LINFO("GPU done! %fms", tim.getSecs() * 1000.0F);
  Image<float> hmin = cmin.exportToImage();
  Image<float> hmax = cmax.exportToImage();
  Image<float> havg = cavg.exportToImage();
  printf("image min %f max %f avg %f, cuda min %f max %f avg %f\n",mi,ma,av,
         hmin.getVal(0),hmax.getVal(0),havg.getVal(0));

  inplaceNormalize(im,0.0F,10.0F);
  cudaInplaceNormalize(cim,0.0F,10.0F);
  Image<float> hnorm = cim.exportToImage();
  printf("Testing inplace normalization\n");
  testDiff(im,hnorm);
  //printImage(im);
  //printImage(hmin);
  //printImage(hmax);
  //printImage(havg);
}

void test_find(Image<float> im, MemoryPolicy mp, int dev)
{
  CudaImage<float> cim = CudaImage<float>(im,mp,dev);
  Point2D<int> p1,p2;
  float ftmp,ftmp2;
  findMin(im,p1,ftmp);
  cudaFindMin(cim,p2,ftmp2);
  printf("Testing Find of Minimum Location on Image = %s [CPU idx:%s,val:%f GPU idx:%s,val:%f]\n",(p1==p2) ? "ACCEPT" : "FAIL",convertToString(p1).c_str(),ftmp,convertToString(p2).c_str(),ftmp2);
  findMax(im,p1,ftmp);
  cudaFindMax(cim,p2,ftmp2);
  printf("Testing Find of Maximum Location on Image = %s [CPU idx:%s,val:%f GPU idx:%s,val:%f]\n",(p1==p2) ? "ACCEPT" : "FAIL",convertToString(p1).c_str(),ftmp,convertToString(p2).c_str(),ftmp2);

}

void test_math(Image<float> im, MemoryPolicy mp, int dev)
{
  float val = 1.2345F;
  Image<float> scalaradd = im + val;
  Image<float> scalarsub = im - val;
  Image<float> scalarmul = im * val;
  Image<float> scalardiv = im / val;
  CudaImage<float> cim = CudaImage<float>(im,mp,dev);
  // Scalar value on CUDA device
  CudaImage<float> cval = CudaImage<float>(1,1,NO_INIT,mp,dev);
  cudaClear(cval,val);
  // Image value on CUDA device
  Image<float> test = Image<float>(im.getDims(),NO_INIT);
  test.clear(val);
  CudaImage<float> ctest = CudaImage<float>(test,mp,dev);
  Image<float> imageadd = im + test;
  Image<float> imagesub = im - test;
  Image<float> imagemul = im * test;
  Image<float> imagediv = im / test;

  CudaImage<float> cscalaradd = cim + cval;
  CudaImage<float> cscalarsub = cim - cval;
  CudaImage<float> cscalarmul = cim * cval;
  CudaImage<float> cscalardiv = cim / cval;
  Image<float> hscalaradd = cscalaradd.exportToImage();
  Image<float> hscalarsub = cscalarsub.exportToImage();
  Image<float> hscalarmul = cscalarmul.exportToImage();
  Image<float> hscalardiv = cscalardiv.exportToImage();
  printf("Testing add/subtract/multiply/divide device scalars\n");
  testDiff(scalaradd,hscalaradd);
  testDiff(scalarsub,hscalarsub);
  testDiff(scalarmul,hscalarmul);
  testDiff(scalardiv,hscalardiv);
  cscalaradd = cscalarsub = cscalarmul = cscalardiv = cim;
  cscalaradd += cval;
  cscalarsub -= cval;
  cscalarmul *= cval;
  cscalardiv /= cval;
  hscalaradd = cscalaradd.exportToImage();
  hscalarsub = cscalarsub.exportToImage();
  hscalarmul = cscalarmul.exportToImage();
  hscalardiv = cscalardiv.exportToImage();
  printf("Testing inplace add/subtract/multiply/divide device scalars\n");
  testDiff(scalaradd,hscalaradd);
  testDiff(scalarsub,hscalarsub);
  testDiff(scalarmul,hscalarmul);
  testDiff(scalardiv,hscalardiv);

  cscalaradd = cim + val;
  cscalarsub = cim - val;
  cscalarmul = cim * val;
  cscalardiv = cim / val;
  hscalaradd = cscalaradd.exportToImage();
  hscalarsub = cscalarsub.exportToImage();
  hscalarmul = cscalarmul.exportToImage();
  hscalardiv = cscalardiv.exportToImage();
  printf("Testing add/subtract/multiply/divide host scalars\n");
  testDiff(scalaradd,hscalaradd);
  testDiff(scalarsub,hscalarsub);
  testDiff(scalarmul,hscalarmul);
  testDiff(scalardiv,hscalardiv);
  cscalaradd = cscalarsub = cscalarmul = cscalardiv = cim;
  cscalaradd += val;
  cscalarsub -= val;
  cscalarmul *= val;
  cscalardiv /= val;
  hscalaradd = cscalaradd.exportToImage();
  hscalarsub = cscalarsub.exportToImage();
  hscalarmul = cscalarmul.exportToImage();
  hscalardiv = cscalardiv.exportToImage();
  printf("Testing inplace add/subtract/multiply/divide host scalars\n");
  testDiff(scalaradd,hscalaradd);
  testDiff(scalarsub,hscalarsub);
  testDiff(scalarmul,hscalarmul);
  testDiff(scalardiv,hscalardiv);

  CudaImage<float> cimageadd = cim;
  CudaImage<float> cimagesub = cim;
  CudaImage<float> cimagemul = cim;
  CudaImage<float> cimagediv = cim;
  cimageadd += ctest;
  cimagesub -= ctest;
  cimagemul *= ctest;
  cimagediv /= ctest;
  Image<float> himageadd = cimageadd.exportToImage();
  Image<float> himagesub = cimagesub.exportToImage();
  Image<float> himagemul = cimagemul.exportToImage();
  Image<float> himagediv = cimagediv.exportToImage();
  printf("Testing inplace add/subtract/multiply/divide images\n");
  testDiff(imageadd,himageadd);
  testDiff(imagesub,himagesub);
  testDiff(imagemul,himagemul);
  testDiff(imagediv,himagediv);

  cimageadd = cim + ctest;
  cimagesub = cim - ctest;
  cimagemul = cim * ctest;
  cimagediv = cim / ctest;
  himageadd = cimageadd.exportToImage();
  himagesub = cimagesub.exportToImage();
  himagemul = cimagemul.exportToImage();
  himagediv = cimagediv.exportToImage();
  printf("Testing add/subtract/multiply/divide images\n");
  testDiff(imageadd,himageadd);
  testDiff(imagesub,himagesub);
  testDiff(imagemul,himagemul);
  testDiff(imagediv,himagediv);
}

void comparePyramids(ImageSet<float> pyr, CudaImageSet<float> cpyr, float rnd_err=ROUNDOFF_MARGIN)
{
  if(pyr.size() != cpyr.size())
    {
      LFATAL("Testing unequal pyramids\n");
    }
 for(uint i=0;i<pyr.size();i++)
  {
    Image<float> ctmp= cpyr[i].exportToImage();
    testDiff(ctmp,pyr[i],rnd_err);
  }

}

void test_pyramids(Image<float> im, MemoryPolicy mp, int dev)
{
  const float rndErrMargin = HIGHER_ROUNDOFF_MARGIN;
  // Sin/cos are going to have worse resolution
  const float trigErrMargin = 3*HIGHER_ROUNDOFF_MARGIN;
  CudaImage<float> cim = CudaImage<float>(im,mp,dev);
  printf("Testing 9 tap Gaussian pyramids\n");
  CudaImageSet<float> cgs = cudaBuildPyrGaussian(cim,0,9,9);
  ImageSet<float> gs = buildPyrGaussian(im,0,9,9);
  comparePyramids(gs,cgs);

  printf("Testing 5 tap Gaussian pyramids\n");
  cgs = cudaBuildPyrGaussian(cim,0,9,5);
  gs = buildPyrGaussian(im,0,9,5);
  comparePyramids(gs,cgs);

  printf("Testing 3 tap Gaussian pyramids\n");
  cgs = cudaBuildPyrGaussian(cim,0,9,3);
  gs = buildPyrGaussian(im,0,9,3);
  comparePyramids(gs,cgs);

  printf("Testing 9 tap Laplacian pyramids\n");
  CudaImageSet<float> cls = cudaBuildPyrLaplacian(cim,0,9,9);
  ImageSet<float> ls = buildPyrLaplacian(im,0,9,9);
  comparePyramids(ls,cls);

  printf("Testing 5 tap Laplacian pyramids\n");
  cls = cudaBuildPyrLaplacian(cim,0,9,5);
  ls = buildPyrLaplacian(im,0,9,5);
  comparePyramids(ls,cls);

  printf("Testing 3 tap Laplacian pyramids\n");
  cls = cudaBuildPyrLaplacian(cim,0,9,3);
  ls = buildPyrLaplacian(im,0,9,3);
  comparePyramids(ls,cls);

  printf("Testing Attenuated borders\n");
  CudaImage<float> ciat = cim;
  Image<float>iat = im;
  inplaceAttenuateBorders(iat,5);
  cudaInplaceAttenuateBorders(ciat,5);
  Image<float> hiat = ciat.exportToImage();
  testDiff(hiat,iat);
  printf("Testing Oriented Filter: RAISING ROUNDOFF ERROR TO %f\n",trigErrMargin);
  iat = orientedFilter(im,2.6F,0.0F,10.0F);
  ciat = cudaOrientedFilter(cim,2.6F,0.0F,10.0F);
  hiat = ciat.exportToImage();
  testDiff(hiat,iat,trigErrMargin);
  printf("Testing an Oriented Pyramid from a 9 tap Laplacian Pyramid: RAISING ROUNDOFF ERROR TO %f\n",trigErrMargin);
  ls = buildPyrLaplacian(im,0,9,9);
  cls = cudaBuildPyrLaplacian(cim,0,9,9);
  ImageSet<float> os = buildPyrOrientedFromLaplacian(ls,9,0);
  CudaImageSet<float> c_os = cudaBuildPyrOrientedFromLaplacian(cls,9,0);
  comparePyramids(os,c_os,trigErrMargin);

  printf("Testing an Oriented Pyramid from scratch: RAISING ROUNDOFF ERROR TO %f\n",trigErrMargin);
  os = buildPyrOriented(im,0,9,9,0);
  c_os = cudaBuildPyrOriented(cim,0,9,9,0);
  comparePyramids(os,c_os,trigErrMargin);

  printf("Testing a Quick Local Average Pyramid ONLY CALCULATES LAST IMAGE OF PYRAMID!!!!: RAISING ROUNDOFF ERROR TO %f\n",rndErrMargin);
  int depth = 7;
  os = buildPyrLocalAvg(im,depth);
  c_os = cudaBuildPyrLocalAvg(cim,depth);
  Image<float> h_os = c_os[depth-1].exportToImage();
  testDiff(os[depth-1],h_os,rndErrMargin);

  printf("Testing a Quick Local Average Pyramid ONLY CALCULATES LAST IMAGE OF PYRAMID!!!!: RAISING ROUNDOFF ERROR TO %f\n",rndErrMargin);
  depth = 7;
  os = buildPyrLocalMax(im,depth);
  c_os = cudaBuildPyrLocalMax(cim,depth);
  h_os = c_os[depth-1].exportToImage();
  testDiff(os[depth-1],h_os,rndErrMargin);

  printf("Testing Reichardt Pyramid: RAISING ROUNDOFF ERROR TO %f\n",trigErrMargin);
  float dx=12.5,dy=7.5;
  ReichardtPyrBuilder<float> reich = ReichardtPyrBuilder<float>(dx,dy,Gaussian5);
  os = reich.build(im,0,9);
  CudaReichardtPyrBuilder<float> creich = CudaReichardtPyrBuilder<float>(dx,dy,Gaussian5);
  c_os = creich.build(cim,0,9);
  comparePyramids(os,c_os,trigErrMargin);
}

void test_centerSurround(Image<float> im, MemoryPolicy mp, int dev)
{
  ImageSet<float> ims = buildPyrLaplacian(im,0,9,9);
  printf("Testing ImageSet conversion to CudaImageSet\n");
  CudaImageSet<float> cims = CudaImageSet<float>(ims,mp,dev);
  CudaImage<float> cim = CudaImage<float>(im,mp,dev);
  comparePyramids(ims,cims,HIGHER_ROUNDOFF_MARGIN);

  printf("Testing Center Surround Absolute\n");
  Image<float> cmap(ims[3].getDims(),ZEROS);
  for (int delta = 3; delta <= 4; delta ++)
    {
      for (int lev = 0; lev <= 2; lev ++)
        {
          Image<float> tmp = centerSurround(ims, lev, lev + delta, true);
          CudaImage<float> ctmp = cudaCenterSurround(cims, lev, lev + delta, true);
          Image<float> htmp = ctmp.exportToImage();
          testDiff(tmp,htmp);
        }
    }
  printf("Testing Center Surround Clamped\n");
  cmap.clear(0);
  for (int delta = 3; delta <= 4; delta ++)
    {
      for (int lev = 0; lev <= 2; lev ++)
        {
          Image<float> tmp = centerSurround(ims, lev, lev + delta, false);
          CudaImage<float> ctmp = cudaCenterSurround(cims, lev, lev + delta, false);
          Image<float> htmp = ctmp.exportToImage();
          testDiff(tmp,htmp);
        }
    }
  printf("Testing Center Surround Directional\n");
  for (int delta = 3; delta <= 4; delta ++)
    {
      for (int lev = 0; lev <= 2; lev ++)
        {
          Image<float> tmppos, tmpneg;
          CudaImage<float> ctmppos,ctmpneg;
          centerSurround(ims, lev, lev + delta, tmppos, tmpneg);
          cudaCenterSurround(cims, lev, lev + delta, ctmppos, ctmpneg);
          Image<float> htmppos = ctmppos.exportToImage();
          Image<float> htmpneg = ctmpneg.exportToImage();
          testDiff(tmppos,htmppos);
          testDiff(tmpneg,htmpneg);
        }
    }
  printf("Testing Downsize\n");
  Image<float> dsz = downSize(im,im.getWidth()/3,im.getHeight()/3,5);
  CudaImage<float> cdsz = cudaDownSize(cim,cim.getWidth()/3,cim.getHeight()/3,5);
  Image<float> hdsz = cdsz.exportToImage();
  testDiff(dsz,hdsz);

  printf("Testing Downsize clean with higher roundoff tolerance %f\n",HIGHER_ROUNDOFF_MARGIN);
  dsz = downSizeClean(im,Dims(im.getWidth()/3,im.getHeight()/3),5);
  cdsz = cudaDownSizeClean(cim,Dims(cim.getWidth()/3,cim.getHeight()/3),5);
  hdsz = cdsz.exportToImage();
  testDiff(dsz,hdsz,HIGHER_ROUNDOFF_MARGIN);

  printf("Testing Bilinear Rescale with higher roundoff tolerance %f\n",HIGHER_ROUNDOFF_MARGIN);
  dsz = rescaleBilinear(im,im.getWidth()/3,im.getHeight()/3);
  cdsz = cudaRescaleBilinear(cim,cim.getWidth()/3,cim.getHeight()/3);
  hdsz = cdsz.exportToImage();
  testDiff(dsz,hdsz,HIGHER_ROUNDOFF_MARGIN);

  printf("Testing RGB Bilinear Rescale with higher roundoff tolerance %f\n",HIGHER_ROUNDOFF_MARGIN);
  Image<PixRGB<float> > imc = toRGB(im);
  CudaImage<PixRGB<float> > cimc = CudaImage<PixRGB<float> >(imc,mp,dev);
  imc = rescaleBilinear(imc,imc.getWidth()/3,imc.getHeight()/3);
  cimc = cudaRescaleBilinear(cimc,cimc.getWidth()/3,cimc.getHeight()/3);
  Image<PixRGB<float> > himc = cimc.exportToImage();
  testDiff(luminance(imc),luminance(himc),HIGHER_ROUNDOFF_MARGIN);
}

void test_localOperators(Image<float> im, MemoryPolicy mp, int dev)
{
  CudaImage<float> cim = CudaImage<float>(im,mp,dev);
  printf("Test Quick Local Average\n");
  Image<float> la = quickLocalAvg(im,7);
  CudaImage<float> cla = cudaQuickLocalAvg(cim,7);
  Image<float> hla = cla.exportToImage();
  testDiff(la,hla);
  printf("Test Quick Local Average 2x2\n");
  la = quickLocalAvg2x2(im);
  cla = cudaQuickLocalAvg2x2(cim);
  hla = cla.exportToImage();
  testDiff(la,hla);
  printf("Test Quick Local Max\n");
  la = quickLocalMax(im,7);
  cla = cudaQuickLocalMax(cim,7);
  hla = cla.exportToImage();
  testDiff(la,hla);
}

void test_kernels(Image<float> im, MemoryPolicy mp, int dev)
{
  CudaImage<float> cim = CudaImage<float>(im,mp,dev);
  float mj_stdev=1.0F;
  float mi_stdev=0.8F;
  float period=1.0F;
  float phase=0.0F;
  float theta=0.0F;
  printf("Test Gabor Kernel Generation\n");
  Image<float> gf = gaborFilter3(mj_stdev,mi_stdev,period,phase,theta);
  CudaImage<float>cgf = cudaGaborFilter3(mp,dev,mj_stdev,mi_stdev,period,phase,theta,-1);
  Image<float> hgf = cgf.exportToImage();
  testDiff(gf,hgf);
  printf("Testing Optimized Convolution of a Gabor with higher roundoff tolerance %f\n",HIGHER_ROUNDOFF_MARGIN);
  Image<float> gcon = optConvolve(im,gf);
  CudaImage<float> cgcon = cudaOptConvolve(cim,cgf);
  Image<float> hgcon = cgcon.exportToImage();
  testDiff(gcon,hgcon,HIGHER_ROUNDOFF_MARGIN);
  printf("Testing Zero Boundary Convolution of a Gabor with higher roundoff tolerance %f\n",HIGHER_ROUNDOFF_MARGIN);
  gcon = convolve(im,gf,CONV_BOUNDARY_ZERO);
  cgcon = cudaConvolve(cim,cgf,CONV_BOUNDARY_ZERO);
  hgcon = cgcon.exportToImage();
  testDiff(gcon,hgcon,HIGHER_ROUNDOFF_MARGIN);
  printf("Testing Clean Boundary Convolution of a Gabor with higher roundoff tolerance %f\n",HIGHER_ROUNDOFF_MARGIN);
  gcon = convolve(im,gf,CONV_BOUNDARY_CLEAN);
  cgcon = cudaConvolve(cim,cgf,CONV_BOUNDARY_CLEAN);
  hgcon = cgcon.exportToImage();
  testDiff(gcon,hgcon,HIGHER_ROUNDOFF_MARGIN);
  printf("Test Gaussian 1D Kernel Generation\n");
  float coef=1.0F; float sigma = 2.0F; int maxhw = 20;
  gf = gaussian<float>(coef,sigma,maxhw);
  cgf = cudaGaussian(mp,dev,coef,sigma,maxhw);
  hgf = cgf.exportToImage();
  testDiff(gf,hgf);
  float rndOffErr = HIGHER_ROUNDOFF_MARGIN;

  printf("Testing Optimized Zero Boundary Convolution of a separable Gaussian with higher roundoff tolerance %f\n",rndOffErr);
  gcon = sepFilter(im,gf,gf,CONV_BOUNDARY_ZERO);
  cgcon = cudaSepFilter(cim,cgf,cgf,CONV_BOUNDARY_ZERO);
  hgcon = cgcon.exportToImage();
  testDiff(gcon,hgcon,rndOffErr);
  printf("Testing NonOptimized Zero Boundary Convolution of a separable Gaussian with higher roundoff tolerance %f\n",rndOffErr);
  gcon = sepFilter(im,gf,gf,CONV_BOUNDARY_ZERO);
  cgcon = cudaSepFilter(cim,cgf,cgf,CONV_BOUNDARY_ZERO,false);
  hgcon = cgcon.exportToImage();
  testDiff(gcon,hgcon,rndOffErr);
  printf("Testing Optimized Clean Boundary Convolution of a separable Gaussian with higher roundoff tolerance %f\n",rndOffErr);
  gcon = sepFilter(im,gf,gf,CONV_BOUNDARY_CLEAN);
  cgcon = cudaSepFilter(cim,cgf,cgf,CONV_BOUNDARY_CLEAN);
  hgcon = cgcon.exportToImage();
  testDiff(gcon,hgcon,rndOffErr);
  printf("Testing NonOptimized Clean Boundary Convolution of a separable Gaussian with higher roundoff tolerance %f\n",rndOffErr);
  gcon = sepFilter(im,gf,gf,CONV_BOUNDARY_CLEAN);
  cgcon = cudaSepFilter(cim,cgf,cgf,CONV_BOUNDARY_CLEAN,false);
  hgcon = cgcon.exportToImage();
  testDiff(gcon,hgcon,rndOffErr);
  printf("Testing Optimized Replicate Boundary Convolution of a separable Gaussian with higher roundoff tolerance %f\n",rndOffErr);
  gcon = sepFilter(im,gf,gf,CONV_BOUNDARY_REPLICATE);
  cgcon = cudaSepFilter(cim,cgf,cgf,CONV_BOUNDARY_REPLICATE);
  hgcon = cgcon.exportToImage();
  testDiff(gcon,hgcon,rndOffErr);
  printf("Testing NonOptimized Replicate Boundary Convolution of a separable Gaussian with higher roundoff tolerance %f\n",rndOffErr);
  gcon = sepFilter(im,gf,gf,CONV_BOUNDARY_REPLICATE);
  cgcon = cudaSepFilter(cim,cgf,cgf,CONV_BOUNDARY_REPLICATE,false);
  hgcon = cgcon.exportToImage();
  testDiff(gcon,hgcon,rndOffErr);
  Timer tim;
  printf("Timing convolution\n");
  tim.reset();
  for(int i=0;i<1;i++)
    gcon = sepFilter(im,gf,gf,CONV_BOUNDARY_CLEAN);
  LINFO("CPU done! %fms", tim.getSecs() * 1000.0F);
  tim.reset();
  for(int i=0;i<1;i++)
    cgcon = cudaSepFilter(cim,cgf,cgf,CONV_BOUNDARY_CLEAN,false);
  LINFO("GPU original done! %fms", tim.getSecs() * 1000.0F);
  tim.reset();
  for(int i=0;i<1;i++)
    cgcon = cudaSepFilter(cim,cgf,cgf,CONV_BOUNDARY_CLEAN);
  LINFO("GPU optimized done! %fms", tim.getSecs() * 1000.0F);
}

void test_hmax(Image<float> im, MemoryPolicy mp, int dev)
{
  float rndOffErr = HIGHER_ROUNDOFF_MARGIN;
  CudaImage<float> cim = CudaImage<float>(im,mp,dev);

  float theta = 40.0F, gamma = 1.0F, div = 0.5F;
  int size = 13;
  Image<float> dg = dogFilterHmax<float>(theta,gamma,size,div);
  CudaImage<float> cdg = CudaImage<float>(dg,mp,dev);
  printf("Testing HMax Style image energy normalized convolution with higher roundoff tolerance %f\n",rndOffErr);
  Image<float> gcon = convolveHmax(im,dg);
  CudaImage<float> cgcon = cudaConvolveHmax(cim,cdg);
  Image<float> hgcon = cgcon.exportToImage();
  testDiff(gcon,hgcon,rndOffErr);

  printf("Testing WindowedPatchDistance Convolution with higher roundoff tolerance %f\n",rndOffErr);
  cgcon = CudaImage<float>(gcon,mp,dev);
  Image<float> gwpd = convolve(im,dg,CONV_BOUNDARY_ZERO);
  CudaImage<float> cgwpd = cudaConvolve(cim,cdg,CONV_BOUNDARY_ZERO);
  Image<float> hgwpd = cgwpd.exportToImage();
  testDiff(gwpd,hgwpd,rndOffErr);

  printf("Testing WindowedPatchDistance Convolution(Clean) with higher roundoff tolerance %f\n",rndOffErr);
  cgcon = CudaImage<float>(gcon,mp,dev);
  gwpd = convolve(im,dg,CONV_BOUNDARY_CLEAN);
  cgwpd = cudaConvolve(cim,cdg,CONV_BOUNDARY_CLEAN);
  hgwpd = cgwpd.exportToImage();
  testDiff(gwpd,hgwpd,rndOffErr);

  Image<float> sp = spatialPoolMax(im,20,20,20,20);
  CudaImage<float> csp = cudaSpatialPoolMax(cim,20,20,20,20);
  Image<float> hsp = csp.exportToImage();
  printf("Testing spatial pool max with higher roundoff tolerance %f\n",rndOffErr);
  testDiff(sp,hsp,rndOffErr);

  Image<float> imr = rotate(im,im.getWidth()/2,im.getHeight()/2,PI/2);
  CudaImage<float> cimr = CudaImage<float>(imr,mp,dev);
  Image<float> mx = takeMax<float>(im,imr);
  CudaImage<float> cmx = cudaTakeMax(cim,cimr);
  Image<float> hmx = cmx.exportToImage();
  printf("Testing pixel by pixel take max  with higher roundoff tolerance %f\n",rndOffErr);
  testDiff(mx,hmx,rndOffErr);

  float sm = sum(im);
  CudaImage<float> csm = cudaGetSum(cim);
  float hsm = cudaGetScalar(csm);
  printf("Testing sum of pixels\n");
  testDiff(sm,hsm);

  dg = dogFilterHmax<float>(theta,gamma,size,div);
  cdg = cudaDogFilterHmax(mp,dev,theta,gamma,size,div);
  Image<float> hdg = cdg.exportToImage();
  printf("Testing DoG HMAX kernel generation\n");
  testDiff(dg,hdg);

  float stddev = 2.0F;
  int halfsize=6;
  dg = dogFilter<float>(stddev,theta,halfsize);
  cdg = cudaDogFilter(mp,dev,stddev,theta,halfsize);
  hdg = cdg.exportToImage();
  printf("Testing DoG kernel generation\n");
  testDiff(dg,hdg);
}

void test_additiveNoise(Image<float> im, MemoryPolicy mp, int dev)
{
  CudaImage<float> cim = CudaImage<float>(im,mp,dev);
  CudaImage<float> rndBuf;
  cudaSetSeed(dev);
  cudaSizeRandomBuffer(rndBuf,mp,dev,cim.size());
  printf("Test Additive Noise\n");
  inplaceAddBGnoise(im,7);
  cudaRandomMT(rndBuf);
  Image<float> hrbf = rndBuf.exportToImage();
  float mi,ma,av; getMinMaxAvg(hrbf,mi,ma,av);
  LINFO("RANDOM NUMBER GENERATOR should be in 0-1 range,w%u h%u avg=%f, min=%f max=%f",hrbf.getWidth(),hrbf.getHeight(), av, mi, ma);
  cudaInplaceAddBGnoise(cim,7,rndBuf);
  Image<float> him = cim.exportToImage();
  cmpNoise(im,him);
}

void test_separable(Image<float> im, MemoryPolicy mp, int dev)
{
  int w = im.getWidth();
  int h = im.getHeight();
  printf("Testing separable filters with large kernels with higher roundoff tolerance %f\n",HIGHER_ROUNDOFF_MARGIN);
  int maxhw = std::max(0,std::min(w,h)/2-1);
  float isig = (std::max(w,h) * 25) * 0.01F;
  CudaImage<float> cim = CudaImage<float>(im,mp,dev);
  CudaImage<float> cg = cudaGaussian(mp,dev,1.5/(isig*sqrt(2.0*M_PI)),isig, maxhw);
  Image<float> g = gaussian<float>(1.5/(isig*sqrt(2.0*M_PI)),isig, maxhw);
  CudaImage<float> cxf = cudaXFilter(cim,cg,cg.size(),CONV_BOUNDARY_CLEAN);
  Image<float> tmp;
  Image<float> xf = sepFilter(im,g,tmp,CONV_BOUNDARY_CLEAN);
  Image<float> hxf = cxf.exportToImage();
  testDiff(xf,hxf,HIGHER_ROUNDOFF_MARGIN);
}

void test_maxNormalize(Image<float> im, MemoryPolicy mp, int dev)
{
  CudaImage<float> cim = CudaImage<float>(im,mp,dev);
  float rndOffErr = HIGHER_ROUNDOFF_MARGIN*3;
  printf("Testing Max Normalize with higher roundoff tolerance %f\n",rndOffErr);
  Image<float> mn = maxNormalize(im,MAXNORMMIN,MAXNORMMAX,VCXNORM_DEFAULT,1);
  CudaImage<float> cmn = cudaMaxNormalize(cim,MAXNORMMIN,MAXNORMMAX,VCXNORM_DEFAULT,1);
  Image<float> hmn = cmn.exportToImage();
  testDiff(mn,hmn,rndOffErr);

}

void test_compressing(Image<float> im, MemoryPolicy mp, int dev)
{
  CudaImage<float> cim = CudaImage<float>(im,mp,dev);
  CudaImage<float> ccompress = cudaLowPass9Dec(cim,true,true);
  CudaImage<float> cbase = cudaDecY(cudaLowPass9y(cudaDecX(cudaLowPass9x(cim))));
  Image<float> base = decY(lowPass9y(decX(lowPass9x(im))));
  Image<float> hbase = cbase.exportToImage();
  Image<float> hcompress = ccompress.exportToImage();
  float rndOffErr = HIGHER_ROUNDOFF_MARGIN*3;
  printf("Testing Baseline lowpass & decimate 9%f\n",rndOffErr);
  testDiff(base,hbase,rndOffErr);
  printf("Testing Compressed lowpass & decimate 9%f\n",rndOffErr);
  testDiff(base,hcompress,rndOffErr);
  //writeImage(base-hcompress);
}

void test_cutpaste(Image<float> im, MemoryPolicy mp, int dev)
{
  Image<float> im_copy = im;
  CudaImage<float> cim = CudaImage<float>(im,mp,dev);
  Rectangle r = Rectangle::tlbrI(10,15,24,35);
  Image<float> crp = crop(im,r);
  CudaImage<float> ccrp = cudaCrop(cim,r);
  Image<float> hcrp = ccrp.exportToImage();
  printf("Testing Cropping\n");
  testDiff(crp,hcrp);
  float dx = 12.5, dy = 7.2;
  Image<float> sim = shiftImage(im,dx,dy);
  CudaImage<float> csim = cudaShiftImage(cim,dx,dy);
  Image<float> hsim = csim.exportToImage();
  printf("Test ShiftImage\n");
  testDiff(sim,hsim);
  Image<float> lit = rotate(crp,crp.getWidth()/2,crp.getHeight()/2,M_PI/4.0);
  CudaImage<float> clit = CudaImage<float>(lit,mp,dev);
  Point2D<int> p2 = Point2D<int>(30,20);
  inplacePaste(im,lit,p2);
  cudaInplacePaste(cim,clit,p2);
  Image<float> hpaste = cim.exportToImage();
  printf("Test Inplace Paste\n");
  testDiff(im,hpaste);
  // Refresh copy of the image to test another inplace operation
  im = im_copy;
  Image<PixRGB<float> > im_color = toRGB(im);
  Image<PixRGB<float> > lit_color = toRGB(lit);
  CudaImage<PixRGB<float> > cim_color = CudaImage<PixRGB<float> >(im_color,mp,dev);
  CudaImage<PixRGB<float> > clit_color = CudaImage<PixRGB<float> >(lit_color,mp,dev);
  inplacePaste(im_color,lit_color,p2);
  cudaInplaceOverlay(cim_color,clit_color,p2);
  hpaste = (cudaLuminance(cim_color)).exportToImage();
  printf("Test Inplace Paste RGB\n");
  testDiff(luminance(im_color),hpaste);


}

void test_drawing(Image<float> im, MemoryPolicy mp, int dev)
{
  float intensity=127.254;
 CudaImage<float> cim = CudaImage<float>(im,mp,dev);
 Rectangle r = Rectangle::tlbrI(10,15,24,35);
 cudaDrawRect(cim,r,intensity,3);
 drawRect(im,r,intensity,3);
 Image<float> him = cim.exportToImage();
 printf("Test Inplace Draw Rectangular Border\n");
 testDiff(im,him);
}

void test_lowpassOptimization(Image<float> im, MemoryPolicy mp, int dev)
{
  Timer tim;
  Rectangle r = Rectangle::tlbrO(0,0,384,384);
  im = crop(im,r);
  CudaImage<float> cim = CudaImage<float>(im,mp,dev);
  CudaImage<float> cnorm, copt, crnd;
  CudaImage<float> ctmp = (cim+402.333F)/12.0F;
  printf("Testing low pass optimization\n");
  // Test optimized lowpass 9
  // tim.reset();
  // for(int i=0;i<1000;i++)
  //   {
  //   cnorm = cudaLowPass9Dec(cim,true,true);
  //   cudaInplaceAddBGnoise(ctmp,7,crnd);
  //   }
  // LINFO("Normal GPU lowpass done! %fms", tim.getSecs() * 1000.0F);
  // tim.reset();
  // for(int i=0;i<1000;i++)
  //   {
  //   copt = cudaLowPass9xyDec(cim);
  //   cudaInplaceAddBGnoise(ctmp,7,crnd);
  //   }
  // LINFO("Optimized GPU lowpass done! %fms", tim.getSecs() * 1000.0F);
  tim.reset();
  for(int i=0;i<1;i++)
    cnorm = cudaLowPass9Dec(cim,true,true);
  LINFO("Normal GPU lowpass done! %fms", tim.getSecs() * 1000.0F);
  tim.reset();
  for(int i=0;i<1;i++)
    copt = cudaLowPass9xyDec(cim);
  LINFO("Texture optimized GPU lowpass done! %fms", tim.getSecs() * 1000.0F);
  testDiff(cnorm.exportToImage(),copt.exportToImage());
}

void unit_test(int argc, char **argv)
{
  if (argc != 2) LFATAL("USAGE: %s <input.pgm>", argv[0]);

  setFpuRoundingMode(FPU_ROUND_NEAR);
  int dev = 0;
  MemoryPolicy mp = GLOBAL_DEVICE_MEMORY;
  CudaDevices::displayProperties(dev);
  //CUT_DEVICE_INIT(dev);

  LINFO("Reading: %s", argv[1]);
  Image<PixRGB<float> > img = Raster::ReadRGB(argv[1]);

  // Compare normal implementation versus CUDA implementation
  // Original Implementation
  Image<float> normalLum = luminanceNTSC(img);
  // CUDA Implementation
  // Copy image to CUDA
  CudaImage<PixRGB<float> > cimg = CudaImage<PixRGB<float> >(img,mp,dev);
  // Run CUDA Implementation and shove it back onto the host
  CudaImage<float> cLum = cudaLuminanceNTSC(cimg);
  Image<float> hcLum = cLum.exportToImage();
  testDiff(normalLum,hcLum);
  // Compare results
  // Test low pass 5 cuda filter against standard
  // testDiff(cudaLowPass5Dec(cLum,true,true).exportToImage(), lowPass5yDecY(lowPass5xDecX(normalLum)));
  // Test low pass 9 filter against standard
  testDiff(cudaLowPass9(cLum,true,true).exportToImage(), lowPass9(normalLum,true,true));
  // Test the Pyramid building
  test_pyramids(normalLum,mp,dev);
  // Test global operators
  test_minmaxavgnorm(normalLum,mp,dev);
  // Test kernel generation
  test_kernels(normalLum,mp,dev);
  // // Test basic scalar and image math +-*/
  // test_math(normalLum,mp,dev);
  // // Test center surround
  // test_centerSurround(normalLum,mp,dev);
  // // Test local operators
  // test_localOperators(normalLum,mp,dev);
  // // Test adding of background noise
  // test_additiveNoise(normalLum,mp,dev);
  // // Test separable filters
  // test_separable(normalLum,mp,dev);
  // // Test max normalize
  // test_maxNormalize(normalLum,mp,dev);
  // // Test compressing of filters
  // test_compressing(normalLum,mp,dev);
  // // Test hmax filters
  // test_hmax(normalLum,mp,dev);
  // // Testing cut paste operations
  // test_cutpaste(normalLum,mp,dev);
  // // Testing drawing operations
  // test_drawing(normalLum,mp,dev);
  // // Test find operations
  // test_find(normalLum,mp,dev);
  // Testing lowpass optimization
  test_lowpassOptimization(normalLum,mp,dev);
}


void toytest()
{
  int dev = 0;
  MemoryPolicy mp = GLOBAL_DEVICE_MEMORY;
  CudaDevices::displayProperties(dev);
  Image<float> img = Image<float>(10,10,NO_INIT);
  img.setVal(0,0,10.0F); img.setVal(0,1,20.0F); img.setVal(0,2,30.0F); img.setVal(0,3,40.0F); img.setVal(0,4,50.0F);
  img.setVal(0,5,20.0F); img.setVal(0,6,30.0F); img.setVal(0,7,40.0F); img.setVal(0,8,50.0F); img.setVal(0,9,60.0F);
  img.setVal(1,0,30.0F); img.setVal(1,1,40.0F); img.setVal(1,2,50.0F); img.setVal(1,3,60.0F); img.setVal(1,4,70.0F);
  img.setVal(1,5,40.0F); img.setVal(1,6,50.0F); img.setVal(1,7,60.0F); img.setVal(1,8,70.0F); img.setVal(1,9,80.0F);
  img.setVal(2,0,50.0F); img.setVal(2,1,60.0F); img.setVal(2,2,70.0F); img.setVal(2,3,80.0F); img.setVal(2,4,90.0F);
  img.setVal(2,5,60.0F); img.setVal(2,6,70.0F); img.setVal(2,7,80.0F); img.setVal(2,8,90.0F); img.setVal(2,9,100.F);
  img.setVal(3,0,70.0F); img.setVal(3,1,80.0F); img.setVal(3,2,90.0F); img.setVal(3,3,100.F); img.setVal(3,4,110.F);
  img.setVal(3,5,80.0F); img.setVal(3,6,90.0F); img.setVal(3,7,100.F); img.setVal(3,8,110.F); img.setVal(3,9,120.F);
  img.setVal(4,0,90.0F); img.setVal(4,1,100.F); img.setVal(4,2,110.F); img.setVal(4,3,120.F); img.setVal(4,4,130.F);
  img.setVal(4,5,80.0F); img.setVal(4,6,90.0F); img.setVal(4,7,100.F); img.setVal(4,8,110.F); img.setVal(4,9,120.F);
  img.setVal(5,0,70.0F); img.setVal(5,1,80.0F); img.setVal(5,2,90.0F); img.setVal(5,3,100.F); img.setVal(5,4,110.F);
  img.setVal(5,5,60.0F); img.setVal(5,6,70.0F); img.setVal(5,7,80.0F); img.setVal(5,8,90.0F); img.setVal(5,9,100.F);
  img.setVal(6,0,50.0F); img.setVal(6,1,60.0F); img.setVal(6,2,70.0F); img.setVal(6,3,80.0F); img.setVal(6,4,90.0F);
  img.setVal(6,5,40.0F); img.setVal(6,6,50.0F); img.setVal(6,7,60.0F); img.setVal(6,8,70.0F); img.setVal(6,9,80.0F);
  img.setVal(7,0,30.0F); img.setVal(7,1,40.0F); img.setVal(7,2,50.0F); img.setVal(7,3,60.0F); img.setVal(7,4,70.0F);
  img.setVal(7,5,20.0F); img.setVal(7,6,30.0F); img.setVal(7,7,40.0F); img.setVal(7,8,50.0F); img.setVal(7,9,60.0F);
  img.setVal(8,0,10.0F); img.setVal(8,1,20.0F); img.setVal(8,2,30.0F); img.setVal(8,3,40.0F); img.setVal(8,4,50.0F);
  img.setVal(8,5,00.0F); img.setVal(8,6,10.0F); img.setVal(8,7,20.0F); img.setVal(8,8,30.0F); img.setVal(8,9,40.0F);
  img.setVal(9,0,00.0F); img.setVal(9,1,00.0F); img.setVal(9,2,10.0F); img.setVal(9,3,20.0F); img.setVal(9,4,30.0F);
  img.setVal(9,5,00.0F); img.setVal(9,6,00.0F); img.setVal(9,7,00.0F); img.setVal(9,8,10.0F); img.setVal(9,9,20.0F);

  CudaImage<float> cimg = CudaImage<float>(img,mp,dev);
  Image<float> cres = cudaLowPass5yDec(cimg).exportToImage();
  Image<float> normres = lowPass5yDecY(img);
  testDiff(normres,cres);
  test_minmaxavgnorm(img,mp,dev);
  test_kernels(img,mp,dev);
  test_math(img,mp,dev);
  test_localOperators(img,mp,dev);
  test_additiveNoise(img,mp,dev);
  test_separable(img,mp,dev);
  test_maxNormalize(img,mp,dev);
  test_compressing(img,mp,dev);
  test_hmax(img,mp,dev);
  test_cutpaste(img,mp,dev);
  test_drawing(img,mp,dev);
  test_find(img,mp,dev);
  //printImages(normres,cres);
}


int main(int argc, char **argv)
{
  if(argc >= 2)
    {
      unit_test(argc,argv);
    }
  else
    toytest();
}
