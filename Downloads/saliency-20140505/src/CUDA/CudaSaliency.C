#include "CudaSaliency.H"
#include "Util/Timer.H"
#include "CUDA/CudaPyramidOps.H"
#include "CUDA/CudaShapeOps.H"
#include "CUDA/CudaTransforms.H"
#include "CUDA/CudaPyrBuilder.H"
#include "CUDA/CudaDrawOps.H"
#include "CUDA/CudaNorm.H"
#include "CUDA/CudaRandom.H"
#include "Component/ModelOptionDef.H"

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
#define sml        4
#define normtyp    VCXNORM_FANCY
#define MAXNORMITERS 1

static const ModelOptionCateg MOC_CSM = {
  MOC_SORTPRI_3, "CudaSaliencyMap-related Options" };

// Used by: CudaSaliency
const ModelOptionDef OPT_CsmCudaDevice =
  { MODOPT_ARG(int), "CsmCudaDevice", &MOC_CSM, OPTEXP_CORE,
    "Number of frames in which the IOR map decays by half",
    "csm-cuda-device", '\0', "<int>", "0" };

// Used by: EnvSaliencyMap
const ModelOptionDef OPT_CsmNMostSalientLoc =
  { MODOPT_ARG(int), "CsmNMostSalientLoc", &MOC_CSM, OPTEXP_CORE,
    "Return a vector with the stop n most salient locations",
    "csm-nmost-salient-loc", '\0', "int", "1" };

// Used by: CudaSaliency
const ModelOptionDef OPT_CsmIorHalfLife =
  { MODOPT_ARG(double), "CsmIorHalfLife", &MOC_CSM, OPTEXP_CORE,
    "Number of frames in which the IOR map decays by half",
    "csm-ior-halflife", '\0', "<double>", "6.5" };

// Used by: CudaSaliency
const ModelOptionDef OPT_CsmIorStrength =
  { MODOPT_ARG(double), "CsmIorStrength", &MOC_CSM, OPTEXP_CORE,
    "Magnitude of IOR (useful range 0..255)",
    "csm-ior-strength", '\0', "<double>", "16.0" };

// Used by: CudaSaliency
const ModelOptionDef OPT_CsmIorRadius =
  { MODOPT_ARG(double), "CsmIorRadius", &MOC_CSM, OPTEXP_CORE,
    "Radius of IOR",
    "csm-ior-radius", '\0', "<double>", "32.0" };

// Used by: CudaSaliency
const ModelOptionDef OPT_CsmPatchSize =
  { MODOPT_ARG(double), "CsmPatchSize", &MOC_CSM, OPTEXP_CORE,
    "Patch size",
    "csm-patch-size", '\0', "<double>", "72.0" };

// Used by: CudaSaliency
const ModelOptionDef OPT_CsmInertiaRadius =
  { MODOPT_ARG(double), "CsmInertiaRadius", &MOC_CSM, OPTEXP_CORE,
    "Radius of inertia blob",
    "csm-inertia-radius", '\0', "<double>", "32.0" };

// Used by: CudaSaliency
const ModelOptionDef OPT_CsmInertiaStrength =
  { MODOPT_ARG(double), "CsmInertiaStrength", &MOC_CSM, OPTEXP_CORE,
    "Initial strength of inertia blob",
    "csm-inertia-strength", '\0', "<double>", "100.0" };

// Used by: CudaSaliency
const ModelOptionDef OPT_CsmInertiaHalfLife =
  { MODOPT_ARG(double), "CsmInertiaHalfLife", &MOC_CSM, OPTEXP_CORE,
    "Number of frames in which the inertia blob decays by half",
    "csm-inertia-halflife", '\0', "<double>", "6.5" };

// Used by: CudaSaliency
const ModelOptionDef OPT_CsmInertiaShiftThresh =
  { MODOPT_ARG(double), "CsmInertiaShiftThresh", &MOC_CSM, OPTEXP_CORE,
    "Distance threshold for inertia shift",
    "csm-inertia-thresh", '\0', "<double>", "5.0" };


// ######################################################################
CudaSaliency::CudaSaliency(OptionManager& mgr,
           const std::string& descrName,
                           const std::string& tagName):
  ModelComponent(mgr, descrName, tagName),
  itsCudaDevice(&OPT_CsmCudaDevice, this),
  itsInertiaLoc(-1,-1),
  itsCurrentInertiaFactor(1.0),
  itsInertiaMap(),
  itsInhibitionMap(),
  itsNMostSalientLoc(&OPT_CsmNMostSalientLoc, this, ALLOW_ONLINE_CHANGES),
  itsInertiaHalfLife(&OPT_CsmInertiaHalfLife, this, ALLOW_ONLINE_CHANGES),
  itsInertiaStrength(&OPT_CsmInertiaStrength, this, ALLOW_ONLINE_CHANGES),
  itsInertiaRadius(&OPT_CsmInertiaRadius, this, ALLOW_ONLINE_CHANGES),
  itsInertiaShiftThresh(&OPT_CsmInertiaShiftThresh, this, ALLOW_ONLINE_CHANGES),
  itsIorHalfLife(&OPT_CsmIorHalfLife, this, ALLOW_ONLINE_CHANGES),
  itsIorStrength(&OPT_CsmIorStrength, this, ALLOW_ONLINE_CHANGES),
  itsIorRadius(&OPT_CsmIorRadius, this, ALLOW_ONLINE_CHANGES),
  itsPatchSize(&OPT_CsmPatchSize, this, ALLOW_ONLINE_CHANGES),
  itsDynamicFactor(1.0)

{
  gotLum = false; gotRGBY = false; gotSaliency = false;
  mp = GLOBAL_DEVICE_MEMORY; dev = -1;
  numMotionDirs = 4;
  numOrientationDirs = 4;
}

void CudaSaliency::setDevice(MemoryPolicy mp_in, int dev_in)
{
  mp = mp_in;
  itsCudaDevice.setVal(dev_in);
  itsSalMaxLoc.resize(itsNMostSalientLoc.getVal());
  itsSalMax.resize(itsNMostSalientLoc.getVal());
}

// ######################################################################
void CudaSaliency::paramChanged(ModelParamBase* const param,
                                  const bool valueChanged,
                                  ParamClient::ChangeStatus* status)
{
  if (param == &itsCudaDevice)
    {
      dev = itsCudaDevice.getVal();
    }
  if (param == &itsNMostSalientLoc)
    {
      itsSalMaxLoc.resize(itsNMostSalientLoc.getVal());
      itsSalMax.resize(itsNMostSalientLoc.getVal());
    }
}


// ######################################################################
void CudaSaliency::start1()
{
  cudaSetSeed(dev);
  cudaSizeRandomBuffer(randBuf,mp,dev);
  reichardtPyr = new CudaReichardtPyrBuilder<float>[numMotionDirs];
  for(int i=0;i<numMotionDirs;i++)
    {
      double direction = 360.0*double(i)/double(numMotionDirs);
      reichardtPyr[i] =
        CudaReichardtPyrBuilder<float>(cos(direction*M_PI/180.0),-sin(direction*M_PI/180.0),
                                       Gaussian5,direction+90.0);
    }
}

// ######################################################################
void CudaSaliency::stop2()
{
  delete [] reichardtPyr;
}

// ######################################################################
CudaSaliency::~CudaSaliency()
{ }

// ######################################################################
void CudaSaliency::doInput(const Image< PixRGB<byte> > img)
{
  LINFO("new input.....");
  Image<PixRGB<float> > fimg = img;
  CudaImage<PixRGB<float> > tmp = CudaImage<PixRGB<float> >(fimg,mp,dev);
  doCudaInput(tmp);
}

void CudaSaliency::clearMaps()
{
  if(outmap.initialized())
    outmap.clear();
  if(intensityMap.initialized())
    intensityMap.clear();
  if(colorMap.initialized())
    colorMap.clear();
  if(orientationMap.initialized())
    orientationMap.clear();
  if(flickerMap.initialized())
    flickerMap.clear();
  if(motionMap.initialized())
    motionMap.clear();
}

void CudaSaliency::doCudaInput(const CudaImage< PixRGB<float> > img)
{
  int inDev = img.getMemoryDevice();
  ASSERT(inDev==dev);
  colima = img;
  clearMaps();
  // Reset the flags
  gotLum = false; gotRGBY = false;
  // Clear the outmap
  // also kill any old output and internals:
  Timer tim;
  //printf("Timing saliency\n");
  tim.reset();
  runSaliency();
  LINFO("Done! %fms", tim.getSecs() * 1000.0F);
}

// ######################################################################
bool CudaSaliency::outputReady()
{
  return gotSaliency;
}

// ######################################################################
Image<float> CudaSaliency::getOutput()
{
  return convmap.exportToImage();
}

CudaImage<float> CudaSaliency::getCudaOutput()
{
  return convmap;
}

CudaImage<float> CudaSaliency::getIMap()
{
  return intensityMap;
}

CudaImage<float> CudaSaliency::getCMap()
{
  return colorMap;
}

CudaImage<float> CudaSaliency::getOMap()
{
  return orientationMap;
}

CudaImage<float> CudaSaliency::getFMap()
{
  return flickerMap;
}

CudaImage<float> CudaSaliency::getMMap()
{
  return motionMap;
}

CudaImage<float> CudaSaliency::getInertiaMap()
{
  return itsInertiaMap;
}

CudaImage<float> CudaSaliency::getInhibitionMap()
{
  return itsInhibitionMap;
}

std::vector<Point2D<int> > CudaSaliency::getSalMaxLoc()
{
  return itsSalMaxLoc;
}

std::vector<float> CudaSaliency::getSalMax()
{
  return itsSalMax;
}

double CudaSaliency::getPatchSize()
{
  return itsPatchSize.getVal();
}

CudaImage<float> CudaSaliency::cudaPostChannel(CudaImage<float> curImage, PyramidType ptyp, float orientation, float weight, CudaImage<float>& outmap)
{
  // compute pyramid:
  CudaImageSet<float> pyr =
    cudaBuildPyrGeneric(curImage, 0, maxdepth, ptyp, orientation);
  CudaImage<float> cmap = processPyramid(pyr);

  // multiply by conspicuity coefficient:
  if (weight != 1.0F) cmap *= weight;
  // Add to saliency map:
  if (outmap.initialized()) outmap += cmap;
  else outmap = cmap;
  return cmap;
}

CudaImage<float> CudaSaliency::processPyramid(CudaImageSet<float> pyr)
{
  // alloc conspicuity map and clear it:
  CudaImage<float> cmap;

  // intensities is the max-normalized weighted sum of IntensCS:
  for (int delta = delta_min; delta <= delta_max; delta ++)
    for (int lev = level_min; lev <= level_max; lev ++)
      {
        CudaImage<float> tmp = cudaCenterSurround(pyr, lev, lev + delta, true);
        tmp = cudaDownSize(tmp, pyr[sml].getWidth(), pyr[sml].getHeight());
        tmp = cudaMaxNormalize(tmp, MAXNORMMIN, MAXNORMMAX, normtyp,MAXNORMITERS);
        if(cmap.initialized())
          cmap += tmp;
        else
          cmap = tmp;
      }

  cudaInplaceAddBGnoise(cmap, 25.0F,randBuf);


  if (normtyp == VCXNORM_MAXNORM)
    cmap = cudaMaxNormalize(cmap, MAXNORMMIN, MAXNORMMAX, normtyp,MAXNORMITERS);
  else
    cmap = cudaMaxNormalize(cmap, 0.0f, 0.0f, normtyp,MAXNORMITERS);


  return cmap;
}

void CudaSaliency::calcInertia(const CudaImage<float> & salMap)
{
  const MemoryPolicy mp = salMap.getMemoryPolicy();
  int dev = salMap.getMemoryDevice();
  ASSERT(mp != HOST_MEMORY);

  if (itsInertiaLoc.i < 0 || itsInertiaLoc.j < 0
      || (itsSalMaxLoc[0].squdist(itsInertiaLoc)
          > (itsInertiaShiftThresh.getVal()
             * itsInertiaShiftThresh.getVal())))
    {
      itsCurrentInertiaFactor = 1.0;
      LDEBUG("inertia shift to (%d,%d)",
             itsSalMaxLoc[0].i, itsSalMaxLoc[0].j);
    }
  else
    {
      const float factor =
        (itsDynamicFactor * itsInertiaHalfLife.getVal()) > 0
        ? pow(0.5, 1.0/(itsDynamicFactor * itsInertiaHalfLife.getVal()))
        : 0.0f;
      itsCurrentInertiaFactor *= factor;
    }

  {
    if (itsInertiaMap.getDims() != salMap.getDims() ||
          itsInertiaMap.getMemoryDevice() != dev)
      itsInertiaMap = CudaImage<float>(salMap.getDims(), ZEROS, mp, dev);

    itsInertiaLoc = itsSalMaxLoc[0];

    const double s =  itsInertiaStrength.getVal() * itsDynamicFactor * itsCurrentInertiaFactor;
    const double r_inv = itsInertiaRadius.getVal() > 0.0
      ? (1.0 / itsInertiaRadius.getVal()) : 0.0;
    Dims tile = CudaDevices::getDeviceTileSize1D(dev);
    int w = itsInertiaMap.getWidth();
    int h = itsInertiaMap.getHeight();
    cuda_c_inertiaMap(itsInertiaMap.getCudaArrayPtr(),s,r_inv,itsInertiaLoc.i,itsInertiaLoc.j,tile.w(),tile.h(),w,h);
  }
}

void CudaSaliency::calcInhibition(const CudaImage<float> & salMap)
{
  const MemoryPolicy mp = salMap.getMemoryPolicy();
  int dev = salMap.getMemoryDevice();
  ASSERT(mp != HOST_MEMORY);
  if (itsDynamicFactor * itsIorStrength.getVal() > 0.0)
    {
      if (itsInhibitionMap.getDims() != salMap.getDims() ||
          itsInhibitionMap.getMemoryDevice() != dev)
        itsInhibitionMap = CudaImage<float>(salMap.getDims(), ZEROS, mp, dev);

      const float factor =
        (itsDynamicFactor * itsIorHalfLife.getVal()) > 0
        ? pow(0.5, 1.0/(itsDynamicFactor * itsIorHalfLife.getVal()))
        : 0.0f;

      Dims tile = CudaDevices::getDeviceTileSize1D(dev);
      int w = itsInhibitionMap.getWidth();
      int h = itsInhibitionMap.getHeight();

      cuda_c_inhibitionMap(itsInhibitionMap.getCudaArrayPtr(),factor,itsDynamicFactor*itsIorStrength.getVal(),itsIorRadius.getVal(),
                           itsSalMaxLoc[0].i,itsSalMaxLoc[0].j,tile.w(),tile.h(),w,h);
    }
  else
    itsInhibitionMap.clear(0);

}


void CudaSaliency::calcIntensity(const CudaImage<PixRGB<float> > & colImage, CudaImage<float>& outmap)
{
  if(gotLum == false)
  {
    curLum = cudaLuminance(colImage);
    // compute pyramid:
    curLumPyr = cudaBuildPyrGeneric(curLum, 0, maxdepth, Gaussian5, 0.0F);
    CudaImage<float> cmap = processPyramid(curLumPyr);
    // multiply by conspicuity coefficient:
    if (IWEIGHT != 1.0F) cmap *= IWEIGHT;
    // Add to saliency map:
    if (outmap.initialized()) outmap += cmap;
    else outmap = cmap;
    intensityMap=cmap;
    gotLum = true;
  }
}


void CudaSaliency::calcColor(const CudaImage<PixRGB<float> > & colImage, CudaImage<float>& outmap)
{
  if(gotRGBY == false)
  {
    cudaGetRGBY(colImage,rg,by,COLOR_THRESH);
    gotRGBY = true;
  }
  CudaImage<float> col=(rg+by)/2.0F;
  colorMap = cudaPostChannel(col,Gaussian5,0.0F,CWEIGHT,outmap);
}

void CudaSaliency::calcOrientation(const CudaImage<PixRGB<float> > & colImage, float orientation, CudaImage<float>& outmap)
{
  if(gotLum == false)
  {
    curLum = cudaLuminance(colImage);
    gotLum = true;
  }
  CudaImage<float> o = cudaPostChannel(curLum,Oriented5,orientation,OWEIGHT,outmap);
  if(orientationMap.initialized())
    orientationMap += o;
  else
    orientationMap = o;
}

void CudaSaliency::calcFlicker(const CudaImage<PixRGB<float> >& colImage, CudaImage<float>& outmap)
{
  CudaImage<float> curImage;
  if(gotLum == false)
  {
    curLum = cudaLuminance(colima);
    gotLum = true;
  }
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
  flickerMap = cudaPostChannel(curImage,Gaussian5,0.0F,FWEIGHT,outmap);
}

void CudaSaliency::calcMotion(const CudaImage<PixRGB<float> > & colImage, int motionIndex)
{
  if(gotLum == false)
  {
    curLum = cudaLuminance(colImage);
    gotLum = true;
  }

  CudaImageSet<float> motPyr = reichardtPyr[motionIndex].build(curLum,0,maxdepth);
  CudaImage<float> m = processPyramid(motPyr);
  if(motionMap.initialized())
    motionMap += m;
  else
    motionMap = m;
}

void CudaSaliency::runSaliency()
{
  Timer tim;
  gotSaliency = false;
  tim.reset();
  calcIntensity(colima,outmap);
  //LINFO("Intensity %fms", tim.getSecs() * 1000.0F);

  tim.reset();
  calcColor(colima,outmap);
  //LINFO("Color %fms", tim.getSecs() * 1000.0F);
  tim.reset();

  //LINFO("Orientation %fms", tim.getSecs() * 1000.0F);
  for(int i=0;i<numOrientationDirs;i++)
    {
      calcOrientation(colima,180.0*double(i)/double(numOrientationDirs),outmap);
    }

  tim.reset();
  calcFlicker(colima,outmap);
  //LINFO("Flicker %fms", tim.getSecs() * 1000.0F);

  for(int i=0;i<numMotionDirs;i++)
    {
      calcMotion(colima,i);
    }

  // Max norm the combined motion maps
  motionMap = cudaMaxNormalize(motionMap, MAXNORMMIN, MAXNORMMAX, normtyp,MAXNORMITERS);
  motionMap *=10;
  // Add to saliency map:
  if (outmap.initialized()) outmap += motionMap;
  else outmap = motionMap;
  // Subtract the inhibition map
  if(itsInhibitionMap.initialized())
    outmap -= itsInhibitionMap;
  cudaInplaceClamp(outmap,0,255);
  convmap = outmap;
  double w_ratio = outmap.getWidth()/double(colima.getWidth());
  double h_ratio = outmap.getHeight()/double(colima.getHeight());
  int smallPatchWidth = itsPatchSize.getVal()*w_ratio;
  int smallPatchHeight = itsPatchSize.getVal()*h_ratio;
  Dims rectDims = Dims(smallPatchHeight,smallPatchHeight);
  CudaImage<float> tmpmap = outmap;

  // Get the N most salient locations
  for(int i=0;i<itsNMostSalientLoc.getVal();i++)
    {
      cudaFindMax(tmpmap,itsSalMaxLoc[i],itsSalMax[i]);
      int modI = std::max(0,std::min(itsSalMaxLoc[i].i-smallPatchWidth/2,int(outmap.getWidth()-smallPatchWidth/2)));
      int modJ = std::max(0,std::min(itsSalMaxLoc[i].j-smallPatchHeight/2,int(outmap.getHeight()-smallPatchHeight/2)));  

      Rectangle tmprect = Rectangle(Point2D<int>(modI,modJ),rectDims);
      if(i+1<itsNMostSalientLoc.getVal())
	cudaDrawFilledRect(tmpmap,tmprect,0);
    }
  // Calculate inertia and inhibition
  calcInhibition(outmap);
  if(!itsInhibitionMap.initialized())
    itsInhibitionMap = CudaImage<float>(outmap.getDims(), ZEROS, mp, dev);
  //calcInertia(outmap);
  gotSaliency = true;
}
