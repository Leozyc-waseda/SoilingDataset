#include "RegSaliency.H"
#include "Image/PyrBuilder.H"
#include "Util/Timer.H"

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
#define MAXNORMITERS 1

// ######################################################################
RegSaliency::RegSaliency(OptionManager& mgr,
           const std::string& descrName,
           const std::string& tagName):
  ModelComponent(mgr, descrName, tagName)
{
  gotLum = false; gotRGBY = false; gotSaliency = false;
  numMotionDirs = 4;
  numOrientationDirs = 4;
  //cudaSetSeed(dev);
  //cudaSizeRandomBuffer(randBuf,mp,dev);
}

// ######################################################################
void RegSaliency::start1()
{
  reichardtPyr = new ReichardtPyrBuilder<float>*[numMotionDirs];
  for(int i=0;i<numMotionDirs;i++)
    {
      double direction = 360.0*double(i)/double(numMotionDirs);
      reichardtPyr[i] =
        new ReichardtPyrBuilder<float>(cos(direction*M_PI/180.0),-sin(direction*M_PI/180.0),
                                       Gaussian5,direction+90.0);
    }
}

// ######################################################################
void RegSaliency::stop2()
{
  for(int i=0;i<numMotionDirs;i++)
    {
      delete reichardtPyr[i];
    }
  delete [] reichardtPyr;
}

// ######################################################################
RegSaliency::~RegSaliency()
{ }

// ######################################################################
void RegSaliency::doInput(const Image< PixRGB<byte> > img)
{
  //LINFO("new input.....");
  Image<PixRGB<float> > fimg = img;
  colima = Image<PixRGB<float> >(fimg);
  if(outmap.initialized())
    outmap.clear();
  // Reset the flags
  gotLum = false; gotRGBY = false;
  // Clear the outmap
  // also kill any old output and internals:
  Timer tim;
  printf("Timing saliency\n");
  tim.reset();
  runSaliency();
  LINFO("Done! %fms", tim.getSecs() * 1000.0F);
}

// ######################################################################
bool RegSaliency::outputReady()
{
  return gotSaliency;
}

// ######################################################################
Image<float> RegSaliency::getOutput()
{
  return convmap;
}

Image<float> RegSaliency::getIMap()
{
  return intensityMap;
}

Image<float> RegSaliency::getCMap()
{
  return colorMap;
}

Image<float> RegSaliency::getOMap()
{
  return orientationMap;
}

Image<float> RegSaliency::getFMap()
{
  return flickerMap;
}

Image<float> RegSaliency::getMMap()
{
  return motionMap;
}

Image<float> RegSaliency::getInertiaMap()
{
  return itsInertiaMap;
}

Image<float> RegSaliency::getInhibitionMap()
{
  return itsInhibitionMap;
}



Image<float> RegSaliency::postChannel(Image<float> curImage, PyramidType ptyp, float orientation, float weight, Image<float>& outmap)
{
  // compute pyramid:
  ImageSet<float> pyr =
    buildPyrGeneric(curImage, 0, maxdepth, ptyp, orientation);

  Image<float> cmap = processPyramid(pyr);

  // multiply by conspicuity coefficient:
  if (weight != 1.0F) cmap *= weight;
  // Add to saliency map:
  if (outmap.initialized()) outmap += cmap;
  else outmap = cmap;
  return cmap;
}


Image<float> RegSaliency::processPyramid(ImageSet<float> pyr)
{
  // alloc conspicuity map and clear it:
  Image<float> cmap;

  // intensities is the max-normalized weighted sum of IntensCS:
  for (int delta = delta_min; delta <= delta_max; delta ++)
    for (int lev = level_min; lev <= level_max; lev ++)
      {
        Image<float> tmp = centerSurround(pyr, lev, lev + delta, true);
        tmp = downSize(tmp, pyr[sml].getWidth(), pyr[sml].getHeight());
        tmp = maxNormalize(tmp, MAXNORMMIN, MAXNORMMAX, normtyp,MAXNORMITERS);
        if(cmap.initialized())
          cmap += tmp;
        else
          cmap = tmp;
      }

  inplaceAddBGnoise(cmap, 25.0F);


  if (normtyp == VCXNORM_MAXNORM)
    cmap = maxNormalize(cmap, MAXNORMMIN, MAXNORMMAX, normtyp,MAXNORMITERS);
  else
    cmap = maxNormalize(cmap, 0.0f, 0.0f, normtyp,MAXNORMITERS);


  return cmap;
}



void RegSaliency::calcIntensity(const Image<PixRGB<float> > & colImage, Image<float>& outmap)
{
  if(gotLum == false)
  {
    curLum = luminance(colImage);
    // compute pyramid:
    curLumPyr = buildPyrGeneric(curLum, 0, maxdepth, Gaussian5, 0.0F);
    Image<float> cmap = processPyramid(curLumPyr);
    // multiply by conspicuity coefficient:
    if (IWEIGHT != 1.0F) cmap *= IWEIGHT;
    // Add to saliency map:
    if (outmap.initialized()) outmap += cmap;
    else outmap = cmap;
    intensityMap=cmap;
    gotLum = true;
  }
}


void RegSaliency::calcColor(const Image<PixRGB<float> > & colImage, Image<float>& outmap)
{

  if(gotRGBY == false)
  {
    getRGBY(colImage,rg,by,COLOR_THRESH);
    gotRGBY = true;
  }
  Image<float> col=(rg+by)/2.0F;
  colorMap = postChannel(col,Gaussian5,0.0F,CWEIGHT,outmap);
}

void RegSaliency::calcOrientation(const Image<PixRGB<float> > & colImage, float orientation, Image<float>& outmap)
{
  if(gotLum == false)
  {
    curLum = luminance(colImage);
    gotLum = true;
  }
  Image<float> o = postChannel(curLum,Oriented5,orientation,OWEIGHT,outmap);
  if(orientationMap.initialized())
    orientationMap += o;
  else
    orientationMap = o;
}

void RegSaliency::calcFlicker(const Image<PixRGB<float> >& colImage, Image<float>& outmap)
{
  Image<float> curImage;
  if(gotLum == false)
  {
    curLum = luminance(colima);
    gotLum = true;
  }
  if (prevLum.initialized() == false)
    {
      prevLum = curLum;
      curImage = Image<float>(curLum.getDims(), ZEROS);
    }
  else
    {
      curImage = curLum - prevLum;
      prevLum = curLum;
    }
  flickerMap = postChannel(curImage,Gaussian5,0.0F,FWEIGHT,outmap);
}

void RegSaliency::calcMotion(const Image<PixRGB<float> > & colImage, int motionIndex)
{
  if(gotLum == false)
  {
    curLum = luminance(colImage);
    gotLum = true;
  }

  ImageSet<float> motPyr = (reichardtPyr[motionIndex])->build(curLum,0,maxdepth);
  Image<float> m = processPyramid(motPyr);
  if(motionMap.initialized())
    motionMap += m;
  else
    motionMap = m;
}


void RegSaliency::runSaliency()
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
  motionMap = maxNormalize(motionMap, MAXNORMMIN, MAXNORMMAX, normtyp,MAXNORMITERS);
  // Add to saliency map:
  if (outmap.initialized()) outmap += motionMap;
  else outmap = motionMap;

  convmap = outmap;
  gotSaliency = true;
}
