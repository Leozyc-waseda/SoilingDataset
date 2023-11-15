

#ifndef COLOR_SEGMENTER_H
#define COLOR_SEGMENTER_H

//#include "BeoSub/HoughTransform.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Util/readConfig.H"


//from BeoSubTaskDecoder

#include "Component/ModelManager.H"
#include "GUI/XWindow.H"
#include "Raster/Raster.H"
#include "rutz/shared_ptr.h"
#include "Image/PixelsTypes.H"
#include "Util/readConfig.H"

#include "Image/ColorOps.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/MathOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Transforms.H"

#include "VFAT/segmentImageTrackMC.H"
#include <iostream>


template <class FLOAT,
          class INT,
          unsigned int SIT_channels> class segmentImageTrackMC;

//class is a combination of the functionality of the isolate color functions and the segmentcolor class
class ColorSegmenter{

public:

  ColorSegmenter();
  ~ColorSegmenter();
  void setupColorSegmenter(const char* inputColor, bool debug);
  //function for segmenting color
  float isolateOrange(Image< PixRGB<byte> > &inputImage,  Image<byte> &outputImage);

private:


  void setupOrange();
  std::vector<bool> frameCounter;
  float fps;
  const char* candidate_color;
  float Hz;
  float res;

  std::vector<float> color;
  std::vector<float> norm;
  std::vector<float> std;
  std::vector<float> adapt;
  std::vector<float> upperBound;
  std::vector<float> lowerBound;

  readConfig colorConf;

  rutz::shared_ptr<XWindow> wini, wino;

  int width, height;

  Image< PixRGB<byte> > ima;
  Image< PixRGB<float> > fima;
  Image< PixRGB<byte> > display;
  Image< PixH2SV2<float> > H2SVimage;

  segmentImageTrackMC<float,unsigned int,4> *segmenter;

  ImageSet< PixRGB<byte> > imgList;
  bool debugmode;
  bool setup;

};



#endif
