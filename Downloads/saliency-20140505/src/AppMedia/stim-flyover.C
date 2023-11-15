/*!@file AppPsycho/psycho-noisecuing.C Psychophysics display for a search for a
  target that is presented in various repeated noise backgrounds */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/psycho-searchGabor.C $
// $Id: psycho-searchGabor.C 10794 2009-02-08 06:21:09Z itti $
//

#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Image/ColorOps.H" // for makeRGB()
#include "Image/CutPaste.H" // for inplacePaste()
#include "Image/DrawOps.H" // for drawLine()
#include "Image/Image.H"
#include "Image/MathOps.H"  // for inplaceSpeckleNoise()
#include "Image/LowPass.H" // for LowPass5x, LowPass5y
#include "Image/ShapeOps.H" // for rescale()
#include "Image/Transforms.H"
#include "Image/Layout.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/EyeTrackerConfigurator.H"
#include "Psycho/EyeTracker.H"
#include "Psycho/PsychoOpts.H"
#include "Image/geom.h"
#include "Psycho/ClassicSearchItem.H"
#include "Psycho/SearchArray.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Raster/Raster.H"
#include "Util/MathFunctions.H"
#include "Util/StringUtil.H"
#include "Util/StringConversions.H"
#include "GUI/GUIOpts.H"

#include <sstream>
#include <ctime>
#include <ctype.h>
#include <vector>
#include <string>
#include <fstream>

using namespace std;

static const ModelOptionCateg MOC_RANDGENIMAGE = {
  MOC_SORTPRI_2, "Options for random image generation" };

static const ModelOptionDef OPT_RandImageDims =
  { MODOPT_ARG(Dims), "GenImageDims", &MOC_RANDGENIMAGE, OPTEXP_CORE,
    "dimensions of the random image",
    "rand-image-dims", '\0', "<width>x<height>", "1280x720" };
/*
static const ModelOptionDef OPT_TextureLibrary =
  { MODOPT_FLAG, "RemoveMagnitude", &MOC_RANDIMAGE, OPTEXP_CORE,
    "remove the phase component of an image",
    "remove-magnitude", '\0', "--[no]remove-phase", "false" };
*/
static const ModelOptionDef OPT_NoiseMagnitude =
  { MODOPT_ARG(float), "NoiseMagnitude", &MOC_RANDGENIMAGE, OPTEXP_CORE,
    "adjust the magnitude of the overlaid noise, from 0.0 to 1.0",
    "noise-magnitude", '\0', "<float>", "0.0"};

static const ModelOptionDef OPT_NoiseColor =
  { MODOPT_ARG(std::string), "NoiseColor", &MOC_RANDGENIMAGE, OPTEXP_CORE,
    "give the color of the overlaid noise, in white, pink, or brown",
    "noise-color", '\0', "[white,pink,brown]", "white"};

//! number of frames in the mask
//#define NMASK 10

// Trial design for contextual cueing experiments.  May class this up soon. 
// Also may make Trial/Experiment classes as ModelComponents of this style.
// Easier to encapsulate experimental designs as separate from trial content.

// But for now it's easier to keep everything public.
enum NoiseColor { WHITE, PINK, BROWN };
struct trialAgenda
{
  bool repeated;
  NoiseColor color;
  geom::vec2d targetloc;
  uint noiseSeed;
  trialAgenda(const bool r, const NoiseColor c, 
              const geom::vec2d t, const uint n)
  {
    repeated = r;
    color = c;
    targetloc = t;
    noiseSeed = n;
  }

  std::string colname() const
  {
    switch(color) {
    case WHITE: return "white";
    case PINK:  return "pink";
    case BROWN: return "brown";
    }
    return "";
  }

  void randomizeNoise(const uint Nbkgds)
  {
    noiseSeed = randomUpToNotIncluding(Nbkgds)+1;
  }

  std::string backgroundFile() const
  {
    std::string stimdir = "/lab/jshen/projects/eye-cuing/stimuli/noiseseeds";
    return sformat("%s/%s%03d.png",stimdir.c_str(),colname().c_str(),noiseSeed);
  }
};

std::string convertToString(const trialAgenda& val)
{
  std::stringstream s; 
  s << val.colname() << " noise, ";
  if (val.repeated)
    s << "repeated, seed " << val.noiseSeed;
  else
    s << "random";

  s << ", target @ (" << val.targetloc.x() << "," << val.targetloc.y() << ")";
  return s.str();
}

// Generate a random integer uniformly in (x,y);
int randomInRange(const int x, const int y)
{
  return randomUpToNotIncluding(y-x-1)+(x+1);
}

// Generate a random point uniformly in d
geom::vec2d randomPtIn(const Dims d);
// Generate a random point uniformly in d
Point2D<int> randomPointIn(const Dims d);

// Generate a random point uniformly in d
geom::vec2d randomPtIn(const Rectangle d)
{
  return geom::vec2d(randomInRange(d.left(),d.rightO()),
                     randomInRange(d.top(),d.bottomO()));
}

Image<byte> plainBkgd(const trialAgenda A)
{
  //skips file validation step
  //  return getPixelComponentImage(Raster::ReadRGB(A.backgroundFile()),0); 
  return Raster::ReadGray(A.backgroundFile(),RASFMT_PNG);
//the second arg can be 0,1,2 since the image is B&W
}

Image<PixRGB<byte> > colorizeBkgd(const trialAgenda A,const uint Nbkgds)
{
  trialAgenda B = A;

  std::vector<Image<byte> > comp;
  for (uint i = 0; i < 3; i++) {
    if(!B.repeated) //randomize noise
      B.randomizeNoise(Nbkgds);  
    else //systematically step
      B.noiseSeed = (B.noiseSeed)%Nbkgds+1;

    //skips file validation step
    comp.push_back(plainBkgd(B));
  }
  return makeRGB(comp[0],comp[1],comp[2]);
}

void drawRandomLine(Image<byte> & im, const byte val, const int thickness);
void extrapolateLine(Dims d, Point2D<int> & X, Point2D<int> & Y);

template <class T>
Image<byte> makeNary(const Image<T>& src, const std::vector<T> thresholds,
                     const std::vector<byte> levels);


Image<byte> texturizeImage(const Image<byte> im, const uint Nlevels);
Image<byte> discretizeImage(const Image<byte> im, const int Nlevels);
Image<byte> getBrodatzTexture(uint seed, const Dims dims);
Image<byte> getStretchedTexture(const std::string filename, const Dims dims);
Image<byte> getTiledTexture(const std::string filename, const Dims dims);

// ######################################################################
static int submain(const int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;  // suppress debug messages

  // Instantiate a ModelManager:
  ModelManager manager("AppMedia: Flyover stimulus");

  // get dimensions of window
  if (manager.parseCommandLine(argc, argv,"<out_stem>", 1, 1) == false)
    return(1);

  //  Image<float> myFloatImg(dims.w(), dims.h(), ZEROS);
  //Image<double> myDoubleImg(dims.w(), dims.h(), ZEROS);
  char filename[255], texfile[255];

  OModelParam<Dims> dims(&OPT_RandImageDims, &manager);
  OModelParam<float> noiseMag(&OPT_NoiseMagnitude, &manager);
  OModelParam<std::string> noiseColor(&OPT_NoiseColor, &manager);

  // get command line parameters for filename
  sprintf(filename, "%s.png",manager.getExtraArg(0).c_str());
  sprintf(texfile, "%s-1.png",manager.getExtraArg(0).c_str());

  /*
  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);
  */

  // let's get all our ModelComponent instances started:
  manager.start();

  // **************** Experimental settings *************** //

  // number of available noise frames in the stimuli folder
  const uint Nnoises = 100;

  /*
  // size of screen - should be no bigger than 1920x1080
  const Dims screenDims = 
    fromStr<Dims>(manager.getOptionValString(&OPT_SDLdisplayDims));
  
  // target/distractor type: we have choice of c o q - t l +
  const std::string Ttype = "T", Dtype = "L";
  const double phiMax = M_PI*5/8, phiMin = M_PI*3/8; 
  
  ClassicSearchItemFactory 
    targetsLeft(SearchItem::FOREGROUND, Ttype, 
                itemsize,
                Range<double>(-phiMax,-phiMin)),
    targetsRight(SearchItem::FOREGROUND, Ttype, 
                 itemsize,
                 Range<double>(phiMin,phiMax)),
    distractors(SearchItem::BACKGROUND, Dtype, 
                itemsize,
                Range<double>(-M_PI/2,M_PI/2));
  */
  // ******************** Trial Design ************************* //

  std::vector<rutz::shared_ptr<trialAgenda> > trials;
  const uint Ntrials = 1;
  const uint Nrepeats = 1;
  NoiseColor colors[Ntrials];
  bool rep[Ntrials];

  //  SearchArray sarray(dims, grid_spacing, min_spacing, itemsize);
  const PixRGB<byte> gray(128,128,128);

  // Design and shuffle trials
  initRandomNumbers();
  for (uint i = 0; i < Ntrials; i++) 
    {
      colors[i] = BROWN; //NoiseColor(i%3); 
      rep[i] = (i < Nrepeats);
    }

  for (uint i = 0; i < Ntrials; i++)
    {
      // a random location for each target
      const geom::vec2d pos = randomPtIn(dims.getVal());
      // a random seed for each trial
      const uint seed = randomInRange(1,Nnoises);
      trials.push_back
            (rutz::shared_ptr<trialAgenda>
             (new trialAgenda(rep[i],colors[i],pos,seed)));
    }

  //tests
  Image<byte> myMap = discretizeImage(plainBkgd(*(trials[0])),4);
  Image<byte> myBkgd = texturizeImage(myMap,4);

  LINFO("writing texture image to %s", filename);
  Raster::WriteGray(myBkgd,filename);
  
  // test texture
  LINFO("writing pattern image to %s", texfile);
  Raster::WriteGray(myMap,texfile);

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################

extern "C" int main(const int argc, char** argv)
{
  // simple wrapper around submain() to catch exceptions (because we
  // want to allow PsychoDisplay to shut down cleanly; otherwise if we
  // abort while SDL is in fullscreen mode, the X server won't return
  // to its original resolution)
  try
    {
      return submain(argc, argv);
    }
  catch (...)
    {
      REPORT_CURRENT_EXCEPTION;
    }

  return 1;
}

// ######################################################################

// draw a random line 
void drawRandomLine(Image<byte> & im, const byte val, const int thickness)
{
  Point2D<int> P = randomPointIn(im.getDims());
  Point2D<int> Q = randomPointIn(im.getDims());

  extrapolateLine(im.getDims(),P,Q);
  drawLine(im, P, Q, val, thickness);
}

// ######################################################################

// extend a line segment to boundaries
void extrapolateLine(Dims d, Point2D<int> & X, Point2D<int> & Y)
{
  // check if X and Y are in d
  Image<byte> foo(d, NO_INIT);
  if(!(foo.coordsOk(X) && foo.coordsOk(Y)) || X == Y) return;

  if (X.i == Y.i) {X.j = 0; Y.j = d.h(); return;}
  else if(X.j == Y.j) {X.i = 0; Y.j = d.w(); return;}
  else {float y_0 = (X.j*Y.i-X.i*Y.j)/(Y.i-X.i);
    float x_0 = (X.i*Y.j-X.j*Y.i)/(Y.j-X.j);
    float slope = (Y.j-X.j)/(Y.i-X.i);
    
    std::vector<Point2D<int> > bounds; 
    bounds.push_back(Point2D<int>(0,y_0));
    bounds.push_back(Point2D<int>(x_0,0));
    bounds.push_back(Point2D<int>(d.w()-1,y_0+(d.w()-1)*slope));
    bounds.push_back(Point2D<int>(x_0+(d.h()-1)/slope,d.h()-1));

    bool Xdone = 0;
    for(int i = 0; i < 4; i++)
      if(foo.coordsOk(bounds[i])) { 
        if(!Xdone) {
          X = bounds[i]; 
          Xdone = true;
        }
        else {
            Y = bounds[i]; 
            break;
        }
      }
  }
}

// ######################################################################

// Generate a random point uniformly in d
geom::vec2d randomPtIn(const Dims d)
{
  return geom::vec2d(randomInRange(0,d.w()),
                     randomInRange(0,d.h()));
}

// Generate a random point uniformly in d
Point2D<int> randomPointIn(const Dims d)
{
  return Point2D<int>(randomInRange(0,d.w()),
                      randomInRange(0,d.h()));
}

// ######################################################################

// inspired from makeBinary in Transforms.C
template <class T>
Image<byte> makeNary(const Image<T>& src, const std::vector<T> thresholds,
                  const std::vector<byte> levels)
{
  ASSERT(thresholds.size() == levels.size() - 1);
  Image<byte> acc(src.getDims(),ZEROS);
  byte floor;
  for(uint i = 0; i < thresholds.size(); i++)
    {
      if(i == 0) 
        {
          floor = levels[0];
        }
    else 
      {
        floor = 0;
      }
    acc += makeBinary(src, thresholds[i],floor,levels[1]);
    }

  return acc;
}

// ######################################################################
// maps discretized image to textured image
Image<byte> texturizeImage(const Image<byte> im, const uint Nlevels)
{
  uint i, seed;
  //  Image<byte> levelImage = discretizeImage(im, Nlevels);
  std::vector<Image<byte> > texBkgds;
  for(i = 0; i < Nlevels; i++)
    {
      seed = randomInRange(0,112); // num brodatz images
      texBkgds.push_back(getBrodatzTexture(seed, im.getDims()));
    }

  byte tiers[Nlevels];
  for(uint i = 0; i < Nlevels; i++) 
      tiers[i] = i*(255/(Nlevels-1));

  return mosaic(im, &texBkgds[0], tiers, Nlevels);
  //return mosaic(levelImage, &texBkgds[0], tiers, Nlevels);
}

// ######################################################################
// discretizes image
Image<byte> discretizeImage(const Image<byte> im, const int Nlevels)
{
  byte imMin, imMax, i;
  getMinMax(im, imMin, imMax);

  const byte Ncuts = Nlevels - 1;
  
  // the ratios that partition the image
  float coeffs[Ncuts];
  for(i = 0; i < Ncuts; i++)
      coeffs[i] = (i+1.0)/(Ncuts+1.0);

  // the values in the noise image that partition the image
  std::vector<byte> cuts;
  for(i = 0; i < Ncuts; i++)
      cuts.push_back(imMax*coeffs[i]+imMin*(1-coeffs[i])); 

  // the mapped values of the outside image
  std::vector<byte> tiers;
  for(i = 0; i <= cuts.size(); i++) 
      tiers.push_back(i*(255/cuts.size()));

  // use makeNary to cut the image
  Image<byte> pattern = makeNary(im,cuts,tiers);

  // draw a random line cutting across the image
  drawRandomLine(pattern, tiers[2], 50);

  return pattern;
  
} 

// ######################################################################
Image<byte> getBrodatzTexture(uint seed, const Dims dims)
{
  char texPath[255];
  
  // there are only 111 images in the brodatz database
  const uint Nimages = 111;
  seed = seed % (Nimages - 1) + 1;

  sprintf(texPath, "/lab/jshen/projects/eye-cuing/stimuli/textures/brodatz/D%u.png",seed);
  return getStretchedTexture(texPath, dims);
}

// ######################################################################
Image<byte> getStretchedTexture(const std::string filename, const Dims dims)
{
  Image<byte> pat = Raster::ReadGray(filename,RASFMT_PNG);
  return rescale(pat, dims);
}

// ######################################################################
Image<byte> getTiledTexture(const std::string filename, const Dims dims)
{
  //filename refers to a simple texture, black and white, PNG
  Image<byte> pat = Raster::ReadGray(filename,RASFMT_PNG);

  const size_t nX = dims.w()/pat.getWidth()+1;
  const size_t nY = dims.h()/pat.getHeight()+1;

  std::vector<Image<byte> > tiles(nX,pat);
  Layout<byte> horiztile(&tiles[0],nX,Layout<byte>::H);

  std::vector<Layout<byte> > rows(nY,horiztile);
  Layout<byte> whole(&rows[0],nY,Layout<byte>::V);

  return crop(whole.render(),Point2D<int>(0,0),dims);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
