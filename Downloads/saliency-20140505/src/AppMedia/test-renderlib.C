/*!@file AppMedia/stim-renderLib.C Psychophysics display for a search for a
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
#include "Image/Transforms.H" // for mosaic()
#include "Image/Layout.H"
#include "Psycho/PsychoDisplay.H"
#include "Psycho/PsychoOpts.H"
#include "Image/geom.h"
#include "Psycho/ClassicSearchItem.H"
#include "Psycho/SearchArray.H"
#include "Component/EventLog.H"
#include "Component/ComponentOpts.H"
#include "Raster/Raster.H"
#include "rutz/shared_ptr.h"
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

// Fill classes for different kinds of textures/colors/gradients
class fillTool {
public:
    fillTool() {}
    virtual Image<byte> fill(const Dims d) const {return Image<byte>(d,NO_INIT);}
    //  virtual void whoami() const {LINFO("I am a fillTool...");}
    virtual ~fillTool() { }
protected:
};

// brodatz only for now
class fillTexture : public fillTool {
public:
    Image<byte> fill(const Dims d) const; 
    virtual ~fillTexture() { }
    void setSeed(const uint c) {seed = c;}
    uint getSeed() const {return seed;} 
    //  void whoami() const {LINFO("I am a fillTexture...");}
protected:
    // brodatz seed
    uint seed;
};

// single, grayscale 'color'
class fillColor : public fillTool {
public:
    virtual ~fillColor() { }

    Image<byte> fill(const Dims d) const;

    void setColor(const byte c) {bg = c;}
    byte getColor() const {return bg;}
    //  void whoami() const {LINFO("I am a fillColor...");}

protected:
  byte bg;
};

// generic class of objects that can be drawn
class renderObject {
public:
  renderObject() {}
    virtual ~renderObject() {}
  virtual void drawOn(Image<byte> &im) const = 0;
  virtual bool inBounds(const Dims &d) const = 0;
  Point2D<int> getLocation() {return loc;} const
  void setLocation(const Point2D<int> &P) {loc = P;}

  int getSize() {return siz;} const
  void setSize(const int &s) {siz = s;}

  float getDirection() {return dir;} const
  void setDirection(const float &t) {dir = t;}

protected:
  Point2D<int> loc; 
  float dir; //angle
  int siz;
};

class renderDot : public renderObject {
public: 
    virtual ~renderDot() {}
  void drawOn(Image<byte> &im) const;
  bool inBounds(const Dims &d) const
  {return 0 <= loc.i - dir && 0 <= loc.j - dir && loc.i + dir < d.w() && loc.j + dir < d.h();}

  void setFill(rutz::shared_ptr<fillTool> f) {filler = f;}
  rutz::shared_ptr<fillTool> getFill() const {return filler;} 
private:
  rutz::shared_ptr<fillTool> filler;
};

// Generate a random integer uniformly in (x,y)
int randomInRange(const int x, const int y);

// Generate a random point uniformly in d
//geom::vec2d randomPtIn(const Dims d);
Point2D<int> randomPointIn(const Dims d);
//geom::vec2d randomPtIn(const Rectangle d);

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
  char filename[255], texfile[255], fillfile[255];

  OModelParam<Dims> dims(&OPT_RandImageDims, &manager);

  // get command line parameters for filename
  sprintf(filename, "%s.png",manager.getExtraArg(0).c_str());
  sprintf(texfile, "%s-tex.png",manager.getExtraArg(0).c_str());
  sprintf(fillfile, "%s-fill.png",manager.getExtraArg(0).c_str());

  // let's get all our ModelComponent instances started:
  manager.start();

  // **************** Experimental settings *************** //

  // number of available noise frames in the stimuli folder
  //  const uint Nnoises = 100;

  //tests
  Image<byte> myBkgd = Image<byte>(dims.getVal(),ZEROS)+128; //gray bkgd
  Image<byte> myMap = myBkgd;
  rutz::shared_ptr<fillTexture> style(new fillTexture);
  rutz::shared_ptr<fillColor> color(new fillColor);

  rutz::shared_ptr<renderDot> dot(new renderDot);  
  dot->setSize(50);

  int rseed, rcol;
  for(int i = 0; i < 50; i++) {
    rseed = randomUpToIncluding(255);
    rcol = randomUpToIncluding(1);
    if(rcol==0) {
      style->setSeed(rseed);
      dot->setFill(style);
    }
    else {
      color->setColor(rseed);
      dot->setFill(color);
    }
    dot->setLocation(randomPointIn(dims.getVal()));
    dot->drawOn(myMap);
  }
  // test texture
  LINFO("writing pattern image to %s", filename);
  Raster::WriteGray(myMap,filename);

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
Image<byte> fillTexture::fill(const Dims d) const {
    return getBrodatzTexture(seed, d);
} 

// ######################################################################
Image<byte> fillColor::fill(const Dims d) const {
  return Image<byte>(d,ZEROS) + bg;
} 

// ######################################################################
void renderDot::drawOn(Image<byte> & im) const
  {
    const byte mask_col = 255;

    Image<byte> mask(im.getDims(), ZEROS);
    drawDisk(mask, loc, siz, mask_col);

    Image<byte> myPrint = filler->fill(im.getDims());
    std::vector<Image<byte> > bgs;
    bgs.push_back(im); bgs.push_back(myPrint);
    byte bg_assigns[2] = {0, mask_col};
    im = mosaic(mask, &bgs[0], bg_assigns, 2);
  }

// ######################################################################
// Generate a random integer uniformly in (x,y) (open interval)
int randomInRange(const int x, const int y)
{
  return randomUpToNotIncluding(y-x-1)+(x+1);
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
