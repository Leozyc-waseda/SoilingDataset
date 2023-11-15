/*!@file SceneUnderstanding/TwoHalfDSketch.H  */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/TwoHalfDSketch.H $
// $Id: TwoHalfDSketch.H 14179 2010-10-28 19:49:24Z lior $
//

#ifndef TwoHalfDSketch_H_DEFINED
#define TwoHalfDSketch_H_DEFINED

#include "Image/OpenCVUtil.H" //Need to be first to avoid type def conf
#include "Util/Types.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "FeatureMatching/OriChamferMatching.H"
#include "plugins/SceneUnderstanding/V2.H"
#include "plugins/SceneUnderstanding/Contours.H"
#include "plugins/SceneUnderstanding/Regions.H"
#include "plugins/SceneUnderstanding/CornersFeatures.H"
#include "plugins/SceneUnderstanding/SMap.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
#include "GUI/ViewPort3D.H"

#include <vector>
#include <string>

class SimEventTwoHalfDSketchPrior;

#define INVALID_PROB -1e10

class TwoHalfDSketch : public SimModule
{
public:

  struct SurfaceState
  {
    Point2D<float> pos;
    Point2D<float> pos2;
    float a;
    float b;
    float e;
    float k1; //shearing in the x direction
    float k2; //shearing in the y direction
    float rot; //rotation
    float start;
    float end;
    double prob;
    int gibs;
    double scale;
    Point2D<float> aspect;
    Point2D<float> shear;
    float matchingScale;
    Polygon polygon;


    SurfaceState () :
      prob(INVALID_PROB)
    {}


    //Get a line up to scale and in the right position
    Line getLine(uint idx)
    {
      ASSERT(idx < polygon.getNumLines());

      Line l = polygon.getLine(idx);
      l.trans(pos);
      l.scale(1.0/matchingScale);
      return l;
    }

    //Get the position up to scale
    Point2D<double> getPos()
    {
      return pos*(1.0/matchingScale);
    }


    Rectangle getBB()
    {
      Rectangle bb;

      if(polygon.getNumLines() > 1)
      {

        Line l = getLine(0); 
        Point2D<int> p1 = (Point2D<int>)l.getP1(); 
        int minX=p1.i;
        int minY=p1.j;
        int maxX=minX;
        int maxY=minY;
        for(uint i=0;i<polygon.getNumLines();i++)
        {
          Line l = getLine(i); 
          Point2D<int> p1 = (Point2D<int>)l.getP1(); 
          if(p1.i < minX) minX = p1.i;
          if(p1.j < minY) minY = p1.j;
          if(p1.i > maxX) maxX = p1.i;
          if(p1.j > maxY) maxY = p1.j;

          Point2D<int> p2 = (Point2D<int>)l.getP2(); 
          if(p2.i < minX) minX = p2.i;
          if(p2.j < minY) minY = p2.j;
          if(p2.i > maxX) maxX = p2.i;
          if(p2.j > maxY) maxY = p2.j;
        }
        // Using inclusive coordinates since it works well when the polygon has 
        bb = Rectangle::tlbrI(minY,minX,maxY,maxX);
      }
      return bb;
    }
  };


  void findMax(const Image<SurfaceState>& src, Point2D<int>& p, SurfaceState& maxval) 
  { 
    ASSERT(src.initialized()); 
    Image<SurfaceState>::const_iterator aptr = src.begin(); 
    const int w = src.getWidth(), h = src.getHeight(); 
    p.i = 0; p.j = 0; maxval = *aptr; 
    for (int j = 0; j < h; j ++) 
      for (int i = 0; i < w; i ++) 
      { 
        if (aptr->prob > maxval.prob) { maxval = *aptr; p.i = i; p.j = j; } 
        aptr++; 
      } 
  } 

  void drawDisk(Image<SurfaceState>& dst, const Point2D<int>& center,
      const int radius)
  {
    if (radius == 1)
    {
      if (dst.coordsOk(center)) dst.setVal(center.i + dst.getWidth() * center.j, SurfaceState());
      return;
    }

    for (int y = -radius; y <= radius; ++y)
    {
      int bound = int(sqrtf(float(squareOf(radius) - squareOf(y))));
      for (int x = -bound; x <= bound; ++x)
        if (dst.coordsOk(x + center.i, y + center.j))
          dst.setVal(x + center.i, y + center.j, SurfaceState());
    }
  }

  Image<float> getProbImage(const Image<SurfaceState>& src)
  {
    Image<float> probImg(src.getDims(), NO_INIT);
    ASSERT(src.initialized()); 

    for(uint i=0; i<src.size(); i++)
      probImg[i] = src[i].prob;

    return probImg;
  }

  

  TwoHalfDSketch(OptionManager& mgr, const std::string& descrName = "TwoHalfDSketch",
      const std::string& tagName = "TwoHalfDSketch");

  //! Destructor
  ~TwoHalfDSketch();

  void evolve(SimEventQueue& q);

  Layout<PixRGB<byte> > getDebugImage(SimEventQueue& q);


  double calcNFA(Line& line);
  
  void drawSurface(const SurfaceState& surface);
  

  Image<SurfaceState>  proposeSurfaces(bool biasMode);
  void calcSurfaceLikelihood(SurfaceState& surface);
  double calcSurfaceEdgeLikelihood(SurfaceState& surface, Image<float>& edges, Image<float>& surfaceLum);
  double getEdgeProb(Point2D<int> loc, float ori);
  
  double calcSurfaceLumLikelihood(SurfaceState& surface, Image<float>& edges, Image<float>& surfaceLum);
  double getSurfaceLumProb(Image<float>& data, Image<float>& model);
  Image<PixRGB<byte> > getSurfaceImage(const SurfaceState& surface);
 
  double getLineProb(Point2D<int> p1, Point2D<int> p2, float ori, int& pixCount);
 
  double getCost(OriChamferMatching& lm, Polygon& poly,Point2D<float> loc, bool biasMode = false);
  
  Image<float> getSurfaceProbImage(Image<SurfaceState>& surfaceState);
  
  
  void printResults(float bias);

  void optimizePolygon(SurfaceState& surfaceState);

  double nfa(int n, int k, double p, double logNT);
  int isaligned(Point2D<int> loc, Image<float>& angles, float theta, float prec);
  double log_gamma_windschitl(double x);
  double log_gamma_lanczos(double x);

  typedef struct
  {
    float vx[4];
    float vy[4];
    float ys,ye;
    int x,y;
  } rect_iter;

  struct rect /* line segment with width */
  {
    float x1,y1,x2,y2;  /* first and second point of the line segment */
    float width;        /* rectangle width */
    float x,y;          /* center of the rectangle */
    float theta;        /* angle */
    float dx,dy;        /* vector with the line segment angle */
    float prec;         /* tolerance angle */
    double p;           /* probability of a point with angle within prec */
    float sum;          /* the sum of mag */
  };
  

  float inter_low(float x, float x1, float y1, float x2, float y2);
  float inter_hi(float x, float x1, float y1, float x2, float y2);
  void ri_del(rect_iter * iter);
  int ri_end(rect_iter * i);
  void ri_inc(rect_iter * i);
  rect_iter * ri_ini(struct rect * r);
  
  

protected:
  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(TwoHalfDSketch, SimEventV2Output);

  SIMCALLBACK_DECLARE(TwoHalfDSketch, SimEventContoursOutput);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(TwoHalfDSketch, SimEventSaveOutput);

  ////! Callback for every time we have a user event
  SIMCALLBACK_DECLARE(TwoHalfDSketch, SimEventUserInput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;


private:
  Image<PixRGB<byte> > itsCurrentFrame;
  Image<float> itsLinesMag;
  Image<float> itsLinesOri;
  Image<float> itsEdgesDT;
  ImageSet<float> itsOriEdgesDT;
  Image<byte> itsSMap;
  std::vector<V2::LineSegment> itsLines;
  std::vector<Contours::Contour> itsContours;
  std::vector<CornersFeatures::CornerState> itsCorners;
  std::vector<SurfaceState> itsSurfaces;
  std::vector<SurfaceState> itsProposals;
  SurfaceState itsUserProposal;
  Polygon itsModel;
  OriChamferMatching itsOriChamferMatcher;
  double itsProposalThreshold;
  double itsAcceptedThreshold;
  Image<float> itsProposalsProb;
  Image<float> itsProposalsIdxs;
  float itsCurrentProb;
  uint itsCurrentIdx;
  bool itsBiasMode;
  float itsBias;
  int itsBiasId;
  
};

/* ############################### V2 sim events ######################## */
class SimEventTwoHalfDSketchOutput : public SimEvent
{
public:
  SimEventTwoHalfDSketchOutput(SimModule* src,
      std::vector<TwoHalfDSketch::SurfaceState>& surfaces) :
    SimEvent(src), itsSurfaces(surfaces)
  {}

  virtual ~SimEventTwoHalfDSketchOutput(){}
  std::vector<TwoHalfDSketch::SurfaceState> getSurfaces() { return itsSurfaces; }

private:
  const std::vector<TwoHalfDSketch::SurfaceState>& itsSurfaces;
};



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
