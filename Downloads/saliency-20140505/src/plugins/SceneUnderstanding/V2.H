/*!@file plugins/SceneUnderstanding/V2.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/V2.H $
// $Id: V2.H 14683 2011-04-05 01:30:59Z lior $
//

#ifndef V2_H_DEFINED
#define V2_H_DEFINED

//#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "Image/TensorOps.H"
#include "plugins/SceneUnderstanding/V1.H"
#include "plugins/SceneUnderstanding/TensorVoting.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
#include "Util/WorkThreadServer.H"
#include <float.h>

//#include "Image/OpenCVUtil.H"

#include <limits>
#include <vector>
#include <string>
#include <stack>
#include <list>

#define NOTDEF   -1000.0
#define BIG_NUMBER 1.0e+300
#define M_3_2_PI 4.71238898038
#define M_2__PI  6.28318530718
#define NOTUSED 0
#define USED    1
#define NOTINI  2

#define MY_ROUND(f) ( (float) floor( (double) (f) + 0.5 ) )

struct NFAInfo
{
  short int ori;
  short int xLoc;
  short int yLoc;
  short int length;
  short int pts;
  short int alg;
};

struct NumEntries
  {
    size_t num;
    short int pts;
    NumEntries() : num(0), pts(0) {}

  };

//A PDF width 4 params
typedef std::map<short int, //ori
        std::map<short int, //xLoc
        std::map<short int, //yLoc,
        std::map<short int, //length, 
        std::map<short int, //alg, 
        NumEntries > > > > > NFAPDFMap;



class Rect
{

  public:

  Rect(const Point2D<int>& center, Dims& d, float ori) :
    itsCenter(center),
    itsSize(d),
    itsOri(ori)
  {
    float distX = itsSize.w()/2;
    float distY = itsSize.h()/2;
    float radius = sqrt((distX*distX) + (distY*distY));

    float thetaO = atan(distX/distY);
    float thetaN1 = thetaO + itsOri;
    float thetaN2 = itsOri - thetaO;
    float Xnew1 = radius*sin(thetaN1);
    float Ynew1 = radius*cos(thetaN1);
    float Xnew2 = radius*sin(thetaN2);
    float Ynew2 = radius*cos(thetaN2);

    itsP1 = Point2D<int>(itsCenter.i+Xnew1, itsCenter.j+Ynew1);
    itsP2 = Point2D<int>(itsCenter.i+Xnew2, itsCenter.j+Ynew2);
    itsP3 = Point2D<int>(itsCenter.i-Xnew1, itsCenter.j-Ynew1);
    itsP4 = Point2D<int>(itsCenter.i-Xnew2, itsCenter.j-Ynew2);

    itsPolygon.clear();
    itsPolygon.push_back(itsP1); itsPolygon.push_back(itsP2);
    itsPolygon.push_back(itsP3); itsPolygon.push_back(itsP4);
  }

  void initIter()
  {
    itsEdgeCount = itsPolygon.size();
    itsNum = itsPolygon.size();
    itsMinY = itsPolygon[0].j; 
    itsStartV1 = 0;
    for(uint c= 1; c < itsPolygon.size(); c++) //Find Top Vertex
    {
      if (itsPolygon[c].j < itsMinY)
      {
        itsMinY = itsPolygon[c].j;
        itsStartV1 = c;
      }
    }
    itsStartV2 = itsStartV1;
    itsEndV1 = itsStartV1 - 1;
    if(itsEndV1 < 0) itsEndV1 = (itsNum-1);

    itsEndV2 = itsStartV2 + 1;
    if (itsEndV2 >= itsNum) itsEndV2 = 0;
    itsMinY = itsPolygon[itsStartV1].j;

    int x1 = itsPolygon[itsStartV1].i; int y1 = itsPolygon[itsStartV1].j;
    int x2 = itsPolygon[itsEndV1].i; int y2 = itsPolygon[itsEndV1].j;

    itsDx1 = ((x2 - x1) << 8) / (y2 - y1 + 1);
    itsCount1 = y2-y1;
    itsXVal1 = x1 << 8;

    int x11 = itsPolygon[itsStartV2].i; int y11 = itsPolygon[itsStartV2].j;
    int x22 = itsPolygon[itsEndV2].i; int y22 = itsPolygon[itsEndV2].j;

    itsDx2 = ((x22 - x11) << 8) / (y22 - y11 + 1);
    itsCount2 = y22-y11;
    itsXVal2 = x11 << 8;
  }

  bool iterEnd()
  {
    if (itsEdgeCount > 1)
      return false;
    else
      return true;
  }

  void incIter(int& x1, int& x2, int& y)
  {
    if (itsEdgeCount > 1)
    {
      while( itsEdgeCount > 1 && (itsCount1 == 0 || itsCount2 == 0) )
      {
        if (itsCount1 == 0)
        {
          itsEdgeCount--;
          itsStartV1 = itsEndV1;
          itsEndV1--;
          if (itsEndV1 < 0) itsEndV1 = itsNum-1;

          itsMinY = itsPolygon[itsStartV1].j;
          int x1 = itsPolygon[itsStartV1].i, y1 = itsPolygon[itsStartV1].j;
          int x2 = itsPolygon[itsEndV1].i, y2 = itsPolygon[itsEndV1].j;
          itsDx1 = ((x2 - x1) << 8) / (abs(y2 - y1) + 1);
          itsCount1 = y2-y1;
          itsXVal1 = x1 << 8;
        }
        if (itsCount2 == 0)
        {
          itsEdgeCount--;
          itsStartV2 = itsEndV2;
          itsEndV2++;
          if(itsEndV2 >= itsNum) itsEndV2 = 0;
          itsMinY = itsPolygon[itsStartV2].j;
          int x11 = itsPolygon[itsStartV2].i, y11 = itsPolygon[itsStartV2].j;
          int x22 = itsPolygon[itsEndV2].i, y22 = itsPolygon[itsEndV2].j;
          itsDx2 = ((x22 - x11) << 8) / (abs(y22 - y11) + 1);
          itsCount2 = y22-y11;
          itsXVal2 = x11 << 8;
        }
      }
      if ( (itsCount1 > 0) &&  (itsCount2 > 0) )
      {
        x1 = itsXVal1 >> 8;
        x2 = itsXVal2 >> 8;
        y = itsMinY;

        itsXVal1 += itsDx1; itsXVal2 += itsDx2;
        itsCount1--; itsCount2--;
        itsMinY++;
      }
    }

  }

  std::vector<Point2D<int> > getPolygon() { return itsPolygon; }



  private:
    Point2D<int> itsCenter;
    Dims itsSize;
    float itsOri;

    Point2D<int> itsP1;
    Point2D<int> itsP2;
    Point2D<int> itsP3;
    Point2D<int> itsP4;


    int itsEdgeCount;
    int itsNum;
    int itsMinY;
    int itsStartV1;
    int itsStartV2;
    int itsEndV1;
    int itsEndV2;

    int itsDx1;
    int itsCount1;
    int itsXVal1;

    int itsDx2;
    int itsCount2;
    int itsXVal2;

    std::vector<Point2D<int> > itsPolygon;
};

class V2 : public SimModule
{
public:

  struct CornerState
  {
    Point2D<float> pos;
    float ori;
    float ang;
    float var;
    float prob;

    int lineSeg1;
    int lineSeg2;

    Point2D<float> endPoint1;
    Point2D<float> endPoint2;


    CornerState() {}
    CornerState(Point2D<float> p, float o, float a) :
      pos(p), ori(o), ang(a)
    {}

    bool operator<(const CornerState& cornerState) const
    {
      return prob < cornerState.prob;
    }

  };

  struct SymmetryState
  {
    int lineSeg1;
    int lineSeg2;
    Point2D<float> center;
    float ang;
    float length;
  };


  /*----------------------------------------------------------------------------*/
  struct coorlist
  {
    int x,y;
    struct coorlist * next;
  };

  /*----------------------------------------------------------------------------*/
  struct point {int x,y;};

  /*----------------------------------------------------------------------------*/
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

  /*----------------------------------------------------------------------------*/
  /*
     rectangle points iterator
     */
  typedef struct
  {
    float vx[4];
    float vy[4];
    float ys,ye;
    int x,y;
  } rect_iter;


  struct LineSegment
  {
    Point2D<float> p1;
    Point2D<float> p2;
    float width;
    float length;
    float ori;
    int color;
    Point2D<float> center;
    std::vector<float> side1Color;
    std::vector<float> side2Color;
    float prob;

    float strength;

    LineSegment(Point2D<float> inP1, Point2D<float> inP2) :
      p1(inP1),
      p2(inP2)
    {}

    LineSegment() {}

    double distance(const LineSegment &line)
    {
      return center.distance(line.center);
    }

    double colorDist(const LineSegment &line)
    {
      float distColor1 = std::numeric_limits<float>::max();
      if (side1Color.size() == 3 && line.side1Color.size() == 3)
        distColor1 = sqrt( squareOf(side1Color[0] - line.side1Color[0]) +
                           squareOf(side1Color[1] - line.side1Color[1]) +
                           squareOf(side1Color[2] - line.side1Color[2]));

      float distColor2 = std::numeric_limits<float>::max();
      if (side2Color.size() == 3 && line.side2Color.size() == 3)
        distColor2 = sqrt( squareOf(side2Color[0] - line.side2Color[0]) +
                           squareOf(side2Color[1] - line.side2Color[1]) +
                           squareOf(side2Color[2] - line.side2Color[2]));

      return std::min(distColor2, distColor1);
    }


    LineSegment& operator=(const LineSegment& line)
    {
      //Assign the new specs for the line, but keep the strength the same
      p1 = line.p1;
      p2 = line.p2;
      width = line.width;
      length = line.length;
      ori = line.ori;
      center = line.center;
      side1Color = line.side1Color;
      side2Color = line.side2Color;

      return *this;

    }

  };

  struct ContourState
  {
    std::vector<LineSegment> lines;
  };



  V2(OptionManager& mgr, const std::string& descrName = "V2",
      const std::string& tagName = "V2");

  //! Destructor
  ~V2();

  void init(Dims numCells);

  void evolve(Image<PixRGB<byte> >& img);
  void evolve(Image<float>& img);
  void evolve(TensorField& tensorField);
  void evolve(SimEventQueue& q);
  void evolveLines();
  void evolveContours();
  void evolve2();

  Layout<PixRGB<byte> > getDebugImage();

  void findLines(const Image<float> &mag, const Image<float>& ori,
      Image<float>& retMag, Image<float>& retOri);

  int findMostProbableLine(const LineSegment& line, float &prob);
  std::vector<LineSegment> proposeLineSegments(ImageSet<float> &LGNInput); //Use the LGN raw input
  std::vector<LineSegment> proposeLineSegments(V1::EdgesState& edgesState); //use the V1 edges

  void trainNFA(V1::EdgesState& edgesState); //use the V1 edges

  std::vector<ContourState> proposeContours(std::vector<LineSegment>& lines);


  std::vector<LineSegment> findNerestLines(std::list<LineSegment>& lines,
      std::stack<Point2D<float> >& endPoints);


  void cornerSymetryDetection(std::vector<LineSegment>& lines);

  void evolveBorderOwner();
  std::vector<V2::ContourState> proposeContoursBO(std::vector<LineSegment>& lines);


  rect getRect(const Point2D<int> p1, const Point2D<int> p2,const int width);
  

protected:
  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(V2, SimEventV1Output);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(V2, SimEventSaveOutput);

  //! Callback for every time we have a user event
  SIMCALLBACK_DECLARE(V2, SimEventUserInput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;

  //! Train the NFA
  OModelParam<bool> itsTrainNFA;
  OModelParam<std::string> itsTrainNFAFile;


  std::vector<LineSegment> itsLines;
  std::vector<ContourState> itsContours;


public:
/*
   compute the direction of the level line at each point.
   it returns:

   - an image_float with the angle at each pixel or NOTDEF.
   - the image_float 'modgrad' (a pointer is passed as argument)
     with the gradient magnitude at each point.
   - a list of pixels 'list_p' roughly ordered by gradient magnitude.
     (the order is made by classing points into bins by gradient magnitude.
      the parameters 'n_bins' and 'max_grad' specify the number of
      bins and the gradient modulus at the highest bin.)
   - a pointer 'mem_p' to the memory used by 'list_p' to be able to
     free the memory.
 */
Image<float> ll_angle(Image<float>& in, float threshold,
                             std::vector<Point2D<int> > &list_p, void ** mem_p,
                             Image<float>& modgrad, int n_bins, int max_grad );
Image<float> ll_angle(const TensorField& tensorField, float threshold,
                             std::vector<Point2D<int> > &list_p,
                             Image<float>& modgrad, int n_bins, int max_grad );

#define RELATIVE_ERROR_FACTOR 100.0

/*----------------------------------------------------------------------------*/
/** Compare doubles by relative error.

    The resulting rounding error after floating point computations
    depend on the specific operations done. The same number computed by
    different algorithms could present different rounding errors. For a
    useful comparison, an estimation of the relative rounding error
    should be considered and compared to a factor times EPS. The factor
    should be related to the cumulated rounding error in the chain of
    computation. Here, as a simplification, a fixed factor is used.
 */
static int double_equal(double a, double b)
{
  double abs_diff,aa,bb,abs_max;

  /* trivial case */
  if( a == b ) return true;

  abs_diff = fabs(a-b);
  aa = fabs(a);
  bb = fabs(b);
  abs_max = aa > bb ? aa : bb;

  /* DBL_MIN is the smallest normalized number, thus, the smallest
     number whose relative error is bounded by DBL_EPSILON. For
     smaller numbers, the same quantization steps as for DBL_MIN
     are used. Then, for smaller numbers, a meaningful "relative"
     error should be computed by dividing the difference by DBL_MIN. */
  if( abs_max < DBL_MIN ) abs_max = DBL_MIN;

  /* equal if relative error <= factor x eps */
  return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON);
}


/*----------------------------------------------------------------------------*/
/*
   find if the point x,y in angles have angle theta up to precision prec
 */
bool isaligned(Point2D<int> loc,const  Image<float>& angles,
               float theta, float prec);
float angle_diff(float a, float b);


/*----------------------------------------------------------------------------*/
/*----------------------------- NFA computation ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*
   Calculates the natural logarithm of the absolute value of
   the gamma function of x using the Lanczos approximation,
   see http://www.rskey.org/gamma.htm.

   The formula used is
     \Gamma(x) = \frac{ \sum_{n=0}^{N} q_n x^n }{ \Pi_{n=0}^{N} (x+n) }
                 (x+5.5)^(x+0.5) e^{-(x+5.5)}
   so
     \log\Gamma(x) = \log( \sum_{n=0}^{N} q_n x^n ) + (x+0.5) \log(x+5.5)
                     - (x+5.5) - \sum_{n=0}^{N} \log(x+n)
   and
     q0 = 75122.6331530
     q1 = 80916.6278952
     q2 = 36308.2951477
     q3 = 8687.24529705
     q4 = 1168.92649479
     q5 = 83.8676043424
     q6 = 2.50662827511
 */
double log_gamma_lanczos(double x);

/*----------------------------------------------------------------------------*/
/*
   Calculates the natural logarithm of the absolute value of
   the gamma function of x using Robert H. Windschitl method,
   see http://www.rskey.org/gamma.htm.

   The formula used is
     \Gamma(x) = \sqrt(\frac{2\pi}{x}) ( \frac{x}{e}
                   \sqrt{ x\sinh(1/x) + \frac{1}{810x^6} } )^x
   so
     \log\Gamma(x) = 0.5\log(2\pi) + (x-0.5)\log(x) - x
                     + 0.5x\log( x\sinh(1/x) + \frac{1}{810x^6} ).

   This formula is good approximation when x > 15.
 */
double log_gamma_windschitl(double x);

/*----------------------------------------------------------------------------*/
/*
   Calculates the natural logarithm of the absolute value of
   the gamma function of x. When x>15 use log_gamma_windschitl(),
   otherwise use log_gamma_lanczos().
 */
#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

/*----------------------------------------------------------------------------*/
/*
   Computes the logarithm of NFA to base 10.

   NFA = NT.b(n,k,p)
   the return value is log10(NFA)

   n,k,p - binomial parameters.
   logNT - logarithm of Number of Tests
 */

#define TABSIZE 100000
double nfa(int n, int k, double p, double logNT);
void rect_copy(struct rect * in, struct rect * out);
float inter_low(float x, float x1, float y1, float x2, float y2);
float inter_hi(float x, float x1, float y1, float x2, float y2);

void ri_del(rect_iter * iter);
int ri_end(rect_iter * i);
void ri_inc(rect_iter * i);
rect_iter * ri_ini(struct rect * r);

int ri_end(rect_iter& itr);
void ri_inc(rect_iter& itr);
void ri_ini(struct rect * r, rect_iter& iter);


double rect_nfa(struct rect * rec, Image<float>& angles, double logNT);
float get_theta( struct point * reg, int reg_size, float x, float y,
                        Image<float>& modgrad, float reg_angle, float prec,
                        float * elongation );
float region2rect( struct point * reg, int reg_size,
                          Image<float>& modgrad, float reg_angle,
                          float prec, double p, struct rect * rec,
                          float* sum_l, float* sum_w, int sum_offset, int sum_res);
void region_grow(Point2D<int> loc, Image<float>& angles, struct point * reg,
                         int * reg_size, float * reg_angle, Image<byte>& used,
                         float prec, int radius,
                         Image<float> modgrad, double p, int min_reg_size );

double rect_improve( struct rect * rec, Image<float>& angles,
                            double logNT, double eps );


/* execute LSD */
/* LSD parameters */
//float q = 2.0;           /* Bound to the quantization error on the
//                            gradient norm.                                 */
//float d = 8.0;           /* Gradient angle tolerance, tau = 180 degree / d */
//double eps = 0.0;        /* Detection threshold, -log10(NFA).              */
//int n_bins = 16256;      /* Number of bins in pseudo-ordering of gradient
//                            modulus. This default value is selected to work
//                            well on images with gray levels in [0,255].    */
//int max_grad = 260100;   /* Gradient modulus in the highest bin. The default
//                            value corresponds to the highest gradient modulus
//                            on images with gray levels in [0,255].         */
//float scale = 1.0;       /* scale the image by Gaussian filter to 'scale'. */
//float sigma_scale = 0.6; /* sigma used in Gaussian filter when scale!=1.0
//                            sigma = sigma_scale/scale.                     */

std::vector<LineSegment> lineSegmentDetection(Image<float>& img,
    float q = 2.0, float d = 8.0, double eps = 0.0,
    int n_bins = 16256, int max_grad = 260100,
    float scale = 1.0, float sigma_scale = 0.6 );

std::vector<LineSegment> lineSegmentDetection(const TensorField& tensorField,
    float q = 2.0, float d = 20, double eps = 0.0,
    int n_bins = 16256, int max_grad = 260100);

private:
  ImageSet<float>  itsLGNInput;
  std::vector<CornerState> itsCornersState;
  std::vector<SymmetryState> itsSymmetries;
  std::vector<TensorVoting> itsTensorFields;
  Image<float> itsEdges;
  Image<float> itsCornersProposal;
  V1::EdgesState itsV1EdgesState;
  bool itsLSDVerbose;
  TensorVoting itsLumTV;
  TensorVoting itsRGTV;
  TensorVoting itsBYTV;
  TensorField itsMaxTF;
  int itsFrameNum;
  std::string itsStoredFrames;
  float itsQuantError;
  float itsAngleTolerance;
  float itsEPS;
  
  rutz::shared_ptr<WorkThreadServer> itsThreadServer;
  std::vector<NFAPDFMap> itsPDF;
  std::vector<Point2D<int> > itsLocations;

#ifdef HAVE_OPENCV
  //CvMemStorage* itsStorage;
#endif

};

/* ############################### V2 sim events ######################## */
class SimEventV2Output : public SimEvent
{
public:
  SimEventV2Output(SimModule* src,
      std::vector<V2::LineSegment>& lines,
      std::vector<V2::CornerState>& corners,
      Image<float>& cornersProposals,
      Image<float>& edges,
      std::vector<TensorVoting>& tensorFields,
      Dims dims) :
    SimEvent(src), itsLines(lines), itsCorners(corners),
    itsCornersProposals(cornersProposals), itsEdges(edges),
    itsTensorFields(tensorFields),
    itsDims(dims)
  {}

  SimEventV2Output(SimModule* src,
      std::vector<V2::LineSegment>& lines, Dims dims) :
    SimEvent(src), itsLines(lines), 
    itsCorners(std::vector<V2::CornerState>()),
    itsCornersProposals(Image<float>()),
    itsEdges(Image<float>()),
    itsDims(dims)
  {}

  virtual ~SimEventV2Output(){}
  std::vector<V2::LineSegment> getLines() { return itsLines; }
  std::vector<V2::CornerState> getCorners() { return itsCorners; }
  Image<float> getCornersProposals() { return itsCornersProposals; }
  Image<float> getEdges() { return itsEdges; }
  std::vector<TensorVoting> getTensorFields() { return itsTensorFields; }
  Dims getDims() { return itsDims; }

private:
  const std::vector<V2::LineSegment>& itsLines;
  const std::vector<V2::CornerState>& itsCorners;
  const Image<float>& itsCornersProposals;
  const Image<float>& itsEdges;
  const std::vector<TensorVoting> itsTensorFields;
  const Dims itsDims;

};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
