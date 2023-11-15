/*!@file SceneUnderstanding/TwoHalfDSketch.C  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/TwoHalfDSketch.C $
// $Id: TwoHalfDSketch.C 14181 2010-10-28 22:46:20Z lior $
//

#ifndef TwoHalfDSketch_C_DEFINED
#define TwoHalfDSketch_C_DEFINED

#include "plugins/SceneUnderstanding/TwoHalfDSketch.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Transforms.H"
#include "Image/fancynorm.H"
#include "Image/Convolutions.H"
#include "Image/MatrixOps.H"
#include "Simulation/SimEventQueue.H"
#include "GUI/DebugWin.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>

const ModelOptionCateg MOC_TwoHalfDSketch = {
  MOC_SORTPRI_3,   "TwoHalfDSketch-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_TwoHalfDSketchShowDebug =
  { MODOPT_ARG(bool), "TwoHalfDSketchShowDebug", &MOC_TwoHalfDSketch, OPTEXP_CORE,
    "Show debug img",
    "twohalfdsketch-debug", '\0', "<true|false>", "false" };

//Define the inst function name
SIMMODULEINSTFUNC(TwoHalfDSketch);


// ######################################################################
TwoHalfDSketch::TwoHalfDSketch(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventV2Output),
  SIMCALLBACK_INIT(SimEventContoursOutput),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  SIMCALLBACK_INIT(SimEventUserInput),
  itsShowDebug(&OPT_TwoHalfDSketchShowDebug, this),
  itsProposalThreshold(-1500),
  itsAcceptedThreshold(-50)
  
{
  itsCurrentProb = -1e10;
  itsCurrentIdx = 0;
  itsBiasMode = false;
  itsBias = -1;
  itsBiasId = -1;
  
  initRandomNumbers();

  itsUserProposal.e=0.8;
  itsUserProposal.a=10; //25;
  itsUserProposal.b=11; //23;
  itsUserProposal.k1=0; //23;
  itsUserProposal.k2=0; //23;
  itsUserProposal.rot = 0;
  itsUserProposal.pos = Point2D<float>(257, 141);
  itsUserProposal.start = -M_PI;
  itsUserProposal.end = M_PI;
  itsUserProposal.gibs=0;
  
  //AppleLogo
  itsModel.addLine(Line(7, 71, 17, 88));
  itsModel.addLine(Line(19, 90, 27, 95)); 
  itsModel.addLine(Line(29, 95, 38, 91));
  itsModel.addLine(Line(39, 91, 55, 95));
  itsModel.addLine(Line(56, 95, 61, 93));
  itsModel.addLine(Line(62, 93, 74, 73));
  itsModel.addLine(Line(74, 72, 65, 62));
  itsModel.addLine(Line(65, 61, 63, 57));
  itsModel.addLine(Line(63, 56, 65, 42));
  itsModel.addLine(Line(66, 42, 71, 36));
  itsModel.addLine(Line(70, 35, 68, 33));
  itsModel.addLine(Line(67, 32, 53, 29));
  itsModel.addLine(Line(52, 30, 45, 32));
  itsModel.addLine(Line(44, 27, 56, 15));
  itsModel.addLine(Line(56, 14, 57, 7));
  itsModel.addLine(Line(56, 6, 53, 7));
  itsModel.addLine(Line(52, 7, 40, 19));
  itsModel.addLine(Line(40, 20, 42, 26));
  itsModel.addLine(Line(44, 33, 25, 29));
  itsModel.addLine(Line(24, 29, 17, 31));
  itsModel.addLine(Line(15, 32, 6, 43));
  itsModel.addLine(Line(5, 45, 5, 64));
  itsModel.addLine(Line(6, 65, 7, 70));

  //Giraff
  //itsModel.lines.push_back(Line(39,44,14,7));
  //itsModel.lines.push_back(Line(26,63,16,26));
  //itsModel.lines.push_back(Line(61,64,87,70));
  //itsModel.lines.push_back(Line(22,80,31,104));
  //itsModel.lines.push_back(Line(74,94,53,94));
  //itsModel.lines.push_back(Line(93,77,96,94));
  //itsModel.lines.push_back(Line(47,49,58,62));
  //itsModel.lines.push_back(Line(1,15,12,3));
  //itsModel.lines.push_back(Line(89,104, 81, 92));
  //itsModel.lines.push_back(Line(29,68,22,79));
  //itsModel.lines.push_back(Line(96,95,99,106));
  //itsModel.lines.push_back(Line(40,91,51,93));
  //itsModel.lines.push_back(Line(3, 18, 13, 20));
  //itsModel.lines.push_back(Line(39,93,38,103));
  //itsModel.lines.push_back(Line(40,45,46,49));
  //itsModel.lines.push_back(Line(92,76,88,71));
  //itsModel.lines.push_back(Line(80,92,75,93));
  //itsModel.lines.push_back(Line(15,25,14,21));
  //itsModel.lines.push_back(Line(28,67,27,64));
  //itsModel.lines.push_back(Line(1, 16, 2, 18));
  //itsModel.lines.push_back(Line(14, 6, 13, 5));
  //itsModel.lines.push_back(Line(11, 2, 10, 3));
  //itsModel.lines.push_back(Line(88, 106, 89, 105));
  //itsModel.lines.push_back(Line(39, 104, 38, 105));
  //itsModel.lines.push_back(Line(60, 63, 59, 63));


  itsModel.setCOM(); //Center to COM
  itsModel.quantize(60); //number of directions in the grid

}

// ######################################################################
TwoHalfDSketch::~TwoHalfDSketch()
{
}

// ######################################################################
void TwoHalfDSketch::onSimEventUserInput(SimEventQueue& q, rutz::shared_ptr<SimEventUserInput>& e)
{

  LINFO("Got event --%s-- %ix%i key=%i",
      e->getWinName(),
      e->getMouseClick().i,
      e->getMouseClick().j,
      e->getKey());

  if (strcmp(e->getWinName(), "TwoHalfDSketch"))
    return;
  SurfaceState& surface = itsUserProposal;

  switch(e->getKey())
  {
    case 111: //98: //111: //up
      surface.pos.i -= 1;
      break;
    case 116: //104: //116: //down
      surface.pos.i += 1;
      break;
    case 113: //100: //113: //left
      surface.pos.j -= 1;
      break;
    case 114: //102: //114: //right
      surface.pos.j += 1;
      break;
    case 21: //=
      break;
    case 20: //-
      break;
    case 38: //a
      break;
    case 52: //z
      break;
    case 39: //s
      break;
    case 53: //x
      break;
    case 40: //d
      break;
    case 54: //c
      break;
    case 10: //1
      surface.a += 1.0;
      break;
    case 24: //q
      surface.a -= 1.0;
      break;
    case 11: //2
      surface.b += 1.0;
      break;
    case 25: //w
      surface.b -= 1.0;
      break;
    case 12: //3
      surface.e += 0.1;
      break;
    case 26: //e
      surface.e -= 0.1;
      break;
    case 13: //4
      surface.k1 += 0.1;
      break;
    case 27: //r
      surface.k1 -= 0.1;
      break;
    case 14: //5
      surface.k2 += 0.1;
      break;
    case 28: //t
      surface.k2 -= 0.1;
      break;
    case 15: //6
      surface.rot += 1*M_PI/180;
      break;
    case 29: //y
      surface.rot -= 1*M_PI/180;
      break;
  }

  LINFO("Pos(%f,%f), param((%0.2f,%0.2f,%0.2f)",
      surface.pos.i, surface.pos.j,
      surface.a, surface.b, surface.e);


  evolve(q);

}


// ######################################################################
void TwoHalfDSketch::onSimEventV2Output(SimEventQueue& q, rutz::shared_ptr<SimEventV2Output>& e)
{
  //Check if we have the smap
  //if (SeC<SimEventSMapOutput> smap = q.check<SimEventSMapOutput>(this))
  //  itsSMap = smap->getSMap();

  //Check if we have the corners
  itsLines = e->getLines();
  Dims dims = e->getDims();

  itsLinesMag = Image<float>(dims, ZEROS);
  itsLinesOri = Image<float>(dims, NO_INIT);
  for(uint i=0; i<itsLinesOri.size(); i++)
    itsLinesOri[i] = NOTDEF;

  std::vector<Line> lines;
  for(uint i=0; i<itsLines.size(); i++)
  {
    V2::LineSegment& ls = itsLines[i];
    drawLine(itsLinesMag, Point2D<int>(ls.p1), Point2D<int>(ls.p2), 255.0F); //ls.length);
    drawLine(itsLinesOri, Point2D<int>(ls.p1), Point2D<int>(ls.p2), ls.ori);
    lines.push_back(Line(ls.p1, ls.p2));
  }
  
  //Build the FDCM
  itsOriChamferMatcher.setLines(lines, 
    60, //Num Orientation to quantize
    5, //Direction cost
    itsLinesMag.getDims());

  evolve(q);

}

void TwoHalfDSketch::onSimEventContoursOutput(SimEventQueue& q, rutz::shared_ptr<SimEventContoursOutput>& e)
{
  LINFO("Contours output");

  itsContours = e->getContours();

  itsLinesMag = Image<float>(e->getImg().getDims(), ZEROS);
  itsLinesOri = Image<float>(e->getImg().getDims(), NO_INIT);
  for(uint i=0; i<itsLinesOri.size(); i++)
    itsLinesOri[i] = NOTDEF;

  std::vector<Line> lines;

  for(uint i=0; i<itsContours.size(); i++)
  {
    std::vector<Point2D<int> > polygon = approxPolyDP(itsContours[i].points, 2);
    for(uint j=0; j<polygon.size()-1; j++)
    {
      drawLine(itsLinesMag, polygon[j], polygon[(j+1)], 255.0F);

      float ori = atan2(polygon[j].j-polygon[(j+1)].j, polygon[(j+1)].i - polygon[j].i);
      if (ori < 0) ori += M_PI;
      if (ori >= M_PI) ori -= M_PI;
      drawLine(itsLinesOri, polygon[j], polygon[(j+1)], ori); 

      lines.push_back(Line(polygon[j], polygon[(j+1)]));
    }
  }

  //Build the FDCM
  itsOriChamferMatcher.setLines(lines, 
    60, //Num Orientation to quantize
    0.5, //Direction cost
    itsLinesMag.getDims());

  evolve(q);

}

double TwoHalfDSketch::getCost(OriChamferMatching& cm,
    Polygon& poly, Point2D<float> loc, bool biasMode)
{

  //Image<PixRGB<byte> > tmp = itsLinesMag;
  //LINFO("Loc %f %f", loc.i, loc.j);
  ////We have a polygon
  //for(uint j=0; j<poly.getNumLines(); j++)
  //{
  //  Line l = poly.getLine(j); //Get the line up to scale and pos
  //  l.trans(loc);
  //  Point2D<int> p1 = (Point2D<int>)l.getP1();
  //  Point2D<int> p2 = (Point2D<int>)l.getP2();
  //  drawLine(tmp, p1,p2, PixRGB<byte>(255,0,0));
  //}



  double totalProb = 0;				
  //double totalHalLength = 0; //The total length we have been hallucinating
  //double factor = 1.0/poly.getLength();
  for (uint i=0 ; i<poly.getNumLines(); i++)
  {
    Line l = poly.getLine(i); //Make a copy so that we dont change the position of the original
    l.trans(loc); //Move the line to the correct location

    Point2D<int> p1 = (Point2D<int>)l.getP1();
    Point2D<int> p2 = (Point2D<int>)l.getP2();
    double sum = cm.getCost(l.getDirectionIdx(), p1, p2);

    double prob = exp(-sum/double(2*l.getLength()));
    if (sum < 0) 
    {
      LINFO("Invalid sum %f", sum);
      prob = 0;
    }
    if (!biasMode)
    {
      if (prob < exp(-500/10))  //If we are very far off, then there might be an edge there, we just dont see it
      {
        //prob = 0.10;
        //totalHalLength+=l.getLength();
      }
    } else {
      //calcNFA(l);
    }
    //Weight the line based on how much it contributes to the overall length
    prob = pow(prob, 1-l.getWeight());
    //LINFO("%i sum=%f l=%f w=%f p=%f", i, sum, l.getLength(), l.getWeight(), prob );
    //Image<PixRGB<byte> > tmp = itsLinesMag;
    //drawLine(tmp, p1, p2, PixRGB<byte>(255,0,0));
    //SHOWIMG(tmp);

    totalProb += log(prob);
  }

  //We need at least 10% of the contour to exist in order to hallucinate
  //TODO this should be a continues function change the totalProb as the totalHalLength decreases
  //if (totalHalLength/poly.getLength() > 0.9)
  //  totalProb = -1000;
  //LINFO("Hal %f %f %f %f", poly.getLength(), totalHalLength, totalHalLength/poly.getLength(), totalProb);
  //LINFO("TOtal prob %f", totalProb);

  //SHOWIMG(tmp);
  return totalProb;
}


double TwoHalfDSketch::calcNFA(Line& line)
{
  //Image<PixRGB<byte> > tmp = itsLinesMag;


  //double theta = line.getOri();
  //double prec = M_PI/8.0;
  //double p = 1.0 / (double) 8;

  //int w = itsLinesMag.getWidth();
  //int h = itsLinesMag.getHeight();
  //double logNT = 5.0 * ( log10( (double) w ) +
  //                       log10( (double) h ) ) / 2.0;


  //int pts = 0;
  //int alg = 0;
  //double nfa_val;


  //rect rec;
  //if (line.p1.j > line.p2.j)
  //{
  //  rec.x1 = line.p2.i*2;
  //  rec.y1 = line.p2.j*2;
  //  rec.x2 = line.p1.i*2;
  //  rec.y2 = line.p1.j*2;
  //} else {
  //  rec.x1 = line.p1.i*2;
  //  rec.y1 = line.p1.j*2;
  //  rec.x2 = line.p2.i*2;
  //  rec.y2 = line.p2.j*2;
  //}

  //rec.theta = theta; //ori*M_PI/180;
  //rec.dx = (float) cos( (double) rec.theta );
  //rec.dy = (float) sin( (double) rec.theta );
  //rec.width=15;

  //rect_iter * i;

  //for(i=ri_ini(&rec); !ri_end(i); ri_inc(i))
  //  if( i->x>=0 && i->y>=0 && i->x < itsLinesMag.getWidth() && i->y < itsLinesMag.getHeight() )
  //  {
  //    Point2D<int> loc(i->x, i->y);
  //    if (itsLinesMag.getVal(loc) > 0)
  //    {
  //      pts++;
  //      if( isaligned(loc,itsLinesOri,theta,prec) )
  //      {
  //        tmp.setVal(loc, PixRGB<byte>(0,255,0));
  //        ++alg;
  //      }
  //    }
  //  }
  //ri_del(i);

  //nfa_val = nfa(pts,alg,p,logNT);
  //LINFO("pts=%i alg=%i NFA %f",pts, alg, nfa_val);

  //SHOWIMG(tmp);

  return 0;
}

/*----------------------------------------------------------------------------*/
float TwoHalfDSketch::inter_low(float x, float x1, float y1, float x2, float y2)
{
  if( x1 > x2 || x < x1 || x > x2 )
    {
      LFATAL("inter_low: x %g x1 %g x2 %g.\n",x,x1,x2);
      LFATAL("Impossible situation.");
    }
  if( x1 == x2 && y1<y2 ) return y1;
  if( x1 == x2 && y1>y2 ) return y2;
  return y1 + (x-x1) * (y2-y1) / (x2-x1);
}

/*----------------------------------------------------------------------------*/
float TwoHalfDSketch::inter_hi(float x, float x1, float y1, float x2, float y2)
{
  if( x1 > x2 || x < x1 || x > x2 )
    {
      LFATAL("inter_hi: x %g x1 %g x2 %g.\n",x,x1,x2);
      LFATAL("Impossible situation.");
    }
  if( x1 == x2 && y1<y2 ) return y2;
  if( x1 == x2 && y1>y2 ) return y1;
  return y1 + (x-x1) * (y2-y1) / (x2-x1);
}

/*----------------------------------------------------------------------------*/
void TwoHalfDSketch::ri_del(rect_iter * iter)
{
  free(iter);
}

/*----------------------------------------------------------------------------*/
int TwoHalfDSketch::ri_end(rect_iter * i)
{
  return (float)(i->x) > i->vx[2];
}

/*----------------------------------------------------------------------------*/
void TwoHalfDSketch::ri_inc(rect_iter * i)
{
  if( (float) (i->x) <= i->vx[2] ) i->y++;

  while( (float) (i->y) > i->ye && (float) (i->x) <= i->vx[2] )
    {
      /* new x */
      i->x++;

      if( (float) (i->x) > i->vx[2] ) return; /* end of iteration */

      /* update lower y limit for the line */
      if( (float) i->x < i->vx[3] )
        i->ys = inter_low((float)i->x,i->vx[0],i->vy[0],i->vx[3],i->vy[3]);
      else i->ys = inter_low((float)i->x,i->vx[3],i->vy[3],i->vx[2],i->vy[2]);

      /* update upper y limit for the line */
      if( (float)i->x < i->vx[1] )
        i->ye = inter_hi((float)i->x,i->vx[0],i->vy[0],i->vx[1],i->vy[1]);
      else i->ye = inter_hi( (float)i->x,i->vx[1],i->vy[1],i->vx[2],i->vy[2]);

      /* new y */
      i->y = (int)((float) ceil( (double) i->ys ));
    }
}

/*----------------------------------------------------------------------------*/
TwoHalfDSketch::rect_iter * TwoHalfDSketch::ri_ini(struct rect * r)
{
  float vx[4],vy[4];
  int n,offset;
  rect_iter * i;

  i = (rect_iter *) malloc(sizeof(rect_iter));
  if(!i) LFATAL("ri_ini: Not enough memory.");

  vx[0] = r->x1 - r->dy * r->width / 2.0;
  vy[0] = r->y1 + r->dx * r->width / 2.0;
  vx[1] = r->x2 - r->dy * r->width / 2.0;
  vy[1] = r->y2 + r->dx * r->width / 2.0;
  vx[2] = r->x2 + r->dy * r->width / 2.0;
  vy[2] = r->y2 - r->dx * r->width / 2.0;
  vx[3] = r->x1 + r->dy * r->width / 2.0;
  vy[3] = r->y1 - r->dx * r->width / 2.0;

  if( r->x1 < r->x2 && r->y1 <= r->y2 ) offset = 0;
  else if( r->x1 >= r->x2 && r->y1 < r->y2 ) offset = 1;
  else if( r->x1 > r->x2 && r->y1 >= r->y2 ) offset = 2;
  else offset = 3;
  /* else if( r->x1 <= r->x2 && r->y1 > r->y2 ) offset = 3; */

  for(n=0; n<4; n++)
    {
      i->vx[n] = vx[(offset+n)%4];
      i->vy[n] = vy[(offset+n)%4];
    }

#define BIG_NUMBER 1.0e+300
  
  /* starting point */
  i->x = (int)(ceil( (double) (i->vx[0]) ) - 1);
  i->y = (int)(ceil( (double) (i->vy[0]) ));
  i->ys = i->ye = -BIG_NUMBER;

  /* advance to the first point */
  ri_inc(i);

  return i;
}



int TwoHalfDSketch::isaligned(Point2D<int> loc, Image<float>& angles, float theta, float prec)
{
  float a = angles.getVal(loc);

  printf("IsAligned %f %f %f ", a*180/M_PI, theta*180/M_PI, prec*180/M_PI);
  if( a == NOTDEF ) return false;

  /* it is assumed that theta and a are in the range [-pi,pi] */
  theta -= a;
  if( theta < 0.0 ) theta = -theta;
  if( theta > M_3_2_PI )
    {
      theta -= M_2__PI;
      if( theta < 0.0 ) theta = -theta;
    }

  printf(" %f < %f \n", theta*180/M_PI, prec*180/M_PI);
  return theta < prec;
}

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
double TwoHalfDSketch::log_gamma_lanczos(double x)
{
  static double q[7] = { 75122.6331530, 80916.6278952, 36308.2951477,
                         8687.24529705, 1168.92649479, 83.8676043424,
                         2.50662827511 };
  double a = (x+0.5) * log(x+5.5) - (x+5.5);
  double b = 0.0;
  int n;

  for(n=0;n<7;n++)
    {
      a -= log( x + (double) n );
      b += q[n] * pow( x, (double) n );
    }
  return a + log(b);
}

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
double TwoHalfDSketch::log_gamma_windschitl(double x)
{
  return 0.918938533204673 + (x-0.5)*log(x) - x
         + 0.5*x*log( x*sinh(1/x) + 1/(810.0*pow(x,6.0)) );
}

/*----------------------------------------------------------------------------*/
/*
   Calculates the natural logarithm of the absolute value of
   the gamma function of x. When x>15 use log_gamma_windschitl(),
   otherwise use log_gamma_lanczos().
 */
//#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

/*----------------------------------------------------------------------------*/
/*
   Computes the logarithm of NFA to base 10.

   NFA = NT.b(n,k,p)
   the return value is log10(NFA)

   n,k,p - binomial parameters.
   logNT - logarithm of Number of Tests
 */
//#define TABSIZE 100000


double TwoHalfDSketch::nfa(int n, int k, double p, double logNT)
{
  static double inv[TABSIZE];   /* table to keep computed inverse values */
  double tolerance = 0.1;       /* an error of 10% in the result is accepted */
  double log1term,term,bin_term,mult_term,bin_tail,err;
  double p_term = p / (1.0-p);
  int i;

  if( n<0 || k<0 || k>n || p<0.0 || p>1.0 )
    LFATAL("Wrong n, k or p values in nfa()");

  if( n==0 || k==0 ) return -logNT;
  if( n==k ) return -logNT - (double) n * log10(p);

  log1term = log_gamma((double)n+1.0) - log_gamma((double)k+1.0)
           - log_gamma((double)(n-k)+1.0)
           + (double) k * log(p) + (double) (n-k) * log(1.0-p);

  term = exp(log1term);
  if( term == 0.0 )              /* the first term is almost zero */
    {
      if( (double) k > (double) n * p )    /* at begin or end of the tail? */
        return -log1term / M_LN10 - logNT; /* end: use just the first term */
      else
        return -logNT;                     /* begin: the tail is roughly 1 */
    }

  bin_tail = term;
  for(i=k+1;i<=n;i++)
    {
      bin_term = (double) (n-i+1) * ( i<TABSIZE ?
                   ( inv[i] == 0.0 ? inv[i] : (inv[i]=1.0/(double)i))
                   : 1.0/(double)i );
      mult_term = bin_term * p_term;
      term *= mult_term;
      bin_tail += term;
      if(bin_term<1.0)
        {
          /* when bin_term<1 then mult_term_j<mult_term_i for j>i.
             then, the error on the binomial tail when truncated at
             the i term can be bounded by a geometric series of form
             term_i * sum mult_term_i^j.                            */
          err = term * ( ( 1.0 - pow( mult_term, (double) (n-i+1) ) ) /
                         (1.0-mult_term) - 1.0 );

          /* one wants an error at most of tolerance*final_result, or:
             tolerance * abs(-log10(bin_tail)-logNT).
             now, the error that can be accepted on bin_tail is
             given by tolerance*final_result divided by the derivative
             of -log10(x) when x=bin_tail. that is:
             tolerance * abs(-log10(bin_tail)-logNT) / (1/bin_tail)
             finally, we truncate the tail if the error is less than:
             tolerance * abs(-log10(bin_tail)-logNT) * bin_tail        */
          if( err < tolerance * fabs(-log10(bin_tail)-logNT) * bin_tail ) break;
        }
    }
  return -log10(bin_tail) - logNT;
}




// ######################################################################
void TwoHalfDSketch::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage(q);
      if (disp.initialized())
        ofs->writeRgbLayout(disp, "TwoHalfDSketch", FrameInfo("TwoHalfDSketch", SRC_POS));
    }
}


// ######################################################################
void TwoHalfDSketch::evolve(SimEventQueue& q)
{

  if (!itsBiasMode)
  {
    Image<SurfaceState> surfaces = proposeSurfaces(false);

    itsProposals.clear();
    for(uint i=0; i<50; i++)
    {
      Point2D<int> maxLoc; SurfaceState maxSurface;
      findMax(surfaces, maxLoc, maxSurface);

      if (maxSurface.prob == INVALID_PROB)
        break; //No more valid surfaces

      itsProposals.push_back(maxSurface);

      //Apply IOR 3/4 the size of the object
      double radius = maxSurface.polygon.getRadius()*0.75;
      drawDisk(surfaces, maxLoc, (int)radius);
    }
  } else {
    optimizePolygon(itsProposals[itsBiasId]);
    LINFO("Optimized match %f\n", itsProposals[itsBiasId].prob);
  }

  printResults(itsBias);

  //sort by prob
  //Update the surfaces
  itsSurfaces.clear();
  for(uint i=0; i<itsProposals.size(); i++)
  {
    if (itsProposals[i].prob > itsAcceptedThreshold)
      itsSurfaces.push_back(itsProposals[i]);
  }

  //Layout<PixRGB<byte> > layout = getDebugImage(q);
  //Image<PixRGB<byte> > tmp = layout.render();
  //SHOWIMG(tmp);

  if (itsSurfaces.size() > 0)
    q.post(rutz::make_shared(new SimEventTwoHalfDSketchOutput(this, itsSurfaces)));

  if (itsProposals.size() > 0 && !itsBiasMode) //Check to see if we can bias
  {
    itsBiasMode = true;
    LINFO("Biasing");

    for(float bias=0; bias<0.30; bias += 0.10)
    {
      for(uint sid=0; sid<itsProposals.size(); sid++)
      {
        itsBias = bias;
        itsBiasId = sid;
        std::vector<V1::SpatialBias> v1Bias;

        Polygon& polygon = itsProposals[sid].polygon;
        for(uint j=0; j<polygon.getNumLines(); j++)
        {
          Line l = itsProposals[sid].getLine(j); //Get the line up to scale and pos

          Point2D<int> p1 = (Point2D<int>)l.getP1();
          Point2D<int> p2 = (Point2D<int>)l.getP2();

          Point2D<int> bP1 = p1;
          Point2D<int> bP2 = (p1+p2)/2;
          Point2D<int> bP3 = (p1+bP2)/2;
          Point2D<int> bP4 = (p2+bP2)/2;
          
          //Bias V1
          {
          V1::SpatialBias sb;
          sb.loc = bP1;
          sb.threshold = bias ;
          sb.dims = Dims(15,15);
          v1Bias.push_back(sb);
          }

          {
          V1::SpatialBias sb;
          sb.loc = bP2;
          sb.threshold = bias ;
          sb.dims = Dims(15,15);
          v1Bias.push_back(sb);
          }

          {
          V1::SpatialBias sb;
          sb.loc = bP3;
          sb.threshold = bias ;
          sb.dims = Dims(15,15);
          v1Bias.push_back(sb);
          }

          {
          V1::SpatialBias sb;
          sb.loc = bP4;
          sb.threshold = bias ;
          sb.dims = Dims(15,15);
          v1Bias.push_back(sb);
          }
        }
        LINFO("Bias %f for %i\n", itsBias, sid);
        q.post(rutz::make_shared(new SimEventV1Bias(this, v1Bias)));
      }

    }
  }

}


Image<float> TwoHalfDSketch::getSurfaceProbImage(Image<SurfaceState>& surfaceState)
{

  Image<float> probImg(surfaceState.getDims(), NO_INIT);
  for(uint i=0; i<probImg.size(); i++)
  {
    if (surfaceState[i].prob < -1000)
      probImg[i] = 0;
    else
      probImg[i] = surfaceState[i].prob;
  }

  return probImg;
}

void TwoHalfDSketch::optimizePolygon(TwoHalfDSketch::SurfaceState& surfaceState)
{

  float matchingScale = 1; //0.5;

  //int x = 285;
  //int y = 328;
  //double scale = 0.9;
  ////float aspectx = 1.9;
  //float aspecty = 1.11;
  //float rot = 2.0;


  double maxProb = getCost(itsOriChamferMatcher, surfaceState.polygon,surfaceState.pos, true);
  if (maxProb > -200 || maxProb < -400)
  {
    surfaceState.prob = maxProb;
    return;
  }
  LINFO("Optimize Polygon %f", maxProb);

  for(float aspectx = surfaceState.aspect.i; aspectx < surfaceState.aspect.i+0.5; aspectx +=0.2)
  {
    for(float aspecty = surfaceState.aspect.j; aspecty < surfaceState.aspect.j+0.5; aspecty +=0.2)
    {
      LINFO("Aspectx %f %f", aspectx, aspecty);
      if (aspectx <= 0.5 || aspecty <= 0.5)
        continue;
      for(double scale = surfaceState.scale-0.2  ; scale< surfaceState.scale+0.2 ; scale += 0.01) 	
      {	
        for(float rot =  surfaceState.rot-5 ; rot< surfaceState.rot+5 ; rot+= 1) 	
        {	
          Polygon model = surfaceState.polygon;
          //LINFO("Rot %f Scale %f aspect %f %f\n", rot, scale, aspectx, aspecty);

          model.rotate(rot*M_PI/180.0);
          model.scale(scale*matchingScale*aspectx, scale*matchingScale*aspecty); 

          model.setWeights();

          for (int x=surfaceState.pos.i-5 ; x<surfaceState.pos.i+5; x++)
          {
            for (int y=surfaceState.pos.j-5 ; y<surfaceState.pos.j+5; y++)
            {
              double prob = getCost(itsOriChamferMatcher, model,Point2D<float>(x,y), true);

              if (prob>maxProb)
              {
                maxProb = prob;
                surfaceState.pos = Point2D<float>(x,y);
                surfaceState.aspect = Point2D<float>(aspectx, aspecty);
                surfaceState.scale = scale;
                surfaceState.rot = rot;
                surfaceState.prob = prob;
                surfaceState.polygon = model;
                surfaceState.matchingScale = matchingScale;
              }
            }
          }
        }
      }
    }

  }
}


Image<TwoHalfDSketch::SurfaceState> TwoHalfDSketch::proposeSurfaces(bool biasMode)
{

  LINFO("Propose surfaces");

  float matchingScale = 1; //0.5;

  Image<SurfaceState>  surfacesImg(itsLinesMag.getDims(), NO_INIT);

  //int aspectx =0;
  //int aspecty = 2;
  //double scale = 1.1;
  //int x = 283;
  //int y = 325;

  for(int aspectx = 0; aspectx < 4; aspectx++)
  {
    for(int aspecty = 0; aspecty < 4; aspecty++)
    {
      for(double scale = 0.2  ; scale< 6.0 ; scale+= 0.1) 	
      {	
        float rot = 0;
        //for(float rot =  -60 ; rot< 60 ; rot+= 5) 	
        {	
          Polygon model = itsModel;
          double ax = pow(1.1, aspectx);
          double ay = pow(1.1, aspecty);
          LINFO("Rot %f Scale %f aspect %f %f\n", rot, scale, ax, ay);

          model.rotate(rot*M_PI/180.0);
          model.scale(scale*matchingScale*ax, scale*matchingScale*ay); 

          model.setWeights();

          double minx, miny, maxx, maxy;
          model.getBoundary(minx, miny, maxx, maxy);

          int width = surfacesImg.getWidth(); 
          int height = surfacesImg.getHeight();

          //Minx/miny is neg since our template is from -w/-h untill w/h
          //Skip the first 4*2 pixels
          for (int x=-(int)(minx-4) ; x<width-((int)maxx+4) ; x += 4)
          {
            for (int y=-(int)(miny-4); y<height-((int)maxy+4); y += 4)
            {

              double prob = getCost(itsOriChamferMatcher, model,Point2D<float>(x,y), biasMode);

              if (prob>itsProposalThreshold)
              {
                if (surfacesImg.coordsOk(x,y))
                  if (prob > surfacesImg.getVal(x,y).prob)
                  {
                    SurfaceState ss;
                    ss.pos = Point2D<float>(x,y);
                    ss.aspect = Point2D<float>(ax, ay);
                    ss.scale = scale;
                    ss.rot = rot;
                    ss.prob = prob;
                    ss.polygon = model;
                    ss.matchingScale = matchingScale;
                    surfacesImg.setVal(x,y, ss);

                  }
              }
            }
          }
        }
      }

    }

  }

  return surfacesImg;
}

void TwoHalfDSketch::calcSurfaceLikelihood(SurfaceState& surface)
{
  Image<float> edges;
  Image<float> lumSurface;
  double edgeProb = calcSurfaceEdgeLikelihood(surface, edges, lumSurface);
  //double surfaceProb = calcSurfaceLumLikelihood(surface, edges, lumSurface);
  surface.prob = edgeProb; // * surfaceProb;
}

double TwoHalfDSketch::calcSurfaceLumLikelihood(SurfaceState& surface, Image<float>& edges, Image<float>& surfaceLum)
{
  //Remove the edges from the surface
  for(uint i=0; i<surfaceLum.size(); i++)
    if (edges[i] > 0)
      surfaceLum[i] = 0;

  return getSurfaceLumProb(itsEdgesDT,surfaceLum);

}

double TwoHalfDSketch::getSurfaceLumProb(Image<float>& data, Image<float>& model)
{

  double prob = 0;
  int pixCount = 0;

  for(int y=0; y < model.getHeight(); y++)
    for(int x=0; x < model.getWidth();  x++)
      if (model.getVal(x,y) > 0)
      {
        prob += (10-data.getVal(x,y));
        pixCount++;
      }
  prob /= (10*pixCount);

  return exp(-prob);
}





double TwoHalfDSketch::calcSurfaceEdgeLikelihood(SurfaceState& surface, Image<float>& edges, Image<float>& surfaceLum)
{
  int pixCount = 0;
  double prob  = 0;

  if (surface.polygon.getNumLines() > 0)
  {

    for(uint j=0; j<surface.polygon.getNumLines(); j++)
    {
      Point2D<int> p1; // = (Point2D<int>)(surface.polygon[j] + surface.pos);
      Point2D<int> p2; // = (Point2D<int>)(surface.polygon[(j+1)%surface.polygon.size()] + surface.pos);

      //Point2D<float> center = (p1 + p2)/2;
      float ori = atan2(p1.j-p2.j, p2.i - p1.i);
      if (ori < 0) ori += M_PI;
      if (ori >= M_PI) ori -= M_PI;

      prob += getLineProb(p1, p2, ori, pixCount);

      //prob += getEdgeProb(Point2D<int>(p1), ori);
      //prob += getEdgeProb(Point2D<int>(center), ori);
    }
    pixCount *= 2;

  } else {

    float nSeg = 20;
    const float dTheta = 2*M_PI / (float)nSeg;

    float a = surface.a;
    float b = surface.b;
    float e = surface.e;
    float k1 = surface.k1;
    float k2 = surface.k2;
    float rot = surface.rot;

    Point2D<float> p = surface.pos;

    for (float theta=surface.start; theta < surface.end; theta += dTheta)
    {
      Point2D<float> p1 = ellipsoid(a,b, e, theta);
      Point2D<float> p2 = ellipsoid(a,b, e, theta + dTheta);

      Point2D<float> tmpPos1;
      Point2D<float> tmpPos2;

      //Sheer
      tmpPos1.i = p1.i + p1.j*k1;
      tmpPos1.j = p1.i*k2 + p1.j;

      tmpPos2.i = p2.i + p2.j*k1;
      tmpPos2.j = p2.i*k2 + p2.j;

      //Rotate and move to p
      p1.i = (cos(rot)*tmpPos1.i - sin(rot)*tmpPos1.j) + p.i;
      p1.j = (sin(rot)*tmpPos1.i + cos(rot)*tmpPos1.j) + p.j;

      p2.i = (cos(rot)*tmpPos2.i - sin(rot)*tmpPos2.j) + p.i;
      p2.j = (sin(rot)*tmpPos2.i + cos(rot)*tmpPos2.j) + p.j;


      Point2D<float> center = (p1 + p2)/2;
      float ori = atan2(p1.j-p2.j, p2.i - p1.i);
      if (ori < 0) ori += M_PI;
      if (ori >= M_PI) ori -= M_PI;

      prob += getEdgeProb(Point2D<int>(p1), ori);
      prob += getEdgeProb(Point2D<int>(center), ori);
      pixCount += 2;
    }
    pixCount *= 1;
  }

  //double pr =  exp(-prob/ double(pixCount));
  //LINFO("%i: PRob %f pixCount %i = %f", (uint)surface.polygon.size(), prob, pixCount*2, pr);

  return exp(-prob/ double(pixCount))*2;
  //return prob/ double(pixCount);


}

double TwoHalfDSketch::getEdgeProb(Point2D<int> loc, float ori)
{

  int numOfEntries = itsOriEdgesDT.size();

  double lambda = (1/6)*M_PI/180; //6 degrees error =  1 pixel distance

  double D = M_PI/numOfEntries;
  int oriIdx = (int)floor(ori/D);

  float minDist = 10000;
  if (itsOriEdgesDT[0].coordsOk(loc))
  {
    for(int i=0; i<numOfEntries; i++)
    {
      float v1 = itsOriEdgesDT[i].getVal(loc) + lambda*angDiff(oriIdx*D, i*D);
      if (v1 < minDist)
        minDist = v1;
    }
  }

  return minDist;
}

double TwoHalfDSketch::getLineProb(Point2D<int> p1, Point2D<int> p2, float ori, int& pixCount)
{

  int numOfEntries = itsOriEdgesDT.size();

  double lambda = (1/2)*M_PI/180; //6 degrees error =  1 pixel distance

  double D = M_PI/numOfEntries;
  int oriIdx = (int)floor(ori/D);

  int dx = p2.i - p1.i, ax = abs(dx) << 1, sx = signOf(dx);
  int dy = p2.j - p1.j, ay = abs(dy) << 1, sy = signOf(dy);
  int x = p1.i, y = p1.j;

  double prob = 0;
  int wSize = 1;
  if (ax > ay)
  {
    int d = ay - (ax >> 1);
    for (;;)
    {
      //search for a max edge prob in a window
      for(int yy = y-wSize; yy < y+wSize; yy++)
        for(int xx = x-wSize; xx < x+wSize; xx++)
        {
          float minDist = 10000;
          if (itsOriEdgesDT[0].coordsOk(xx,yy))
          {
            for(int i=0; i<numOfEntries; i++)
            {
              float v1 = itsOriEdgesDT[i].getVal(xx,yy) + lambda*angDiff(oriIdx*D, i*D);
              if (v1 < minDist)
                minDist = v1;
            }
          }
          prob += minDist;
          pixCount++;
        }

      if (x == p2.i) return prob;
      if (d >= 0) { y += sy; d -= ax; }
      x += sx; d += ay;
    }
  } else {
    int d = ax - (ay >> 1);
    for (;;)
    {
      for(int yy = y-wSize; yy < y+wSize; yy++)
        for(int xx = x-wSize; xx < x+wSize; xx++)
        {
          float minDist = 10000;
          if (itsOriEdgesDT[0].coordsOk(xx,yy))
          {
            for(int i=0; i<numOfEntries; i++)
            {
              float v1 = itsOriEdgesDT[i].getVal(xx,yy) + lambda*angDiff(oriIdx*D, i*D);
              if (v1 < minDist)
                minDist = v1;
            }
          }
          prob += minDist;
          pixCount++;
        }
      if (y == p2.j) return prob;

      if (d >= 0) { x += sx; d -= ay; }
      y += sy; d += ax;
    }
  }

  return prob;
  
}


void TwoHalfDSketch::printResults(float bias)
{
  for(uint i=0; i<itsProposals.size(); i++)
  {
    Rectangle bb = itsProposals[i].getBB();

    if (i == (uint)itsBiasId)
      bias = itsBias;
    else
      bias = -1;

    printf("Result: %i %i %f %i %i %i %i %f %f %f\n",
        i, (uint)itsBiasId,
        bias,
        bb.topLeft().i,
        bb.topLeft().j,
        bb.bottomRight().i,
        bb.bottomRight().j,
        itsProposals[i].prob,
        itsProposals[i].pos.i,
        itsProposals[i].pos.j

        );
  }

}


Layout<PixRGB<byte> > TwoHalfDSketch::getDebugImage(SimEventQueue& q)
{
  Layout<PixRGB<byte> > outDisp;

  Image<float> input = itsLinesMag;
  inplaceNormalize(input, 0.0F, 255.0F);

  Image<PixRGB<byte> > worldFrame = input;
  for(uint i=0; i<itsSurfaces.size(); i++)
  {
    if (itsSurfaces[i].polygon.getNumLines()> 0)
    {
      Polygon& polygon = itsSurfaces[i].polygon;
      //We have a polygon
      for(uint j=0; j<polygon.getNumLines(); j++)
      {
        Line l = itsProposals[i].getLine(j); //Get the line up to scale and pos
        Point2D<int> p1 = (Point2D<int>)l.getP1();
        Point2D<int> p2 = (Point2D<int>)l.getP2();
        drawLine(worldFrame, p1, p2, PixRGB<byte>(0,255,0));
      }

    } else {
      drawSuperquadric(worldFrame,
          Point2D<int>(itsSurfaces[i].pos),
          itsSurfaces[i].a, itsSurfaces[i].b, itsSurfaces[i].e, 
          PixRGB<byte>(0,255,0),
          itsSurfaces[i].rot, itsSurfaces[i].k1, itsSurfaces[i].k2,
          itsSurfaces[i].start,itsSurfaces[i].end);
    }
    //char msg[255];
    //sprintf(msg, "%0.2f", itsSurfaces[i].prob);
    //writeText(worldFrame, (Point2D<int>)itsSurfaces[i].getPos(), msg,
    //    PixRGB<byte>(255,255,255),
    //    PixRGB<byte>(0,0,0));

  }

  Image<PixRGB<byte> > proposalsFrame = input;
  float maxProb = -100000;
  for(uint i=0; i<itsProposals.size(); i++)
  {
    LINFO("Propsal %i p:%fx%f s:%f a:%fx%f r:%f p:%f", i,
        itsProposals[i].pos.i, itsProposals[i].pos.j,
        itsProposals[i].scale,
        itsProposals[i].aspect.i, itsProposals[i].aspect.j,
        itsProposals[i].rot,
        itsProposals[i].prob);
    //if (itsProposals[i].prob > 0.00)
    {
      if (itsProposals[i].prob > maxProb)
        maxProb = itsProposals[i].prob;

      if (itsProposals[i].polygon.getNumLines() > 0)
      {
        Polygon& polygon = itsProposals[i].polygon;
        //We have a polygon
        for(uint j=0; j<polygon.getNumLines(); j++)
        {
          Line l = itsProposals[i].getLine(j); //Get the line up to scale and pos
          Point2D<int> p1 = (Point2D<int>)l.getP1();
          Point2D<int> p2 = (Point2D<int>)l.getP2();
          drawLine(proposalsFrame, p1,p2, PixRGB<byte>(255,0,0));
        }
      } else {

        float nSeg = 20;
        const float dTheta = 2*M_PI / (float)nSeg;

        float a = itsProposals[i].a;
        float b = itsProposals[i].b;
        float e = itsProposals[i].e;
        float k1 = itsProposals[i].k1;
        float k2 = itsProposals[i].k2;
        float rot = itsProposals[i].rot;
        float start = itsProposals[i].start;
        float end = itsProposals[i].end;

        Point2D<float> p = itsProposals[i].pos;

        for (float theta=start; theta < end; theta += dTheta)
        {
          Point2D<float> p1 = ellipsoid(a,b, e, theta);
          Point2D<float> tmpPos;

          //Sheer
          tmpPos.i = p1.i + p1.j*k1;
          tmpPos.j = p1.i*k2 + p1.j;

          //Rotate and move to p
          p1.i = (cos(rot)*tmpPos.i - sin(rot)*tmpPos.j) + p.i;
          p1.j = (sin(rot)*tmpPos.i + cos(rot)*tmpPos.j) + p.j;

          drawCircle(proposalsFrame, (Point2D<int>)p1, 3, PixRGB<byte>(255,0,0));
        }
      }

      char msg[255];
      sprintf(msg, "%0.2f", itsProposals[i].prob);
      writeText(proposalsFrame, (Point2D<int>)itsProposals[i].getPos(), msg,
          PixRGB<byte>(255,255,255),
          PixRGB<byte>(0,0,0));
    }

  }
  char msg[255];
  sprintf(msg, "Max: %0.2f", maxProb);
  writeText(proposalsFrame, Point2D<int>(0,proposalsFrame.getHeight()-20), msg,
      PixRGB<byte>(255,255,255),
      PixRGB<byte>(0,0,0));


  outDisp = proposalsFrame; //hcat(toRGB(Image<byte>(input)), proposalsFrame);
  outDisp = hcat(outDisp, worldFrame);

  return outDisp;

}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

