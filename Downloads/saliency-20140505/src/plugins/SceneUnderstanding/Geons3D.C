/*!@file SceneUnderstanding/Geons3D.C  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/Geons3D.C $
// $Id: Geons3D.C 13875 2010-09-03 00:54:58Z lior $
//

#ifndef Geons3D_C_DEFINED
#define Geons3D_C_DEFINED

#include "plugins/SceneUnderstanding/Geons3D.H"

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

const ModelOptionCateg MOC_Geons3D = {
  MOC_SORTPRI_3,   "Geons3D-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_Geons3DShowDebug =
  { MODOPT_ARG(bool), "Geons3DShowDebug", &MOC_Geons3D, OPTEXP_CORE,
    "Show debug img",
    "geons3d-debug", '\0', "<true|false>", "false" };

//Define the inst function name
SIMMODULEINSTFUNC(Geons3D);


// ######################################################################
Geons3D::Geons3D(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventTwoHalfDSketchOutput),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  SIMCALLBACK_INIT(SimEventUserInput),
  SIMCALLBACK_INIT(SimEventGeons3DPrior),
  itsShowDebug(&OPT_Geons3DShowDebug, this)
{

  itsViewPort = new ViewPort3D(320,240, true, false, true);

  double trans[3][4] = {
    {-0.999143, -0.041185, -0.004131, -1.142130},
    {-0.017002, 0.499358, -0.866229, 17.284269},
    {0.037739, -0.865416, -0.499630, 220.977236}};

  itsViewPort->setCamera(trans);

  //itsGHough.readTable("hough.dat");
 // itsGHough2.readTable("hough2.dat");

  initRandomNumbers();

  itsLearn = true;

  //itsViewPort->setWireframeMode(false);
  //itsViewPort->setLightsMode(false);
  //itsViewPort->setFeedbackMode(false);

  ////Build lookup table from ground plane
  //itsViewPort->initFrame();
  //itsViewPort->drawGround(Point2D<float>(1000, 1000), PixRGB<byte>(256,256,256));
  //itsTableDepth = flipVertic(itsViewPort->getDepthFrame());

  //itsTrainingThreshold = 0.65;

  //itsThreadServer.reset(new WorkThreadServer("Geons3D", 4));



  //The user proposals
  GeonState gs;
  gs.pos = Point3D<float>(-31,-34,0);
  gs.rot = Point3D<float>(10,0,0);

  gs.superQuadric.its_a1 = 25; 
  gs.superQuadric.its_a2 = 30;
  gs.superQuadric.its_a3 = 25;
  gs.superQuadric.its_alpha = -1; //25;
  gs.superQuadric.its_n = 0.0;
  gs.superQuadric.its_e = 1.0;
  gs.superQuadric.its_u1 = -M_PI ;
  gs.superQuadric.its_u2 = M_PI;
  gs.superQuadric.its_v1 = -M_PI;
  gs.superQuadric.its_v2 = M_PI;
  gs.superQuadric.its_s1 = 0.0f;
  gs.superQuadric.its_t1 = 0.0f;
  gs.superQuadric.its_s2 = 1.0f;
  gs.superQuadric.its_t2 = 1.0f;
  
  itsProposals.push_back(gs);
}

// ######################################################################
Geons3D::~Geons3D()
{
}


// ######################################################################
void Geons3D::onSimEventTwoHalfDSketchOutput(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventTwoHalfDSketchOutput>& e)
{

  //Check if we have metadata
  if (SeC<SimEventLGNOutput> lgn = q.check<SimEventLGNOutput>(this))
  {
    rutz::shared_ptr<GenericFrame::MetaData> metaData = lgn->getMetaData();
    if (metaData.get() != 0) {
      itsObjectsData.dyn_cast_from(metaData);
    }
    itsLGNInput = lgn->getCells();
  }

  //Check if we have the smap
  if (SeC<SimEventSMapOutput> smap = q.check<SimEventSMapOutput>(this))
    itsSMap = smap->getSMap();

  itsSurfaces = e->getSurfaces();

  evolve(q);

}

// ######################################################################
void Geons3D::onSimEventGeons3DPrior(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventGeons3DPrior>& e)
{
  itsGeonsState = e->getGeons();
  itsGlobalRotation = e->getRotation();
  itsGlobalPos = e->getPos();

}

// ######################################################################
void Geons3D::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage(q);
      ofs->writeRgbLayout(disp, "Geons3D", FrameInfo("Geons3D", SRC_POS));
    }
}

// ######################################################################
void Geons3D::onSimEventUserInput(SimEventQueue& q, rutz::shared_ptr<SimEventUserInput>& e)
{

  LINFO("Got event %s %ix%i key=%i",
      e->getWinName(),
      e->getMouseClick().i,
      e->getMouseClick().j,
      e->getKey());

  if (strcmp(e->getWinName(), "Geons3D"))
    return;
  
  itsLearn = !itsLearn;
  //LINFO("Learning: %i", itsLearn);
  if (e->getMouseClick().isValid())
  {
  }

  GeonState& geon = itsProposals[0];
  switch(e->getKey())
  {
    case 104: //up
      geon.pos.x -= 1.0;
      break;
    case 98: //down
      geon.pos.x += 1.0;
      break;
    case 100: //left
      geon.pos.y -= 1.0;
      break;
    case 102: //right
      geon.pos.y += 1.0;
      break;
    case 21: //=
      geon.pos.z += 1.0;
      break;
    case 20: //-
      geon.pos.z -= 1.0;
      break;
    case 38: //a
      geon.rot.x += 5;
      break;
    case 52: //z
      geon.rot.x -= 5;
      break;
    case 39: //s
      geon.rot.y += 5;
      break;
    case 53: //x
      geon.rot.y -= 5;
      break;
    case 40: //d
      geon.rot.z += 5;
      break;
    case 54: //c
      geon.rot.z -= 5;
      break;

    case 10: //1
      geon.superQuadric.its_a1 += 1; 
      break;
    case 24: //q
      geon.superQuadric.its_a1 -= 1; 
      break;
    case 11: //2
      geon.superQuadric.its_a2 += 1; 
      break;
    case 25: //w
      geon.superQuadric.its_a2 -= 1; 
      break;
    case 12: //3
      geon.superQuadric.its_a3 += 1; 
      break;
    case 26: //e
      geon.superQuadric.its_a3 -= 1; 
      break;
    case 13: //4
      geon.superQuadric.its_n += 0.1; 
      break;
    case 27: //r
      geon.superQuadric.its_n -= 0.1; 
      break;
    case 14: //5
      geon.superQuadric.its_e += 0.1; 
      break;
    case 28: //t
      geon.superQuadric.its_e -= 0.1; 
      break;
    case 15: //5
      geon.superQuadric.its_alpha += 0.1; 
      break;
    case 29: //t
      geon.superQuadric.its_alpha -= 0.1; 
      break;
  }

  LINFO("Pos(%0.2f,%0.2f,%0.2f), rotation(%0.2f,%0.2f,%0.2f) (%f,%f,%f,%f,%f,%f)",
      geon.pos.x, geon.pos.y, geon.pos.z,
      geon.rot.x, geon.rot.y, geon.rot.z,
      geon.superQuadric.its_a1, geon.superQuadric.its_a2, geon.superQuadric.its_a3,
      geon.superQuadric.its_n, geon.superQuadric.its_e,
      geon.superQuadric.its_alpha);


  //evolve(q);

}

void Geons3D::calcGeonLikelihood(GeonState& geon)
{
  Image<float> edges;
  Image<float> surface;
  double edgeProb = calcGeonEdgeLikelihood(geon, edges, surface);
  double surfaceProb = calcGeonSurfaceLikelihood(geon, edges, surface);

  geon.prob = edgeProb * surfaceProb;

}

void Geons3D::drawGeon(const GeonState& geon)
{
}

void Geons3D::renderScene(const GeonState& geon, std::vector<ViewPort3D::Line>& lines, Image<PixRGB<byte> >& frame)
{
  itsViewPort->setWireframeMode(true);
  itsViewPort->setLightsMode(true);
  itsViewPort->setFeedbackMode(true);
  itsViewPort->initFrame();

  drawGeon(geon);
  lines = itsViewPort->getFrameLines();

  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); 
  drawGeon(geon);

  frame =  flipVertic(itsViewPort->getFrame());
}

double Geons3D::calcGeonEdgeLikelihood(GeonState& geon, Image<float>& edges, Image<float>& surface)
{

  itsViewPort->setWireframeMode(true);
  itsViewPort->setLightsMode(true);
  itsViewPort->setFeedbackMode(true);
  itsViewPort->initFrame();

  drawGeon(geon);
  std::vector<ViewPort3D::Line> lines = itsViewPort->getFrameLines();

  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); 
  drawGeon(geon);

  Image<PixRGB<byte> > frame =  flipVertic(itsViewPort->getFrame());
  surface = luminance(frame);


  Image<float> edgesMag, edgesOri;
  gradientSobel(surface, edgesMag, edgesOri);

  Image<float> mag(itsEdgesDT.getDims(),ZEROS);
  Image<float> ori(itsEdgesDT.getDims(),ZEROS);
  
  inplaceNormalize(edgesMag, 0.0F, 100.0F);
  for(uint i=0; i<lines.size(); i++)
  {

    double dx = lines[i].p2.i - lines[i].p1.i;
    double dy = lines[i].p2.j - lines[i].p1.j;
    double ang = atan2(dx,dy) + M_PI/2;
    //Change orientation from 0 to M_PI
    if (ang < 0) ang += M_PI;
    if (ang >= M_PI) ang -= M_PI;

    //Get the center of the line
    Point2D<float> center = lines[i].p1 + Point2D<float>(dx/2,dy/2); 
    if (edgesMag.coordsOk(Point2D<int>(lines[i].p1)) &&
        edgesMag.coordsOk(Point2D<int>(lines[i].p2)) &&
        edgesMag.getVal(Point2D<int>(lines[i].p1)) > 10 &&
        edgesMag.getVal(Point2D<int>(lines[i].p2)) > 10 &&
        edgesMag.getVal(Point2D<int>(center)) > 10)
    {
      drawLine(mag, Point2D<int>(lines[i].p1), Point2D<int>(lines[i].p2), 1.0F);
      drawLine(ori, Point2D<int>(lines[i].p1), Point2D<int>(lines[i].p2), float(ang));
    }
  }


  edges = mag;

  return getEdgeProb(itsEdgesDT, ori, mag);

}

double Geons3D::calcGeonSurfaceLikelihood(GeonState& geon, Image<float>& edges, Image<float>& surface)
{
  //Remove the edges from the surface
  for(uint i=0; i<surface.size(); i++)
    if (edges[i] > 0)
      surface[i] = 0;

  return getSurfaceProb(itsEdgesDT,surface);

}


double Geons3D::getEdgeProb(Image<float>& mag,  Image<float>& modelOri, Image<float>& modelMag)
{

  //SHOWIMG(data);
  double prob = 0;
  int pixCount = 0;

  //double r = 20;

  int numOfEntries = itsOriEdgesDT.size();
  double D = M_PI/numOfEntries;

  //SHOWIMG(modelOri);
  //SHOWIMG(itsLinesOri);

  ImageSet<float> modelOriImg(numOfEntries, modelMag.getDims(), ZEROS);

  for(int y=0; y < modelMag.getHeight(); y++)
    for(int x=0; x < modelMag.getWidth();  x++)
      if (modelMag.getVal(x,y) > 0)
      {

        float phi = modelOri.getVal(x,y);

        int oriIdx = (int)floor(phi/D);

        modelOriImg[oriIdx].setVal(x,y, 255.0F);

        int oriIdxMin = (oriIdx-1)%numOfEntries;
        if (oriIdxMin < 0) 
          oriIdxMin += numOfEntries;
        int oriIdxMax = (oriIdx+1)%numOfEntries;

        float v1 = itsOriEdgesDT[oriIdxMin].getVal(x,y);
        float v2 = itsOriEdgesDT[oriIdx].getVal(x,y);
        float v3 = itsOriEdgesDT[oriIdxMax].getVal(x,y);

        prob += std::min(v1, std::min(v2,v3));
        //prob += ( (v1*2) + v2 + (v3*2) )/5;

        //prob += itsEdgesDT.getVal(x,y);
        pixCount++;
      }

  return exp(-prob/ double(pixCount*20));
}

double Geons3D::getSurfaceProb(Image<float>& data, Image<float>& model)
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


std::vector<Geons3D::GeonState> Geons3D::proposeGeons(Rectangle& attenLoc)
{

  std::vector<Geons3D::GeonState> geons;

  return geons;

}

void Geons3D::testLikelihood()
{

  //For Frame 390
  //Image<PixRGB<byte> > img = calcGeonLikelihood(geon);
  //double trueLocProb = geon.prob;


  //geon.pos = Point3D<float>(26.80,-44.90,15.00);
  //img = calcGeonLikelihood(geon);
  //double emptyLocProb = geon.prob;

  //geon.pos = Point3D<float>(5.90,-251.11,15.00);
  //img = calcGeonLikelihood(geon);
  //double noiseLocProb = geon.prob;

  //geon.pos = Point3D<float>(-6.014052,-206.406448,15.00);
  //geon.rotation = Point3D<float>(0,0,55.973370);
  //img = calcGeonLikelihood(geon);
  //double noise2LocProb = geon.prob;
  //
  //LINFO("TrueLocProb  %f", trueLocProb);
  //LINFO("EmptyLocProb %f", emptyLocProb);
  //LINFO("NoiseLocProb %f", noiseLocProb);
  //LINFO("Noise2LocProb %f", noise2LocProb);

  //for frame 0

  //LINFO("Get true prob");
  //calcGeonLikelihood(geon);
  //double trueLocProb = geon.prob;

  //geon.pos = Point3D<float>(-4.964483, 45.255852,5.188423);
  //geon.rotation = Point3D<float>(0,0,85);

  //LINFO("Get bad prob");
  //calcGeonLikelihood(geon);
  //double badLocProb = geon.prob;

  //LINFO("True prob %f", trueLocProb);
  //LINFO("Bad prob %f", badLocProb);
}

// ######################################################################
void Geons3D::evolve(SimEventQueue& q)
{

  //std::vector<Geons3D::GeonState> geonsState = proposeGeons();
  //for(uint j=0; j<geonsState.size(); j++)
  //{
  //  calcGeonLikelihood(geonsState[j]);
  //  totalEvaluated++;
  //}

}

Image<PixRGB<byte> > Geons3D::getGeonImage(GeonState& geon)
{

  itsViewPort->setWireframeMode(false);
  itsViewPort->setLightsMode(true);
  itsViewPort->setFeedbackMode(false);

  itsViewPort->initFrame();

  itsViewPort->setColor(PixRGB<byte>(255,100,100));
  glTranslatef(geon.pos.x, geon.pos.y, geon.pos.z);
  glRotatef(geon.rot.x, 1,0,0);
  glRotatef(geon.rot.y, 0,1,0);
  glRotatef(geon.rot.z, 0,0,1);

  if (geon.superQuadric.its_alpha >= 0)
    geon.superQuadric.solidToroid();
  else
    geon.superQuadric.solidEllipsoid();
  
  return  flipVertic(itsViewPort->getFrame());
}


Layout<PixRGB<byte> > Geons3D::getDebugImage(SimEventQueue& q)
{
  Layout<PixRGB<byte> > outDisp;

  //Image<float> input = itsLinesMag;
  //inplaceNormalize(input, 0.0F, 255.0F);
  //Image<PixRGB<byte> > tmp = toRGB(Image<byte>(input));

  Image<PixRGB<byte> > inputFrame = itsLGNInput[0];
  float nSeg = 20;
  const float dTheta = 2*M_PI / (float)nSeg;

  for(uint i=0; i<itsSurfaces.size(); i++)
  {
    const TwoHalfDSketch::SurfaceState& surface = itsSurfaces[i];
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

      drawLine(inputFrame, (Point2D<int>)p1, (Point2D<int>)p2, PixRGB<byte>(255,0,0));

    }
  }

  Image<PixRGB<byte> > proposalsImg(inputFrame.getDims(), ZEROS);; 
  for(uint i=0; i<itsProposals.size(); i++)
  {
    Image<PixRGB<byte> > frame = getGeonImage(itsProposals[i]);
    proposalsImg += rescale(frame, inputFrame.getDims());
  }

  outDisp = hcat(inputFrame, proposalsImg);
  
  return outDisp;

}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

