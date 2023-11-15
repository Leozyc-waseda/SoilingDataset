/*!@file SceneUnderstanding/V4d.C non-accidental features */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/V4d.C $
// $Id: V4d.C 13551 2010-06-10 21:56:32Z itti $
//

#ifndef V4d_C_DEFINED
#define V4d_C_DEFINED

#include "plugins/SceneUnderstanding/V4d.H"

#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Convolutions.H"
#include "Image/fancynorm.H"
#include "Image/Point3D.H"
#include "SIFT/FeatureVector.H"
#include "Simulation/SimEventQueue.H"
#include "Neuro/EnvVisualCortex.H"
#include "GUI/DebugWin.H"
#include "Util/CpuTimer.H"
#include "Util/JobServer.H"
#include "Util/JobWithSemaphore.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>

const ModelOptionCateg MOC_V4d = {
  MOC_SORTPRI_3,   "V4d-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_V4dShowDebug =
  { MODOPT_ARG(bool), "V4dShowDebug", &MOC_V4d, OPTEXP_CORE,
    "Show debug img",
    "v4d-debug", '\0', "<true|false>", "false" };

namespace
{
  class GHTJob : public JobWithSemaphore
  {
  public:
    GHTJob(V4d* v4d, Image<float>& acc, int angIdx, std::vector<V1::EdgeState>& rTable) :
      itsV4d(v4d),
      itsAcc(acc),
      itsAngIdx(angIdx),
      itsRTable(rTable)
    {}

    virtual ~GHTJob() {}

    virtual void run()
    {
      itsV4d->voteForFeature(itsAcc, itsAngIdx, itsRTable);

      this->markFinished();
    }

    virtual const char* jobType() const { return "GHTJob"; }
    V4d* itsV4d;
    Image<float>& itsAcc;
    int itsAngIdx;
    std::vector<V1::EdgeState>& itsRTable;
  };

  class LikelihoodJob : public JobWithSemaphore
  {
  public:
    LikelihoodJob(V4d* v4d, V4d::NAFState& nafState) :
      itsV4d(v4d),
      itsNAFState(nafState)
    {}

    virtual ~LikelihoodJob() {}

    virtual void run()
    {
      itsV4d->getParticleLikelihood(itsNAFState);
      this->markFinished();
    }

    virtual const char* jobType() const { return "LikelihoodJob"; }

    V4d* itsV4d;
    V4d::NAFState& itsNAFState;


  };
}


// ######################################################################
V4d::V4d(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventV2Output),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  SIMCALLBACK_INIT(SimEventUserInput),
  SIMCALLBACK_INIT(SimEventV4BiasOutput),
  itsShowDebug(&OPT_V4dShowDebug, this),
  itsMaxVal(0),
  itsGHTAngStep(1),
  itsObjectsDist(370.00)
{

  itsThreadServer.reset(new WorkThreadServer("V4d", 100));

  //Camera parameters
  itsCamera = Camera(Point3D<float>(0,0,0.0),
                     Point3D<float>(0, 0,0),
                     450, //Focal length
                     320, //width
                     240); //height

//  ///NAF
  itsNAFeatures.resize(3);


  FeatureTemplate rightVertixFeature;
  rightVertixFeature.featureType = RIGHT_VERTIX;
  rightVertixFeature.outline.push_back(Point3D<float>(11.0,0.0, 0.0));
  rightVertixFeature.outline.push_back(Point3D<float>(0.0,0, 0.0));
  rightVertixFeature.outline.push_back(Point3D<float>(0.0,11.0, 0.0));

  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(0,4), 0));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(0,5), 0));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(0,6), 0));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(0,7), 0));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(0,8), 0));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(0,9), 0));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(0,10), 0));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(0,11), 0));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(0,12), 0));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(4,0), M_PI/2));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(5,0), M_PI/2));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(6,0), M_PI/2));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(7,0), M_PI/2));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(8,0), M_PI/2));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(9,0), M_PI/2));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(10,0), M_PI/2));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(11,0), M_PI/2));
  rightVertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(12,0), M_PI/2));

  alignToCenterOfMass(rightVertixFeature);
  itsNAFeatures[RIGHT_VERTIX] = rightVertixFeature;

  FeatureTemplate vertixFeature;
  vertixFeature.featureType = VERTIX;
  vertixFeature.outline.push_back(Point3D<float>(0.0,0, 0.0));
  vertixFeature.outline.push_back(Point3D<float>(0.0,11.0, 0.0));
  vertixFeature.outline.push_back(Point3D<float>(11.0,0.0, 0.0));

  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(5,0), M_PI/2));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(6,0), M_PI/2));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(7,0), M_PI/2));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(8,0), M_PI/2));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(9,0), M_PI/2));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(10,0), M_PI/2));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(11,0), M_PI/2));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(12,0), M_PI/2));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(13,0), M_PI/2));

  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(5,-5), -M_PI/4));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(6,-6), -M_PI/4));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(7,-7), -M_PI/4));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(8,-8), -M_PI/4));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(9,-9), -M_PI/4));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(10,-10), -M_PI/4));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(11,-11), -M_PI/4));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(12,-12), -M_PI/4));
  vertixFeature.edges.push_back(V1::EdgeState(Point2D<int>(13,-13), -M_PI/4));

  alignToCenterOfMass(vertixFeature);
  itsNAFeatures[VERTIX] = vertixFeature;

  FeatureTemplate arcFeature;
  arcFeature.featureType = ARC;
  arcFeature.outline.push_back(Point3D<float>(0.0,0, 0.0));
  arcFeature.outline.push_back(Point3D<float>(5.0,0.0, 0.0));

  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(5,0), 1.885212));
  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(3,1), 2.082733));
  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(4,1), 1.941421));
  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(5,1), 1.841154));

  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(1,2), 2.076972));
  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(2,2), 1.954276));
  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(3,2), 1.959018));
  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(4,2), 1.903953));
  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(5,2), 1.816486));

  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(0,3), 2.045129));
  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(1,3), 2.006197));
  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(2,3), 1.965587));
  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(3,3), 1.968628));
  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(4,3), 1.873842));
  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(5,3), 1.809913));

  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(0,4), 1.992570));
  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(1,4), 1.990212));
  arcFeature.edges.push_back(V1::EdgeState(Point2D<int>(2,4), 2.023316));


  alignToCenterOfMass(arcFeature);
  itsNAFeatures[ARC] = arcFeature;

  //Image<float> tmp(320,240, ZEROS);
  //for(uint i=0; i<vertixFeature.edges.size(); i++)
  //{
  //  LINFO("%i %i %f",
  //      vertixFeature.edges[i].pos.i,
  //      vertixFeature.edges[i].pos.j,
  //      vertixFeature.edges[i].ori + M_PI/2);
  //  drawLine(tmp, vertixFeature.edges[i].pos + Point2D<int>(320/2,240/2), vertixFeature.edges[i].ori + M_PI/2, 5.0F, 30.0F);
  //}
  //SHOWIMG(tmp);

  buildRTables();
  itsFeaturesParticles.resize(1000);
  for(uint i=0; i<itsFeaturesParticles.size(); i++)
  {
    itsFeaturesParticles[i].pos = Point3D<float>(-53.444447,0.000000,370.000000);
    itsFeaturesParticles[i].rot = 35*M_PI/180;
    //itsFeaturesParticles[i].pos = Point3D<float>(0,0,370);
    itsFeaturesParticles[i].featureType = VERTIX;
    itsFeaturesParticles[i].weight = 1.0/double(itsFeaturesParticles.size());
    itsFeaturesParticles[i].prob = 0;
  }

  itsHashedEdgesState.set_empty_key(-1);

  itsDebugImg = Image<PixRGB<byte> >(320,240,ZEROS);
}

// ######################################################################
V4d::~V4d()
{

}


// ######################################################################
void V4d::alignToCenterOfMass(FeatureTemplate& featureTemplate)
{
  //Set the  0,0 location around the center of mass
  //get the center of the object;
  float centerX = 0, centerY = 0, centerZ = 0;
  for(uint i=0; i<featureTemplate.outline.size(); i++)
  {
    centerX += featureTemplate.outline[i].x;
    centerY += featureTemplate.outline[i].y;
    centerZ += featureTemplate.outline[i].z;
  }
  centerX /= featureTemplate.outline.size();
  centerY /= featureTemplate.outline.size();
  centerZ /= featureTemplate.outline.size();

  for(uint i=0; i<featureTemplate.outline.size(); i++)
    featureTemplate.outline[i] -= Point3D<float>(centerX, centerY, centerZ);

}


// ######################################################################
void V4d::buildRTables()
{
  for(uint fid=0; fid<itsNAFeatures.size(); fid++)
  {
    if (itsNAFeatures[fid].outline.size() > 0)
    {
      NAFState naFeature;
      naFeature.pos = Point3D<float>(0, 0, itsObjectsDist);
      naFeature.rot = 0;
      naFeature.featureType = (NAF_TYPE)fid;
      naFeature.prob = 0;

      std::vector<Point2D<int> > outline = getImageOutline(naFeature);
      Image<float> tmp(320,240, ZEROS);
      for(uint i=0; i<outline.size()-1; i++)
        drawLine(tmp, outline[i], outline[i+1], 1.0F, 1);

      //build R table
      for(int x=0; x<tmp.getWidth(); x+=1)
        for(int y=0; y<tmp.getHeight(); y+=1)
        {
          if (tmp.getVal(x,y) == 1.0)
          {
            int rx = x-tmp.getWidth()/2;
            int ry = y-tmp.getHeight()/2;

            RTableEntry rTableEntry;
            rTableEntry.loc = Point2D<int>(rx, ry);
            rTableEntry.rot = atan2(ry, rx);
            itsNAFeatures[fid].rTable.push_back(rTableEntry);
          }
        }
    }
  }
}

// ######################################################################
void V4d::onSimEventV4BiasOutput(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventV4BiasOutput>& e)
{
  itsBias = e->getCells();
  //LINFO("Got bias of size %lu", itsBias.size());
}

// ######################################################################
void V4d::onSimEventV2Output(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventV2Output>& e)
{
  std::vector<V1::EdgeState> edgesState; // = e->getEdges();
  std::vector<V2::CornerState> cornersState = e->getCorners();
  itsEdgesState = edgesState;

  LINFO("Build priority queue for corners");
  //Corners queue should be empty (We get all corners during proposals)
  ASSERT(cornersState.size());
  for(uint i=0; i<cornersState.size(); i++)
    itsCornersState.push(cornersState[i]);

  //LINFO("Found %lu corners\n",
  //    itsCornersState.size());

  Image<float> tmp(320,240,ZEROS);
  itsHashedEdgesState.clear();
  for(uint i=0; i<edgesState.size(); i++)
  {
    int key = edgesState[i].pos.i + 320*edgesState[i].pos.j;
    tmp.setVal(edgesState[i].pos, edgesState[i].prob);
    itsHashedEdgesState[key] = edgesState[i];
  }
  //SHOWIMG(tmp);

  //evolveSift();
  evolve();

  //q.post(rutz::make_shared(new SimEventV4dOutput(this, itsFeaturesParticles)));

}

// ######################################################################
void V4d::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage();
      ofs->writeRgbLayout(disp, "V4d", FrameInfo("V4d", SRC_POS));

      //ofs->writeRGB(itsDebugImg, "V4dDebug", FrameInfo("V4dDebug", SRC_POS));
    }
}

// ######################################################################
void V4d::onSimEventUserInput(SimEventQueue& q, rutz::shared_ptr<SimEventUserInput>& e)
{

  LINFO("Got event %s %ix%i key=%i",
      e->getWinName(),
      e->getMouseClick().i,
      e->getMouseClick().j,
      e->getKey());

  Image<float> mag(320,240,ZEROS);
  Image<float> ori(320,240,ZEROS);

  Point2D<int> loc = e->getMouseClick();

  if (loc.i > 320)
    loc.i -= 320;
  //Draw the edges
  for(uint i=0; i<itsEdgesState.size(); i++)
  {
    mag.setVal(itsEdgesState[i].pos, itsEdgesState[i].prob);
    ori.setVal(itsEdgesState[i].pos, itsEdgesState[i].ori);
  }

   //Histogram OV(360);
   //float mainOri = calculateOrientationVector(loc, OV);
   //LINFO("Main ori %f", mainOri);

   //createDescriptor(loc, OV, 0);

  Image<float> magC = crop(mag, loc - 10, Dims(20,20));
  Image<float> oriC = crop(ori, loc - 10, Dims(20,20));
  //oriC *= 180.0/M_PI;

  for(int y=0; y<oriC.getHeight(); y++)
    for(int x=0; x<oriC.getWidth(); x++)
    {
      if (magC.getVal(x,y) > 0.1)
        printf("%i ", (int)(oriC.getVal(x,y)*180/M_PI + 90));
    }
  printf("\n");

  double sumSin = 0, sumCos = 0;
  float numElem = 0;
  for(uint i=0; i<oriC.size(); i++)
  {
    if (magC[i] > 0.1)
    {
      printf("%0.1f ", oriC[i]*180/M_PI+90);
      sumSin += sin(oriC[i]+ M_PI/2);
      sumCos += cos(oriC[i] + M_PI/2);
      numElem++;
    }
  }
  double sum = sqrt( squareOf(sumSin/numElem) + squareOf(sumCos/numElem));
  printf("\n");
  LINFO("sin %f cos %f Var %f", sumSin, sumCos, sum);


  SHOWIMG(magC);
  SHOWIMG(oriC);

  //Image<float> magN = maxNormalize(magC, 0.0F, 1.0F, VCXNORM_FANCYONE);
  //Image<float> oriN = maxNormalize(oriC, 0.0F, 1.0F, VCXNORM_FANCYONE);

  //for(uint i=0; i<oriN.size(); i++)
  //{
  //  if (oriN[i] > 0.1)
  //    magC[i] = 128;
  //}
  //
  //SHOWIMG(magN);
  //SHOWIMG(oriC);
  //SHOWIMG(oriN);
  //SHOWIMG(magC);


  //if (e->getMouseClick().isValid())
  //{
  //  Point3D<float>  iPoint = itsCamera.inverseProjection((Point2D<float>)e->getMouseClick(), itsObjectsDist);
  //  itsFeaturesParticles[0].pos = iPoint;
  //}

  //switch(e->getKey())
  //{
  //  case 98: itsFeaturesParticles[0].rot += 5*M_PI/180; break;
  //  case 104: itsFeaturesParticles[0].rot -= 5*M_PI/180; break;
  //}

  //itsFeaturesParticles[0].featureType = VERTIX;
  //itsFeaturesParticles[0].weight = 1.0/double(itsFeaturesParticles.size());
  //itsFeaturesParticles[0].prob = 0;

}

// ######################################################################
void V4d::evolveSift()
{
  Image<float> tmp(320,240,ZEROS);

  dense_hash_map<int, V1::EdgeState>::const_iterator iter;
  for(iter = itsHashedEdgesState.begin(); iter != itsHashedEdgesState.end(); iter++)
  {
    V1::EdgeState edgeState = iter->second;

    if (edgeState.prob > 0)
    {
      tmp.setVal(edgeState.pos, edgeState.prob);
  //    Histogram OV(36);

  //    calculateOrientationVector(edgeState.pos, OV);
  //    //LINFO("Main ori %f", mainOri);

  //    createDescriptor(edgeState.pos, OV, 0);
    }
  }
  LINFO("Done");
  SHOWIMG(tmp);

  IplImage* img = img2ipl(tmp);
  IplImage* hImg = cvCreateImage( cvSize(img->width, img->height), IPL_DEPTH_32F, 1 );

  cvCornerHarris(img, hImg, 5, 5, 0);
  SHOWIMG(ipl2float(hImg));


  SHOWIMG(ipl2float(hImg));

  //Point2D<int> loc(68,81);
//  Point2D<int> loc(127,127);
//  Histogram OV(36);
//
//  float mainOri = calculateOrientationVector(loc, OV);
//  LINFO("Main ori %f", mainOri);
//
//   createDescriptor(loc, OV, 0);

}

float V4d::createDescriptor(Point2D<int>& loc, Histogram& OV, float mainOri)
{
  //TODO hash me
  Image<float> perc(320,240, ZEROS);
  Image<float> ori(320,240, ZEROS);
  dense_hash_map<int, V1::EdgeState>::const_iterator iter;
  for(iter = itsHashedEdgesState.begin(); iter != itsHashedEdgesState.end(); iter++)
  {
    V1::EdgeState edgeState = iter->second;
    perc.setVal(edgeState.pos, edgeState.prob);
    ori.setVal(edgeState.pos, edgeState.ori + M_PI/2);
  }


  float s = 1.5;
  // compute the effective blurring sigma corresponding to the
  // fractional scale s:
  const float sigma = 2;

  const int xi = int(loc.i + 0.5f);
  const int yi = int(loc.j + 0.5f);


  const float sinAngle = sin(mainOri), cosAngle = cos(mainOri);

  // check this scale
  const int radius = int(5.0F * sigma + 0.5F); // NOTE: Lowe uses radius=8?
  const float gausssig = float(radius); // 1/2 width of descript window
  const float gaussfac = - 0.5F / (gausssig * gausssig);

  // Scan a window of diameter 2*radius+1 around the point of
  // interest, and we will cumulate local samples into a 4x4 grid
  // of bins, with interpolation. NOTE: rx and ry loop over a
  // square that is assumed centered around the point of interest
  // and rotated to the gradient orientation (realangle):

  int scale = abs(int(s));
  scale = scale > 5 ? 5 : scale;

  FeatureVector fv;

  Image<float> tmp(radius*2+1, radius*2+1, ZEROS);
  for (int ry = -radius; ry <= radius; ry++)
    for (int rx = -radius; rx <= radius; rx++)
    {
      // rotate the point:
      const float newX = rx * cosAngle - ry * sinAngle;
      const float newY = rx * sinAngle + ry * cosAngle;

      // get the coords in the image frame of reference:
      const float orgX = newX + float(xi);
      const float orgY = newY + float(yi);

      // if outside the image, forget it:
      if (perc.coordsOk(orgX, orgY) == false) continue;

      //LINFO("%i %i %i", rx, ry, radius*2);
      tmp.setVal(rx+radius, ry+radius, perc.getValInterp(orgX, orgY));

      // find the fractional coords of the corresponding bin
      // (we subdivide our window into a 4x4 grid of bins):
      const float xf = 2.0F + 2.0F * float(rx) / float(radius);
      const float yf = 2.0F + 2.0F * float(ry) / float(radius);


      // find the Gaussian weight from distance to center and
      // get weighted gradient magnitude:
      const float gaussFactor = expf((newX*newX+newY*newY) * gaussfac);
      const float weightedMagnitude =
        gaussFactor * perc.getValInterp(orgX, orgY);

      // get the gradient orientation relative to the keypoint
      // orientation and scale it for 8 orientation bins:
      float gradAng = ori.getValInterp(orgX, orgY) - mainOri;

      gradAng=fmod(gradAng, 2*M_PI); //bring the range from 0 to M_PI

      //convert from -M_PI to M_PI
      if (gradAng < 0.0) gradAng+=2*M_PI; //convert to -M_PI to M_PI
      if (gradAng >= M_PI) gradAng-=2*M_PI;
      //split to eight bins
      const float orient = (gradAng + M_PI) * 8 / (2 * M_PI);

      /*
      //reflect the angle to convert from 0 to M_PI
      if (gradAng >= M_PI) gradAng-=M_PI;
      //split to four bins
      const float orient = (gradAng + M_PI) * 4 / (2 * M_PI);
      */

      // will be interpolated into 2 x 2 x 2 bins:
      fv.addValue(xf, yf, orient, weightedMagnitude);


    }
  SHOWIMG(tmp);

  // normalize, clamp, scale and convert to byte:
  std::vector<byte> oriVec;
  fv.toByteKey(oriVec);

  SHOWIMG(fv.getFeatureVectorImage(oriVec));

  return 0;
}

float V4d::calculateOrientationVector(Point2D<int>& loc, Histogram& OV)
{

  //TODO hash me
  Image<float> perc(320,240, ZEROS);
  Image<float> ori(320,240, ZEROS);
  dense_hash_map<int, V1::EdgeState>::const_iterator iter;
  for(iter = itsHashedEdgesState.begin(); iter != itsHashedEdgesState.end(); iter++)
  {
    V1::EdgeState edgeState = iter->second;
    perc.setVal(edgeState.pos, edgeState.prob);
    ori.setVal(edgeState.pos, edgeState.ori);
  }

  int histSize = 36;

  // compute the effective blurring sigma corresponding to the
  // fractional scale s:
  const float sigma = 1;

  const float sig = 1.5F * sigma, inv2sig2 = - 0.5F / (sig * sig);
  const int dimX = perc.getWidth(), dimY = perc.getHeight();

  const int xi = int(loc.i + 0.5f);
  const int yi = int(loc.j + 0.5f);

  const int rad = int(3.0F * sig);
  const int rad2 = rad * rad;

  // search bounds:
  int starty = yi - rad; if (starty < 0) starty = 0;
  int stopy = yi + rad; if (stopy >= dimY) stopy = dimY-1;

  // 1. Calculate orientation vector
  for (int ind_y = starty; ind_y <= stopy; ind_y ++)
  {
    // given that y, get the corresponding range of x values that
    // lie within the disk (and remain within the image):
    const int yoff = ind_y - yi;
    const int bound = int(sqrtf(float(rad2 - yoff*yoff)) + 0.5F);
    int startx = xi - bound; if (startx < 0) startx = 0;
    int stopx = xi + bound; if (stopx >= dimX) stopx = dimX-1;

    for (int ind_x = startx; ind_x <= stopx; ind_x ++)
    {
      const float dx = float(ind_x) - loc.i, dy = float(ind_y) - loc.j;
      const float distSq = dx * dx + dy * dy;

      // get gradient:
      const float gradVal = perc.getVal(ind_x, ind_y);

      // compute the gaussian weight for this histogram entry:
      const float gaussianWeight = expf(distSq * inv2sig2);

      // add this orientation to the histogram
      // [-pi ; pi] -> [0 ; 2pi]
      float angle = ori.getVal(ind_x, ind_y) + M_PI;

      // [0 ; 2pi] -> [0 ; 36]
      angle = 0.5F * angle * histSize / M_PI;
      while (angle < 0.0F) angle += histSize;
      while (angle >= histSize) angle -= histSize;

      OV.addValueInterp(angle, gaussianWeight * gradVal);
    }
  }


  // smooth the orientation histogram 3 times:
  for (int i = 0; i < 3; i++) OV.smooth();

  // find the max in the histogram:
  float maxPeakValue = OV.findMax();
  float peakAngle = 0;
  int ai =0;
  for (int bin = 0; bin < histSize; bin++)
  {
    // consider the peak centered around 'bin':
    const float midval = OV.getValue(bin);

    // if current value much smaller than global peak, forget it:
    //if (midval !=  maxPeakValue) continue;
    if (midval < 0.8F * maxPeakValue) continue;

    // get value to the left of current value
    const float leftval = OV.getValue((bin == 0) ? histSize-1 : bin-1);

    // get value to the right of current value
    const float rightval = OV.getValue((bin == histSize-1) ? 0 : bin+1);

    // interpolate the values to get the orientation of the peak:
    //  with f(x) = ax^2 + bx + c
    //   f(-1) = x0 = leftval
    //   f( 0) = x1 = midval
    //   f(+1) = x2 = rightval
    //  => a = (x0+x2)/2 - x1
    //     b = (x2-x0)/2
    //     c = x1
    // f'(x) = 0 => x = -b/2a
    const float a  = 0.5f * (leftval + rightval) - midval;
    const float b  = 0.5f * (rightval - leftval);
    float realangle = float(bin) - 0.5F * b / a;

    realangle *= 2.0F * M_PI / histSize; // [0:36] to [0:2pi]
    realangle -= M_PI;                      // [0:2pi] to [-pi:pi]
    peakAngle = realangle;
    ai++;
    LINFO("Ang %i %f", ai, realangle*180/M_PI + 90);
  }

  return peakAngle;

}

// ######################################################################
void V4d::evolve()
{

  ///Resample
  //for(uint i=0; i<itsFeaturesParticles.size(); i++)
  //  printf("%i %f %f %f %e %e;\n",i,
  //      itsFeaturesParticles[i].pos.x,
  //      itsFeaturesParticles[i].pos.y,
  //      itsFeaturesParticles[i].rot,
  //      itsFeaturesParticles[i].weight, itsFeaturesParticles[i].prob);
  //getchar();

  //resampleParticles(itsFeaturesParticles);
  //proposeParticlesHarris(itsFeaturesParticles, 0.0F);
  proposeParticles(itsFeaturesParticles, 0.0F);

  //Evaluate the particles;
  evaluateParticles(itsFeaturesParticles);

 // LINFO("Particle (%fx%fx%f) rot %f Prob %e",
 //     itsFeaturesParticles[0].pos.x,
 //     itsFeaturesParticles[0].pos.y,
 //     itsFeaturesParticles[0].pos.z,
 //     itsFeaturesParticles[0].rot*180/M_PI,
 //     itsFeaturesParticles[0].prob);
 // LINFO("\n\n");

}

// ######################################################################
float V4d::evaluateParticles(std::vector<NAFState>& particles)
{

  LINFO("Evaluate Particles");
  CpuTimer timer;
  timer.reset();

  std::vector<rutz::shared_ptr<LikelihoodJob> > jobs;
  for(uint p=0; p<particles.size(); p++)
  {
    //jobs.push_back(rutz::make_shared(new LikelihoodJob(this, particles[p])));
    //itsThreadServer->enqueueJob(jobs.back());
    getParticleLikelihood(particles[p]);
  }

  //wait for jobs to finish
  while(itsThreadServer->size() > 0)
    usleep(10000);

  timer.mark();
  LINFO("Total time %0.2f sec", timer.real_secs());

  //Normalize the particles;
  double sum = 0;
  double Neff = 0; //Number of efictive particles
  for(uint i=0; i<particles.size(); i++)
    sum += particles[i].prob;

  for(uint i=0; i<particles.size(); i++)
  {
    particles[i].weight = particles[i].prob/sum;
    Neff += squareOf(particles[i].weight);
  }

  Neff = 1/Neff;


  return Neff;

}

void V4d::GHT(std::vector<V4d::GHTAcc>& accRet, const FeatureTemplate& featureTemplate)
{
  ImageSet<float> acc(int(360/itsGHTAngStep), Dims(320,240), ZEROS);


  //Image<float> tmp2(320,240,ZEROS);
  //for(iter = itsHashedEdgesState.begin(); iter != itsHashedEdgesState.end(); iter++)
  //{
  //  V1::EdgeState edgeState = iter->second;
  //  tmp2.setVal(edgeState.pos, edgeState.prob);
  //}

  //IplImage* img2 = img2ipl(tmp2);
  //IplImage* hImg = cvCreateImage( cvSize(img2->width, img2->height), IPL_DEPTH_32F, 1 );

  //cvCornerHarris(img2, hImg, 5, 5, 0);
  //

  //Image<float> harris = ipl2float(hImg);
  //inplaceNormalize(harris, 0.0F, 1.0F);
  //SHOWIMG(harris);


  //float totalSum = 0;
  CpuTimer timer;
  timer.reset();
  std::vector<V1::EdgeState> rTable = featureTemplate.edges;

  //Parallel votes
  std::vector<rutz::shared_ptr<GHTJob> > jobs;

  for(int angIdx=0; angIdx<int(360/itsGHTAngStep); angIdx++)
  {
    jobs.push_back(rutz::make_shared(new GHTJob(this, acc[angIdx], angIdx, rTable)));
    itsThreadServer->enqueueJob(jobs.back());
    //voteForFeature(acc[angIdx], angIdx, rTable);
  }

  //wait for jobs to finish
  while(itsThreadServer->size() > 0)
    usleep(10000);

  //for (size_t i = 0; i < jobs.size(); ++i)
  //  jobs[i]->wait();
  timer.mark();
  LINFO("Total time %0.2f sec", timer.real_secs());

  //Image<float> tmp(320, 240, ZEROS);
  //for(int angIdx=0; angIdx<int(360/itsGHTAngStep); angIdx++)
  //{
  //  for(uint i=0; i<acc[angIdx].size(); i++)
  //  {
  //    if (acc[angIdx].getVal(i) > tmp.getVal(i))
  //      tmp.setVal(i, acc[angIdx].getVal(i));
  //  }
  //}
  //SHOWIMG(tmp);


  for(uint i=0; i<acc.size(); i++)
    for(int y=0; y<acc[i].getHeight(); y++)
      for(int x=0; x<acc[i].getWidth(); x++)
        if (acc[i].getVal(x,y) > 0)
        {
          GHTAcc ghtAcc;
          ghtAcc.pos = Point2D<int>(x,y);
          ghtAcc.ang = i*itsGHTAngStep;
          if (featureTemplate.featureType == VERTIX)
            ghtAcc.ang = i*itsGHTAngStep + 45;
          ghtAcc.scale = -1;
          ghtAcc.featureType = featureTemplate.featureType;
          ghtAcc.sum = acc[i].getVal(x,y); ///totalSum;
          accRet.push_back(ghtAcc);
        }
}

float V4d::voteForFeature(Image<float>& acc, int angIdx, std::vector<V1::EdgeState>& rTable)
{
  float sum = 0;

  for(uint ei = 0; ei < itsEdgesState.size(); ei++)
  {
    V1::EdgeState& edgeState = itsEdgesState[ei];

    if (edgeState.prob > 0) // && harris.getVal(edgeState.pos) > 0.1)
    {
      for(uint j=0; j<rTable.size(); j++)
      {
        //Rotate
        float ang = (angIdx*itsGHTAngStep)*M_PI/180;

        float diffRot = edgeState.ori - (rTable[j].ori - ang);
        diffRot = atan(sin(diffRot)/cos(diffRot)); //wrap to 180 deg to 0

        float stddevRot = 1;
        int sizeRot = int(ceil(stddevRot * sqrt(-2.0F * log(exp(-5.0F)))));

        if (fabs(diffRot) < sizeRot*M_PI/180) //TODO change to a for loop with hash
        {
          float rRot = exp(-((diffRot*diffRot)/(stddevRot*stddevRot)));

          // drawLine(acc[0], edgeState.pos, Point2D<int>(a0, b0), 1.0F);
          //acc[0].setVal(a0, b0, 0.5);

          float stddevX = 0.1;
          float stddevY = 0.1;
          int sizeX = int(ceil(stddevX * sqrt(-2.0F * log(exp(-5.0F)))));
          int sizeY = int(ceil(stddevY * sqrt(-2.0F * log(exp(-5.0F)))));

          //Apply a variance over position
          for(int y=edgeState.pos.j-sizeY; y<edgeState.pos.j+sizeY; y++)
          {
            float diffY = y-edgeState.pos.j;
            float ry = exp(-((diffY*diffY)/(stddevY*stddevY)));
            for(int x=edgeState.pos.i-sizeX; x<edgeState.pos.i+sizeX; x++)
            {
              float diffX = x-edgeState.pos.i;
              float rx = exp(-((diffX*diffX)/(stddevX*stddevX)));
              float weight = rRot*rx*ry;

              int a0 = x - int((float)rTable[j].pos.i*cos(ang) - (float)rTable[j].pos.j*sin(ang));
              int b0 = y - int((float)rTable[j].pos.i*sin(ang) + (float)rTable[j].pos.j*cos(ang));

              if (acc.coordsOk(a0, b0))
              {
                sum += edgeState.prob + weight;
                float val = acc.getVal(a0, b0);
                val += weight*edgeState.prob; //Add the strength of the feature
                acc.setVal(a0, b0, val);
              }
            }
          }
        }
      }
    }
  }

  return sum;
}

Image<PixRGB<byte> > V4d::showParticles(const std::vector<NAFState>& particles)
{
  //LINFO("Show part %lu %lu", itsBias.size(), particles.size());
  Image<float> probImg(320,240, ZEROS);
  Image<PixRGB<byte> > particlesImg(probImg.getDims(),ZEROS);
  for(uint i=0; i<particles.size(); i++)
  {
    Point2D<int> loc = (Point2D<int>)itsCamera.project(particles[i].pos);
    if (particlesImg.coordsOk(loc) &&
        particles[i].weight > probImg.getVal(loc) )
    {
      PixRGB<byte> col;
      if (particles[i].featureType == RIGHT_VERTIX)
        col = PixRGB<byte>(0, 255, 0);
      else if (particles[i].featureType == VERTIX)
        col = PixRGB<byte>(255, 0, 0);
      else
        col = PixRGB<byte>(0, 0, 255);

      probImg.setVal(loc, particles[i].weight);

      particlesImg.setVal(loc, col);
    }
  }
  inplaceNormalize(probImg, 0.0F, 255.0F);

  //particlesImg *= probImg;
  //for(uint i=0; i<probImg.size(); i++)
  //{
  //  int pColor = (int)probImg.getVal(i);
  //  particlesImg.setVal(i, PixRGB<byte>(0,pColor,0));
  //}

  //Image<float> biasProbImg(320,240, ZEROS);
  //float h = 2; //smoothing paramter
  //for(int y=0; y<240; y++)
  //  for(int x=0; x<320; x++)
  //  {
  //    Point2D<int> sampleLoc(x,y);

  //    //Particles
  //    float sum = 0;
  //    for(uint i=0; i<particles.size(); i++)
  //    {
  //      Point2D<int> loc = (Point2D<int>)itsCamera.project(particles[i].pos);
  //      sum += (1.0F/sqrt(2*M_PI)*exp(-0.5*squareOf(sampleLoc.distance(loc)/h)))*particles[i].weight;
  //    }
  //    float val = 1.0/(float(particles.size())*h) * sum;

  //    //Bias
  //    float biasSum = 0;
  //    for(uint i=0; i<itsBias.size(); i++)
  //    {
  //      if (itsBias[i].weight > 0)
  //      {
  //        Point2D<int> loc = (Point2D<int>)itsCamera.project(itsBias[i].pos);
  //        biasSum += (1.0F/sqrt(2*M_PI)*exp(-0.5*squareOf(sampleLoc.distance(loc)/h))); //*itsBias[i].weight;
  //      }
  //    }
  //    float biasVal = 1.0/(float(itsBias.size())*h) * biasSum;

  //    probImg.setVal(sampleLoc, biasVal+val);
  //  }

  inplaceNormalize(probImg, 0.0F, 255.0F);
  particlesImg = toRGB(probImg);

  //LINFO("Done part");

  return particlesImg;

}

void V4d::proposeParticles(std::vector<NAFState>& particles, const double Neff)
{

  LINFO("Propose Particles");
  //SEt the veriance to the number of effective particles
  //Basicly we always want all the particles to cover the space with
  //some probability
  //double posVar = 10*Neff/geonParticles.size();
//  LINFO("NEff %0.2f %lu",
//      Neff, geonParticles.size());
//
  float probThresh = 0.5; //1.0e-10;

  //If we have good hypothisis then just adjest them by a small amount
  uint particlesAboveProb = 0;
  for(uint i=0; i<particles.size(); i++)
  {
    if (particles[i].prob > probThresh)
    {
      particles[i].pos.x +=  randomDoubleFromNormal(1.0);
      particles[i].pos.y +=  randomDoubleFromNormal(1.0);
      particles[i].pos.z =  itsObjectsDist + randomDoubleFromNormal(0.05);
      particles[i].rot   +=  randomDoubleFromNormal(1)*M_PI/180;
      particles[i].weight = 1.0/(float)particles.size();
      particlesAboveProb++;
    }
  }


  //LINFO("Particles Above prob %i/%lu",
  //    particlesAboveProb, particles.size());


  if (particlesAboveProb < particles.size())
  {
    LINFO("GHT ");
    std::vector<GHTAcc> acc;
    //GHT(acc, itsNAFeatures[RIGHT_VERTIX]);
    //GHT(acc, itsNAFeatures[VERTIX]);
    //GHT(acc, itsNAFeatures[ARC]);


    Image<float> tmp(320,240,ZEROS);
    for(uint i=0; i<itsEdgesState.size(); i++)
      tmp.setVal(itsEdgesState[i].pos, itsEdgesState[i].prob);
    inplaceNormalize(tmp, 0.0F, 255.0F);


    Dims win(15,15);
    float hist[360];
    for(int i=0; i<360; i++)
      hist[i] = 0;
    while(!itsCornersState.empty())
    {
      V2::CornerState cornerState = itsCornersState.top(); itsCornersState.pop();
      tmp.setVal((Point2D<int>)cornerState.pos, cornerState.prob);
      drawCircle(tmp, (Point2D<int>)cornerState.pos, 6, (float)255.0);
      SHOWIMG(tmp);

      Image<float> edges(win,ZEROS);
      for(int wy=0; wy<win.h(); wy++)
        for(int wx=0; wx<win.w(); wx++)
        {
          int x = (int)cornerState.pos.i + wx - (win.w()/2);
          int y = (int)cornerState.pos.j + wy - (win.h()/2);

          int key = x + 320 * y;
          if (key > 0)
          {
            dense_hash_map<int, V1::EdgeState>::const_iterator iter = itsHashedEdgesState.find(key);
            if (iter != itsHashedEdgesState.end())
            {
              edges.setVal(wx,wy, iter->second.prob);
              int ang = (int)(iter->second.ori*180.0/M_PI);
              if (ang < 0) ang += 360;
              if (ang > 360) ang -= 360;
              hist[ang] += iter->second.prob;
              //drawLine(edges, Point2D<int>(wx,wy), (float)iter->second.ori, (float)5, (float)255.0);
            }
          }
        }
      for(int i=1; i<359; i++)
        if (hist[i] > 1)
          if (hist[i] > hist[i-1] && hist[i] > hist[i+1])
            printf("%i %0.1f\n",i, hist[i]);
      printf("\n");

      SHOWIMG(edges);

    }



    LINFO("GHT Done ");
    //if (acc.size() == 0)
    //  return;

    CpuTimer timer;
    timer.reset();
    ////Normalize to values from 0 to 1
    normalizeAcc(acc);

    //Combine the bias with the proposal
    //if (itsBias.size() > 0)
    //  mergeBias(acc);



   // //Sample from the accumilator
   // Image<float> tmp(320,240,ZEROS);
   // for(uint i=0; i<acc.size(); i++)
   //   tmp.setVal(acc[i].pos, tmp.getVal(acc[i].pos) + acc[i].sum);
   // SHOWIMG(tmp);

    ////Sample from acc

    LINFO("Build priority queue");
    std::priority_queue <GHTAcc> pAcc;
    for(uint i=0; i<acc.size(); i++)
      pAcc.push(acc[i]);

    LINFO("Get particules");
    //tmp.clear();
    for(uint i=0; i<particles.size(); i++)
    {
      if (particles[i].prob <= probThresh)
      {
        //add this point to the list
        if (pAcc.empty())
          break;
        GHTAcc p = pAcc.top(); pAcc.pop();

        NAFState nafState;
        Point3D<float>  iPoint = itsCamera.inverseProjection(Point2D<float>(p.pos), itsObjectsDist);
        nafState.pos.x = iPoint.x; // + randomDoubleFromNormal(1);
        nafState.pos.y = iPoint.y; // + randomDoubleFromNormal(1);
        nafState.pos.z =  itsObjectsDist; // + randomDoubleFromNormal(0.05);
        nafState.rot = p.ang*M_PI/180; // ((p.ang + 180) + randomDoubleFromNormal(5))*M_PI/180;
        nafState.featureType = p.featureType;
        nafState.weight = 1.0/(float)particles.size();
        nafState.prob = 1.0e-50;

                                //check to see how lcose this particle is to the reset of the group

                                //int nearPart = 0;
                                //float partDist = particles[nearPart].distance(nafState);
                                //for(uint p=1; p<particles.size(); p++)
                                //{
                                //        float distance = particles[p].distance(nafState);
                                //        if (distance < partDist)
                                //        {
                                //                partDist = distance;
                                //                nearPart = p;
                                //        }
                                //}

                                //if (partDist > 3) //we dont have this particle in our list, addit
                                {
                                        particles[i].pos = nafState.pos;
                                        particles[i].rot = nafState.rot;
                                        particles[i].featureType = nafState.featureType;
                                        particles[i].weight = nafState.weight;
                                        particles[i].prob = nafState.prob;
                                }

                                //std::vector<Point2D<int> > outline = getImageOutline(particles[i]);
        //for(uint i=0; i<outline.size()-1; i++)
        //  drawLine(tmp, outline[i], outline[i+1], 1.0F, 1);
      }
    }
    timer.mark();
    LINFO("Get particles: Total time %0.2f sec", timer.real_secs());
    //SHOWIMG(tmp);
  }
  LINFO("Done proposeing");
}

void V4d::mergeBias(std::vector<GHTAcc>& acc)
{
  LINFO("Merge bias");
  CpuTimer timer;
  timer.reset();
  //find the nearset bias, if it close then increase the probability of that location
  //if not, then add this bias particle to the list to be concidered for sampleing


  ImageSet<float> tmpAcc(360*2, Dims(320,240), ZEROS);

  for(uint p=0; p<itsBias.size(); p++)
  {
    if (itsBias[p].prob > 0) //only look at worty particles
    {
      GHTAcc ghtAcc;
      Point2D<int> pos = (Point2D<int>)itsCamera.project(itsBias[p].pos);
      int ang = int(itsBias[p].rot*180/M_PI);

      int fType = 0;
      switch(itsBias[p].featureType)
      {
        case RIGHT_VERTIX: fType = 0; break;
        case VERTIX: fType = 1; break;
        case ARC: fType = 1; break;
      }
      if (tmpAcc[fType*360 + ang].coordsOk(pos))
              tmpAcc[fType*360 + ang].setVal(pos,itsBias[p].prob);
    }
  }

  //Image<float> tmp(320, 240, ZEROS);
  //for(uint j=0; j<tmpAcc.size(); j++)
  //{
  //  for(uint i=0; i<tmpAcc[j].size(); i++)
  //  {
  //    if (tmpAcc[j].getVal(i) > tmp.getVal(i))
  //      tmp.setVal(i, tmpAcc[j].getVal(i));
  //  }
  //}
  //SHOWIMG(tmp);


  for(uint i=0; i<acc.size(); i++)
  {
      Point2D<int> pos = acc[i].pos;
      int rot = acc[i].ang;
      float sum = acc[i].sum;

      int fType = 0;
      switch(acc[i].featureType)
      {
        case RIGHT_VERTIX: fType = 0; break;
        case VERTIX: fType = 1; break;
        case ARC: fType = 1; break;
      }

      if (rot > 360) rot -= 360;
      if (rot == 360) rot = 0;

                        if (tmpAcc[fType*360 + rot].coordsOk(pos))
                        {
                                float prior = tmpAcc[fType*360 + rot].getVal(pos);

                                float post = sum+prior;

                                tmpAcc[fType*360 + rot].setVal(pos, post);
                        }
  }

  //tmp.clear();
  //for(uint j=0; j<tmpAcc.size(); j++)
  //{
  //  for(uint i=0; i<tmpAcc[j].size(); i++)
  //  {
  //    if (tmpAcc[j].getVal(i) > tmp.getVal(i))
  //      tmp.setVal(i, tmpAcc[j].getVal(i));
  //  }
  //}
  //SHOWIMG(tmp);



  acc.clear();
  for(uint i=0; i<tmpAcc.size(); i++)
    for(int y=0; y<tmpAcc[i].getHeight(); y++)
      for(int x=0; x<tmpAcc[i].getWidth(); x++)
        if (tmpAcc[i].getVal(x,y) > 0)
        {
          GHTAcc ghtAcc;
          ghtAcc.pos = Point2D<int>(x,y);

          if (i > 360)
          {
            ghtAcc.featureType = VERTIX;
            ghtAcc.ang =(i - 360);
          }
          else
          {
            ghtAcc.featureType = RIGHT_VERTIX;
            ghtAcc.ang =  i;
          }
          ghtAcc.scale = -1;
          ghtAcc.sum = tmpAcc[i].getVal(x,y); ///totalSum;
          acc.push_back(ghtAcc);
        }

  normalizeAcc(acc);

  //std::vector<GHTAcc> tmpAcc;

  //for(uint p=0; p<itsBias.size(); p++)
  //{
  //  if (itsBias[p].prob > 0) //only look at worty particles
  //  {
  //    findAngMerge(acc, itsBias[p], tmpAcc);
  //  }
  //}
  ////merge the temp acc and the acc
  //for(uint i=0; i<tmpAcc.size(); i++)
  //  acc.push_back(tmpAcc[i]);
  timer.mark();
  LINFO("Merge bias: Total time %0.2f sec", timer.real_secs());

}

void V4d::findAngMerge(std::vector<GHTAcc>& acc, const NAFState& biasNafState, std::vector<GHTAcc> tmpAcc)
{
  //find for the closest value in the acc
  int nearPart = 0;
  float partDist = 1.0e100;
  for(uint i=0; i<acc.size(); i++)
  {
    NAFState nafState;
    nafState.pos = itsCamera.inverseProjection(Point2D<float>(acc[i].pos), itsObjectsDist);
    nafState.rot = acc[i].ang*M_PI/180;
    nafState.featureType = acc[i].featureType;

    float distance = nafState.distance(biasNafState);
    if (distance < partDist)
    {
      partDist = distance;
      nearPart = i;
    }
  }

  //If we are close, then increase this acc probability, we we will sample it more
  if (partDist < 10)
    acc[nearPart].sum += (0.1 + biasNafState.prob);
  else
  {
    GHTAcc ghtAcc;
    ghtAcc.pos = (Point2D<int>)itsCamera.project(biasNafState.pos);
    ghtAcc.ang = int(biasNafState.rot*180/M_PI);
    ghtAcc.featureType = biasNafState.featureType;
    ghtAcc.scale = -1;
    ghtAcc.sum = 0.1 + biasNafState.prob;
    //LINFO("Add to acc with sum %f (prob %f)", ghtAcc.sum, itsBias[p].prob);
    tmpAcc.push_back(ghtAcc); //Add to temp, so that we dont search for it int the next iteration
  }

}

void V4d::harrisDetector(std::vector<V4d::GHTAcc>& accRet, const FeatureTemplate& featureTemplate)
{
  Image<float> tmp(320,240,ZEROS);
  dense_hash_map<int, V1::EdgeState>::const_iterator iter;
  for(iter = itsHashedEdgesState.begin(); iter != itsHashedEdgesState.end(); iter++)
    {
      V1::EdgeState edgeState = iter->second;
      tmp.setVal(edgeState.pos, edgeState.prob);
    }

    IplImage* img = img2ipl(tmp);
    IplImage* hImg = cvCreateImage( cvSize(img->width, img->height), IPL_DEPTH_32F, 1 );

    cvCornerHarris(img, hImg, 5, 5, 0);
    SHOWIMG(ipl2float(hImg));

    Image<float> acc = ipl2float(hImg);

    inplaceNormalize(acc, 0.0F, 1.0F);
    for(int y=0; y<acc.getHeight(); y++)
      for(int x=0; x<acc.getWidth(); x++)
        if (acc.getVal(x,y) > 0)
        {
          GHTAcc ghtAcc;
          ghtAcc.pos = Point2D<int>(x,y);
          ghtAcc.ang = 0; //i*itsGHTAngStep;
          //ghtAcc.ang = i*itsGHTAngStep;
          //if (featureTemplate.featureType == VERTIX)
          //  ghtAcc.ang = i*itsGHTAngStep + 45;
          ghtAcc.scale = -1;
          ghtAcc.featureType = featureTemplate.featureType;
          ghtAcc.sum = acc.getVal(x,y);
          accRet.push_back(ghtAcc);
        }

}
void V4d::proposeParticlesHarris(std::vector<NAFState>& particles, const double Neff)
{

  LINFO("Propose Particles");
  float probThresh = 0.5; //1.0e-10;

  //If we have good hypothisis then just adjest them
  uint particlesAboveProb = 0;
  for(uint i=0; i<particles.size(); i++)
  {
    if (particles[i].prob > probThresh)
    {
      particles[i].pos.x +=  randomDoubleFromNormal(1.0);
      particles[i].pos.y +=  randomDoubleFromNormal(1.0);
      particles[i].pos.z =  itsObjectsDist + randomDoubleFromNormal(0.05);
      particles[i].rot   +=  randomDoubleFromNormal(1)*M_PI/180;
      particles[i].weight = 1.0/(float)particles.size();
      particlesAboveProb++;
    }
  }

  //LINFO("Particles Above prob %i/%lu",
  //    particlesAboveProb, particles.size());

  if (particlesAboveProb < particles.size())
  {
    std::vector<GHTAcc> acc;
    harrisDetector(acc, itsNAFeatures[RIGHT_VERTIX]);

    for(uint i=0; i<particles.size(); i++)
    {
      if (particles[i].prob <= probThresh)
      {
        //find max;
        float maxVal = acc[0].sum; int maxIdx = 0;
        for(uint j=0; j<acc.size(); j++)
        {
          if (acc[j].sum > maxVal)
          {
            maxVal = acc[j].sum;
            maxIdx = j;
          }
        }

        //add this point to the list
        GHTAcc p = acc.at(maxIdx);
        //  LINFO("Max at %i,%i  %i %i", p.pos.i, p.pos.j, p.ang, p.featureType);
        Point3D<float>  iPoint = itsCamera.inverseProjection(Point2D<float>(p.pos), itsObjectsDist);
        particles[i].pos.x = iPoint.x; // + randomDoubleFromNormal(1);
        particles[i].pos.y = iPoint.y; // + randomDoubleFromNormal(1);
        particles[i].pos.z =  itsObjectsDist; // + randomDoubleFromNormal(0.05);
        particles[i].rot = ((p.ang + 180) + randomDoubleFromNormal(5))*M_PI/180;
        particles[i].featureType = p.featureType;
        particles[i].weight = 1.0/(float)particles.size();
        particles[i].prob = 1.0e-50;
        acc[maxIdx].sum = 0;
      }
    }
  }
}

void V4d::normalizeAcc(std::vector<GHTAcc>& acc)
{

  double sum = 0;
  for(uint i=0; i<acc.size(); i++)
    sum += acc[i].sum;

  for(uint i=0; i<acc.size(); i++)
    acc[i].sum /= sum;
}

void V4d::resampleParticles2(std::vector<NAFState>& particles)
{
  std::vector<NAFState> newParticles;

  while(newParticles.size() < particles.size())
  {
    //Find max
    int maxP = 0; double maxVal = particles[maxP].weight;
    for(uint j=1; j<particles.size(); j++)
    {
      if (particles[j].weight > maxVal)
      {
        maxP = j;
        maxVal = particles[j].weight;
      }
    }

    if (maxVal > 0)
    {
      for(int np=0; np<10; np++)
      {
        //Add the particle to the list
        NAFState p = particles[maxP];
        p.weight     = 1.0/double(particles.size());
        newParticles.push_back(p);
      }

      //IOR
      float sigmaX = 0.1, sigmaY = 0.1;
      float fac = 0.5f / (M_PI * sigmaX * sigmaY);
      float vx = -0.5f / (sigmaX * sigmaX);
      float vy = -0.5f / (sigmaY * sigmaY);

      for(uint j=0; j<particles.size(); j++)
      {
        float x = particles[j].pos.x;
        float y = particles[j].pos.y;

        float vydy2 = y - particles[maxP].pos.y; vydy2 *= vydy2 * vy;
        float dx2 = x - particles[maxP].pos.x; dx2 *= dx2;
        particles[j].weight -= particles[j].weight * fac*expf(vx * dx2 + vydy2);
        if (particles[j].weight  < 0)
          particles[j].weight = 0;

      }
      particles[maxP].weight = 0;
    } else {
        NAFState p = particles[0];
        p.weight     = 1.0/double(particles.size());
        newParticles.push_back(p);
    }

  }

  particles = newParticles;

}


void V4d::resampleParticles(std::vector<NAFState>& particles)
{

  LINFO("Resample particles");
  std::vector<NAFState> newParticles;

  if(particles.size() == 0)
    return;

  //Calculate a Cumulative Distribution Function for our particle weights
  uint i = 0;
  double c = particles.at(0).weight;
  double U = randomDouble()* 1.0/double(particles.size());

  for(uint j=0; j < particles.size(); j++)
  {

    while(U > c)
    {
      i++;
      c += particles.at(i).weight;
    }

    NAFState p = particles.at(i);
    p.weight     = 1.0/double(particles.size());
    newParticles.push_back(p);

    U += 1.0/double(particles.size());
  }

  particles = newParticles;

}

std::vector<Point2D<int> > V4d::getImageOutline(NAFState& nafState)
{

  FeatureTemplate featureCameraOutline = itsNAFeatures[nafState.featureType];

  //Transofrm the object relative to the camera
  for(uint i=0; i<featureCameraOutline.outline.size(); i++)
  {
    float x = featureCameraOutline.outline[i].x;
    float y = featureCameraOutline.outline[i].y;
    float z = featureCameraOutline.outline[i].z;
    featureCameraOutline.outline[i].x = (cos(nafState.rot)*x - sin(nafState.rot)*y) + nafState.pos.x;
    featureCameraOutline.outline[i].y = (sin(nafState.rot)*x + cos(nafState.rot)*y) + nafState.pos.y;
    featureCameraOutline.outline[i].z = z + nafState.pos.z;
  }

  //Project the object to camera cordinats
  std::vector<Point2D<int> > outline;
  for(uint i=0; i<featureCameraOutline.outline.size(); i++)
  {
    Point2D<float> loc = itsCamera.project(featureCameraOutline.outline[i]);
    outline.push_back(Point2D<int>(loc));
  }

  return outline;

}
/*
void V4d::getParticleLikelihood2(NAFState& particle)
{

  while(particle.rot > M_PI*2)
    particle.rot =- M_PI*2;

  while(particle.rot < 0)
    particle.rot =+ M_PI*2;

  //Transofrm the object position relative to the camera
  std::vector<Point2D<int> > outline = getImageOutline(particle);

  double totalProb = 1; //.0e-5;
  for(uint i=0; i<outline.size()-1; i++)
  {
    Point2D<int> pLoc1 = outline[i];
    Point2D<int> pLoc2 = outline[i+1];
    double prob = getLineProbability(pLoc1, pLoc2);
    totalProb *= prob;
  }
  particle.prob = totalProb;

  //Apply bias by finding the closest particle, and if it is within some threahold, increase
  //the probability by a factor of this amount.

  if (itsBias.size() > 0)
  {
    int nearPart = 0;
    float partDist = itsBias[nearPart].distance(particle);

    for(uint p=1; p<itsBias.size(); p++)
    {
      float distance = itsBias[p].distance(particle);
      if (distance < partDist)
      {
        partDist = distance;
        nearPart = p;
      }
    }

    //If we are close, then increase this particle probability
    if (partDist < 10)
      particle.prob += (0.1 + itsBias[nearPart].prob);
    if (particle.prob > 1)
      particle.prob = 1;
  }
}
*/

void V4d::getParticleLikelihood(NAFState& particle)
{
  while(particle.rot > M_PI*2)
    particle.rot =- M_PI*2;

  while(particle.rot < 0)
    particle.rot =+ M_PI*2;

  FeatureTemplate featureTemplate = itsNAFeatures[particle.featureType];


  Image<float> tmp(320,240,ZEROS);
  double totalProb = 1;
  //Get the position of the feature in the image
  Point2D<float> featureLoc = itsCamera.project(particle.pos);
  float featureRot = particle.rot;

        //LINFO("Feature rot %f", featureRot*180/M_PI);
  if (particle.featureType == VERTIX)
    featureRot -= M_PI/4;
        int badValues = 0;
  for(uint i=0; i<featureTemplate.edges.size(); i++)
  {
    //Map the feature to position in the image
    V1::EdgeState featureEdgeState = featureTemplate.edges[i];

    int x = featureEdgeState.pos.i;
    int y = featureEdgeState.pos.j;


    featureEdgeState.pos.i = (int)(featureLoc.i + (float)x*cos(featureRot) - (float)y*sin(featureRot));
    featureEdgeState.pos.j = (int)(featureLoc.j + (float)x*sin(featureRot) + (float)y*cos(featureRot));

    featureEdgeState.ori = featureEdgeState.ori - featureRot + M_PI/2;

    int nearEdge = 0;
    //float edgeDist = itsEdgesState[nearEdge].distance(featureEdgeState);

    double edgeDist = 1.0e100;
    for(uint i = 0; i < itsEdgesState.size(); i++)
    {
            if (itsEdgesState[i].prob > 0)
                        {
                                //double dPoint = itsEdgesState[i].pos.squdist(featureEdgeState.pos);
                                float distance = 1.0e100;
                                if (itsEdgesState[i].pos == featureEdgeState.pos)
                                //if (itsEdgesState[i].distance(featureEdgeState) < 2)
                                {
                                        double dRot = itsEdgesState[i].ori + M_PI/2 - featureEdgeState.ori;
                                        dRot = atan(sin(dRot)/cos(dRot))*180/M_PI; //wrap to 180 deg to 0
                                        distance = dRot;
                                }

                                //float distance = sqrt(dRot*dRot);
                                //if (dPoint > 0)
                                //  distance = 0;

                                //float distance = sqrt(dPoint + (dRot*dRot));

                                if (distance < edgeDist)
                                {
                                        nearEdge = i;
                                        edgeDist = distance;
                                }
                        }
                }

                if (edgeDist == 1.0e100)
                        badValues++;
                else
                        if (badValues < 3)
                        {

                                float sig = 10.75F;
                                float prob  = (1.0F/(sig*sqrt(2*M_PI))*exp(-0.5*squareOf(edgeDist)/squareOf(sig)));
                                totalProb  *= prob;
                        } else {
                                totalProb = 0;
                        }

                //LINFO("dist %f prob %e", edgeDist, prob);
    //LINFO("Edges %ix%i %f",
    //    featureEdgeState.pos.i, featureEdgeState.pos.j, featureEdgeState.ori*180/M_PI + 360);
    //LINFO("Neare %ix%i %f",
    //    itsEdgesState[nearEdge].pos.i - (int)featureLoc.i,
                //                 itsEdgesState[nearEdge].pos.j - (int)featureLoc.j,
                //                 itsEdgesState[nearEdge].ori);
    ////    edgeDist);
    ////drawLine(tmp, itsEdgesState[nearEdge].pos, itsEdgesState[nearEdge].ori + M_PI/2, 5.0F, itsEdgesState[nearEdge].prob);
    //drawLine(tmp, featureEdgeState.pos, featureEdgeState.ori, 5.0F, 1.0F);
    //tmp.setVal(featureEdgeState.pos, 1.0F);
  }

        //totalProb = 1;
  particle.prob = totalProb;
        //LINFO("Total prob %e", totalProb);

  //inplaceNormalize(tmp, 0.0F, 255.0F);
  //itsDebugImg = toRGB(tmp);

  //Apply bias by finding the closest particle, and if it is within some threahold, increase
  //the probability by a factor of this amount.

  if (itsBias.size() > 0)
  {
    int nearPart = 0;
    float partDist = itsBias[nearPart].distance(particle);

    for(uint p=1; p<itsBias.size(); p++)
    {
      float distance = itsBias[p].distance(particle);
      if (distance < partDist)
      {
        partDist = distance;
        nearPart = p;
      }
    }

    //If we are close, then increase this particle probability
    if (partDist < 10)
      particle.prob += (0.1 + itsBias[nearPart].prob);
    if (particle.prob > 1)
      particle.prob = 1;
  }
}


double V4d::getLineProbability(const Point2D<int>& p1, const Point2D<int>& p2)
{

  int dx = p2.i - p1.i, ax = abs(dx) << 1, sx = signOf(dx);
  int dy = p2.j - p1.j, ay = abs(dy) << 1, sy = signOf(dy);
  int x = p1.i, y = p1.j;

  const int w = 320;

  double prob = 1; //1.0e-5;

  int wSize = 1;
  if (ax > ay)
  {
    int d = ay - (ax >> 1);
    for (;;)
    {
      //search for a max edge prob in a window
      float maxProb = 0; //1.0e-20;
      for(int yy = y-wSize; yy < y+wSize; yy++)
        for(int xx = x-wSize; xx < x+wSize; xx++)
        {
          int key = xx + w * yy;
          if (key > 0)
          {
            dense_hash_map<int, V1::EdgeState>::const_iterator iter = itsHashedEdgesState.find(key);
            if (iter != itsHashedEdgesState.end())
            {
              if (iter->second.prob > maxProb)
                maxProb = iter->second.prob;
            }
          }
        }
      prob *= maxProb;

      if (x == p2.i) return prob;
      if (d >= 0) { y += sy; d -= ax; }
      x += sx; d += ay;
    }
  } else {
    int d = ax - (ay >> 1);
    for (;;)
    {
      float maxProb = 0; //1.0e-20;
      for(int yy = y-wSize; yy < y+wSize; yy++)
        for(int xx = x-wSize; xx < x+wSize; xx++)
        {
          int key = xx + w * yy;
          if (key > 0)
          {
            dense_hash_map<int, V1::EdgeState>::const_iterator iter = itsHashedEdgesState.find(key);
            if (iter != itsHashedEdgesState.end())
            {
              if (iter->second.prob > maxProb)
                maxProb = iter->second.prob;
            }
          }
        }
      prob *= maxProb;
      if (y == p2.j) return prob;

      if (d >= 0) { x += sx; d -= ay; }
      y += sy; d += ax;
    }
  }

  return prob;


}

Layout<PixRGB<byte> > V4d::getDebugImage()
{
  Layout<PixRGB<byte> > outDisp;

  //Show the gabor states
  Image<float> perc(320,240, ZEROS);

  //Draw the edges
  for(uint i=0; i<itsEdgesState.size(); i++)
    perc.setVal(itsEdgesState[i].pos, itsEdgesState[i].prob);

  inplaceNormalize(perc, 0.0F, 255.0F);

  Image<PixRGB<byte> > NAFDisp = toRGB(Image<byte>(perc));

  //Get min max
  float minVal = itsFeaturesParticles[0].prob;
  float maxVal = itsFeaturesParticles[0].prob;
  for(uint p=0; p<itsFeaturesParticles.size(); p++)
  {
    if (itsFeaturesParticles[p].prob < minVal)
      minVal = itsFeaturesParticles[p].prob;
    if (itsFeaturesParticles[p].prob > maxVal)
      maxVal = itsFeaturesParticles[p].prob;
  }

  float scale = maxVal - minVal;
  float nScale = (255.0F - 0.0F)/scale;
  for(uint p=0; p<itsFeaturesParticles.size(); p++)
  {
    //set the color to the probability
    float normProb = 0.0F + ((itsFeaturesParticles[p].prob - minVal) * nScale);
    //if (normProb > 0) //itsFeaturesParticles[p].prob > 0)
    if (itsFeaturesParticles[p].prob > 0)
    {
      normProb = 255.0;
      PixRGB<byte> col;
      if (itsFeaturesParticles[p].featureType == RIGHT_VERTIX)
        col = PixRGB<byte>(0,int(normProb),0);
      else if (itsFeaturesParticles[p].featureType == VERTIX)
        col = PixRGB<byte>(int(normProb),0, 0);
      else
        col = PixRGB<byte>(0,0,int(normProb));

      std::vector<Point2D<int> > outline = getImageOutline(itsFeaturesParticles[p]);
      for(uint i=0; i<outline.size()-1; i++)
        drawLine(NAFDisp, outline[i], outline[i+1], col, 1);
    }
  }

  //draw bias

//  for(uint p=0; p<itsBias.size(); p++)
//  {
//      PixRGB<byte> col;
//      if (itsFeaturesParticles[p].featureType == RIGHT_VERTIX)
//        col = PixRGB<byte>(0,255,255);
//      else
//        col = PixRGB<byte>(255,0, 255);
//
//      std::vector<Point2D<int> > outline = getImageOutline(itsBias[p]);
//      for(uint i=0; i<outline.size()-1; i++)
//        drawLine(NAFDisp, outline[i], outline[i+1], col, 1);
//  }
  //Show the normalized particles
  Image<PixRGB<byte> > particlesImg = showParticles(itsFeaturesParticles);

  outDisp = hcat(NAFDisp, particlesImg);

  return outDisp;

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

