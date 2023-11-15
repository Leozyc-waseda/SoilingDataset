/*!@file SceneUnderstanding/V4.C  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/V4.C $
// $Id: V4.C 13551 2010-06-10 21:56:32Z itti $
//

#ifndef V4_C_DEFINED
#define V4_C_DEFINED

#include "plugins/SceneUnderstanding/V4.H"

#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Convolutions.H"
#include "Image/fancynorm.H"
#include "Image/Point3D.H"
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
#include <queue>

const ModelOptionCateg MOC_V4 = {
  MOC_SORTPRI_3,   "V4-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_V4ShowDebug =
{ MODOPT_ARG(bool), "V4ShowDebug", &MOC_V4, OPTEXP_CORE,
  "Show debug img",
  "v4-debug", '\0', "<true|false>", "false" };

namespace
{
  class GHTJob : public JobWithSemaphore
  {
    public:

      GHTJob(V4* v4, Image<float>& acc, int angIdx, std::vector<V4::RTableEntry>& rTable) :
        itsV4(v4),
        itsAcc(acc),
        itsAngIdx(angIdx),
        itsRTable(rTable)
    {}

      virtual ~GHTJob() {}

      virtual void run()
      {
        itsV4->voteForFeature(itsAcc, itsAngIdx, itsRTable);

        this->markFinished();
      }

      virtual const char* jobType() const { return "GHTJob"; }
      V4* itsV4;
      Image<float>& itsAcc;
      int itsAngIdx;
      std::vector<V4::RTableEntry>& itsRTable;
  };
}


// ######################################################################
V4::V4(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventV2Output),
  SIMCALLBACK_INIT(SimEventV4dOutput),
  SIMCALLBACK_INIT(SimEventUserInput),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  itsShowDebug(&OPT_V4ShowDebug, this),
  itsMaxVal(0),
  itsGHTAngStep(5),
  itsObjectsDist(370.00)
{
  itsThreadServer.reset(new WorkThreadServer("V4", 100));


  itsDebugImg = Image<PixRGB<byte> >(302,240,ZEROS);

  //Camera parameters
  itsCamera = Camera(Point3D<float>(0,0,0.0),
      Point3D<float>(0, 0,0),
      450, //Focal length
      320, //width
      240); //height

  ///Geons
  itsGeons.resize(3);

  GeonOutline triangle;
  triangle.geonType = TRIANGLE;
  triangle.outline.push_back(Point3D<float>(0.0,0, 0.0));
  triangle.outline.push_back(Point3D<float>(0.0,-40.0, 0.0));
  triangle.outline.push_back(Point3D<float>(40.0, 0.0, 0.0));

  triangle.NAFTemplate.push_back(
      V4d::NAFState(Point3D<float>(0,0,0), 135*M_PI/180, V4d::VERTIX,
        Point3D<float>(10,10,1), 1*M_PI/180));
  triangle.NAFTemplate.push_back(
      V4d::NAFState(Point3D<float>(0,40.0,0), -90*M_PI/180, V4d::RIGHT_VERTIX,
        Point3D<float>(10,10,1), 1*M_PI/180));
  triangle.NAFTemplate.push_back(
      V4d::NAFState(Point3D<float>(40.0,40.0,0), -90*M_PI/180, V4d::VERTIX,
        Point3D<float>(10,10,1), 1*M_PI/180));


  alignToCenterOfMass(triangle);
  itsGeons[TRIANGLE] = triangle;


  //Add a square Geon
  GeonOutline square;
  square.geonType = SQUARE;
  square.outline.push_back(Point3D<float>(0.0,0, 0.0));
  square.outline.push_back(Point3D<float>(0.0, 30.0, 0.0));
  square.outline.push_back(Point3D<float>(30, 30.0, 0.0));
  square.outline.push_back(Point3D<float>(30,0.0, 0.0));

  square.NAFTemplate.push_back(
      V4d::NAFState(Point3D<float>(0,0,0), 0*M_PI/180, V4d::RIGHT_VERTIX,
        Point3D<float>(10,10,1), 1*M_PI/180));
  square.NAFTemplate.push_back(
      V4d::NAFState(Point3D<float>(0,30.0,0), -90*M_PI/180, V4d::RIGHT_VERTIX,
        Point3D<float>(10,10,1), 1*M_PI/180));
  square.NAFTemplate.push_back(
      V4d::NAFState(Point3D<float>(30.0,30.0,0), 180*M_PI/180, V4d::RIGHT_VERTIX,
        Point3D<float>(10,10,1), 1*M_PI/180));
  square.NAFTemplate.push_back(
      V4d::NAFState(Point3D<float>(30.0,0.0,0), 90*M_PI/180, V4d::RIGHT_VERTIX,
        Point3D<float>(10,10,1), 1*M_PI/180));

  alignToCenterOfMass(square);
  itsGeons[SQUARE] = square;

  //Add a circle Geon
  GeonOutline circle;
  circle.geonType = CIRCLE;
  circle.outline.push_back(Point3D<float>(0.0,0, 0.0));
  circle.outline.push_back(Point3D<float>(0.0, 30.0, 0.0));
  circle.outline.push_back(Point3D<float>(30, 30.0, 0.0));
  circle.outline.push_back(Point3D<float>(30,0.0, 0.0));

  circle.NAFTemplate.push_back(
      V4d::NAFState(Point3D<float>(-11.511111,-9.044445,0.000000), 5.829400, V4d::ARC,
        Point3D<float>(10,10,1), 1*M_PI/180));
  circle.NAFTemplate.push_back(
      V4d::NAFState(Point3D<float>(-12.333334,12.333332,0.000000), 4.572762, V4d::ARC,
        Point3D<float>(10,10,1), 1*M_PI/180));
  circle.NAFTemplate.push_back(
      V4d::NAFState(Point3D<float>(15.622223,14.799999,0.000000), 2.757620, V4d::ARC,
        Point3D<float>(10,10,1), 1*M_PI/180));
  circle.NAFTemplate.push_back(
      V4d::NAFState(Point3D<float>(13.155556,-12.333336,0.000000), 1.134464, V4d::ARC,
        Point3D<float>(10,10,1), 1*M_PI/180));

  alignToCenterOfMass(circle);
  itsGeons[CIRCLE] = circle;

  itsGeonsParticles.resize(1000);
  for(uint i=0; i<itsGeonsParticles.size(); i++)
  {
    itsGeonsParticles[i].pos = Point3D<float>(0,0,0);
    itsGeonsParticles[i].rot = 0;
    itsGeonsParticles[i].geonType = CIRCLE;
    itsGeonsParticles[i].weight = 1.0/double(itsGeonsParticles.size());
    itsGeonsParticles[i].prob = 0;
  }


  itsBestProb = 0; //Out bet probability so far
  itsHashedEdgesState.set_empty_key(-1);
  buildRTables();

  //  for(uint geonId=0; geonId<itsGeons.size(); geonId++)
  //    showGeon(itsGeons[geonId]);
  //
}

// ######################################################################
V4::~V4()
{

}

// ######################################################################
void V4::alignToCenterOfMass(GeonOutline& featureTemplate)
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

  if (featureTemplate.NAFTemplate.size() > 0)
  {
    centerX = 0; centerY = 0; centerZ = 0;
    for(uint i=0; i<featureTemplate.NAFTemplate.size(); i++)
    {
      centerX += featureTemplate.NAFTemplate[i].pos.x;
      centerY += featureTemplate.NAFTemplate[i].pos.y;
      centerZ += featureTemplate.NAFTemplate[i].pos.z;
    }
    centerX /= featureTemplate.NAFTemplate.size();
    centerY /= featureTemplate.NAFTemplate.size();
    centerZ /= featureTemplate.NAFTemplate.size();

    for(uint i=0; i<featureTemplate.NAFTemplate.size(); i++)
    {
      featureTemplate.NAFTemplate[i].pos -= Point3D<float>(centerX, centerY, centerZ);
      //change angle to 0 - 360;
      if (featureTemplate.NAFTemplate[i].rot < 0)
        featureTemplate.NAFTemplate[i].rot += M_PI*2;
    }
  }

}

// ######################################################################
void V4::buildRTables()
{

  //Position relative to the camera
  for(uint geonId=0; geonId<itsGeons.size(); geonId++)
  {
    if (itsGeons[geonId].outline.size() > 0)
    {
      Point3D<float> pos(0,0,itsObjectsDist);
      float rot = 0;

      for(uint fid = 0; fid < itsGeons[geonId].NAFTemplate.size(); fid++)
      {
        V4d::NAFState nafState = itsGeons[geonId].NAFTemplate[fid];

        nafState.rot += rot;
        float x = nafState.pos.x;
        float y = nafState.pos.y;
        float z = nafState.pos.z;
        nafState.pos.x = (cos(rot)*x - sin(rot)*y) + pos.x;
        nafState.pos.y = (sin(rot)*x + cos(rot)*y) + pos.y;
        nafState.pos.z = z + pos.z;

        Point2D<int> loc = (Point2D<int>)itsCamera.project(nafState.pos);

        RTableEntry rTableEntry;
        rTableEntry.featureType = nafState.featureType;
        rTableEntry.loc.i = loc.i - 320/2;
        rTableEntry.loc.j = loc.j - 240/2;
        rTableEntry.rot = nafState.rot;

        itsGeons[geonId].rTable.push_back(rTableEntry);
      }
      //showGeon(itsGeons[geonId], (GEON_TYPE)geonId);
    }
  }


}

// ######################################################################
void V4::showGeon(GeonOutline& geon)
{

  //Position relative to the camera
  Point3D<float> pos(0,0,itsObjectsDist);
  float rot = 0;

  Image<PixRGB<byte> > geonDisp(320, 240, ZEROS);

  //for(uint fid = 0; fid < geon.NAFTemplate.size(); fid++)
  //{
  //  V4d::NAFState nafState = geon.NAFTemplate[fid];

  //  nafState.rot += rot;
  //  float x = nafState.pos.x;
  //  float y = nafState.pos.y;
  //  float z = nafState.pos.z;
  //  nafState.pos.x = (cos(rot)*x - sin(rot)*y) + pos.x;
  //  nafState.pos.y = (sin(rot)*x + cos(rot)*y) + pos.y;
  //  nafState.pos.z = z + pos.z;

  //  std::vector<Point2D<int> > outline = itsV4dCells.getImageOutline(nafState);

  //  for(uint i=0; i<outline.size()-1; i++)
  //    drawLine(geonDisp, outline[i], outline[i+1], PixRGB<byte>(0, 255,0), 1);
  //}

  GeonState geonState;
  geonState.pos = pos;
  geonState.rot = rot;
  geonState.geonType  = geon.geonType;

  std::vector<Point2D<int> > outline = getImageGeonOutline(geonState);
  for(uint i=0; i<outline.size(); i++)
    drawLine(geonDisp, outline[i], outline[(i+1)%outline.size()], PixRGB<byte>(255, 0,0), 1);

  SHOWIMG(geonDisp);
}


// ######################################################################
void V4::init(Dims numCells)
{

}

// ######################################################################
std::vector<V4::GeonState> V4::getOutput()
{
  return itsGeonsParticles;
}


// ######################################################################
void V4::onSimEventV2Output(SimEventQueue& q,
    rutz::shared_ptr<SimEventV2Output>& e)
{
  std::vector<V1::EdgeState> edgesState; // = e->getEdges();

  itsHashedEdgesState.clear();
  for(uint i=0; i<edgesState.size(); i++)
  {
    int key = edgesState[i].pos.i + 320*edgesState[i].pos.j;
    itsHashedEdgesState[key] = edgesState[i];
  }

}

// ######################################################################
void V4::onSimEventV4dOutput(SimEventQueue& q,
    rutz::shared_ptr<SimEventV4dOutput>& e)
{
  itsNAFParticles = e->getCells();
  evolve();


  q.post(rutz::make_shared(new SimEventV4Output(this, itsGeonsParticles)));

  //Send out the bias
  std::vector<V4d::NAFState> bias = getBias();
  q.post(rutz::make_shared(new SimEventV4BiasOutput(this, bias)));
}

// ######################################################################
void V4::onSimEventUserInput(SimEventQueue& q, rutz::shared_ptr<SimEventUserInput>& e)
{

  LINFO("Got event %s %ix%i key=%i",
      e->getWinName(),
      e->getMouseClick().i,
      e->getMouseClick().j,
      e->getKey());

  if (e->getMouseClick().isValid())
  {
    Point3D<float>  iPoint = itsCamera.inverseProjection((Point2D<float>)e->getMouseClick(), itsObjectsDist);
    itsGeonsParticles[0].pos = iPoint;
  }

  itsGeonsParticles[0].geonType = CIRCLE;
  itsGeonsParticles[0].weight = 1.0/double(itsGeonsParticles.size());
  itsGeonsParticles[0].prob = 0;

}

// ######################################################################
void V4::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
  {
    // get the OFS to save to, assuming sinfo is of type
    // SimModuleSaveInfo (will throw a fatal exception otherwise):
    nub::ref<FrameOstream> ofs =
      dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
    Layout<PixRGB<byte> > disp = getDebugImage();
    ofs->writeRgbLayout(disp, "V4", FrameInfo("V4", SRC_POS));
    //      ofs->writeRGB(itsDebugImg, "V4Debug", FrameInfo("V4Debug", SRC_POS));

  }
}


// ######################################################################
void V4::setInput(const std::vector<V1::EdgeState> &edgesState)
{

  itsHashedEdgesState.clear();
  for(uint i=0; i<edgesState.size(); i++)
  {
    int key = edgesState[i].pos.i + 320*edgesState[i].pos.j;
    itsHashedEdgesState[key] = edgesState[i];
  }
}

// ######################################################################
void V4::setInput(const std::vector<V4d::NAFState> &nafStates)
{
  itsNAFParticles = nafStates;

  evolve();
}



// ######################################################################
void V4::evolve()
{

  ////Resample
  //resampleParticles2(itsGeonsParticles);
  proposeParticles(itsGeonsParticles, 0.0F);

  //////Evaluate the particles;
  evaluateGeonParticles(itsGeonsParticles);

}

// ######################################################################
std::vector<V4d::NAFState> V4::getBias()
{
  std::vector<V4d::NAFState> bias;
  double totalProb = 0;
  for(uint p=0; p<itsGeonsParticles.size(); p++)
  {

    if (itsGeonsParticles[p].prob > 0)
    {
      Point2D<int> loc = (Point2D<int>)itsCamera.project(itsGeonsParticles[p].pos);
      float rot = itsGeonsParticles[p].rot + M_PI;
      GeonOutline geonOutline = itsGeons[itsGeonsParticles[p].geonType];

      for(uint rIdx = 0; rIdx < geonOutline.rTable.size(); rIdx++)
      {
        float ang =  rot;
        float a0 = loc.i - (geonOutline.rTable[rIdx].loc.i*cos(ang) - geonOutline.rTable[rIdx].loc.j*sin(ang));
        float b0 = loc.j - (geonOutline.rTable[rIdx].loc.i*sin(ang) + geonOutline.rTable[rIdx].loc.j*cos(ang));

        V4d::NAFState nafState;
        nafState.pos = itsCamera.inverseProjection(Point2D<float>(a0,b0), itsObjectsDist);
        nafState.rot = rot + geonOutline.rTable[rIdx].rot + M_PI;
        nafState.featureType = geonOutline.rTable[rIdx].featureType;
        nafState.prob = itsGeonsParticles[p].prob;
        totalProb += nafState.prob;

        while (nafState.rot < 0)
          nafState.rot += M_PI*2;
        while(nafState.rot > M_PI*2)
          nafState.rot -= M_PI*2;

        bias.push_back(nafState);
      }
    }
  }

  for(uint i=0; i<bias.size(); i++)
    bias[i].prob /= totalProb;

  return bias;

}

// ######################################################################
float V4::evaluateGeonParticles(std::vector<GeonState>& geonParticles)
{

  LINFO("Evaluate particles");
  for(uint p=0; p<geonParticles.size(); p++)
    getGeonLikelihood(geonParticles[p]);
  //getOutlineLikelihood(geonParticles[p]);

  //Normalize the particles;
  double sum = 0;
  double Neff = 0; //Number of efictive particles
  for(uint i=0; i<geonParticles.size(); i++)
    sum += geonParticles[i].prob;

  for(uint i=0; i<geonParticles.size(); i++)
  {
    geonParticles[i].weight = geonParticles[i].prob/sum;
    Neff += squareOf(geonParticles[i].weight);
  }

  Neff = 1/Neff;

  return Neff;

}


void V4::GHT(std::vector<V4::GHTAcc>& accRet, GeonOutline &geonOutline)
{
  ImageSet<float> acc(360, Dims(320,240), ZEROS);

  CpuTimer timer;
  timer.reset();

  //Parallel votes
  std::vector<rutz::shared_ptr<GHTJob> > jobs;

  for(int angIdx = 0; angIdx < 360; angIdx++)
  {
    jobs.push_back(rutz::make_shared(new GHTJob(this, acc[angIdx], angIdx, geonOutline.rTable)));
    itsThreadServer->enqueueJob(jobs.back());
    //voteForFeature(acc[angIdx], angIdx, geonOutline.rTable);
  }
  ////wait for jobs to finish
  while(itsThreadServer->size() > 0)
    usleep(10000);

  timer.mark();
  LINFO("Total time %0.2f sec", timer.real_secs());


  //Image<float> tmp(320, 240, ZEROS);
  //for(uint angIdx=0; angIdx<acc.size(); angIdx++)
  //{
  //  for(uint i=0; i<acc[angIdx].size(); i++)
  //  {
  //    if (acc[angIdx].getVal(i) > tmp.getVal(i))
  //      tmp.setVal(i, acc[angIdx].getVal(i));
  //  }
  //}
  //SHOWIMG(tmp);


  for(uint rot=0; rot<acc.size(); rot++)
    for(int y=0; y<acc[rot].getHeight(); y++)
      for(int x=0; x<acc[rot].getWidth(); x++)
        if (acc[rot].getVal(x,y) > 0)
        {
          GHTAcc ghtAcc;
          ghtAcc.geonType = geonOutline.geonType;
          ghtAcc.pos = Point2D<int>(x,y);
          ghtAcc.ang = rot;
          ghtAcc.scale = -1;
          ghtAcc.sum = acc[rot].getVal(x,y);
          accRet.push_back(ghtAcc);
        }

}

void V4::voteForFeature(Image<float>& acc, int angIdx, std::vector<RTableEntry>& rTable)
{


  for(uint p=0; p < itsNAFParticles.size(); p++)
  {
    V4d::NAFState nafState = itsNAFParticles[p];
    if (nafState.prob > 0)
    {
      Point2D<int> loc = (Point2D<int>)itsCamera.project(nafState.pos);

      //      std::vector<Point2D<int> > outline = itsV4dCells.getImageOutline(nafState);
      //      for(uint i=0; i<outline.size()-1; i++)
      //        drawLine(tmp, outline[i], outline[i+1], 1.0F, 1);

      for(uint rIdx = 0; rIdx < rTable.size(); rIdx++)
      {
        if (rTable[rIdx].featureType == nafState.featureType)
        {
          //LINFO("Next feature");
          float ang = angIdx * M_PI/180;

          float featureAng = rTable[rIdx].rot + ang;
          if (featureAng < 0)
            featureAng += M_PI*2;
          if (featureAng > M_PI*2)
            featureAng -= M_PI*2;

          float diffRot = acos(cos(nafState.rot - featureAng));



          //float diffRot = nafState.rot - (rTable[rIdx].rot - ang);
          //diffRot = atan(sin(diffRot)/cos(diffRot)); //wrap to 180 deg to 0

          float stddevRot = 1.5;
          int sizeRot = int(ceil(stddevRot * sqrt(-2.0F * log(exp(-5.0F)))));

          if (fabs(diffRot) < sizeRot*M_PI/180) //TODO change to a for loop with hash
          {
            float rRot = exp(-((diffRot*diffRot)/(stddevRot*stddevRot)));

            //Apply a variance over position and angle
            //TODO change to a veriance in feature position, not its endpoint
            float stddevX = 1.5;
            float stddevY = 1.5;
            int sizeX = int(ceil(stddevX * sqrt(-2.0F * log(exp(-5.0F)))));
            int sizeY = int(ceil(stddevY * sqrt(-2.0F * log(exp(-5.0F)))));

            for(int y=loc.j-sizeY; y<loc.j+sizeY; y++)
            {
              float diffY = y-loc.j;
              float ry = exp(-((diffY*diffY)/(stddevY*stddevY)));
              for(int x=loc.i-sizeX; x<loc.i+sizeX; x++)
              {
                float diffX = x-loc.i;
                float rx = exp(-((diffX*diffX)/(stddevX*stddevX)));
                //float weight = nafState.prob + rRot*rx*ry;
                float weight = rRot*rx*ry;

                int a0 = x - int(rTable[rIdx].loc.i*cos(ang) - rTable[rIdx].loc.j*sin(ang));
                int b0 = y - int(rTable[rIdx].loc.i*sin(ang) + rTable[rIdx].loc.j*cos(ang));
                if (acc.coordsOk(a0, b0))
                {
                  float val = acc.getVal(a0, b0);
                  val += weight;
                  acc.setVal(a0, b0, val);
                }
              }
            }
          }
        }
      }
    }
  }
}

Image<PixRGB<byte> > V4::showParticles(const std::vector<GeonState>& particles)
{
  Image<float> probImg(320,240, ZEROS);
  Image<PixRGB<byte> > particlesImg(probImg.getDims(),ZEROS);
  for(uint i=0; i<particles.size(); i++)
  {
    Point2D<int> loc = (Point2D<int>)itsCamera.project(particles[i].pos);
    if (particlesImg.coordsOk(loc) &&
        particles[i].weight > probImg.getVal(loc) )
    {
      PixRGB<byte> col;
      if (particles[i].geonType == SQUARE)
        col = PixRGB<byte>(0, 255, 0);
      else
        col = PixRGB<byte>(255, 0, 0);
      probImg.setVal(loc, particles[i].weight);

      particlesImg.setVal(loc, col);
    }
  }
  inplaceNormalize(probImg, 0.0F, 255.0F);

  particlesImg *= probImg;

  return particlesImg;
}

void V4::proposeParticles(std::vector<GeonState>& particles, const double Neff)
{

  LINFO("Propose Particles");
  //SEt the veriance to the number of effective particles
  //Basicly we always want all the particles to cover the space with
  //some probability
  //double posVar = 10*Neff/geonParticles.size();
  //  LINFO("NEff %0.2f %lu",
  //      Neff, geonParticles.size());
  //
  float probThresh = 1.0e-10;

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
    LINFO("GHT ");
    std::vector<GHTAcc> acc;
    GHT(acc, itsGeons[TRIANGLE]);
    GHT(acc, itsGeons[SQUARE]);
    GHT(acc, itsGeons[CIRCLE]);
    LINFO("GHT Done ");

    if (acc.size() == 0)
      return;
    ////Normalize to values from 0 to 1
    normalizeAcc(acc);

    //Image<float> tmp(320,240,ZEROS);
    //for(uint i=0; i<acc.size(); i++)
    //{
    //  if (acc[i].sum > tmp.getVal(acc[i].pos))
    //    tmp.setVal(acc[i].pos, acc[i].sum);
    //}
    //SHOWIMG(tmp);

    std::priority_queue <GHTAcc> pAcc;
    for(uint i=0; i<acc.size(); i++)
      pAcc.push(acc[i]);

    ////Sample from acc
    for(uint i=0; i<particles.size(); i++)
    {
      if (particles[i].prob <= probThresh)
      {
        //add this point to the list
        if (pAcc.empty())
          break;
        GHTAcc p = pAcc.top(); pAcc.pop();

        Point3D<float>  iPoint = itsCamera.inverseProjection(Point2D<float>(p.pos), itsObjectsDist);
        particles[i].pos.x = iPoint.x + randomDoubleFromNormal(1);
        particles[i].pos.y = iPoint.y + randomDoubleFromNormal(1);
        particles[i].pos.z =  itsObjectsDist + randomDoubleFromNormal(0.05);
        particles[i].rot =  (p.ang + randomDoubleFromNormal(1))*M_PI/180;
        particles[i].geonType = p.geonType;
        particles[i].weight = 1.0/(float)particles.size();
        particles[i].prob = 1.0e-50;
      }
    }
  }
}

void V4::normalizeAcc(std::vector<GHTAcc>& acc)
{

  double sum = 0;
  for(uint i=0; i<acc.size(); i++)
    sum += acc[i].sum;

  for(uint i=0; i<acc.size(); i++)
    acc[i].sum /= sum;
}

void V4::resampleParticles2(std::vector<GeonState>& particles)
{
  std::vector<GeonState> newParticles;

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
        GeonState p = particles[maxP];
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
      GeonState p = particles[0];
      p.weight     = 1.0/double(particles.size());
      newParticles.push_back(p);
    }

  }

  particles = newParticles;

}

void V4::resampleParticles(std::vector<GeonState>& geonParticles)
{
  LINFO("Resample");

  std::vector<GeonState> newParticles;

  //Calculate a Cumulative Distribution Function for our particle weights
  std::vector<double> CDF;
  CDF.resize(geonParticles.size());

  CDF.at(0) = geonParticles.at(0).weight;
  for(uint i=1; i<CDF.size(); i++)
    CDF.at(i) = CDF.at(i-1) + geonParticles.at(i).weight;

  uint i = 0;
  double u = randomDouble()* 1.0/double(geonParticles.size());

  for(uint j=0; j < geonParticles.size(); j++)
  {
    while(u > CDF.at(i))
      i++;

    GeonState p = geonParticles.at(i);
    p.weight     = 1.0/double(geonParticles.size());
    newParticles.push_back(p);

    u += 1.0/double(geonParticles.size());
  }

  geonParticles = newParticles;

}

std::vector<Point2D<int> > V4::getImageGeonOutline(GeonState& geon)
{

  GeonOutline cameraOutline = itsGeons[geon.geonType];

  //Transofrm the object relative to the camera
  for(uint i=0; i<itsGeons[geon.geonType].outline.size(); i++)
  {
    float x = itsGeons[geon.geonType].outline[i].x;
    float y = itsGeons[geon.geonType].outline[i].y;
    float z = itsGeons[geon.geonType].outline[i].z;
    cameraOutline.outline[i].x = (cos(geon.rot)*x - sin(geon.rot)*y) + geon.pos.x;
    cameraOutline.outline[i].y = (sin(geon.rot)*x + cos(geon.rot)*y) + geon.pos.y;
    cameraOutline.outline[i].z = z + geon.pos.z;
  }

  //Project the object to camera cordinats
  std::vector<Point2D<int> > outline;
  for(uint i=0; i<cameraOutline.outline.size(); i++)
  {
    Point2D<float> loc = itsCamera.project(cameraOutline.outline[i]);
    outline.push_back(Point2D<int>(loc));
  }

  return outline;

}


void V4::getOutlineLikelihood(GeonState& geon)
{

  //Transofrm the object position relative to the camera
  std::vector<Point2D<int> > geonOutline = getImageGeonOutline(geon);

  double totalProb = 1; //.0e-5;
  for(uint i=0; i<geonOutline.size(); i++)
  {
    Point2D<int> pLoc1 = geonOutline[i];
    Point2D<int> pLoc2 = geonOutline[(i+1)%geonOutline.size()];
    double prob = getLineProbability(pLoc1, pLoc2);
    totalProb *= prob;
  }
  geon.prob = totalProb;
}


void V4::getGeonLikelihood(GeonState& geon)
{
  double totalProb = 1;
  Point2D<int> loc = (Point2D<int>)itsCamera.project(geon.pos);
  float rot = geon.rot + M_PI;
  GeonOutline geonOutline = itsGeons[geon.geonType];

  //Image<float> tmp(320,240,ZEROS);

  for(uint rIdx = 0; rIdx < geonOutline.rTable.size(); rIdx++)
  {
    float ang = rot;
    float a0 = loc.i - (geonOutline.rTable[rIdx].loc.i*cos(ang) - geonOutline.rTable[rIdx].loc.j*sin(ang));
    float b0 = loc.j - (geonOutline.rTable[rIdx].loc.i*sin(ang) + geonOutline.rTable[rIdx].loc.j*cos(ang));

    V4d::NAFState nafState;
    nafState.pos = itsCamera.inverseProjection(Point2D<float>(a0,b0), itsObjectsDist);
    nafState.rot = rot + geonOutline.rTable[rIdx].rot + M_PI;
    nafState.featureType = geonOutline.rTable[rIdx].featureType;
    nafState.prob = -1;

    while (nafState.rot < 0)
      nafState.rot += M_PI*2;
    while(nafState.rot > M_PI*2)
      nafState.rot -= M_PI*2;

    //Find the closest naf
    int nearPart = 0;
    float partDist = 1.0e100;
    for(uint i=0; i<itsNAFParticles.size(); i++)
    {
      if (itsNAFParticles[i].prob > 0)
      {
        float distance = nafState.distance(itsNAFParticles[i]);
        if (distance < partDist)
        {
          partDist = distance;
          nearPart = i;
        }
      }
    }

    float sig = 0.75F;

    //        float prob  = itsNAFParticles[nearPart].prob*(1.0F/(sig*sqrt(2*M_PI))*exp(-0.5*squareOf(partDist)/squareOf(sig)));
    float prob  = (1.0F/(sig*sqrt(2*M_PI))*exp(-0.5*squareOf(partDist)/squareOf(sig)));
    totalProb *= prob;

    //                LINFO("Near %f,%f,%f rot %f type %i",
    //                                 itsNAFParticles[nearPart].pos.x - geon.pos.x,
    //                                 itsNAFParticles[nearPart].pos.y - geon.pos.y,
    //                                 itsNAFParticles[nearPart].pos.z - geon.pos.z,
    //                                 itsNAFParticles[nearPart].rot,
    //                                 itsNAFParticles[nearPart].featureType);
    //                LINFO("Dist %f prob %e", partDist, prob);
    //
    //                if (tmp.coordsOk(a0,b0))
    //                        tmp.setVal(Point2D<int>(a0,b0), 1.0F);
    //
    //    Point2D<int> loc = (Point2D<int>)itsCamera.project(itsNAFParticles[nearPart].pos);
    //    drawLine(tmp, loc, itsNAFParticles[nearPart].rot, 5.0F, 1.0F);


  }
  //        LINFO("Total prob %e", totalProb);
  //
  //  inplaceNormalize(tmp, 0.0F, 255.0F);
  //  itsDebugImg = toRGB(tmp);

  geon.prob = totalProb;


}


double V4::getLineProbability(const Point2D<int>& p1, const Point2D<int>& p2)
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
      float maxProb = 1.0e-20;
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
      //if (maxProb == 0) return 0;
      prob *= maxProb;

      if (x == p2.i) return prob;
      if (d >= 0) { y += sy; d -= ax; }
      x += sx; d += ay;
    }
  } else {
    int d = ax - (ay >> 1);
    for (;;)
    {
      float maxProb = 1.0e-20;
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
      //if (maxProb == 0) return 0;
      if (y == p2.j) return prob;

      if (d >= 0) { x += sx; d -= ay; }
      y += sy; d += ax;
    }
  }

  return prob;


}

Layout<PixRGB<byte> > V4::getDebugImage()
{
  Layout<PixRGB<byte> > outDisp;

  //Show the gabor states
  Image<float> perc(320,240, ZEROS);

  //Draw the edges
  dense_hash_map<int, V1::EdgeState>::const_iterator iter;
  for(iter = itsHashedEdgesState.begin(); iter != itsHashedEdgesState.end(); iter++)
    perc.setVal(iter->second.pos, iter->second.prob);

  inplaceNormalize(perc, 0.0F, 255.0F);

  Image<PixRGB<byte> > geonDisp = toRGB(Image<byte>(perc));

  //inplace normalize
  //Get min max
  //float minVal = itsGeonsParticles[0].prob;
  //float maxVal = itsGeonsParticles[0].prob;
  //for(uint p=0; p<itsGeonsParticles.size(); p++)
  //{
  //  if (itsGeonsParticles[p].prob < minVal)
  //    minVal = itsGeonsParticles[p].prob;
  //  if (itsGeonsParticles[p].prob > maxVal)
  //    maxVal = itsGeonsParticles[p].prob;
  //}


  //float scale = maxVal - minVal;
  //float nScale = (255.0F - 0.0F)/scale;

  for(uint p=0; p<itsGeonsParticles.size(); p++)
  {
    //set the color to the probability
    //float normProb = 0.0F + ((itsGeonsParticles[p].prob - minVal) * nScale);
    //if (normProb > 0) //itsGeonsParticles[p].prob > 0)
    if (itsGeonsParticles[p].prob > 0)
    {
      float normProb = 255.0;
      PixRGB<byte> col;
      if (itsGeonsParticles[p].geonType == TRIANGLE)
        col = PixRGB<byte>(0,int(normProb),0);
      else if (itsGeonsParticles[p].geonType == SQUARE)
        col = PixRGB<byte>(int(normProb),0,0);
      else
        col = PixRGB<byte>(0,0,int(normProb));

      if (itsGeonsParticles[p].geonType == CIRCLE)
      {
        Point2D<float> loc = itsCamera.project(itsGeonsParticles[p].pos);
        drawCircle(geonDisp, Point2D<int>(loc), 20, col);
      } else {
        std::vector<Point2D<int> > outline = getImageGeonOutline(itsGeonsParticles[p]);
        for(uint i=0; i<outline.size(); i++)
          drawLine(geonDisp, outline[i], outline[(i+1)%outline.size()], col, 1);
      }
    }
  }


  //Show the normalized particles
  Image<PixRGB<byte> > particlesImg = showParticles(itsGeonsParticles);

  char msg[255];
  sprintf(msg, "V4");
  writeText(geonDisp, Point2D<int>(0,0), msg, PixRGB<byte>(255,255,255), PixRGB<byte>(0,0,0));

  outDisp = hcat(geonDisp, particlesImg);

  return outDisp;

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

