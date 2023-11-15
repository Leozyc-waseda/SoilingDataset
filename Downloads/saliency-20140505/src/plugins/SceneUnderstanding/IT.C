/*!@file SceneUnderstanding/IT.C  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/IT.C $
// $Id: IT.C 13765 2010-08-06 18:56:17Z lior $
//

#ifndef IT_C_DEFINED
#define IT_C_DEFINED

#include "Image/DrawOps.H"
#include "Image/MathOps.H"
//#include "Image/OpenCVUtil.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Convolutions.H"
#include "Image/fancynorm.H"
#include "Image/Point3D.H"
#include "plugins/SceneUnderstanding/IT.H"
#include "Neuro/EnvVisualCortex.H"
#include "GUI/DebugWin.H"
#include "Util/CpuTimer.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>
#include <queue>

const ModelOptionCateg MOC_IT = {
  MOC_SORTPRI_3,   "IT-Related Options" };

const ModelOptionDef OPT_ITShowDebug =
  { MODOPT_ARG(bool), "ITShowDebug", &MOC_IT, OPTEXP_CORE,
    "Show debug img",
    "it-debug", '\0', "<true|false>", "false" };

// ######################################################################
IT::IT(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventV4Output),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  itsShowDebug(&OPT_ITShowDebug, this),
  itsObjectsDist(370.00)
{

  //Camera parameters
  itsCamera = Camera(Point3D<float>(0,0,0.0),
                     Point3D<float>(0, 0,0),
                     450, //Focal length
                     320, //width
                     240); //height

  itsHashedGeonsState.set_empty_key(-1);

  itsObjects.resize(4);


  //Add objects
  //House object
  Object houseObject;
  V4::GeonState geon;
  geon.pos = Point3D<float>(0, -15, 0);
  geon.rot = 0.75*M_PI;
  geon.geonType = V4::TRIANGLE;
  geon.posSigma = Point3D<float>(2, 2, 2);
  geon.rotSigma = 10*M_PI/180;
  houseObject.geons.push_back(geon);

  geon.pos = Point3D<float>(0, 15, 0);
  geon.rot = 0;
  geon.geonType = V4::SQUARE;
  geon.posSigma = Point3D<float>(2, 2, 2);
  geon.rotSigma = 10*M_PI/180;
  houseObject.geons.push_back(geon);
  houseObject.objectType = HOUSE;
  itsObjects[HOUSE] = houseObject;

  //Woman object
  Object womanObject;

  geon.pos = Point3D<float>(0, -15, 0);
  geon.rot = 0;
  geon.geonType = V4::CIRCLE;
  geon.posSigma = Point3D<float>(2, 2, 2);
  geon.rotSigma = 10*M_PI/180;
  womanObject.geons.push_back(geon);

  geon.pos = Point3D<float>(0, 15, 0);
  geon.rot = 0.75*M_PI;
  geon.geonType = V4::TRIANGLE;
  geon.posSigma = Point3D<float>(2, 2, 2);
  geon.rotSigma = 10*M_PI/180;
  womanObject.geons.push_back(geon);
  womanObject.objectType = WOMAN;
  itsObjects[WOMAN] = womanObject;


  //Hat object
  Object hatObject;
  geon.pos = Point3D<float>(0, -15, 0);
  geon.rot = 0.75*M_PI;
  geon.geonType = V4::TRIANGLE;
  geon.posSigma = Point3D<float>(2, 2, 2);
  geon.rotSigma = 10*M_PI/180;
  hatObject.geons.push_back(geon);

  geon.pos = Point3D<float>(0, 15, 0);
  geon.rot = 0;
  geon.geonType = V4::CIRCLE;
  geon.posSigma = Point3D<float>(2, 2, 2);
  geon.rotSigma = 10*M_PI/180;
  hatObject.geons.push_back(geon);
  hatObject.objectType = HAT;
  itsObjects[HAT] = hatObject;

  //Man object
  Object manObject;

  geon.pos = Point3D<float>(0, -15, 0);
  geon.rot = 0;
  geon.geonType = V4::CIRCLE;
  geon.posSigma = Point3D<float>(2, 2, 2);
  geon.rotSigma = 10*M_PI/180;
  manObject.geons.push_back(geon);

  geon.pos = Point3D<float>(0, 15, 0);
  geon.rot = 0;
  geon.geonType = V4::SQUARE;
  geon.posSigma = Point3D<float>(2, 2, 2);
  geon.rotSigma = 10*M_PI/180;
  manObject.geons.push_back(geon);
  manObject.objectType = MAN;
  itsObjects[MAN] = manObject;

  buildRTables();

  itsObjectsParticles.resize(1000);
  for(uint i=0; i<itsObjectsParticles.size(); i++)
  {
    itsObjectsParticles[i].objectType = HOUSE;
    itsObjectsParticles[i].weight = 1.0/100.0;
  }



}

// ######################################################################
IT::~IT()
{

}

// ######################################################################
void IT::buildRTables()
{
  //Position relative to the camera
  for(uint objId=0; objId<itsObjects.size(); objId++)
  {
                Point3D<float> pos(0,0,itsObjectsDist);
                float rot = 0;

                for(uint gid = 0; gid < itsObjects[objId].geons.size(); gid++)
                {
                        V4::GeonState geonState = itsObjects[objId].geons[gid];

                        geonState.rot += rot;
                        float x = geonState.pos.x;
                        float y = geonState.pos.y;
                        float z = geonState.pos.z;
                        geonState.pos.x = (cos(rot)*x - sin(rot)*y) + pos.x;
                        geonState.pos.y = (sin(rot)*x + cos(rot)*y) + pos.y;
                        geonState.pos.z = z + pos.z;

                        Point2D<int> loc = (Point2D<int>)itsCamera.project(geonState.pos);

                        RTableEntry rTableEntry;
                        rTableEntry.geonType = geonState.geonType;
                        rTableEntry.loc.i = loc.i - 320/2;
                        rTableEntry.loc.j = loc.j - 240/2;
                        rTableEntry.rot = geonState.rot;

                        itsObjects[objId].rTable.push_back(rTableEntry);
                }
                //Image<PixRGB<byte> > img(320,240,ZEROS);
                //showObject(img, itsObjects[objId], pos, rot);
                //SHOWIMG(img);
        }
}

// ######################################################################
void IT::init(Dims numCells)
{

}

// ######################################################################
void IT::onSimEventV4Output(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventV4Output>& e)
{
  std::vector<V4::GeonState> geonsState = e->getCells();


  itsHashedGeonsState.clear();

        itsGeonsState.clear();
  for(uint i=0; i<geonsState.size(); i++)
  {
                if (geonsState[i].prob > 0)
                {
                        itsGeonsState.push_back(geonsState[i]);
                        //TODO use a 3D hash key
                        int key = i; //(int)(geonsState[i].pos.x + 1000.0*geonsState[i].pos.y);
                        itsHashedGeonsState[key] = geonsState[i];
                }
  }

  evolve();

  //q.post(rutz::make_shared(new SimEventITOutput(this, itsFeaturesParticles)));


}

// ######################################################################
void IT::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage();
      ofs->writeRgbLayout(disp, "IT", FrameInfo("IT", SRC_POS));
    }
}

// ######################################################################
void IT::setInput(const std::vector<V4::GeonState> &geonsState)
{
  itsHashedGeonsState.clear();
  for(uint i=0; i<geonsState.size(); i++)
  {
    //TODO use a 3D hash key
    int key = i; //(int)(geonsState[i].pos.x + 1000.0*geonsState[i].pos.y);
    itsHashedGeonsState[key] = geonsState[i];
  }

  evolve();
}

// ######################################################################
void IT::showObject(Image<PixRGB<byte> > &img, Object& object, Point3D<float>& pos, float rot)
{

        PixRGB<byte> col;
        switch (object.objectType)
        {
                case HOUSE: col = PixRGB<byte>(255,0,0); break;
                case WOMAN: col = PixRGB<byte>(0,255,0); break;
                case HAT: col = PixRGB<byte>(0,255,255); break;
                case MAN: col = PixRGB<byte>(0,0,255);  break;
                default: break;
        }
  //Position relative to the camera
  for(uint geonId=0; geonId<object.geons.size(); geonId++)
  {
    //Transform the object relative to the camera;
    V4::GeonState geonState = object.geons[geonId];

    geonState.rot += rot;
    float x = geonState.pos.x;
    float y = geonState.pos.y;
    float z = geonState.pos.z;
    geonState.pos.x = (cos(rot)*x - sin(rot)*y) + pos.x;
    geonState.pos.y = (sin(rot)*x + cos(rot)*y) + pos.y;
    geonState.pos.z = z + pos.z;

    Point2D<int> loc = (Point2D<int>)itsCamera.project(geonState.pos);

                        switch(geonState.geonType)
                        {
                                case V4::SQUARE:
                                        drawRectOR(img, Rectangle(loc - Point2D<int>(20,20), Dims((int)40,(int)40)),
                                                        col, 1, geonState.rot);
                                        break;
                                case V4::CIRCLE:
                                        if (object.objectType != HAT)
                                                drawCircle(img, loc, 20, col);
                                        break;
                                case V4::TRIANGLE:
                                        {
                                                std::vector<Point3D<float> > outline;
                                                outline.push_back(Point3D<float>(0.0,0, 0.0));
                                                outline.push_back(Point3D<float>(0.0,-40.0, 0.0));
                                                outline.push_back(Point3D<float>(40.0, 0.0, 0.0));

                                                //get the center of the object;
                                                float centerX = 0, centerY = 0, centerZ = 0;
                                                for(uint i=0; i<outline.size(); i++)
                                                {
                                                        centerX += outline[i].x;
                                                        centerY += outline[i].y;
                                                        centerZ += outline[i].z;
                                                }
                                                centerX /= outline.size();
                                                centerY /= outline.size();
                                                centerZ /= outline.size();

                                                for(uint i=0; i<outline.size(); i++)
                                                        outline[i] -= Point3D<float>(centerX, centerY, centerZ);

                                                for(uint i=0; i<outline.size(); i++)
                                                {
                                                        float x = outline[i].x;
                                                        float y = outline[i].y;
                                                        float z = outline[i].z;
                                                        outline[i].x = (cos(geonState.rot)*x - sin(geonState.rot)*y) + geonState.pos.x;
                                                        outline[i].y = (sin(geonState.rot)*x + cos(geonState.rot)*y) + geonState.pos.y;
                                                        outline[i].z = z + geonState.pos.z;
                                                }

                                                //Project the object to camera cordinats
                                                std::vector<Point2D<int> > points;

                                                for(uint i=0; i<outline.size(); i++)
                                                {
                                                        Point2D<float> loc = itsCamera.project(outline[i]);
                                                        points.push_back(Point2D<int>(loc));
                                                }

                                                for(uint i=0; i<points.size(); i++)
                                                        drawLine(img, points[i], points[(i+1)%points.size()], col, 1);

                                        }
                                        break;
                        }
  }

}

// ######################################################################
void IT::evolve()
{

  ////Resample
  //resampleParticles(itsObjectsParticles);
  proposeParticles(itsObjectsParticles, 0.0F);

  ////Evaluate the particles;
  evaluateObjectParticles(itsObjectsParticles);

}

// ######################################################################
float IT::evaluateObjectParticles(std::vector<ObjectState>& objectParticles)
{

  for(uint p=0; p<objectParticles.size(); p++)
    getObjectLikelihood(objectParticles[p]);

  //Normalize the particles;
  double sum = 0;
  double Neff = 0; //Number of efictive particles
  for(uint i=0; i<objectParticles.size(); i++)
    sum += objectParticles[i].prob;

  for(uint i=0; i<objectParticles.size(); i++)
  {
    objectParticles[i].weight = objectParticles[i].prob/sum;
    Neff += squareOf(objectParticles[i].weight);
  }

  Neff = 1/Neff;


  return Neff;

}

void IT::GHT(std::vector<GHTAcc>& accRet, OBJECT_TYPE objectType)
{
  ImageSet<float> acc(360, Dims(320,240), ZEROS);

  CpuTimer timer;
  timer.reset();
        //for(int angIdx = 0; angIdx < 360; angIdx++)
        //{
  ////  jobs.push_back(rutz::make_shared(new GHTJob(this, acc[angIdx], angIdx, geonOutline.rTable)));
  ////  itsThreadServer->enqueueJob(jobs.back());
        //        //voteForFeature(acc[angIdx], angIdx, geonOutline.rTable);
        //}

        int angIdx = 0;
        voteForFeature(acc[angIdx], angIdx, itsObjects[objectType].rTable);

  timer.mark();
  LINFO("Total time %0.2f sec", timer.real_secs());


//  Image<float> tmp(320, 240, ZEROS);
//  for(uint angIdx=0; angIdx<acc.size(); angIdx++)
//  {
//    for(uint i=0; i<acc[angIdx].size(); i++)
//    {
//      if (acc[angIdx].getVal(i) > tmp.getVal(i))
//        tmp.setVal(i, acc[angIdx].getVal(i));
//    }
//  }
//  SHOWIMG(tmp);


        for(uint rot=0; rot<acc.size(); rot++)
                for(int y=0; y<acc[rot].getHeight(); y++)
                        for(int x=0; x<acc[rot].getWidth(); x++)
                                if (acc[rot].getVal(x,y) > 0)
                                {
                                        GHTAcc ghtAcc;
                                        ghtAcc.objectType = objectType;
                                        ghtAcc.pos = Point2D<int>(x,y);
                                        ghtAcc.ang = rot;
                                        ghtAcc.scale = -1;
                                        ghtAcc.sum = acc[rot].getVal(x,y);
                                        accRet.push_back(ghtAcc);
                                }

}

void IT::voteForFeature(Image<float>& acc, int angIdx, std::vector<RTableEntry>& rTable)
{

        for(uint g=0; g < itsGeonsState.size(); g++)
        {
                V4::GeonState geonState = itsGeonsState[g];
                if (geonState.prob > 0)
                {
                        Point2D<int> loc = (Point2D<int>)itsCamera.project(geonState.pos);


                        for(uint rIdx = 0; rIdx < rTable.size(); rIdx++)
                        {
                                if (rTable[rIdx].geonType == geonState.geonType)
                                {
                                        float ang = angIdx * M_PI/180;

                                        float featureAng = rTable[rIdx].rot + ang;
                                        if (featureAng < 0)
                                                featureAng += M_PI*2;
                                        if (featureAng > M_PI*2)
                                                featureAng -= M_PI*2;

                                        float diffRot = acos(cos(geonState.rot - featureAng));

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

void IT::normalizeAcc(std::vector<GHTAcc>& acc)
{

  double sum = 0;
  for(uint i=0; i<acc.size(); i++)
    sum += acc[i].sum;

  for(uint i=0; i<acc.size(); i++)
    acc[i].sum /= sum;
}

// ######################################################################
Image<float> IT::showParticles(const std::vector<ObjectState>& objectParticles)
{
  Image<float> probImg(320,240, ZEROS);

  for(uint i=0; i<objectParticles.size(); i++)
  {
    Point2D<int> loc = (Point2D<int>)itsCamera.project(objectParticles[i].pos);
    if (probImg.coordsOk(loc))
      probImg.setVal(loc, probImg.getVal(loc) + objectParticles[i].weight);
  }
  inplaceNormalize(probImg, 0.0F, 255.0F);

  return probImg;

}

// ######################################################################
void IT::proposeParticles(std::vector<ObjectState>& objectParticles, const double Neff)
{
  LINFO("Propose Particles");

  float probThresh = 1.0e-2;

  //If we have good hypothisis then just adjest them
  uint objectsAboveProb = 0;

  for(uint i=0; i<objectParticles.size(); i++)
  {
    if (objectParticles[i].prob > probThresh)
    {
      objectParticles[i].pos.x +=  randomDoubleFromNormal(1.0);
      objectParticles[i].pos.y +=  randomDoubleFromNormal(1.0);
      objectParticles[i].pos.z =  itsObjectsDist + randomDoubleFromNormal(0.05);
      objectParticles[i].rot   +=  randomDoubleFromNormal(1.0)*M_PI/180;
      objectParticles[i].weight = 1.0/(float)objectParticles.size();
      objectsAboveProb++;
    }
  }

  //LINFO("Objects Above prob %i/%lu",
  //    objectsAboveProb, objectParticles.size());

  if (objectsAboveProb < objectParticles.size())
  {

    LINFO("GHT sampleing");
    std::vector<GHTAcc> acc;
    GHT(acc, HOUSE);
    GHT(acc, WOMAN);
    GHT(acc, HAT);
    GHT(acc, MAN);
                //LINFO("Acc size %lu", acc.size());
    LINFO("GHT Done ");

    if (acc.size() == 0)
            return;
                normalizeAcc(acc);

    std::priority_queue <GHTAcc> pAcc;
    for(uint i=0; i<acc.size(); i++)
       pAcc.push(acc[i]);

    ////Sample from acc
    for(uint i=0; i<objectParticles.size(); i++)
    {
      if (objectParticles[i].prob <= probThresh)
      {
        //add this point to the list
        if (pAcc.empty())
          break;
        GHTAcc p = pAcc.top(); pAcc.pop();

        Point3D<float>  iPoint = itsCamera.inverseProjection(Point2D<float>(p.pos), itsObjectsDist);
        objectParticles[i].pos.x = iPoint.x + randomDoubleFromNormal(1);
        objectParticles[i].pos.y = iPoint.y + randomDoubleFromNormal(1);
        objectParticles[i].pos.z =  itsObjectsDist + randomDoubleFromNormal(0.05);
        objectParticles[i].rot =  (p.ang + randomDoubleFromNormal(1))*M_PI/180;
        objectParticles[i].objectType = p.objectType;
        objectParticles[i].weight = 1.0/(float)objectParticles.size();
        objectParticles[i].prob = 1.0e-50;
      }
    }
  }
}

// ######################################################################
void IT::resampleParticles(std::vector<ObjectState>& objectParticles)
{
  LINFO("Resample");

  std::vector<ObjectState> newParticles;

  //Calculate a Cumulative Distribution Function for our particle weights
  std::vector<double> CDF;
  CDF.resize(objectParticles.size());

  CDF.at(0) = objectParticles.at(0).weight;
  for(uint i=1; i<CDF.size(); i++)
    CDF.at(i) = CDF.at(i-1) + objectParticles.at(i).weight;

  uint i = 0;
  double u = randomDouble()* 1.0/double(objectParticles.size());

  for(uint j=0; j < objectParticles.size(); j++)
  {
    while(u > CDF.at(i))
      i++;

    ObjectState p = objectParticles.at(i);
    p.weight     = 1.0/double(objectParticles.size());
    newParticles.push_back(p);

    u += 1.0/double(objectParticles.size());
  }

  objectParticles = newParticles;

}

// ######################################################################
void IT::getObjectLikelihood(ObjectState& objectState)
{

  Object object = itsObjects[objectState.objectType];
  Point2D<int> loc = (Point2D<int>)itsCamera.project(objectState.pos);
  float rot = objectState.rot + M_PI;


  double totalProb = 1;

  for(uint rIdx=0; rIdx<object.rTable.size(); rIdx++)
  {
          float ang = rot;
          float a0 = loc.i - (object.rTable[rIdx].loc.i*cos(ang) - object.rTable[rIdx].loc.j*sin(ang));
          float b0 = loc.j - (object.rTable[rIdx].loc.i*sin(ang) + object.rTable[rIdx].loc.j*cos(ang));

    //Get the expected geon
    V4::GeonState geonState;
          geonState.pos = itsCamera.inverseProjection(Point2D<float>(a0,b0), itsObjectsDist);
          geonState.rot = rot + object.rTable[rIdx].rot + M_PI;
          geonState.geonType = object.rTable[rIdx].geonType;
          geonState.prob = -1;

          while (geonState.rot < 0)
                  geonState.rot += M_PI*2;
          while(geonState.rot > M_PI*2)
                  geonState.rot -= M_PI*2;


                //Find the closest naf
                int nearPart = 0;
                float partDist = 1.0e100;
                for(uint i=0; i<itsGeonsState.size(); i++)
                {
                        if (itsGeonsState[i].prob > 0)
                        {
                                float distance = geonState.distance(itsGeonsState[i]);
                                if (distance < partDist)
                                {
                                                partDist = distance;
                                                nearPart = i;
                                }
                        }
                }

                float sig = 0.75F;
                float prob  = (1.0F/(sig*sqrt(2*M_PI))*exp(-0.5*squareOf(partDist)/squareOf(sig)));
//                LINFO("Dist %f %e", partDist, prob);
                totalProb *= prob;
  }

//        LINFO("Total prob : %e", totalProb);

  objectState.prob = totalProb;


}
// ######################################################################
void IT::fixAngle(float& ang)
{
  if (ang > M_PI)
    ang -= 2*M_PI;

  if (ang < M_PI)
    ang += 2*M_PI;
}


// ######################################################################
V4::GeonState IT::findNearestGeon(const V4::GeonState& geonState)
{

  V4::GeonState nearestGeon;
  nearestGeon.prob = -1;
  float minDistance = 1.0e100;

  dense_hash_map<int, V4::GeonState>::const_iterator iter;
  for(iter = itsHashedGeonsState.begin(); iter != itsHashedGeonsState.end(); iter++)
  {
    if (geonState.geonType == iter->second.geonType && iter->second.prob > 0.5)
    {

      //geon distance
      float dist = getGeonDistance(geonState, iter->second);
      if (dist < minDistance)
      {
        minDistance = dist;
        nearestGeon = iter->second;
      }
    }
  }

  return nearestGeon;
}

// ######################################################################
double IT::getGeonDistance(const V4::GeonState& g1, const V4::GeonState& g2)
{

  double posDist = g1.pos.squdist(g2.pos);
  double rotDist = 0;

  return sqrt(posDist + rotDist);

}

// ######################################################################
Layout<PixRGB<byte> > IT::getDebugImage()
{
  Layout<PixRGB<byte> > outDisp;

  //Show the gabor states
  Image<float> perc(320,240, ZEROS);

  Image<PixRGB<byte> > objectDisp = toRGB(Image<byte>(perc));

  //for(uint i=0; i<itsObjectsParticles.size(); i++)
  //{
  //  if (itsObjectsParticles[i].weight > 0)
  //  {
  //    Object object=itsObjects[itsObjectsParticles[i].objectType];
  //    ObjectState objectState = itsObjectsParticles[i];
        //                showObject(objectDisp, object, objectState.pos, objectState.rot);

        //                Point2D<int> textLoc = (Point2D<int>)itsCamera.project(objectState.pos);
        //
        //                char msg[255];
        //                msg[0] = NULL;
        //                switch (itsObjectsParticles[i].objectType)
        //                {
        //                        case HOUSE: sprintf(msg, "HOUSE"); break;
        //                        case WOMAN: sprintf(msg, "WOMAN"); break;
        //                        case HAT: sprintf(msg, "HAT"); textLoc.j -= 15; break;
        //                        case MAN: sprintf(msg, "MAN"); break;
        //                        default:
        //                                break;
        //                }
        //                writeText(objectDisp, textLoc, msg, PixRGB<byte>(255,255,255), PixRGB<byte>(0,0,0));
        //
  //  }
  //}


        //Show only the max object from each type
        for(uint obj = 0; obj < itsObjects.size(); obj++)
        {
                int maxObj = -1;
                float max = 0;

                for(uint i=0; i<itsObjectsParticles.size(); i++)
                {
                        if (itsObjectsParticles[i].weight > 0 && itsObjectsParticles[i].objectType == itsObjects[obj].objectType)
                        {
                                if (itsObjectsParticles[i].weight > max)
                                {
                                        max = itsObjectsParticles[i].weight;
                                        maxObj = i;
                                }
                        }
                }

                //Show the max object
                if (maxObj != -1)
                {
                        Object object=itsObjects[itsObjectsParticles[maxObj].objectType];
                        ObjectState objectState = itsObjectsParticles[maxObj];
                        showObject(objectDisp, object, objectState.pos, objectState.rot);

                        Point2D<int> textLoc = (Point2D<int>)itsCamera.project(objectState.pos);
                        textLoc.i -= 10;

                        char msg[255];
                        msg[0] = 0;
                        switch (itsObjectsParticles[maxObj].objectType)
                        {
                                case HOUSE: sprintf(msg, "HOUSE"); break;
                                case WOMAN: sprintf(msg, "WOMAN"); break;
                                case HAT: sprintf(msg, "HAT"); textLoc.j -= 20; break;
                                case MAN: sprintf(msg, "MAN"); break;
                                default:
                                                                        break;
                        }
                        writeText(objectDisp, textLoc, msg, PixRGB<byte>(255,255,255), PixRGB<byte>(0,0,0));

                }

        }


  Image<float> particles = showParticles(itsObjectsParticles);

  char msg[255];
  sprintf(msg, "IT");
  writeText(objectDisp, Point2D<int>(0,0), msg, PixRGB<byte>(255,255,255), PixRGB<byte>(0,0,0));

  outDisp = hcat(objectDisp, toRGB(Image<byte>(particles)));

  return outDisp;

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

