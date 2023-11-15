/*!@file SceneUnderstanding/GHough.C  Generalized Hough */



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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/FeatureMatching/GHough.C $
// $Id: GHough.C 13815 2010-08-22 17:58:48Z lior $
//

#ifndef GHough_C_DEFINED
#define GHough_C_DEFINED

#include "FeatureMatching/GHough.H"
#include "Image/DrawOps.H"
#include "GUI/DebugWin.H"
#include "Image/FilterOps.H"
#include <fcntl.h>
#include <cstdio>


namespace
{
  pthread_mutex_t itsAccLock;

  class GHTJob : public JobWithSemaphore
  {
  public:

    GHTJob(GHough* ghough, int id, const GHough::RTable& rTable, const std::vector<GHough::Feature>& features,
        std::vector<GHough::Acc>& acc) :
      itsGHough(ghough),
      itsId(id),
      itsRTable(rTable),
      itsFeatures(features),
      itsAcc(acc)
    {}

    virtual ~GHTJob() {}

    virtual void run()
    {
      float maxVotes = 0;
      std::vector<GHough::Acc> acc = itsGHough->getVotes(itsId, itsRTable, itsFeatures, maxVotes);

      pthread_mutex_lock(&itsAccLock);
      for(uint j=0; j<acc.size(); j++)
        itsAcc.push_back(acc[j]);
      pthread_mutex_unlock(&itsAccLock);

      this->markFinished();
    }

    virtual const char* jobType() const { return "GHTJob"; }

    GHough* itsGHough;
    int itsId;
    const GHough::RTable& itsRTable;
    const std::vector<GHough::Feature>& itsFeatures;
    std::vector<GHough::Acc>& itsAcc;
  };
}

// ######################################################################
GHough::GHough()  :
  itsNumEntries(20)
{

  //itsThreadServer.reset(new WorkThreadServer("GHough", 10));
  
  if (0 != pthread_mutex_init(&itsAccLock, NULL))
    LFATAL("pthread_mutex_init() failed");

  itsSOFM = new SOFM("Corners", 360, 25,25 );
  
}

// ######################################################################
GHough::~GHough()
{

  if (0 != pthread_mutex_destroy(&itsAccLock))
    LERROR("pthread_mutex_destroy() failed");
}

Point2D<float> GHough::addModel(int& id, const Image<byte>& img, const Image<float>& ang,
    Point3D<float> pos, Point3D<float> rot)
{

  Model model;
  model.id = 0;
  model.pos = pos;
  model.rot = rot;


  Point2D<float> imgLoc;
  RTable rTable = createRTable(img, ang, model.imgPos, model.numFeatures, imgLoc);
  if (rTable.entries.size() > 0)
  {
    model.rTables.push_back(rTable);
    itsModels.push_back(model);
    id = itsModels.size()-1;
  }

  return imgLoc;

}

Point2D<float> GHough::addModel(int& id, int type, const std::vector<GHough::Feature>& features, 
    Point3D<float> pos, Point3D<float> rot)
{

  Model model;
  model.id = 0;
  model.pos = pos;
  model.rot = rot;
  model.type = type;

  Point2D<float> imgLoc;
  model.numFeatures = features.size();
  RTable rTable = createRTable(features, model.imgPos, imgLoc);
  if (rTable.featureEntries.size() > 0)
  {
    model.rTables.push_back(rTable);
    itsModels.push_back(model);
    id = itsModels.size()-1;
  }

  return imgLoc;

}

void GHough::addModel(int id, const Image<float>& img)
{

  Image<float> mag, ori;
  gradientSobel(img, mag, ori);

  ////Non maximal suppersion
  mag = nonMaxSuppr(mag, ori);
  SHOWIMG(mag);

  //createInvRTable(mag, ori);

  Model model;
  model.id = id;

  //RTable rTable = createRTable(mag, ori, model.imgPos, model.numFeatures);
  //if (rTable.entries.size() > 0)
  //{
  //  model.rTables.push_back(rTable);
  //  itsModels.push_back(model);
  //}

}

Point2D<int> GHough::addModel(int id, const std::vector<Point2D<int> >& polygon)
{

  Point2D<int> center;
 
  Image<float> mag(320, 240, ZEROS);
  Image<float> ori(320, 240, ZEROS);

  for(uint i=0; i<polygon.size(); i++)
  {
    Point2D<int> p1 = polygon[i];
    Point2D<int> p2 = polygon[(i+1)%polygon.size()];

    float ang = atan2(p1.j-p2.j, p2.i - p1.i);
    drawLine(mag, p1, p2, 255.0F);
    drawLine(ori, p1, p2, ang);
  }

  Model model;
  model.id = id;

  RTable rTable = createRTable(mag, ori, center, model.numFeatures);
  if (rTable.entries.size() > 0)
  {
    model.rTables.push_back(rTable);
    itsModels.push_back(model);
  }

  return center;

}


std::vector<GHough::Acc> GHough::getVotes(const Image<float>& img)
{

  //std::vector<Acc> acc;

  Image<float> mag, ori;
  gradientSobel(img, mag, ori);

  ////Non maximal suppersion
  mag = nonMaxSuppr(mag, ori);

  //Image<float> acci = getInvVotes(mag, ori);
  //Point2D<int> loc; float max;
  //findMax(acci, loc, max);
  ////drawCircle(acci, loc, 10, 1550.0F);
  //SHOWIMG(acci);

  std::vector<GHough::Acc> acc;

  for(uint i=0; i<itsModels.size(); i++)
  {
    for(uint j=0; j<itsModels[i].rTables.size(); j++)
    {
      std::vector<GHough::Acc> accScale =  getVotes(i, itsModels[i].rTables[j], mag, ori);
      for(uint j=0; j<accScale.size(); j++)
        acc.push_back(accScale[j]);
    }
  }

  //sort the acc
  std::sort(acc.begin(), acc.end(), AccCmp());

  drawCircle(mag, acc[0].pos, 3, 255.0F);
  SHOWIMG(mag);

  return acc;
}

std::vector<GHough::Acc> GHough::getVotes(const Image<float>& mag, const Image<float>& ori)
{

  std::vector<GHough::Acc> acc;

  for(uint i=0; i<itsModels.size(); i++)
  {
    for(uint j=0; j<itsModels[i].rTables.size(); j++)
    {
      std::vector<GHough::Acc> accScale =  getVotes(i, itsModels[i].rTables[j], mag, ori);
      for(uint j=0; j<accScale.size(); j++)
        acc.push_back(accScale[j]);
    }
  }

  //sort the acc
  //std::sort(acc.begin(), acc.end(), AccCmp());

  return acc;
}


void GHough::setPosOffset(int id, Point3D<float> pos)
{

  itsModels[id].pos = pos;

}

GHough::RTable GHough::createRTable(const Image<byte>& img, const Image<float>& ang,
    Point2D<float>& imgPos, int& numFeatures, Point2D<float>& imgLoc)
{
  RTable rTable;

  //Compute refrance Point
  Point2D<int> center(0,0);
  int numOfPixels = 0;

  imgLoc = Point2D<float>(0,0);
  for(int y=0; y<img.getHeight(); y++)
    for(int x=0; x<img.getWidth(); x++)
    {
      if (img.getVal(x,y) > 0)
      {
        center.i += x;
        center.j += y;
        numOfPixels++;

        if (y > imgLoc.j)
          imgLoc = Point2D<float>(x,y);
      }
    }
  numFeatures = numOfPixels;
  if (numOfPixels > 0)
    center /= numOfPixels;
  else
    return rTable;

  //LINFO("Learn pos");
  //Image<PixRGB<byte> > tmp = img;
  //drawCircle(tmp, Point2D<int>(imgLoc), 3, PixRGB<byte>(255,0,0));
  //SHOWIMG(tmp);


  imgPos.i = imgLoc.i - center.i;
  imgPos.j = imgLoc.j - center.j;


  double D=M_PI/itsNumEntries;

  for(int y=0; y<img.getHeight(); y++)
    for(int x=0; x<img.getWidth(); x++)
    {
      if (img.getVal(x,y) > 0)
      {
        double phi = ang.getVal(x,y);
        int i = (int)round(phi/D);
        rTable.entries[i].push_back(Point2D<float>(x-center.i, y-center.j));
      }
    }
  return rTable;
}

GHough::RTable GHough::createRTable(const Image<byte>& img, const Image<float>& ang,
    Point2D<int>& center, int& numFeatures)
{
  RTable rTable;

  //Compute refrance Point
  center.i = 0; center.j = 0;
  int numOfPixels = 0;

  for(int y=0; y<img.getHeight(); y++)
    for(int x=0; x<img.getWidth(); x++)
    {
      if (img.getVal(x,y) > 0)
      {
        center.i += x;
        center.j += y;
        numOfPixels++;
      }
    }
  numFeatures = numOfPixels;
  if (numOfPixels > 0)
    center /= numOfPixels;
  else
    return rTable;

  double D=M_PI/itsNumEntries;

  for(int y=0; y<img.getHeight(); y++)
    for(int x=0; x<img.getWidth(); x++)
    {
      if (img.getVal(x,y) > 0)
      {
        double phi = ang.getVal(x,y);

        if (phi < 0) phi += M_PI;
        if (phi > M_PI) phi -= M_PI;

        int i = (int)round(phi/D);
        rTable.entries[i].push_back(Point2D<float>(x-center.i, y-center.j));
      }
    }
  return rTable;
}

GHough::RTable GHough::createRTable(const std::vector<Feature>& features,
    Point2D<float>& imgPos, Point2D<float>& imgLoc)
{
  RTable rTable;

  //Compute refrance Point
  imgLoc = Point2D<float>(0,0);
  Point2D<float> center(0,0);
  for(uint i=0; i<features.size(); i++)
  {
    center += features[i].loc;
    if (features[i].loc.j > imgLoc.j)
      imgLoc = features[i].loc;
  }
  center /= features.size();

  //LINFO("Learn pos");
  //Image<PixRGB<byte> > tmp(150,150,ZEROS);
  //for(uint i=0; i<features.size(); i++)
  //  tmp.setVal(Point2D<int>(features[i].loc), PixRGB<byte>(0,255,0));
  //drawCircle(tmp, Point2D<int>(imgLoc), 3, PixRGB<byte>(255,0,0));
  //SHOWIMG(tmp);

  imgPos.i = imgLoc.i - center.i;
  imgPos.j = imgLoc.j - center.j;

  for(uint i=0; i<features.size(); i++)
  {
    long idx = getIndex(features[i].values);

    Feature f;
    f.loc = features[i].loc - center;
    f.values = features[i].values;
    rTable.featureEntries[idx].push_back(f);
  }

  return rTable;
}

long GHough::getIndex(const std::vector<float>& values)
{

  double D=2*M_PI/45; //itsNumEntries;

  //Generate a histogram
  int hist[360];
  for(uint i=0; i<360; i++)
    hist[i] = 0;

  for(uint i=0; i<values.size(); i++)
  {
    float ang = values[i];
    if (ang < 0) ang += 2*M_PI;
    if (ang > 2*M_PI) ang -= M_PI*2;
    int idx = (int)round(ang/D);
    hist[idx]++;
  }

  //Show histogram

  //for(uint i=0; i<360; i++)
  //  if (hist[i] > 0)
  //    printf("%i:%i ", i, hist[i]);
  //printf("\n");


  //Generate the index from the histogram
  long idx = 0;
  for(uint i=0; i<360; i++)
  {
    if (hist[i] > 0)
      idx = (360*idx) + i; 
  }
  //LINFO("Index %ld", idx);

  return idx;

}

Point3D<float> GHough::getModelRot(const int id)
{
  if (id >= 0 && id < (int)itsModels.size())
    return itsModels[id].rot;
  else
  {
    LFATAL("Invalid model id %i", id);
    return Point3D<float>(0,0,0);
  }

}

Point2D<float> GHough::getModelImgPos(const int id)
{
  if (id >= 0 && id < (int)itsModels.size())
    return itsModels[id].imgPos;
  else
  {
    LFATAL("Invalid model id %i", id);
    return Point2D<float>(0,0);
  }
}

int GHough::getModelType(const int id)
{
  if (id >= 0 && id < (int)itsModels.size())
    return itsModels[id].type;
  else
  {
    LFATAL("Invalid model id %i", id);
    return -1;
  }
}

Point3D<float> GHough::getModelPosOffset(const int id)
{
  if (id >= 0 && id < (int)itsModels.size())
    return itsModels[id].pos;
  else 
  {
    LFATAL("Invalid model id %i", id);
    return Point3D<float>(0,0,0);
  }
}

std::vector<GHough::Acc> GHough::getVotes(const Image<byte>& img, const Image<float>& ang)
{
  std::vector<GHough::Acc> acc;

  for(uint i=0; i<itsModels.size(); i++)
  {
    for(uint j=0; j<itsModels[i].rTables.size(); j++)
    {
      std::vector<GHough::Acc> accScale =  getVotes(i, itsModels[i].rTables[j], img, ang);
      for(uint j=0; j<accScale.size(); j++)
        acc.push_back(accScale[j]);
    }
  }

  //sort the acc
  std::sort(acc.begin(), acc.end(), AccCmp());

  return acc;

}

std::vector<GHough::Acc> GHough::getVotes(const std::vector<Feature>& features)
{

  //Show the features
  Image<PixRGB<byte> > cornersImg(320, 240, ZEROS);
  for(uint i=0; i<features.size(); i++)
  {
    for(uint ai=0; ai<features[i].values.size(); ai++)
    {
      int x1 = int(cos(features[i].values[ai])*30.0/2.0);
      int y1 = int(sin(features[i].values[ai])*30.0/2.0);
      Point2D<float> p1(features[i].loc.i-x1, features[i].loc.j+y1);

      drawLine(cornersImg, Point2D<int>(features[i].loc), Point2D<int>(p1), PixRGB<byte>(0,255,0));
    }
  }
  SHOWIMG(cornersImg);
  
  //itsSOFM->RandomWeights(0,1);
  //itsSOFM->ReadNet("hough.sofm");
  //for(uint i=0; i<10; i++)
  //trainSOFM();


  CpuTimer timer;
  timer.reset();
  
  //std::vector<rutz::shared_ptr<GHTJob> > jobs;
  
  std::vector<GHough::Acc> acc;
  uint numModels = 0;
  for(uint i=0; i<itsModels.size(); i++)
  {

    for(uint j=0; j<itsModels[i].rTables.size(); j++)
    {
      uint numFeatures = getNumFeatures(i);
      if (numFeatures < 3)
        continue;
      if (itsModels[i].type != 1)
        continue;

      float maxVotes  =0;
      //Image<PixRGB<byte> > rTableImg = getRTableImg(i);
      //SHOWIMG(rTableImg);
      std::vector<GHough::Acc> accScale =  getVotes2(i, itsModels[i].rTables[j], features, maxVotes);

      //if (accScale.size() > 0 && maxVotes > 0.01)
      {
        LINFO("Model %i/%i rt %i nf %i maxVotes %f rot %f,%f,%f", i,
            (uint)itsModels.size(), j, numFeatures, maxVotes,
            itsModels[i].rot.x,itsModels[i].rot.y,itsModels[i].rot.z);
        Image<PixRGB<byte> > rTableImg = getRTableImg(i);
        SHOWIMG(rTableImg);

        Image<float> accImg = getAccImg(accScale);
        inplaceNormalize(accImg, 0.0F, 255.0F);
        Image<PixRGB<byte> > tmp = accImg;
        tmp += cornersImg;
        SHOWIMG(tmp);
      }

      for(uint j=0; j<accScale.size(); j++)
        acc.push_back(accScale[j]);

      //jobs.push_back(rutz::make_shared(new GHTJob(this, i, itsModels[i].rTables[j], features, acc)));
      //itsThreadServer->enqueueJob(jobs.back());

      numModels++;

    }
  }

  ////wait for jobs to finish
  //while(itsThreadServer->size() > 0)
  //  usleep(10000);

  timer.mark();
  LINFO("Total time %0.2f sec for %i models (%i proposals)", timer.real_secs(), numModels, (uint)acc.size());

  //sort the acc
  std::sort(acc.begin(), acc.end(), AccCmp());

  return acc;

}


std::vector<GHough::Acc> GHough::getVotes(int id, const RTable& rTable,
    const Image<byte>& img, const Image<float>& ang)
{
  double D=M_PI/itsNumEntries;


  std::map<unsigned long, Acc> tmpAcc;

  Image<float> accImg(320, 240, ZEROS);

  for(int y=0; y<img.getHeight(); y++)
    for(int x=0; x<img.getWidth(); x++)
    {
      if (img.getVal(x,y) > 0)
      {
        double phi = ang.getVal(x,y);
        if (phi < 0) phi += M_PI;
        if (phi > M_PI) phi -= M_PI;
        int i = (int)round(phi/D);

        std::map<int, std::vector<Point2D<float> > >::const_iterator iter = 
          rTable.entries.find(i);

        if (iter != rTable.entries.end())
        {
          ////Vote
          for(uint j=0; j<iter->second.size(); j++)
          {
            Point2D<float> loc =iter->second[j];
            Point2D<int> voteLoc(int(x-loc.i),int(y-loc.j)); 
            if (voteLoc.i > 0 && voteLoc.i < 512 &&
                voteLoc.j > 0 && voteLoc.j < 512)
            {
              if (accImg.coordsOk(voteLoc))
                accImg.setVal(voteLoc, accImg.getVal(voteLoc) + 1);
              unsigned long key = id*512*512 + voteLoc.j*512 + voteLoc.i; 
              std::map<unsigned long, Acc>::iterator it = tmpAcc.find(key);
              if (it != tmpAcc.end())
                it->second.votes++;
              else
                tmpAcc[key] = Acc(id, voteLoc,1);
            }
          }
        }
      }
    }

  std::vector<Acc> acc;

  for(uint i=0; i<100; i++)
  {
    Point2D<int> maxLoc; float maxVal;
    findMax(accImg, maxLoc, maxVal);

    acc.push_back(Acc(0, maxLoc, maxVal));
    drawDisk(accImg, maxLoc, 10, 0.0F);
    //SHOWIMG(accImg);
  }


  //std::map<unsigned long, Acc>::iterator it;
  //for(it = tmpAcc.begin(); it != tmpAcc.end(); it++)
  //{
  //  it->second.prob = float(it->second.votes)/float(itsModels[it->second.id].numFeatures);
  //  //if (it->second.prob > 0.20)
  //  {
  //    acc.push_back(it->second);
  //  }
  //}

  return acc;
}


std::vector<GHough::Acc> GHough::getVotes(int id, const RTable& rTable, const std::vector<Feature>& features, float& maxVotes)
{
  std::map<unsigned long, Acc> tmpAcc;

  //Used for appliing a variance over position
  //TODO change to a veriance in feature position, not its endpoint
  float stddevX = 1.1;
  float stddevY = 1.1;
  int voteSizeX = int(ceil(stddevX * sqrt(-2.0F * log(exp(-5.0F)))));
  int voteSizeY = int(ceil(stddevY * sqrt(-2.0F * log(exp(-5.0F)))));

  Image<PixRGB<byte> > tmp(320,240,ZEROS);

  int numFeatures = 0;
  for(uint fi=0; fi<features.size(); fi++)
  {
    long idx = getIndex(features[fi].values);


    /************/
    for(uint ai=0; ai<features[fi].values.size(); ai++)
    {
      int x1 = int(cos(features[fi].values[ai])*30.0/2.0);
      int y1 = int(sin(features[fi].values[ai])*30.0/2.0);
      Point2D<float> p1(features[fi].loc.i-x1, features[fi].loc.j+y1);

      drawLine(tmp, Point2D<int>(features[fi].loc), Point2D<int>(p1), PixRGB<byte>(0,255,0));
    }

    LINFO("Feature %i indx %ld\n", fi, idx);
    SHOWIMG(tmp);
    /****************/



    std::map<long, std::vector<Feature > >::const_iterator iter = 
      rTable.featureEntries.find(idx);

    if (iter != rTable.featureEntries.end())
    {
      LINFO("Found match");
      ////Vote
      for(uint j=0; j<iter->second.size(); j++)
      {
        Point2D<float> loc =iter->second[j].loc;

        Point2D<int> voteLoc(int(features[fi].loc.i-loc.i),int(features[fi].loc.j-loc.j)); 
        numFeatures ++;

        
        //Vote in a gaussien unsertinty
        for(int y=voteLoc.j-voteSizeY; y<voteLoc.j+voteSizeY; y++)
        {
          float diffY = y-voteLoc.j;
          float ry = exp(-((diffY*diffY)/(stddevY*stddevY)));
          for(int x=voteLoc.i-voteSizeX; x<voteLoc.i+voteSizeX; x++)
          {
            float diffX = x-voteLoc.i;
            float rx = exp(-((diffX*diffX)/(stddevX*stddevX)));
            //float weight = nafState.prob + rRot*rx*ry;
            float weight = rx*ry;

            if (x > 0 && x < 512 &&
                y > 0 && y < 512)
            {
              unsigned long key = id*512*512 + y*512 + x; 

              std::map<unsigned long, Acc>::iterator it = tmpAcc.find(key);
              if (it != tmpAcc.end())
                it->second.votes += weight;
              else
                tmpAcc[key] = Acc(id, x,y, weight);
            }
          }
        }
      }
    }
  }

  std::vector<Acc> acc;


  std::map<unsigned long, Acc>::iterator it;
  for(it = tmpAcc.begin(); it != tmpAcc.end(); it++)
  {
    it->second.prob = float(it->second.votes)/float(numFeatures);
    //LINFO("id:%i geons %i votes %i features %i prob %f", 
    //    it->second.id,
    //    getModelType( it->second.id),
    //    it->second.votes,
    //    numFeatures,
    //    it->second.prob);
    if (it->second.votes > maxVotes)
      maxVotes = it->second.votes;

    if (it->second.votes > 1)
    {
      acc.push_back(it->second);
    }
  }

  return acc;
}

std::vector<GHough::Acc> GHough::getVotes2(int id, const RTable& rTable, const std::vector<Feature>& features, float& maxVotes)
{
  std::map<unsigned long, Acc> tmpAcc;

  //Used for appliing a variance over position
  //TODO change to a veriance in feature position, not its endpoint
  float stddevX = 0.5;
  float stddevY = 0.5;
  int voteSizeX = int(ceil(stddevX * sqrt(-2.0F * log(exp(-5.0F)))));
  int voteSizeY = int(ceil(stddevY * sqrt(-2.0F * log(exp(-5.0F)))));

  Image<PixRGB<byte> > tmp(320,240,ZEROS);

  int numFeatures = 0;
  for(uint fi=0; fi<features.size(); fi++)
  {
    //Build a GMM
    std::vector<GaussianDef> gmmF;
    double weight = 1.0/double(features[fi].values.size()); //equal weight
    for(uint i=0; i<features[fi].values.size(); i++)
      gmmF.push_back(GaussianDef(weight, features[fi].values[i], 1*M_PI/180)); //1 deg variance

    //Find all the features that are closest to this one and vote

    std::map<long, std::vector<Feature> >::const_iterator iter;
    for(iter = rTable.featureEntries.begin(); iter != rTable.featureEntries.end(); iter++)
    {
      for(uint k=0; k<iter->second.size(); k++)
      {
        const Feature& feature = iter->second[k];
        if (feature.values.size() > 1)
        {
          std::vector<GaussianDef> gmmG;
          double weight = 1.0/double(feature.values.size()); //equal weight
          for(uint j=0; j<feature.values.size(); j++)
            gmmG.push_back(GaussianDef(weight, feature.values[j], (1*M_PI/180))); //2 deg variance

          double dist = L2GMM(gmmF, gmmG);
          if (dist < 2)
          {
            Point2D<float> loc =iter->second[k].loc;
            Point2D<int> voteLoc(int(features[fi].loc.i-loc.i),int(features[fi].loc.j-loc.j)); 
            numFeatures ++;

            //Vote in a gaussien unsertinty
            for(int y=voteLoc.j-voteSizeY; y<voteLoc.j+voteSizeY; y++)
            {
              float diffY = y-voteLoc.j;
              float ry = exp(-((diffY*diffY)/(stddevY*stddevY)));
              for(int x=voteLoc.i-voteSizeX; x<voteLoc.i+voteSizeX; x++)
              {
                float diffX = x-voteLoc.i;
                float rx = exp(-((diffX*diffX)/(stddevX*stddevX)));
                //float weight = nafState.prob + rRot*rx*ry;
                float weight = rx*ry;

                if (x > 0 && x < 512 &&
                    y > 0 && y < 512)
                {
                  unsigned long key = id*512*512 + y*512 + x; 

                  std::map<unsigned long, Acc>::iterator it = tmpAcc.find(key);
                  if (it != tmpAcc.end())
                    it->second.votes += weight;
                  else
                    tmpAcc[key] = Acc(id, x,y, weight);
                }
              }
            }
          }

        }
      }
    }


  }

  std::vector<Acc> acc;


  std::map<unsigned long, Acc>::iterator it;
  for(it = tmpAcc.begin(); it != tmpAcc.end(); it++)
  {
    it->second.prob = float(it->second.votes)/float(numFeatures);
    //LINFO("id:%i geons %i votes %i features %i prob %f", 
    //    it->second.id,
    //    getModelType( it->second.id),
    //    it->second.votes,
    //    numFeatures,
    //    it->second.prob);
    if (it->second.votes > maxVotes)
      maxVotes = it->second.votes;

    if (it->second.votes > 1)
    {
      acc.push_back(it->second);
    }
  }

  return acc;
}

//Image<float> GHough::getRTableImg(const int id)
//{
//
//  Image<float> img(320,240,ZEROS);
//  for(uint tbl=0; tbl<itsModels[id].rTables.size() && tbl < 1; tbl++)
//  {
//    RTable rTable = itsModels[id].rTables[tbl];
//
//    std::map<int, std::vector<Point2D<float> > >::const_iterator iter;
//    for(iter = rTable.entries.begin(); iter != rTable.entries.end(); iter++)
//    {
//      //int ori = iter->first;
//      for(uint k=0; k<iter->second.size(); k++)
//      {
//        Point2D<int> loc = Point2D<int>(iter->second[k]) + Point2D<int>(320/2, 240/2);
//        img.setVal(loc, 255.0F);
//      }
//    }
//  }
//
//  return img;
//
//}

Image<float> GHough::getAccImg(std::vector<GHough::Acc>& acc)
{
  Image<float> img(320,240,ZEROS);


  for(uint i=0; i<acc.size(); i++)
    img.setVal(acc[i].pos, acc[i].votes);

  Point2D<int> loc; float val;
  findMax(img, loc, val);
  LINFO("Max at %ix%i val %f", loc.i, loc.j, val);
  
  return img;
}

Image<PixRGB<byte> > GHough::getRTableImg(const int id)
{

  Image<PixRGB<byte> > img(320,240,ZEROS);
  for(uint tbl=0; tbl<itsModels[id].rTables.size() && tbl < 1; tbl++)
  {
    const RTable& rTable = itsModels[id].rTables[tbl];

    std::map<long, std::vector<Feature> >::const_iterator iter;
    for(iter = rTable.featureEntries.begin(); iter != rTable.featureEntries.end(); iter++)
    {
      //int ori = iter->first;
      for(uint k=0; k<iter->second.size(); k++)
      {
        const Feature& feature = iter->second[k];
        Point2D<int> loc = Point2D<int>(feature.loc) + Point2D<int>(320/2, 240/2);
        drawCircle(img, loc, 3, PixRGB<byte>(255,0,0));

        for(uint ai=0; ai<feature.values.size(); ai++)
        {
          int x1 = int(cos(feature.values[ai])*30.0/2.0);
          int y1 = int(sin(feature.values[ai])*30.0/2.0);
          Point2D<float> p1(loc.i-x1, loc.j+y1);

          drawLine(img, Point2D<int>(loc), Point2D<int>(p1), PixRGB<byte>(0,255,0));
        }
      }
    }
  }
  return img;
}


void GHough::trainSOFM()
{

  Image<PixRGB<byte> > tmp = itsSOFM->getWeightsImage();
  SHOWIMG(tmp);

  std::vector<Feature> features;

  for(uint obj=0; obj<itsModels.size(); obj++)
  {
    for(uint tbl=0; tbl<itsModels[obj].rTables.size() && tbl < 1; tbl++)
    {
      const RTable& rTable = itsModels[obj].rTables[tbl];

      std::map<long, std::vector<Feature> >::const_iterator iter;
      for(iter = rTable.featureEntries.begin(); iter != rTable.featureEntries.end(); iter++)
      {
        for(uint k=0; k<iter->second.size(); k++)
        {
          const Feature& feature = iter->second[k];
          if (feature.values.size() > 1)
            features.push_back(feature);
          //Point2D<int> loc = Point2D<int>(feature.loc) + Point2D<int>(320/2, 240/2);

          //itsSOFM->setInput(hist);
          //itsSOFM->Propagate();

          //double winnerVal;
          //Point2D<int> winnerId = itsSOFM->getWinner(winnerVal);

          //itsSOFM->organize(hist);
          
        }
      }
    }
    //LINFO("Obj %i/%lu done", obj, itsModels.size());

  //  Image<PixRGB<byte> > tmp = itsSOFM->getWeightsImage();
  //  SHOWIMG(tmp);
  }
  //itsSOFM->WriteNet("hough.sofm");

  //Image<float> amap = itsSOFM->getActMap();
  //SHOWIMG(amap);
  //

  int idx = 1203;

  //Build a GMM
  std::vector<GaussianDef> gmmF;
  double weight = 1.0/double(features[idx].values.size()); //equal weight
  for(uint i=0; i<features[idx].values.size(); i++)
    gmmF.push_back(GaussianDef(weight, features[idx].values[i], 1*M_PI/180)); //2 deg variance

  Image<PixRGB<byte> > qImg(320,240,ZEROS);
  for(uint ai=0; ai<features[idx].values.size(); ai++)
  {
    int x1 = int(cos(features[idx].values[ai])*30.0/2.0);
    int y1 = int(sin(features[idx].values[ai])*30.0/2.0);
    Point2D<float> p1(320/2-x1, 240/2+y1);

    drawLine(qImg, Point2D<int>(320/2,240/2), Point2D<int>(p1), PixRGB<byte>(0,255,0));
  }
  SHOWIMG(qImg);

  LINFO("Features %i", (uint)features.size());
  for(uint i=0; i<features.size(); i++)
  {
    printf("Feature c: ");
    for(uint j=0; j<features[idx].values.size(); j++)
      printf("%f ", features[idx].values[j]*180/M_PI);
    printf("\n");

    printf("Feature %i: ", i);
    for(uint j=0; j<features[i].values.size(); j++)
      printf("%f ", features[i].values[j]*180/M_PI);
    printf("\n");


    std::vector<GaussianDef> gmmG;
    double weight = 1.0/double(features[i].values.size()); //equal weight
    for(uint j=0; j<features[i].values.size(); j++)
      gmmG.push_back(GaussianDef(weight, features[i].values[j], (1*M_PI/180))); //2 deg variance

    double dist = L2GMM(gmmF, gmmG);
    LINFO("Dist %f", dist);
    LINFO("\n");

    if (dist < 2)
    {
      Image<PixRGB<byte> > tmp2 = qImg;
      for(uint ai=0; ai<features[i].values.size(); ai++)
      {
        int x1 = int(cos(features[i].values[ai])*30.0/2.0);
        int y1 = int(sin(features[i].values[ai])*30.0/2.0);
        Point2D<float> p1(320/2-x1, 240/2+y1);

        drawLine(tmp2, Point2D<int>(320/2,240/2), Point2D<int>(p1), PixRGB<byte>(255,0,0));
      }
      SHOWIMG(tmp2);
    }
  }
    
}


uint GHough::getNumFeatures(const int id)
{

  uint numFeatures=0;
  for(uint tbl=0; tbl<itsModels[id].rTables.size() && tbl < 1; tbl++)
  {
    const RTable& rTable = itsModels[id].rTables[tbl];

    std::map<long, std::vector<Feature> >::const_iterator iter;
    for(iter = rTable.featureEntries.begin(); iter != rTable.featureEntries.end(); iter++)
    {
      //int ori = iter->first;
      for(uint k=0; k<iter->second.size(); k++)
      {
        numFeatures++;
      }
    }
  }
  return numFeatures;

}

//Image<float> GHough::getCorners()
//{
//
//  float hist[360];
//
//  for(uint obj=0; obj<itsModels.size(); obj++)
//    for(uint tbl=0; tbl<itsModels[id].rTables.size() && tbl < 1; tbl++)
//    {
//      RTable rTable = itsModels[id].rTables[tbl];
//
//      std::map<long, std::vector<Feature> >::const_iterator iter;
//      for(iter = rTable.featureEntries.begin(); iter != rTable.featureEntries.end(); iter++)
//      {
//        for(uint j=0; j<iter->second.size(); j++)
//        {
//          Point2D<float> loc =iter->second[j].loc;
//          numFeatures ++;
//          Point2D<int> voteLoc(int(features[fi].loc.i-loc.i),int(features[fi].loc.j-loc.j)); 
//          if (voteLoc.i > 0 && voteLoc.i < 512 &&
//              voteLoc.j > 0 && voteLoc.j < 512)
//          {
//            unsigned long key = id*512*512 + voteLoc.j*512 + voteLoc.i; 
//            std::map<unsigned long, Acc>::iterator it = tmpAcc.find(key);
//            if (it != tmpAcc.end())
//              it->second.votes++;
//            else
//              tmpAcc[key] = Acc(id, voteLoc);
//          }
//        }
//      }
//    }
//
//  return Image<float>();
//}


void GHough::createInvRTable(const Image<byte>& img, const Image<float>& ang)
{

  //Compute refrance Point
  Point2D<int> center(0,0);
  int numOfPixels = 0;

  for(int y=0; y<img.getHeight(); y++)
    for(int x=0; x<img.getWidth(); x++)
    {
      if (img.getVal(x,y) > 0)
      {
        center.i += x;
        center.j += y;
        numOfPixels++;
      }
    }
  center /= numOfPixels;

  int entries = 5;
  double D=M_PI/entries;

  Model model;
  model.id = 0;

  for(int y=0; y<img.getHeight(); y++)
    for(int x=0; x<img.getWidth(); x++)
    {
      if (img.getVal(x,y) > 0)
      {
        Point2D<int> w2 = findInvFeature(x,y, img, ang);

        if (w2.isValid())
        {
          //Compute beta
          double phi = tan(ang.getVal(x,y));
          double phj = tan(ang.getVal(w2));

          double beta=1.57;
          if ((1.0+phi*phj) != 0)
            beta=atan((phi-phj)/(1.0+phi*phj));

          //compute k
          double ph=1.57;
          if((x-center.i)!=0)
            ph=atan(float(y-center.j)/float(x-center.i));
          double k=ph-ang.getVal(x,y);

          //Insert into the table
          int i=(int)round((beta+(M_PI/2))/D);

          model.invRTable[i].push_back(k);

        }
      }
    }

  itsModels.push_back(model);
}


Point2D<int> GHough::findInvFeature(const int x, const int y, const Image<float>&img, const Image<float>& ang)
{
  double alpha = M_PI/4;
  //Find a second feature point
  Point2D<int> w2(-1,-1);
  double phi = ang.getVal(x,y) + M_PI/2;
  double m = tan(phi-alpha);

  //Follow a line (in bouth direction of x,y) 
  //and find the next feature from that location
  if (m>-1 && m<1)
  {
    for(int i=3; i<img.getWidth(); i++)
    {
      int c = x + i;
      int j=(int)round(m*(x-c)+y);
      Point2D<int> loc(c,j);

      if (img.coordsOk(loc) && img.getVal(loc) > 0)
      {
        w2 = loc; //We found the feature
        break;
      } else {
        //Look in the other direction
        c = x - i;
        j=(int)round(m*(x-c)+y);
        loc = Point2D<int>(c,j);
        if (img.coordsOk(loc) && img.getVal(loc) > 0)
        {
          w2 = loc; //We found the feature
          break;
        }
      }
    }
  } else {
    for(int i=3; i<img.getHeight(); i++)
    {
      int c = y + i;
      int j=(int)round(x+(y-c)/m);
      Point2D<int> loc(j,c);

      if (img.coordsOk(loc) && img.getVal(loc) > 0)
      {
        w2 = loc; //We found the feature
        break;
      } else {
        //Look in the other direction
        c = y - i;
        j=(int)round(x+(y-c)/m);
        loc = Point2D<int>(j,c);
        if (img.coordsOk(loc) && img.getVal(loc) > 0)
        {
          w2 = loc; //We found the feature
          break;
        }
      }
    }
  }

  return w2;
}

Image<float> GHough::getInvVotes(const Image<byte>& img,
    const Image<float>& ang)
{
  int entries = 5;
  double D=M_PI/entries;

  Image<float> acc(img.getDims(), ZEROS);

  for(int y=0; y<img.getHeight(); y++)
    for(int x=0; x<img.getWidth(); x++)
    {
      if (img.getVal(x,y) > 0)
      {
        Point2D<int> w2 = findInvFeature(x,y,img, ang);

        if (w2.isValid())
        {
          //Compute beta
          double phi = tan(ang.getVal(x,y));
          double phj = tan(ang.getVal(w2));

          double beta=1.57;
          if ((1+phi*phj) != 0)
            beta=atan((phi-phj)/(1+phi*phj));

          //Read from rTable
          int i=(int)round((beta+(M_PI/2))/D);

          //Search for k
          std::map<int, std::vector<double> >::iterator iter =
            itsModels[0].invRTable.find(i);

          for(uint j=0; j<iter->second.size(); j++)
          {
            float k=iter->second[j];
            //lines of votes
            float m=tan(k+ang.getVal(x,y));
            if (m>-1 && m<1)
            {
              for(int x0=1; x0<img.getWidth(); x0++)
              {
                int y0=(int)round(y+m*(x0-x));
                if(y0>0 && y0<img.getHeight())
                  acc.setVal(x0,y0, acc.getVal(x0,y0)+1);
              }
            } else {
              for(int y0=0; y0<img.getHeight(); y0++)
              {
                int x0=(int)round(x+(y0-y)/m);
                if(x0>0 && x0<img.getWidth())
                  acc.setVal(x0,y0, acc.getVal(x0,y0)+1);
              }
            }
          }

        }
      }
    }

  return acc;
}


void GHough::writeTable(const char* filename)
{
  int fd;

  if ((fd = creat(filename, 0644)) == -1)
    LFATAL("Can not open %s for saving\n", filename);

  //Write the Dims of the table
  size_t numModels = itsModels.size();
  int ret = write(fd, (char *) &numModels, sizeof(size_t));

  for(uint i=0; i<numModels; i++)
  {
    ret = write(fd, (char *) &itsModels[i].id, sizeof(int));
    ret = write(fd, (char *) &itsModels[i].type, sizeof(int));
    ret = write(fd, (char *) &itsModels[i].pos, sizeof(Point3D<float>));
    ret = write(fd, (char *) &itsModels[i].rot, sizeof(Point3D<float>));
    ret = write(fd, (char *) &itsModels[i].imgPos, sizeof(Point2D<float>));
    ret = write(fd, (char *) &itsModels[i].numFeatures, sizeof(int));


    size_t numTables = itsModels[i].rTables.size();
    ret = write(fd, (char *) &numTables, sizeof(size_t));

    for(uint j=0; j<numTables; j++)
    {
      //Size of each table
      RTable& rTable = itsModels[i].rTables[j];

      //Entries
      size_t numEntries = rTable.entries.size();
      ret = write(fd, (char *) &numEntries, sizeof(size_t));

      std::map<int, std::vector<Point2D<float> > >::const_iterator iter;
      for(iter = rTable.entries.begin(); iter != rTable.entries.end(); iter++)
      {
        ret = write(fd, (char *) &iter->first, sizeof(int));

        size_t numPos = iter->second.size();
        ret = write(fd, (char *) &numPos, sizeof(size_t));

        for(uint k=0; k<numPos; k++)
          ret = write(fd, (char *) &iter->second[k], sizeof(Point2D<float>));
      }

      //Feature Entries
      size_t numFeatureEntries = rTable.featureEntries.size();
      ret = write(fd, (char *) &numFeatureEntries, sizeof(size_t));

      std::map<long, std::vector<Feature> >::const_iterator fiter;
      for(fiter = rTable.featureEntries.begin(); fiter != rTable.featureEntries.end(); fiter++)
      {
        ret = write(fd, (char *) &fiter->first, sizeof(long));

        size_t numFeatures = fiter->second.size();
        ret = write(fd, (char *) &numFeatures, sizeof(size_t));

        for(uint k=0; k<numFeatures; k++)
        {
          ret = write(fd, (char *) &fiter->second[k].loc, sizeof(Point2D<float>));

          size_t numValues = fiter->second[k].values.size();
          ret = write(fd, (char *) &numValues, sizeof(size_t));
          for(uint ii=0; ii<numValues; ii++)
            ret = write(fd, (char *) &fiter->second[k].values[ii], sizeof(float));
        }
      }

    }
  }

  close(fd);
}

void GHough::readTable(const char* filename)
{
  int fd;
  if ((fd = open(filename, 0, 0644)) == -1) return;

  LINFO("Reading from %s", filename);

  //Write the Dims of the table
  size_t numModels;
  int ret = read(fd, (char *) &numModels, sizeof(size_t));


  itsModels.clear();

  for(uint i=0; i<numModels; i++)
  {
    Model model;
    ret = read(fd, (char *) &model.id, sizeof(int));
    ret = read(fd, (char *) &model.type, sizeof(int));
    ret = read(fd, (char *) &model.pos, sizeof(Point3D<float>));
    ret = read(fd, (char *) &model.rot, sizeof(Point3D<float>));
    ret = read(fd, (char *) &model.imgPos, sizeof(Point2D<float>));
    ret = read(fd, (char *) &model.numFeatures, sizeof(int));

    size_t numTables;
    ret = read(fd, (char *) &numTables, sizeof(size_t));

    int totalFeatures = 0;

    for(uint j=0; j<numTables; j++)
    {
      //Size of each table

      RTable rTable;

      //features
      size_t numEntries;
      ret = read(fd, (char *) &numEntries, sizeof(size_t));
      for(uint k=0; k<numEntries; k++)
      {
        int key;
        ret = read(fd, (char *) &key, sizeof(int));

        size_t numPos;
        ret = read(fd, (char *) &numPos, sizeof(size_t));

        for(uint ii=0; ii<numPos; ii++)
        {
          Point2D<float> loc;
          ret = read(fd, (char *) &loc, sizeof(Point2D<float>));
          rTable.entries[key].push_back(loc);
        }
      }

      //Feature Entries
      size_t numFeatureEntries;
      ret = read(fd, (char *) &numFeatureEntries, sizeof(size_t));
      for(uint k=0; k<numFeatureEntries; k++)
      {
        long key;
        ret = read(fd, (char *) &key, sizeof(long));

        size_t numFeatures;
        ret = read(fd, (char *) &numFeatures, sizeof(size_t));

        for(uint ii=0; ii<numFeatures; ii++)
        {
          Feature f;
          ret = read(fd, (char *) &f.loc, sizeof(Point2D<float>));
          totalFeatures++;

          size_t numValues;
          ret = read(fd, (char *) &numValues, sizeof(size_t));

          for(uint jj=0; jj<numValues; jj++)
          {
            float value;
            ret = read(fd, (char *) &value, sizeof(float));
            f.values.push_back(value);
          }

          rTable.featureEntries[key].push_back(f);
        }
      }

      model.rTables.push_back(rTable);
    }
    itsModels.push_back(model);
  }
  LINFO("Added %i models", (uint)itsModels.size());
  close(fd);
}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

