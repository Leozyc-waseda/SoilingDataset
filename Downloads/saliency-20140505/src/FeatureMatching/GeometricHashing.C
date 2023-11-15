/*!@file SceneUnderstanding/GeometricHashing.C  Shape from shading */


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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/FeatureMatching/GeometricHashing.C $
// $Id: GeometricHashing.C 12962 2010-03-06 02:13:53Z irock $
//

#ifndef GeometricHashing_C_DEFINED
#define GeometricHashing_C_DEFINED

#include "FeatureMatching/GeometricHashing.H"
#include "Image/DrawOps.H"
#include "GUI/DebugWin.H"
#include <fcntl.h>

// ######################################################################
GeometricHashing::GeometricHashing() :
  itsTableWidth(0.10),
  itsNumFeatures(0),
  itsNumBinEntries(0)
{
  itsHashTable = Image<TableEntry>(32,32,ZEROS);
  itsTableWidth = 1/32.0;
}

// ######################################################################
GeometricHashing::~GeometricHashing()
{

}

//void GeometricHashing::addModel(const Model& model)
//{
//  itsModels.push_back(model);
//
//  //uint p1 = 0;
//  //uint p2 = 1;
//  for(uint p1 = 0; p1 < model.v.size(); p1++)
//    for(uint p2 = p1; p2 < model.v.size(); p2++)
//    {
//      if (p1 == p2)
//        continue;
//      //Change basis
//      std::vector<Point2D<float> > featureLoc = changeBasis(model.v, p1, p2);
//
//      //Add entries to hash table
//      for(uint i=0; i<featureLoc.size(); i++)
//      {
//  //      LINFO("%i %fx%f",
//  //          i, featureLoc[i].i, featureLoc[i].j);
//        //Dont insert the basis points, since they are always at the same place
//        if (i != p1 && i != p2)
//          insertToHashTable(featureLoc[i], p1, p2, model.id, model.rot);
//      }
//    }
//}

void GeometricHashing::addModel(const Model& model)
{
  itsModels.push_back(model);

  //Find the minimum bounding box
  Point2D<int> tl = model.v[0];
  Point2D<int> br = model.v[0];
  for(uint i=0; i<model.v.size(); i++)
  {
    if (br.i > model.v[i].i)
      br.i = model.v[i].i;
    if (br.j > model.v[i].j)
      br.j = model.v[i].j;

    if (tl.i < model.v[i].i)
      tl.i = model.v[i].i;
    if (tl.j < model.v[i].j)
      tl.j = model.v[i].j;
  }

  std::vector<Point2D<float> > featureLoc = changeBasis(model.v, tl, br);
  for(uint i=0; i<featureLoc.size(); i++)
    insertToHashTable(featureLoc[i], 0, 0, model.id, model.rot);

}

void GeometricHashing::insertToHashTable(const Point2D<float> loc,
    int p1, int p2, int modelId,
    Point3D<float> rot)
{

  ModelTableEntry mte;
  mte.modelId = modelId;
  mte.rot = rot;
  mte.featureLoc = loc;
  mte.basisP1 = p1;
  mte.basisP2 = p2;

  Image<GeometricHashing::TableEntry>::iterator iter = findInHash(loc);
  if (iter != NULL)
    iter->modelEntries.push_back(mte);
}


Image<GeometricHashing::TableEntry>::iterator
  GeometricHashing::findInHash(const Point2D<float>& loc)
{
  //Find the table location;
  int i = (itsHashTable.getWidth()/2) + int( floor(loc.i / itsTableWidth) );
  int j = (itsHashTable.getHeight()/2) + int( floor(loc.j / itsTableWidth) );

  Image<GeometricHashing::TableEntry>::iterator  iter = NULL;

  if (itsHashTable.coordsOk(i,j))
  {
    iter = itsHashTable.beginw();
    iter += (i + j*itsHashTable.getWidth());
  }
  return iter;
}

//std::vector<GeometricHashing::Acc> GeometricHashing::getVotes(std::vector<Point2D<int> >& input)
//{
//  std::vector<Acc> acc;
//
//  //Choose two points from the image to become the basis
//  for(uint p1 = 0; p1 < input.size(); p1++)
//    for(uint p2 = p1; p2 < input.size(); p2++)
//    {
//      if (p1 == p2)
//        continue;
//      //Change basis of input
//      std::vector<Point2D<float> > newInput = changeBasis(input, p1, p2);
//
//      //For each input find the value in the hash table and count the votes
//      for(uint i=0; i<newInput.size(); i++)
//      {
//        Image<GeometricHashing::TableEntry>::iterator iter =
//          findInHash(newInput[i]);
//        if (iter != NULL)
//        {
//          for(uint j=0; j<iter->modelEntries.size(); j++)
//          {
//            ModelTableEntry mte = iter->modelEntries[j];
//
//            //find the acc;
//            bool found = false;
//            for(uint i=0; i<acc.size(); i++)
//            {
//              if (acc[i].P1 == mte.basisP1 &&
//                  acc[i].P2 == mte.basisP2 &&
//                  acc[i].modelId == mte.modelId &&
//                  acc[i].rot == mte.rot &&
//                  acc[i].inputP1 == p1 &&
//                  acc[i].inputP2 == p2)
//              {
//                found = true;
//                acc[i].inputP1 = p1;
//                acc[i].inputP2 = p2;
//                acc[i].votes++;
//              }
//            }
//
//            if (!found)
//            {
//              Acc newAcc;
//              newAcc.P1 = mte.basisP1;
//              newAcc.P2 = mte.basisP2;
//              newAcc.modelId = mte.modelId;
//              newAcc.rot = mte.rot;
//              newAcc.votes = 1;
//              newAcc.inputP1 = p1;
//              newAcc.inputP2 = p2;
//              acc.push_back(newAcc);
//            }
//          }
//        }
//      }
//    }
//
//  return acc;
//}

std::vector<GeometricHashing::Acc> GeometricHashing::getVotes(std::vector<Point2D<int> >& input)
{
  std::vector<Acc> acc;

  if (input.size() <= 0)
    return acc;

  //Find the minimum bounding box
  Point2D<int> tl = input[0];
  Point2D<int> br = input[0];
  for(uint i=0; i<input.size(); i++)
  {
    if (br.i > input[i].i)
      br.i = input[i].i;
    if (br.j > input[i].j)
      br.j = input[i].j;

    if (tl.i < input[i].i)
      tl.i = input[i].i;
    if (tl.j < input[i].j)
      tl.j = input[i].j;
  }

  std::vector<Point2D<float> > newInput = changeBasis(input, tl, br);

  Point2D<float> center(tl.i+(br.i - tl.i)/2,
                        tl.j+(br.j - tl.j)/2);


  //For each input find the value in the hash table and count the votes
  for(uint i=0; i<newInput.size(); i++)
  {
    Image<GeometricHashing::TableEntry>::iterator iter =
      findInHash(newInput[i]);
    if (iter != NULL)
    {
      for(uint j=0; j<iter->modelEntries.size(); j++)
      {
        ModelTableEntry mte = iter->modelEntries[j];

        //find the acc;
        bool found = false;
        for(uint i=0; i<acc.size(); i++)
        {
          if (acc[i].modelId == mte.modelId &&
              acc[i].rot == mte.rot)
          {
            found = true;
            acc[i].votes++;
          }
        }

        if (!found)
        {
          Acc newAcc;
          newAcc.modelId = mte.modelId;
          newAcc.rot = mte.rot;
          newAcc.votes = 1;
          newAcc.center = center;
          acc.push_back(newAcc);
        }
      }
    }
  }

  return acc;
}

Image<PixRGB<byte> > GeometricHashing::getHashTableImage()
{

  int binWidth = 10;
  Image<PixRGB<byte> > img(itsHashTable.getDims()*binWidth, ZEROS);
  drawGrid(img, binWidth, binWidth, 1, 1, PixRGB<byte>(0,0,255));
  drawLine(img, Point2D<int>(0, img.getHeight()/2),
                Point2D<int>(img.getWidth(), img.getHeight()/2),
                PixRGB<byte>(0,0,255),2);
  drawLine(img, Point2D<int>(img.getWidth()/2, 0),
                Point2D<int>(img.getWidth()/2, img.getHeight()),
                PixRGB<byte>(0,0,255),2);

  float scale = binWidth/itsTableWidth;
  for(uint i=0; i<itsHashTable.size(); i++)
  {
    TableEntry te = itsHashTable[i];
    for(uint j=0; j<te.modelEntries.size(); j++)
    {
      ModelTableEntry mte = te.modelEntries[j];
      int x = (img.getWidth()/2) + (int)(scale*mte.featureLoc.i);
      int y = (img.getHeight()/2) - (int)(scale*mte.featureLoc.j);
      drawCircle(img, Point2D<int>(x,y), 3, PixRGB<byte>(0,255,0));
    }
  }

  return img;

}


std::vector<Point2D<float> > GeometricHashing::changeBasis(
    const std::vector<Point2D<int> >& featureLoc,
    int p1, int p2)
{

  std::vector<Point2D<float> > newFeatureLoc;

  float modelScale = 1/sqrt( squareOf(featureLoc[p2].i - featureLoc[p1].i) +
                             squareOf(featureLoc[p2].j - featureLoc[p1].j) );

  float ang = 0;
  if ((featureLoc[p2].i - featureLoc[p1].i) != 0)
    ang = -atan((featureLoc[p2].j - featureLoc[p1].j)/(featureLoc[p2].i - featureLoc[p1].i));

  //Rotate the model
  for(uint i=0; i<featureLoc.size(); i++)
  {
    float x = modelScale*(featureLoc[i].i);
    float y = modelScale*(featureLoc[i].j);

    newFeatureLoc.push_back(
        Point2D<float>((x * cos(ang) - y * sin(ang)),
                       (y * cos(ang) + x * sin(ang))));
  }

  //Offset so the midpoint is between points center
  Point2D<float> center(newFeatureLoc[p1].i+(newFeatureLoc[p2].i - newFeatureLoc[p1].i)/2,
                        newFeatureLoc[p1].j+(newFeatureLoc[p2].j - newFeatureLoc[p1].j)/2);
  for(uint i=0; i<newFeatureLoc.size(); i++)
    newFeatureLoc[i] -= center;


  return newFeatureLoc;
}

std::vector<Point2D<float> > GeometricHashing::changeBasis(
    const std::vector<Point2D<int> >& featureLoc,
    Point2D<int> tl,
    Point2D<int> br)
{

  std::vector<Point2D<float> > newFeatureLoc;

  //Offset so the midpoint is between points center
  Point2D<float> center(tl.i+(br.i - tl.i)/2,
                        tl.j+(br.j - tl.j)/2);

  float modelScale = 1/sqrt( squareOf(br.i - tl.i) +
                             squareOf(br.j - tl.j) );

  //Transelate the model to 0,0 and scale it
  float ang = 0;
  for(uint i=0; i<featureLoc.size(); i++)
  {
    float x = modelScale*(featureLoc[i].i - center.i);
    float y = modelScale*(featureLoc[i].j - center.j);

    newFeatureLoc.push_back(
        Point2D<float>((x * cos(ang) - y * sin(ang)),
                       (y * cos(ang) + x * sin(ang))));
  }

  return newFeatureLoc;
}

void GeometricHashing::writeTable(const char* filename)
{
  int fd;

  if ((fd = creat(filename, 0644)) == -1)
    LFATAL("Can not open %s for saving\n", filename);

  //Write the Dims of the table
  int ret;
  int width = itsHashTable.getWidth();
  int height = itsHashTable.getHeight();
  ret = write(fd, (char *) &width, sizeof(int));
  ret = write(fd, (char *) &height, sizeof(int));

  //Write each entry
  for(int j=0; j<height; j++)
    for(int i=0; i<width; i++)
    {
      GeometricHashing::TableEntry te = itsHashTable.getVal(i,j);
      //Write the size of the model entries
      size_t numEntries = te.modelEntries.size();
      ret = write(fd, (char *) &numEntries, sizeof(size_t));
      //Write the entries
      for(uint idx=0; idx<numEntries; idx++)
      {
        ModelTableEntry mte = te.modelEntries[idx];
        ret = write(fd, (char *) &mte, sizeof(ModelTableEntry));
      }
    }
}

void GeometricHashing::readTable(const char* filename)
{
  int fd;
  if ((fd = open(filename, 0, 0644)) == -1) return;

  int ret;
  LINFO("Reading from %s", filename);

  //Write the Dims of the table
  int width;
  int height;
  ret = read(fd, (char *) &width, sizeof(int));
  ret = read(fd, (char *) &height, sizeof(int));
  LINFO("Width %i %i", width, height);

  itsHashTable = Image<TableEntry>(width,height,ZEROS);

  //Write each entry
  for(int j=0; j<height; j++)
    for(int i=0; i<width; i++)
    {
      //Write the size of the model entries
      size_t numEntries;
      ret = read(fd, (char *) &numEntries, sizeof(size_t));

      //Write the entries
      for(uint idx=0; idx<numEntries; idx++)
      {
        ModelTableEntry mte;
        ret = read(fd, (char *) &mte, sizeof(ModelTableEntry));

        Image<GeometricHashing::TableEntry>::iterator  iter = NULL;
        iter = itsHashTable.beginw();
        iter += (i + j*itsHashTable.getWidth());
        iter->modelEntries.push_back(mte);
      }
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

