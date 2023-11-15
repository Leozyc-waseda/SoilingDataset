/*!@file NeovisionII/ColorObjRecBrain.C A mean RGB color classifier */

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
// Primary maintainer for this file: Lior Elazary
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/ObjRec/ColorObjRecBrain.C $
// $Id: ColorObjRecBrain.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Image/Image.H"
#include "Image/ColorOps.H"
#include "Image/Pixels.H"
#include "Image/CutPaste.H"
#include "Image/ShapeOps.H"
#include "Image/Rectangle.H"
#include "Image/MathOps.H"
#include "Image/MatrixOps.H"
#include "Image/Transforms.H"
#include "Image/Convolutions.H"
#include "nub/ref.h"
#include "Util/Types.H"
#include "Util/MathFunctions.H"
#include "Util/log.H"
#include "TestSuite/ObjRecBrain.h"


#include <stdio.h>
#include <vector>
#include <string>
#include <limits>

struct ObjectDBData
{
  char name[255];
  PixRGB<float> meanRGB;

};

struct ObjectRecData
{
  std::string name;
  PixRGB<float> meanRGB;
};

float distanceSq(PixRGB<float>& c1, PixRGB<float>& c2)
{
  return (squareOf((c1[0] - c2[0])) +
      squareOf((c1[1] - c2[1])) +
      squareOf((c1[2] - c2[2])));
}


class ColorBrain : public ObjRecBrain
{

  public:
  ColorBrain(std::string dbFile) :
    itsObjectsDBFile(dbFile)
  {
  }

  ~ColorBrain()
  {
  }

  void preTraining()
  {
    itsFP = fopen(itsObjectsDBFile.c_str(), "wb");
    if (itsFP == NULL)
      LFATAL("Error loading data file");
  }

  void onTraining(Image<PixRGB<byte> > &img, ObjectData& objData)
  {
    ObjectDBData obj;
    strcpy(obj.name, objData.name.c_str());
    //Use the mean RGB as the feature value
    obj.meanRGB = meanRGB(img);


    fwrite(&obj, 1, sizeof(ObjectDBData), itsFP);

  };
  void postTraining()
  {
    fclose(itsFP);
  }


  void preRecognition()
  {
    itsFP = fopen(itsObjectsDBFile.c_str(), "r");
    if (itsFP == NULL)
      LFATAL("Error loading data file");

    itsObjects.clear();
    ObjectDBData obj;
    while( fread(&obj, 1, sizeof(ObjectDBData), itsFP) > 0)
    {
      ObjectRecData objData;
      objData.name = obj.name;
      objData.meanRGB = obj.meanRGB;
      itsObjects.push_back(objData);
    }
    fclose(itsFP);
  }

  ObjectData onRecognition(Image<PixRGB<byte> > &img)
  {

    ObjectData obj;
    if (itsObjects.size() > 0)
    {
      PixRGB<float> fv = meanRGB(img);

      //A simple NN search
      //Get the ratio of the min distance between the first nearest object
      //and the second different object
      std::string firstObj = itsObjects[0].name;
      float minDist1 = distanceSq(itsObjects[0].meanRGB, fv);
      for(uint i=1; i<itsObjects.size(); i++)
      {
        float dist = distanceSq(itsObjects[i].meanRGB, fv);
        if (dist < minDist1)
        {
          minDist1 = dist;
          firstObj = itsObjects[i].name;
        }
      }

      //Find the score for the next best object
      std::string secondObj;
      float minDist2 = std::numeric_limits<float>::max();

      for(uint i=0; i<itsObjects.size(); i++)
      {
        if (itsObjects[i].name != firstObj)
        {
          float dist = distanceSq(itsObjects[i].meanRGB, fv);
          if (dist < minDist2)
          {
            minDist2 = dist;
            secondObj = itsObjects[i].name;
          }
        }
      }

      obj.name = firstObj;
      obj.confidence = minDist2/(minDist1+2.22044604925031e-16); //avoid divide by 0 (eps from matlab)

      //LINFO("fv (%f,%f,%f) first best=%s %f : second best=%s %f confidence=%f",
      //    fv[0], fv[1], fv[2],
      //    firstObj.c_str(), minDist1,
      //    secondObj.c_str(), minDist2,
      //    obj.confidence);
    } else {
      obj.confidence = -1;
      obj.name = "unknown";
    }

    return obj;
  }

  void postRecognition()
  {
  }

  private:

    FILE* itsFP;
    std::vector<ObjectRecData> itsObjects;
    std::string itsObjectsDBFile;

};

//Create and destory the brain
extern "C" ObjRecBrain* createObjRecBrain(std::string dbFile)
{
  return new ColorBrain(dbFile);
}

extern "C" void destoryObjRecBrain(ObjRecBrain* brain)
{
  delete brain;
}


int main(const int argc, const char **argv)
{
  LFATAL("Use test-ObjRec");
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
