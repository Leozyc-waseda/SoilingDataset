/*!@file TestSuite/TestBrain.C Test Brain for object rec code */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/ObjRec/RandomObjRecBrain.C $
// $Id: RandomObjRecBrain.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Image/Image.H"
#include "Image/ColorOps.H"
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

struct ObjectDBData
{
  char name[255];
};

class RandomBrain : public ObjRecBrain
{

  public:
  RandomBrain(std::string dbFile) :
    itsObjectsDBFile(dbFile)
  {
    initRandomNumbersZero(); //To have the same output all the time
  }

  ~RandomBrain()
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
    while(1)
    {
      ObjectDBData obj;
      int ret = fread(&obj, 1, sizeof(ObjectDBData), itsFP);
      if (ret == 0)
        break;
      itsObjects.push_back(obj.name);
    }
    fclose(itsFP);
  }

  ObjectData onRecognition(Image<PixRGB<byte> > &img)
  {

    ObjectData obj;
    if (itsObjects.size() > 0)
      obj.name = itsObjects.at(randomUpToNotIncluding(itsObjects.size()));
    else
      obj.name = "unknown";

    obj.confidence = 100;

    return obj;
  }

  void postRecognition()
  {
  }

  private:
    FILE* itsFP;
    std::vector<std::string> itsObjects;
    std::string itsObjectsDBFile;

};

//Create and destory the brain
extern "C" ObjRecBrain* createObjRecBrain(std::string dbFile)
{
  return new RandomBrain(dbFile);
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
