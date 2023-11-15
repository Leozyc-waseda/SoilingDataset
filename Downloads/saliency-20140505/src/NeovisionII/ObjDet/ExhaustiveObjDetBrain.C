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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/ObjDet/ExhaustiveObjDetBrain.C $
// $Id: ExhaustiveObjDetBrain.C 12962 2010-03-06 02:13:53Z irock $
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
#include "TestSuite/ObjDetBrain.h"


#include <stdio.h>
#include <vector>
#include <string>

struct ObjectDBData
{
  char name[255];
};

class ExhaustiveBrain : public ObjDetBrain
{

  public:
  ExhaustiveBrain( const int argc, const char **argv)
  {
  }

  ~ExhaustiveBrain()
  {
  }

  void preTraining()
  {
  }

  void onTraining(Image<PixRGB<byte> > &img, ObjectData& objData)
  {
  }

  void postTraining()
  {
  }


  void preDetection()
  {
  }

  std::vector<DetLocation> onDetection(Image<PixRGB<byte> > &img)
  {

    std::vector<DetLocation> smap;

    for(int j=0; j<img.getHeight(); j++)
      for(int i=0; i<img.getWidth(); i++)
      {
        float val = img.size() - ((j*img.getWidth()) + i);
        smap.push_back(DetLocation(i,j, val));
      }

    return smap;
  }

  void postDetection()
  {
  }

  private:

};

//Create and destory the brain
extern "C" ObjDetBrain* createObjDetBrain( const int argc, const char **argv)
{
  return new ExhaustiveBrain(argc, argv);
}

extern "C" void destoryObjDetBrain(ObjDetBrain* brain)
{
  delete brain;
}


int main(const int argc, const char **argv)
{
  LFATAL("Use test-ObjDet");
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
