/*! @file SceneUnderstanding/test-GHough.C Test the GHough */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/FeatureMatching/test-GHough.C $
// $Id: test-GHough.C 13821 2010-08-24 00:30:37Z lior $
//

#include "Image/Image.H"
#include "Component/ModelManager.H"
#include "Raster/Raster.H"
#include "GUI/DebugWin.H"
#include "FeatureMatching/GHough.H"
#include "Image/Rectangle.H"
#include "Image/FilterOps.H"

#include <signal.h>
#include <sys/types.h>

int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager manager("Test GHough");

  if (manager.parseCommandLine(
        (const int)argc, (const char**)argv, "", 0, 0) == false)
    return 1;

  manager.start();

  GHough gHough;


  Image<PixRGB<byte> > car = Raster::ReadRGB("car.pnm");

  std::vector<Point2D<int> > carTempl;
  carTempl.push_back(Point2D<int>(151,105));
  carTempl.push_back(Point2D<int>(222,100));
  carTempl.push_back(Point2D<int>(240,112));
  carTempl.push_back(Point2D<int>(273,113));
  carTempl.push_back(Point2D<int>(279,133));
  carTempl.push_back(Point2D<int>(270,135));
  carTempl.push_back(Point2D<int>(265,122));
  carTempl.push_back(Point2D<int>(247,123));
  carTempl.push_back(Point2D<int>(241,137));
  carTempl.push_back(Point2D<int>(189,140));
  carTempl.push_back(Point2D<int>(182,129));
  carTempl.push_back(Point2D<int>(163,130));
  carTempl.push_back(Point2D<int>(157,143));
  carTempl.push_back(Point2D<int>(141,144));
  carTempl.push_back(Point2D<int>(143,123));
  carTempl.push_back(Point2D<int>(151,106));


  //Center the template
  Point2D<int> center(0,0);
  for(uint i=0; i<carTempl.size(); i++)
    center += carTempl[i];
  center /= carTempl.size();

  for(uint i=0; i<carTempl.size(); i++)
    carTempl[i] -= center;


  drawOutlinedPolygon(car, carTempl, PixRGB<byte>(0,255,0),
      Point2D<int>(207,124), 2*M_PI/180, 1);
  SHOWIMG(car);


  gHough.addModel(1, carTempl);

  Image<float> inCar = luminance(car);
  SHOWIMG(inCar);

  gHough.getVotes(inCar);


  //Image<float> modelImage(320,240,ZEROS); //The input image

  ////Draw a shape in the image
  //drawSuperquadric(modelImage,
  //    Point2D<int>(150,100),
  //    20, //shape width
  //    40, //shape height 
  //    0.8,  //shape param
  //    255.0F, //color
  //    0, //angle
  //    0, //sheer x
  //    0, //sheer y
  //    -M_PI, //start
  //    M_PI, //end
  //    20);

  ////Learn the shape
  //SHOWIMG(modelImage);
  //gHough.addModel(1, modelImage);


  ////FInd the shape
  //Image<float> inImage(320,240,ZEROS); //The input image

  ////Draw a shape in the image
  //drawSuperquadric(inImage,
  //    Point2D<int>(100,70),
  //    10, //shape width
  //    20, //shape height 
  //    0.8,  //shape param
  //    255.0F, //color
  //    45*M_PI/180, //angle
  //    0, //sheer x
  //    0, //sheer y
  //    -M_PI, //start
  //    M_PI, //end
  //    20);

  ////drawSuperquadric(inImage,
  ////    Point2D<int>(250,70),
  ////    10, //shape width
  ////    20, //shape height 
  ////    2.8,  //shape param
  ////    255.0F, //color
  ////    0, //angle
  ////    0, //sheer x
  ////    0, //sheer y
  ////    -M_PI, //start
  ////    M_PI, //end
  ////    20);

  ////Learn the shape
  //SHOWIMG(inImage);
  //gHough.getVotes(inImage);

  ////Create a model object
  //Image<float> tmp(320,240, ZEROS);
  //Image<float> mag, ori;


  //for(uint i=0; i<10; i++)
  //{
  //    tmp.clear();
  //    int d = int((i/10.0F)*50);
  //    drawRectOR(tmp, Rectangle(Point2D<int>(128,128), Dims(d, d)),
  //      1.0F, d, 0);
  //    gradientSobel(tmp, mag, ori);
  //    ghough.addModel(i, mag, ori);
  //}

  //for(uint i=0; i<10; i++)
  //{
  //    tmp.clear();
  //    int d = int((i/10.0F)*50);
  //    drawDisk(tmp, Point2D<int>(128,128), d, 1.0F);
  //    gradientSobel(tmp, mag, ori);
  //    ghough.addModel(20+i, mag, ori);
  //}

  //ghough.writeTable("test.dat");
  //ghough.readTable("test.dat");



  //for(int x=50; x<200; x+=10)
  //  for(int y=50; y<200; y+=10)
  //    for(int rot=0; rot<1; rot+=10)
  //  {
  //    Image<float> img(320,240, ZEROS);
  //    drawRectOR(img, Rectangle(Point2D<int>(x,y), Dims(15, 15)),
  //          1.0F, 15, rot);

  //    drawDisk(img, Point2D<int>(100,100), 30, 1.0F);

  //    //Draw some random lines
  //    for(uint i=0; i<50; i++)
  //    {
  //      Point2D<int> p1(randomUpToIncluding(319), randomUpToIncluding(239));
  //      Point2D<int> p2(randomUpToIncluding(319), randomUpToIncluding(239));
  //      drawLine(img, p1, p2, 1.0F);
  //    }

  //    SHOWIMG(img);

  //    Image<float> mag, ori;
  //    gradientSobel(img, mag, ori);

  //    std::vector<GHough::Acc> acc = ghough.getVotes(mag, ori);
  //    Image<float> accImg(mag.getDims(), ZEROS);
  //    for(uint i=0; i<acc.size(); i++)
  //      if (accImg.coordsOk(acc[i].pos))
  //        accImg.setVal(acc[i].pos, accImg.getVal(acc[i].pos) + acc[i].votes);

  //    SHOWIMG(accImg);

  //    inplaceNormalize(img, 0.0F, 255.0F);

  //    Image<PixRGB<byte> > results = img;
  //    for(uint i=0; i<10; i++)
  //    { //Show the top 10
  //      drawCircle(results, acc[i].pos, 3, PixRGB<byte>(0,255,0));
  //      LINFO("%i: Max is %i",i, acc[i].id);
  //    }
  //    SHOWIMG(results);

  //  }


  manager.stop();

  return 0;
}

