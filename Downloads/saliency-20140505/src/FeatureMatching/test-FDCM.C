/*! @file SceneUnderstanding/test-FDCM.C Test the FDCM */

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
// $HeadURL: $
// $Id: $
//

#include "Image/Image.H"
#include "Component/ModelManager.H"
#include "Raster/Raster.H"
#include "GUI/DebugWin.H"
#include "FeatureMatching/GHough.H"
#include "FeatureMatching/OriChamferMatching.H"
#include "Image/Rectangle.H"
#include "Image/FilterOps.H"

#include <signal.h>
#include <sys/types.h>

int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager manager("Test FDCM");

  if (manager.parseCommandLine(
        (const int)argc, (const char**)argv, "", 0, 0) == false)
    return 1;

  manager.start();

  //LMLineMatcher lineMatcher;
  //lineMatcher.Configure("para_line_matcher.txt");
  

  //Generate a random image
  for(float ori=360; ori>10; ori-=10)
  {
    Image<PixRGB<byte> > inputImg(320,240,ZEROS);
    Point2D<int> pos(100,100);
    float len=50;
    float x1 = cos(ori*M_PI/180)*len/2;
    float y1 = sin(ori*M_PI/180)*len/2;

    Point2D<float> p1(pos.i-x1, pos.j+y1);
    Point2D<float> p2(pos.i+x1, pos.j-y1);

    drawLine(inputImg, (Point2D<int>)p1, (Point2D<int>)p2, PixRGB<byte>(255,0,0));


    //Build the FDCM
    std::vector<Line> lines;
    lines.push_back(Line(p1,p2));
    OriChamferMatching matcher(lines,
        60, //Num Orientations
        2.5, //Direction cost
        inputImg.getDims());
    
  
    //lineMatcher.computeIDT3(inputImg.getWidth(), inputImg.getHeight(),
    //    lines.size(), &lines[0]);

    //Build the template model and match

    Image<float> costImg(inputImg.getDims(), NO_INIT);
    costImg.clear(1e10);
    for(int y=0; y<inputImg.getHeight(); y++)
    {
      for(int x=0; x<inputImg.getWidth(); x++)
      {
        for(int ori=0; ori < 360; ori++)
        {
          Point2D<int> pos(x,y);
          float len=50;
          float x1 = cos((ori)*M_PI/180)*len/2;
          float y1 = sin((ori)*M_PI/180)*len/2;

          Point2D<float> p1(pos.i-x1, pos.j+y1);
          Point2D<float> p2(pos.i+x1, pos.j-y1);

          Polygon model;
          model.addLine(Line(p1,p2));
          model.quantize(60);

          for(uint i=0; i<model.getNumLines(); i++)
          {
            Line l = model.getLine(i);

            float sum = matcher.getCost(l.getDirectionIdx(), Point2D<int>(l.getP1()), Point2D<int>(l.getP2()));
            //float sum2 = matcher.getCostFast(l.getDirectionIdx(), Point2D<int>(l.getP1()), Point2D<int>(l.getP2()));
            //LINFO("Line %i cost %f %f",i, sum, costImg.getVal(x,y));
            if (sum < costImg.getVal(x,y))
              costImg.setVal(x,y,sum);
          }
        }
      }
      LINFO("Y %i", y);
    }
    SHOWIMG(costImg);


    //SHOWIMG(inputImg);
  }

    


  manager.stop();

  return 0;
}

