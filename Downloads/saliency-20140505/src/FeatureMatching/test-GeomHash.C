/*! @file SceneUnderstanding/test-GeomHash */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/FeatureMatching/test-GeomHash.C $
// $Id: test-GeomHash.C 12962 2010-03-06 02:13:53Z irock $
//

#include "FeatureMatching/GeometricHashing.H"
#include "Image/Point2D.H"
#include "GUI/DebugWin.H"
#include "Util/MathFunctions.H"

#include <signal.h>
#include <sys/types.h>

void drawInput(std::vector<Point2D<float> >& input,
    std::vector<Point2D<float> >& model)
{
  float binWidth = 20;
  Image<PixRGB<byte> > img(int(32*binWidth), int(32*binWidth), ZEROS);
  drawGrid(img, (int)binWidth, (int)binWidth, 1, 1, PixRGB<byte>(0,0,255));
  drawLine(img, Point2D<int>(0, img.getHeight()/2),
                Point2D<int>(img.getWidth(), img.getHeight()/2),
                PixRGB<byte>(0,0,255),2);
  drawLine(img, Point2D<int>(img.getWidth()/2, 0),
                Point2D<int>(img.getWidth()/2, img.getHeight()),
                PixRGB<byte>(0,0,255),2);

  //draw the Input
  float scale = 20;
  for(uint i=0; i<input.size(); i++)
  {
    int x = (img.getWidth()/2) + (int)(scale*input[i].i);
    int y = (img.getHeight()/2) - (int)(scale*input[i].j);
    drawCircle(img, Point2D<int>(x,y), 3, PixRGB<byte>(255,0,0));
  }

  //draw the Model
  for(uint i=0; i<model.size(); i++)
  {
    int x = (img.getWidth()/2) + (int)(scale*model[i].i);
    int y = (img.getHeight()/2) - (int)(scale*model[i].j);
    drawCircle(img, Point2D<int>(x,y), 5, PixRGB<byte>(0,255,0));
  }

  SHOWIMG(img);
}
int main(const int argc, const char **argv)
{

  GeometricHashing gHash;

  GeometricHashing::Model model;

  //model.v.push_back(Point2D<float>(5+2,2+3.464102));
  //model.v.push_back(Point2D<float>(5+7.556922, 2+3.488973));
  //model.v.push_back(Point2D<float>(5+-1.110512, 2+6.876537));
  //model.v.push_back(Point2D<float>(5+-2.000000, 2+-3.464102));
  //model.v.push_back(Point2D<float>(5+6.022947, 2+-1.167949));

  //std::vector<Point2D<float> > input;
  //input.push_back(Point2D<float>(1, 7));
  //input.push_back(Point2D<float>(9, 6));
  //input.push_back(Point2D<float>(1+2,1+3.464102));
  //input.push_back(Point2D<float>(1+7.556922, 1+3.488973));
  //input.push_back(Point2D<float>(1+-1.110512, 1+6.876537));
  //input.push_back(Point2D<float>(1+-2.000000, 1+-3.464102));
  //input.push_back(Point2D<float>(1+6.022947, 1+-1.167949));
  //input.push_back(Point2D<float>(1+1.022947, 1+-3.167949));
  //input.push_back(Point2D<float>(1+2.022947, 1+-2.167949));
  //input.push_back(Point2D<float>(1+3.022947, 1+-7.167949));
  //input.push_back(Point2D<float>(-1.022947, -3.167949));
  //input.push_back(Point2D<float>(-2.022947, -2.167949));
  //
  //gHash.addModel(model);
  //Image<PixRGB<byte> > img = gHash.getHashTableImage();
  //LINFO("Show Hash table\n");
  ////SHOWIMG(img);

  //LINFO("Show Input");
  //std::vector<Point2D<float> > modelF;
  ////drawInput(input, modelF);

  ////Find the model in the input
  //std::vector<GeometricHashing::Acc> acc = gHash.getVotes(input);
  //if (acc.size() > 0)
  //{
  //  //Find the max
  //  GeometricHashing::Acc maxAcc = acc[0];
  //  for(uint i=1; i<acc.size(); i++)
  //    if (acc[i].votes > maxAcc.votes)
  //      maxAcc = acc[i];

  //  LINFO("Found model at: %i %i %i %i (%i %i)\n",
  //      maxAcc.P1, maxAcc.P2, maxAcc.modelId, maxAcc.votes,
  //      maxAcc.inputP1, maxAcc.inputP2);

  //  //Change the basis of the model to find the input
  //  Point2D<float> p1 = input[maxAcc.inputP1]; //Get the input basis
  //  Point2D<float> p2 = input[maxAcc.inputP2]; //Get the input basis

  //  //Find the transformation
  //  float modelScale = sqrt( squareOf(p2.i - p1.i) + squareOf(p2.j - p1.j) );
  //  float ang = atan((p2.j - p1.j)/(p2.i - p1.i));
  //  Point2D<float> center(p1.i+(p2.i - p1.i)/2, p1.j+(p2.j - p1.j)/2);
  //
  //  //Change the basis of the model to the basis that we found
  //  modelF = gHash.changeBasis(model.v, maxAcc.P1, maxAcc.P2);

  //  //Change the basis of the mode to the input
  //  for(uint i=0; i<modelF.size(); i++)
  //  {
  //    float x = modelScale*(modelF[i].i);
  //    float y = modelScale*(modelF[i].j);
  //    modelF[i] = Point2D<float>((x * cos(ang) - y * sin(ang)),
  //          (y * cos(ang) + x * sin(ang)));
  //  }
  //  for(uint i=0; i<modelF.size(); i++)
  //    modelF[i] += center;

  //  drawInput(input, modelF);
  //}


  return 0;
}

