/*! @file FeatureMatching/test-PGH.C Test the PGH */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/FeatureMatching/test-PGH.C $
// $Id: test-PGH.C 12985 2010-03-09 00:18:49Z lior $
//

#include "Image/Image.H"
#include "Component/ModelManager.H"
#include "Raster/Raster.H"
#include "GUI/DebugWin.H"
#include "FeatureMatching/PGH.H"
#include "Image/Rectangle.H"
#include "Image/FilterOps.H"

#include <signal.h>
#include <sys/types.h>

std::vector<PGH::Line>  transform(std::vector<PGH::Line>& lines, Point2D<int> pos,
    float scale, float rot)
{
  std::vector<PGH::Line> newLines = lines;

  //Get the center of mass so we can rotte around that


  Point2D<int> com(0,0);
  for(uint i=0; i<newLines.size(); i++)
    com += newLines[i].pos;
  com /= newLines.size();

  for(uint i=0; i<lines.size(); i++)
  {

    ////Rotate 

    float x = scale*(newLines[i].pos.i - com.i);
    float y = scale*(newLines[i].pos.j - com.j);
    newLines[i].pos.i = com.i + int(x * cos(rot) - y * sin(rot));
    newLines[i].pos.j = com.j + int(y * cos(rot) + x * sin(rot));

    newLines[i].pos += pos; //Transelate

    newLines[i].length *= scale;
    newLines[i].ori -= rot;
  }


  return newLines;

}

int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager manager("Test SFS");

  if (manager.parseCommandLine(
        (const int)argc, (const char**)argv, "", 0, 0) == false)
    return 1;

  manager.start();


  PGH pgh;

  //Create a model object
  std::vector<PGH::Line> model;
  //Building
  model.push_back(PGH::Line(Point2D<int>(100, 100), 50, 0));
  model.push_back(PGH::Line(Point2D<int>(100, 50), 50, 0));
  model.push_back(PGH::Line(Point2D<int>(75, 75), 50, M_PI/2));
  model.push_back(PGH::Line(Point2D<int>(125, 75), 50, M_PI/2));

  //Roof
  model.push_back(PGH::Line(Point2D<int>(80, 40), 60, M_PI/3.7));
  model.push_back(PGH::Line(Point2D<int>(120, 40), 60, -M_PI/3.7));

  //Show model
  Image<PixRGB<byte> > img(320, 240, ZEROS);
  for(uint i=0; i<model.size(); i++)
  {
    drawLine(img, model[i].pos, model[i].ori, model[i].length, PixRGB<byte>(255,0,0));
  }
  SHOWIMG(img);


  std::vector<PGH::Line> newLines2 = transform(model,Point2D<int>(50,50), 1.0F, float(M_PI/8));

  pgh.addModel(newLines2, 0);

  for(float rot = 0; rot < 2*M_PI; rot+=10*M_PI/180)
  {
    Point2D<int> pos(50,50);
    float scale = 1;
    std::vector<PGH::Line> newLines = transform(model,pos, scale, rot);

    img = Image<PixRGB<byte> >(320, 240, ZEROS);
    for(uint i=0; i<model.size(); i++)
    {
      drawLine(img, newLines[i].pos, newLines[i].ori, newLines[i].length,
          PixRGB<byte>(255,0,0));
    }

    pgh.matchModel(newLines);

    SHOWIMG(img);
  }



  manager.stop();

  return 0;
}

