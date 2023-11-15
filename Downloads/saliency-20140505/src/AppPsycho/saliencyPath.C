/*!@file AppPsycho/saliencyPath.C finds a saliency path in an image
 */

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
// Primary maintainer for this file: T. Nathan Mundhenk <mundhenk@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/saliencyPath.C $
// $Id: saliencyPath.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "GUI/XWindow.H"
#include "Raster/Raster.H"
#include "Util/Assert.H"
#include "Util/Timer.H"
#include "Util/log.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/DrawOps.H"
#include "Image/ColorOps.H"

#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <cfloat>

#define DISKSIZE         50
#define SALPOINTS        8
#define OUT_SPEED        10
#define OUT_ACCEL        1
#define OUT_IMAGE_SIZE_X 128
#define OUT_IMAGE_SIZE_Y 128

using namespace std;

int main(const int argc, const char **argv)
{

  if(argc < 1)
    std::cerr << "USAGE: saliencyPath image.pgm\n";

  string imageFile       = argv[1];
  const uint salPoints   = SALPOINTS;

  Image<float> salImage          = Raster::ReadGray(imageFile.c_str());
  Image<PixRGB<float> > outImage = luminance(salImage);

  std::vector<float> xpos(salPoints,0);
  std::vector<float> ypos(salPoints,0);
  std::vector<float> sval(salPoints,0.0F);


  // find the n most salient points
  for(uint i = 0; i < salPoints; i++)
  {
    for(uint x = 0; x < (uint)salImage.getWidth(); x++)
    {
      for(uint y = 0; y < (uint)salImage.getHeight(); y++)
      {
        // if this point is better, then store it
        if(salImage.getVal(x,y) > sval[i])
        {
          sval[i] = salImage.getVal(x,y);
          ypos[i] = y;
          xpos[i] = x;
        }
      }
    }
    // for each salient location, disk it so we don't count it twice
    drawDisk(salImage,Point2D<int>((int)xpos[i],(int)ypos[i]),DISKSIZE,0.0F);
  }

  std::vector<uint> cpos(salPoints,0);
  std::vector<bool> used(salPoints,false);
  std::vector<uint> best(salPoints,0);

  float bestVal = FLT_MAX;
  bool  end     = false;
  float dist    = 0;

  /* By setting salPoints as a constant at compile time, if we use
     loop unrolling, this portion of the code will be optimized for
     indexing into the vectors.
  */

  while(!end)
  {
    // increment over all permutations like an odometer
    cpos[0]++;
    for(uint i = 0; i < salPoints-1; i++)
    {
      if(cpos[i] == 10)
      {
        cpos[i] = 0;
        cpos[i+1]++;
      }
    }

    // when the last permutation "flips" then end
    if(cpos[salPoints-1] == 10)
      end = true;

    // to keep points unique, don't use the same point twice
    for(uint i = 0; i < salPoints; i++)
    {
      used[i] = false;
    }

    dist = 0;
    for(uint i = 0; i < salPoints - 1; i++)
    {
      // record this point as being used
      used[cpos[i]] = true;
      // compute distance over current permutation only if
      // it is unique
      if(used[cpos[i+1]] == false)
      {
        dist += sqrt(pow(ypos[cpos[i]] - ypos[cpos[i+1]],2)
                   + pow(xpos[cpos[i]] - xpos[cpos[i+1]],2));
      }
      else
      {
        dist = FLT_MAX;
        break;
      }
    }

    // if the distance is superior, then store it
    if(dist < bestVal)
    {
      best    = cpos;
      bestVal = dist;
    }
  }

  string A            = ".results.txt";
  string B            = ".path";
  string outFileName  = imageFile + A;
  string outImageName = imageFile + B;
  ofstream outFile(outFileName.c_str(), ios::out);
  //outFile << salImage.getWidth() << " " << salImage.getHeight() << "\n";
  outFile << OUT_IMAGE_SIZE_X << " " << OUT_IMAGE_SIZE_Y << "\n";
  outFile << salPoints           << "\n";

  std::cerr << "computing lum image\n";

  PixRGB<float> red(255.0F,0.0F,0.0F);
  PixRGB<float> blue(0.0F,0.0F,255.0F);

  //output best path to file and an output image
  for(uint i = 0; i < salPoints; i++)
  {
    std::cerr << i << "\n";
    outFile << xpos[best[i]]     << " " << ypos[best[i]]        << " "
            << OUT_SPEED         << " " << OUT_ACCEL            << "\n";
    drawCircle(outImage,Point2D<int>((int)xpos[best[i]]
                                ,(int)ypos[best[i]]),25,red,2);
    if(i < salPoints-1)
    {
      drawArrow(outImage,Point2D<int>((int)xpos[best[i]],(int)ypos[best[i]]),
                Point2D<int>((int)xpos[best[i+1]],(int)ypos[best[i+1]]),blue,2);
    }
    std::cerr << ".\n";
  }
  Raster::WriteRGB(outImage,outImageName,RASFMT_PNG);
  outFile.close();
  std::cerr << "DONE!";
}
