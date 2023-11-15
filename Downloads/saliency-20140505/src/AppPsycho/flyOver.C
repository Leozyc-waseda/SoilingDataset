/*!@file AppPsycho/flyOver.C simulates a fly over satalite images
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/flyOver.C $
// $Id: flyOver.C 7157 2006-09-15 07:55:58Z itti $
//
#include "GUI/XWindow.H"
#include "Raster/Raster.H"
#include "Util/Assert.H"
#include "Util/Timer.H"
#include "Util/log.H"
#include "Image/Image.H"
#include "Image/Pixels.H"

#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

using namespace std;

//**********************************************************************
class projectIO
{
private:
  unsigned int inputX, inputY;
  // nested vector class container, this one holds integers
  std::vector<std::vector<int> > inputData;
public:
  projectIO();
  ~projectIO();
  // Read in a project file
  std::vector<std::vector<int> > readFile(string infile);
};

//**********************************************************************
projectIO::projectIO()
{}

//**********************************************************************
projectIO::~projectIO()
{}

//**********************************************************************
/* This is a blind file reader. While we can compute the file size on the
   fly by checking for EOL, this makes the file size explicite and readable.
   In general, it really doesn't matter which way its done.
*/
std::vector<std::vector<int> > projectIO::readFile(string infile)
{

  std::cerr << "Reading in file " << infile << "\n";
  // Create a file handle and open a file for reading
  ifstream inputFile(infile.c_str(), ios::in);
  unsigned int marker = 0;
  string in, inX, inY, size;

  // read in the first two tokens as a height
  // and width of this matrix
  // (1) atoi converts the char to an integer
  // (2) c_str() gets a char from a string
  inputFile >> inX >> inY >> size;
  // The X size of the matrix file
  inputX = 4;
  // The Y size of the matrix file
  inputY = atoi(size.c_str());
  // resize the vector container
  std::vector<int> filler(inputY+2,0);
  inputData.resize(inputX,filler);

  inputData[0][0] = atoi(inX.c_str());
  inputData[1][0] = atoi(inY.c_str());
  inputData[0][1] = atoi(size.c_str());

  // While not end of file, place each white space
  // seperated token into a string
  while(inputFile >> in)
  {
    // for each token, place it in the vector
    // at its X and Y position
    unsigned int currentX = marker % inputX;
    unsigned int currentY = marker / inputX;
    inputData[currentX][currentY+2] = atoi(in.c_str());
    marker++;
  }

  inputFile.close();
  return inputData;
}

//**********************************************************************

int main(const int argc, const char **argv)
{

  if(argc < 2)
    std::cerr << "USAGE: flyOver image.png path_file.txt\n";
  string satImageFile = argv[1];
  //string movieFile    = argv[2];
  string infile       = argv[2];
  uint fsizex, fsizey, tpoints;
  uint sizex,sizey;
  uint frameNumber = 0;

  projectIO proj;
  std::vector<std::vector<int> > inputData;
  inputData = proj.readFile(infile);

  Image<PixRGB<float> > satImage;

  fsizex  = inputData[0][0];
  fsizey  = inputData[1][0];
  tpoints = inputData[0][1];

  std::cout << "Image Frame size (x,y) " << fsizex
            << " x "                     << fsizey
            << "\n"
            << "Travel Points "          << tpoints
            << "\n";

  for(uint i = 0; i < tpoints; i++)
  {
    // center the frame
    inputData[0][i+2] = inputData[0][i+2] - (int)floor(fsizex/2);
    inputData[1][i+2] = inputData[1][i+2] - (int)floor(fsizey/2);

    cout << "(" << inputData[0][i+2] << "," << inputData[1][i+2] << ")"
         << " Speed (pixels per frame) " << inputData[2][i+2]
         << " Ramp (frames) "  << inputData[3][i+2]
         << "\n";
  }

  std::cerr << "READING IN SATALITE IMAGE\n"
            << "Depending on image size, this may take a while\n";
  satImage = Raster::ReadRGB(satImageFile.c_str(), RASFMT_PNG);
  std::cerr << "Done\n";

  sizex = satImage.getWidth();
  sizey = satImage.getHeight();

  std::cout << "Sat image size (" << sizex << "," << sizey << ")\n";
  Image<PixRGB<float> > frame;
  frame.resize(fsizex,fsizey);
  for(uint i = 0; i < tpoints - 1; i++)
  {
    const float distx  = inputData[0][i+2] - inputData[0][i+3];
    const float disty  = inputData[1][i+2] - inputData[1][i+3];
    const float dist   = sqrt(distx*distx + disty+disty);
    const float speed  = inputData[2][i+2];
    const float accel  = inputData[3][i+2];
    const float frames = dist/speed;
    float currx        = inputData[0][i+2];
    float curry        = inputData[1][i+2];
    //! current speed
    float currsp = 0;
    //! How far to travel so far
    float dtot   = dist;
    //! at what point to begin to slow down (ramp function)
    //float slow   = floor(((accel*(speed + 1))/(2*accel))*speed);
    // approx how far we are out when we need to begin to stop
    const float slow = floor(((0.5F + (speed/accel)) *
                              (speed + (accel/2)))/2);
    float ndec = 0;
    bool stop    = false;
    bool doStop  = false;

    std::cerr << "Position (" << currx << "," << curry << ")"
              << "Approx frames " << frames << "\n";

    // while we are one or more pixel from the target
    while(!stop)
    {
      uint fcurrx = (int)floor(currx);
      uint fcurry = (int)floor(curry);
      std::cerr << "Pos: "  << fcurrx << "," << fcurry
                << " Spd: " << currsp
                << " Dst: " << dtot << "\n";
      // from this position, get the current frame
      for(uint x = fcurrx; x < fcurrx + fsizex; x++)
      {
        for(uint y = fcurry; y < fcurry + fsizey; y++)
        {
          PixRGB<float> pix = satImage.getVal(x,y);
          frame.setVal(x - fcurrx,y - fcurry,pix);
        }
      }

      // check to see if we need to slow down
      if(dtot < slow+speed)
      {
        if(!doStop)
        {
          // figure out decelleration from current position
          ndec   = speed/(((2*dtot)/(speed + accel/2))-0.5F);
          doStop = true;
        }
        else
        {
          currsp = currsp - ndec;
          if(currsp < 1)
            currsp = 1;
        }
      }
      // else if we are below full speed, speed up
      else if(currsp < speed)
      {
        currsp = currsp + accel;
        if(currsp > speed) currsp = speed;
      }

      // compute new x and new y based on distance to travel and
      // trajectory
      float ratio = disty/distx;

      float movex = sqrt((currsp*currsp)/(1+(ratio*ratio)));
      float movey = fabs(ratio*movex);

      // compute new x coord
      if(distx < 0)
      {
        currx = currx + movex;
        if(currx > inputData[0][i+3])
          stop = true;
      }
      else
      {
        currx = currx - movex;
        if(currx < inputData[0][i+3])
          stop = true;
      }

      // compute new y coord
      if(disty < 0)
      {
        curry = curry + movey;
        if(curry > inputData[1][i+3])
          stop = true;
      }
      else
      {
        curry = curry - movey;
        if(curry < inputData[1][i+3])
          stop = true;
      }

      // compute how much further we have to go
      float ndistx  = currx - inputData[0][i+3];
      float ndisty  = curry - inputData[1][i+3];
      dtot          = sqrt(ndistx*ndistx + ndisty*ndisty);

      // get the string for this frame
      char c[100];
      string a = "frame";
      string b = ".";
      if(frameNumber < 10)
        sprintf(c,"00000%d",frameNumber);
      else if(frameNumber < 100)
        sprintf(c,"0000%d",frameNumber);
      else if(frameNumber < 1000)
        sprintf(c,"000%d",frameNumber);
      else if(frameNumber < 10000)
        sprintf(c,"00%d",frameNumber);
      else if(frameNumber < 100000)
        sprintf(c,"0%d",frameNumber);
      else
        sprintf(c,"%d",frameNumber);
      string Myname = a + b + c;
      Raster::WriteRGB(frame,Myname,RASFMT_PNM);
      frameNumber++;
    }
  }
}









