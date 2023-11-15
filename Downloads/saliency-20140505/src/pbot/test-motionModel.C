/*! @file pbot/test-motionModel.C test the various motion models */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/pbot/test-motionModel.C $
// $Id: test-motionModel.C 10794 2009-02-08 06:21:09Z itti $
//


#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/Transforms.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Rectangle.H"
#include "Image/ShapeOps.H"
#include "Image/ColorOps.H"
#include "Image/Rectangle.H"
#include "Image/MathOps.H"
#include "Image/Layout.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "GUI/XWinManaged.H"

#define KEY_UP 98
#define KEY_DOWN 104
#define KEY_LEFT 100
#define KEY_RIGHT 102

struct State
{
  int x;
  int y;  //robot position
  int ori; //Robot angle in deg

  State(int _x, int _y, int _ori) :
    x(_x), y(_y), ori(_ori)
  {}
};


void drawRobot(Image<PixRGB<byte> > &img, const State& robotState)
{
  drawCircle(img,
      Point2D<int>(img.getWidth()/2 + robotState.x, img.getHeight()/2+robotState.y),
      15, PixRGB<byte>(0,255,0));


  float ori = (float)robotState.ori*M_PI/180;
  int x1 = int(cos(ori)*10);
  int y1 = int(sin(ori)*10);

  drawLine(img,
      Point2D<int>(img.getWidth()/2 + robotState.x, img.getHeight()/2+robotState.y),
      Point2D<int>(img.getWidth()/2 + robotState.x+x1, img.getHeight()/2+robotState.y+y1),
      PixRGB<byte>(255,0,0));


}

State moveRobot (State& curState, float tranVel, float rotVel)
{

  rotVel += 1.0e-10; //Avoid divide by 0
  float dt = 1.0;
  State newState(0,0,0);
  float ori = (float)curState.ori*M_PI/180;
  newState.x = curState.x + (int)(-tranVel/rotVel*sin(ori) + tranVel/rotVel*sin(ori+rotVel*dt));
  newState.y = curState.y + (int)(tranVel/rotVel*cos(ori) - tranVel/rotVel*cos(ori+rotVel*dt));
  newState.ori = curState.ori + (int)((rotVel*dt)*180/M_PI);
  return newState;

}

float sample(float b)
{
  int dist = 1; //normal dist
  switch (dist)
  {
    case 0: //normal dist
      {
          float sum =0;
          //normal dist
          for(int i=0; i<12; i++)
            sum += (randomDouble()*b*2)-b;
          return 0.5*sum;
      }
    case 1: //Triangular dist
      {
        return sqrt(6.0)/2.0 *
          (((randomDouble()*b*2)-b) + ((randomDouble()*b*2)-b));
      }
  }

  return 0;
}


void drawSamples(Image<PixRGB<byte> > &img, float tranVel, float rotVel, const State& curState)
{
  rotVel += 1.0e-10; //Avoid divide by 0
  float dt = 1.0;

  float alpha_1 = 1.5; float alpha_2 = 0;
  float alpha_3 = 0.5; float alpha_4 = 0;
  float alpha_5 = 0.2; float alpha_6 = 0;


  float tranVel_hat = tranVel + sample(alpha_1+alpha_2);
  float rotVel_hat = rotVel + sample(alpha_3 + alpha_4);
  float gamma_hat = sample(alpha_5 + alpha_6);

  State predictState(0,0,0);
  float ori = (float)curState.ori*M_PI/180;
  predictState.x = curState.x + (int)(-tranVel_hat/rotVel_hat*sin(ori) + tranVel_hat/rotVel_hat*sin(ori+rotVel_hat*dt));
  predictState.y = curState.y + (int)(tranVel_hat/rotVel_hat*cos(ori) - tranVel_hat/rotVel_hat*cos(ori+rotVel_hat*dt));
  predictState.ori = curState.ori + (int)((rotVel_hat*dt)*180/M_PI) + (int)((gamma_hat*dt)*180/M_PI);

  Point2D<int> pPt(img.getWidth()/2 + predictState.x, img.getHeight()/2+predictState.y);
  if (img.coordsOk(pPt))
    img.setVal(pPt, PixRGB<byte>(255,0,255));

}


int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;
  ModelManager *mgr = new ModelManager("Test Motion Models");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
  mgr->addSubComponent(ofs);

  mgr->exportOptions(MC_RECURSE);

  if (mgr->parseCommandLine(
        (const int)argc, (const char**)argv, "", 0, 0) == false)
    return 1;

  mgr->start();

  initRandomNumbers();

  State robotState(0,0,0);


  while(1)
  {
    Image<PixRGB<byte> > world(512,512, ZEROS);
    LINFO("State: %i,%i  %i", robotState.x, robotState.y, robotState.ori);
    drawRobot(world, robotState);

    float transVel = 100;
    float rotVel = 1;

    for(int i=0; i<100; i++)
      drawSamples(world, transVel, rotVel, robotState);

    robotState = moveRobot(robotState, transVel, rotVel);

    ofs->writeRGB(world, "Output", FrameInfo("Output", SRC_POS));
    getchar();

  }
  mgr->stop();

  return 0;
}

