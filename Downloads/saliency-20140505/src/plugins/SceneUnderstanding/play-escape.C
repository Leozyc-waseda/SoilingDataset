/*! @file SceneUnderstanding/play-escape.C Play the escape game using POMDP */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/play-escape.C $
// $Id: play-escape.C 13765 2010-08-06 18:56:17Z lior $
//

//#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/Transforms.H"
#include "Image/DrawOps.H"
#include "Image/ShapeOps.H"
#include "Image/ColorOps.H"
#include "Image/Rectangle.H"
#include "Image/MathOps.H"
#include "Image/Layout.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "GUI/AutomateXWin.H"
#include "GUI/ImageDisplayStream.H"
#include "GUI/XWinManaged.H"
#include "GUI/DebugWin.H"
#include "Neuro/getSaliency.H"
#include "plugins/SceneUnderstanding/POMDP.H"
#include "plugins/SceneUnderstanding/Ganglion.H"

#define KEY_UP 98
#define KEY_DOWN 104
#define KEY_LEFT 100
#define KEY_RIGHT 102

int getKey(nub::ref<OutputFrameSeries> &ofs)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow("Output")
    : rutz::shared_ptr<XWinManaged>();
  return uiwin->getLastKeyPress();
}


//Get the input image
Image<PixRGB<byte> > getImage(AutomateXWin &xwin)
{
  int idnum = getIdum();

  Image<PixRGB<byte> > img = xwin.getImage();

 // inplaceColorSpeckleNoise(img, int(img.getSize()*0.50F)); //25% speckle noise

 //Add Noise
  Image<PixRGB<byte> >::iterator ptr = img.beginw(), stop = img.endw();
  while(ptr != stop)
  {
    *ptr += (int)(25*gasdev(idnum));
    ++ptr;
  }

  return img;

}

void doAction(AutomateXWin &xwin,const int action)
{

  LINFO("Perform action");
  //with 80% probability the action will succeed, otherwise it will go to a
  //perperdiculer location

  float prob = randomDouble();

  int newAction = action;
  if (prob > 0.80)
  {
    switch(action)
    {
      case POMDP::NORTH:
      case POMDP::SOUTH:
        (prob > 0.90) ? newAction = POMDP::EAST : newAction = POMDP::WEST;
        break;
      case POMDP::EAST:
      case POMDP::WEST:
        (prob > 0.90) ? newAction = POMDP::NORTH : newAction = POMDP::SOUTH;
        break;
    }
    LINFO("Action %i newaction %i", action, newAction);
  }

  switch(newAction)
  {
    xwin.setFocus();
    case POMDP::NORTH: xwin.sendKey(KEY_UP); break;
    case POMDP::SOUTH: xwin.sendKey(KEY_DOWN); break;
    case POMDP::WEST: xwin.sendKey(KEY_LEFT); break;
    case POMDP::EAST: xwin.sendKey(KEY_RIGHT); break;
  }

}


int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;
  ModelManager *mgr = new ModelManager("Test ObjRec");

  nub::ref<GetSaliency> getSaliency(new GetSaliency(*mgr));
  mgr->addSubComponent(getSaliency);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
  mgr->addSubComponent(ofs);

  POMDP pomdp;
  Ganglion ganglionCells;

  mgr->exportOptions(MC_RECURSE);

  if (mgr->parseCommandLine(
        (const int)argc, (const char**)argv, "", 0, 0) == false)
    return 1;

  mgr->start();

  AutomateXWin xwin("escape 200606190");

  bool runPOMDP = false;
  Layout<PixRGB<byte> > outDisp;

  Image< PixRGB<byte> > inputImg = getImage(xwin);
  pomdp.init();
  ganglionCells.init(inputImg.getDims());




  float surprise = 0;
  while(1)
  {

    inputImg = getImage(xwin);

    ganglionCells.setInput(inputImg);

    if (runPOMDP)
    {

      //get The current Perception
      LINFO("Get perception");
      Image<float> agentPercep = pomdp.getPerception(0);
      SHOWIMG(agentPercep);
      //SHOWIMG(scaleBlock(agentPercep, agentPercep.getDims()*5));

      int action = -1; //pomdp.getPropAction();

      if (action == -1)
      {
        //choose a random action
        action = randomUpToNotIncluding(4)+1;
      } else {
        LINFO("Make prediction");
        //Make predication
        //Image<float> predict = pomdp.makePrediction(action);
        //SHOWIMG(scaleBlock(predict, predict.getDims()*5));
      }
      action = 0;
      LINFO("Action is %i\n", action);


      LINFO("Bayes filter to update perception");
      //float surprise = pomdp.bayesFilter(action, inputImg);
      float surprise = pomdp.particleFilter(action, inputImg);
      LINFO("Surprise %f", surprise);

      LINFO("Get perception");
      agentPercep = pomdp.getPerception(0);
      SHOWIMG(agentPercep);


      //LINFO("Update perception");
      ////Get Data and update Perception
      //inputImg = xwin.getImage();
      //surprise = pomdp.updatePerception(inputImg);
      //LINFO("Surprise %f\n", surprise);

      //if (surprise > 0)
      //{
      //  LINFO("What happend?, Learning");
      //  //pomdp.updateStateTransitions(action);
      //  sleep(5);
      //} else {
      //  //if we are exploring and there are no suprises then
      //  //we know about the current world, try to find a solution
      //  if (pomdp.isExploring())
      //  {
      //    runPOMDP = false;
      //  }
      //}



     // SHOWIMG(scaleBlock(agentPercep, agentPercep.getDims()*5));


      LINFO("Show found objects");
      //Show the found objects
      Point2D<int> agentState = pomdp.getAgentState();
      Point2D<int> goalState = pomdp.getGoalState();

      if (agentState.isValid())
        drawCircle(inputImg, agentState, 20, PixRGB<byte>(255,0,0));

      if (goalState.isValid())
        drawCircle(inputImg, goalState, 20, PixRGB<byte>(0,255,0));

      LINFO("%ix%i %ix%i\n", agentState.i, agentState.j,
          goalState.i, goalState.j);

      //Show supprise


    }

    char msg[255];
    sprintf(msg, "Surprise %0.2f", surprise);
    writeText(inputImg, Point2D<int>(0,0), msg, PixRGB<byte>(255), PixRGB<byte>(127) );
    //outDisp = inputImg;

    Image<float> gangIn = ganglionCells.getInput();
    Image<float> gangOut = ganglionCells.getOutput();
    Image<float> gangPerc = ganglionCells.getGanglionCells();

    //Display result
    inplaceNormalize(gangIn, 0.0F, 255.0F);
    inplaceNormalize(gangPerc, 0.0F, 255.0F);
    inplaceNormalize(gangOut, 0.0F, 255.0F);

    outDisp = hcat(toRGB(Image<byte>(gangIn)), toRGB(Image<byte>(gangPerc)));
    outDisp = hcat(outDisp, toRGB(Image<byte>(gangOut)));

    ofs->writeRgbLayout(outDisp, "Output", FrameInfo("output", SRC_POS));


    int key = getKey(ofs);

    if (key != -1)
    {
      xwin.setFocus();
     // xwin.sendKey(key);

      switch (key)
      {
        case 27: //r to restart game
          if (runPOMDP)
            runPOMDP = false;
          else
          {
            runPOMDP = true; //run the controller
            LINFO("POMDP running");

            sleep(1);
            for(int i=0; i<10; i++) //wait for reset
            {
              outDisp = inputImg;
              ofs->writeRgbLayout(outDisp, "Output", FrameInfo("output", SRC_POS));
            }
            inputImg = xwin.getImage();

            //LINFO("Get Initial Observation");
            //if (pomdp.makeObservation(inputImg))
            //{
            ////  //New input, plan the actions
            ////  Point2D<int> agentState = pomdp.getAgentState();

            ////  //new Observation, calculate optimal solution
            ////  pomdp.valueIteration();
            ////  pomdp.doPolicy(agentState);
            //}
          }
          break;

        case KEY_UP: doAction(xwin, POMDP::NORTH); break;
        case KEY_DOWN: doAction(xwin, POMDP::SOUTH); break;
        case KEY_RIGHT: doAction(xwin, POMDP::EAST); break;
        case KEY_LEFT: doAction(xwin, POMDP::WEST); break;

      }
      LINFO("Key = %i\n", key);
    }


  }
  mgr->stop();

  return 0;

}

