/*!@file AppMedia/app-mouseChaserColorHist.C get estimate of mouse position in temperature preference assay using color histogram */

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
// Primary maintainer for this file: Farhan Baluch <fbaluch at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-mouseChaserColorHist.C $
// $Id:   $
//

#ifndef APPMEDIA_APP_MOUSE_CHASER_COLORHIST_C_DEFINED
#define APPMEDIA_APP_MOUSE_CHASER_COLORHIST_C_DEFINED

#include "Component/ModelManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/DeviceOpts.H"
#include "GUI/XWindow.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/ShapeOps.H"
#include "Image/ColorOps.H"
#include "Image/DrawOps.H"
#include "Image/Range.H"
#include "Raster/Raster.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"

#include "Media/FrameSeries.H"
#include "Media/MediaOpts.H"
#include "Raster/GenericFrame.H"
#include "Transport/FrameInfo.H"
#include <cstdio>
#include <numeric>
#include <iostream>
#include <fstream>

Point2D<int> getMeanMotion(const Image<float>& itsImg,float thresh);
Point2D<int> featTrack(Image<PixRGB<byte> >& crnt, Point2D<int> prevTrackPt,
                       PixRGB<float> itsMeanRGB, Dims rectDims);
float euclidDist(PixRGB<float> a, PixRGB<float> b);

int prevDir = 0;
int main(int argc,const char**argv)
{

     //Instantiate a ModelManager:
  ModelManager *mgr = new ModelManager("Animal Tracker");
  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(*mgr));
  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));

  mgr->addSubComponent(ifs);
  mgr->addSubComponent(ofs);

  mgr->setOptionValString(&OPT_FrameGrabberFPS, "30");
  mgr->exportOptions(MC_RECURSE);

 // Parse command-line:
  if (mgr->parseCommandLine(argc, argv, "", 0, 0) == false) return(1);
  
   // do post-command-line configs:
  Dims imageDims = ifs->peekDims();
  LINFO("w=%d,h=%d",imageDims.w(),imageDims.h());
  
  Dims layer_screen(800,800);
  XWindow layers(layer_screen, 0, 0, "layers"); //preview window

  // let's get all our ModelComponent instances started:
  mgr->start();

  const FrameState is = ifs->updateNext();
  if(is == FRAME_COMPLETE)
      LFATAL("frames completed!");
 
  Image <PixRGB<byte> > crnt,prev;
  Image <PixRGB<byte> > dispImg; 
  std::vector<Image<PixRGB<byte> > > frames;

  prev = ifs->readRGB();
  frames.push_back(prev);

  if(!prev.initialized())
     LFATAL("frame killed");
      //figure out plate region threshold
  bool happy=false,happyBlob=false;
  int crit = imageDims.w()/2;
  int x,y;
  Dims rectDims(imageDims.w()/8,imageDims.h()/8);
  
  dispImg = prev;
  drawLine(dispImg,Point2D<int>(crit,0),
           Point2D<int>(crit,imageDims.h()),PixRGB<byte>(255,0,0),2);
  drawGrid(dispImg,10,10,1,PixRGB<byte> (255,0,0));
  layers.drawImage(dispImg,0,0);
  std::string saveName; 
  saveName = mgr->getOptionValString(&OPT_InputFrameSource);
  saveName = saveName + ".dat";
  std::ofstream outFile(saveName.c_str());

  while(!happy && !happyBlob)
  {
      Image<PixRGB<byte> > testImg = prev;
      
      int choice, choiceX, choiceY;
      LINFO("Input a threshold image center is at %d: ",imageDims.w()/2);
      std::cin >> choice;
      if(choice == -1)
          happy =true;
      else
      {
          drawLine(testImg,Point2D<int>(choice,0),
                   Point2D<int>(choice,imageDims.h()),PixRGB<byte>(255,0,0),2);
          drawGrid(testImg,10,10,0.5,PixRGB<byte> (255,0,0));
          layers.drawImage(testImg,0,0);
          crit = choice;   
      }

      LINFO("Input an x pos: ");
      std::cin >> choiceX;
      LINFO("Input an y pos: ");
      std::cin >> choiceY;

      if(choiceX == -1)
          happyBlob =true;
      else
      {
          Rectangle r1(Point2D<int>(choiceX,choiceY),rectDims);
          drawRect(testImg,r1,PixRGB<byte>(0,255,0),1);
          layers.drawImage(testImg,0,0);
          x = choiceX;
          y = choiceY;
       }
  }
          
  
  LINFO("processing...");

  Timer T,frameTim;
  T.reset();
  int fTotal=1;
  
  frameTim.reset();
  
  std::vector<int> trackX;
  std::vector<int> trackY;

  trackX.push_back(x);
  trackY.push_back(y);
  
  Point2D<int> prevTrackPt, crntTrackPt;
  prevTrackPt = Point2D<int>(x,y);

  Image<PixRGB<float> >  target= crop(prev,prevTrackPt,rectDims,false);
  PixRGB<float> itsMeanRGB = meanRGB(target);
  int leftCnt=0,rightCnt=0;
  
  Image<PixRGB<byte> > meanBkg =prev;
  
  while(is != FRAME_COMPLETE)
  {
      const FrameState is = ifs->updateNext();
      if(is == FRAME_COMPLETE)
          break;
      
          //grab the images
      crnt = ifs->readRGB();
            
      if(!crnt.initialized())
          break;
         
      crntTrackPt = featTrack(crnt, prevTrackPt, itsMeanRGB,rectDims);
      prevTrackPt = crntTrackPt;
      
      Rectangle r1(crntTrackPt,rectDims);
      drawRect(crnt,r1,PixRGB<byte>(255,0,0),1);
      Point2D<int> cross = crntTrackPt;
      cross.i += rectDims.w()/2;
      cross.j += rectDims.h()/2;

      outFile << cross.i << "\t" << cross.j << std::endl;
      //LINFO("crit is %d cross.i =%d cross.j=%d",crit,cross.i,cross.j);
      if (cross.i < crit)
        {
          leftCnt++;
          writeText(crnt, Point2D<int> (2,0),"LEFT",
                    PixRGB<byte>(0,255,0),PixRGB<byte>(0,0,0),
                    SimpleFont::FIXED(9));
        }
      else
        {
          rightCnt++;
           writeText(crnt, Point2D<int> (2,0),"RIGHT",
                    PixRGB<byte>(0,255,0),PixRGB<byte>(0,0,0),
                    SimpleFont::FIXED(9));
        }
      
      drawCross(crnt, cross, PixRGB<byte>(255,255,0), 3, 2);
      drawLine(crnt, Point2D<int>(crit,0), Point2D<int>(crit,imageDims.h()),
               PixRGB<byte>(255,0,0),2);
      //frames.push_back(crnt);
      //      layers.drawImage(crnt,0,0);
      ofs->writeRgbLayout(crnt,"MouseChaser", FrameInfo("output",SRC_POS));
      fTotal++;
  }
  
  LINFO("Total movie time %f: \ntime to compute motion: %f",frameTim.getSecs(), T.getSecs());
  T.reset();

  LINFO("Result cnts: LEFT = %d, RIGHT= %d ",leftCnt, rightCnt);
  float total = (float) (leftCnt + rightCnt);
  
  float leftP = (float)leftCnt/total;
  float rightP = (float)rightCnt/total;
  
  LINFO("Results: LEFT = %.2f %%, RIGHT= %.2f %% total=%f", leftP*100.0F, rightP*100.0F,total);
  
  /*for(std::vector<Image<PixRGB<byte> > >::iterator it = frames.begin();
      it!=frames.end()-1;++it)
  {
      layers.drawImage(*it,0,0);
      usleep(250);
      }*/
  
  mgr->stop();  
  return 0;
}

//###########################Feat Track################################/
/*Tracks color blob across a frame--tests 8 positions to find best match
for previous track point*/

Point2D<int> featTrack(Image<PixRGB<byte> > &crnt, Point2D<int> prevTrackPt,
                       PixRGB<float> itsMeanRGB, Dims rectDims)
{
    int x = prevTrackPt.i;
    int y = prevTrackPt.j;
    
    std::vector<Image<PixRGB<byte> > > regions;    
    std::vector<float> colorDists;
    std::vector<Point2D<int> > thePoints;
    
    //int pd = rectDims.w()/5;
    int pd =40;
      
    for (int i = x-pd;i <= x+pd; i+=5)
      for (int j = y-pd; j<=y+pd; j+=5)
        if (crnt.coordsOk(i, j) & 
            crnt.coordsOk(i + rectDims.w()-1, j + rectDims.h()-1))
            thePoints.push_back(Point2D<int>(i,j));

    std::vector<Point2D<int> >::iterator pItr = thePoints.begin();
    
    while(pItr != thePoints.end())
        regions.push_back(crop(crnt,*pItr++,rectDims,false));
   
    for(std::vector<Image<PixRGB<byte> > >::iterator it = regions.begin();
        it!=regions.end(); ++it)
      {
        int dist =  euclidDist(itsMeanRGB,meanRGB(*it));
        Image<PixRGB<float> > temp = *it; 
        dist = dist + euclidDist(temp.getVal(Point2D<int>(rectDims.w()/2,
                                                         rectDims.h()/2)),
                                 itsMeanRGB);
        colorDists.push_back(dist);
      }
    //LINFO("min of the dists was %f", *(std::min_element(colorDists.begin(),
    //                                                    colorDists.end())));
        //LINFO("min dist is at %td", std::min_element(colorDists.begin(),colorDists.end())-colorDists.begin());
          
    int min = *(std::min_element(colorDists.begin(),colorDists.end()));
    if (min > 120)
      return prevTrackPt;
    else
      {
        prevDir = std::min_element(colorDists.begin(),colorDists.end())-
          colorDists.begin();
        return thePoints[std::min_element(colorDists.begin(),colorDists.end())-
                     colorDists.begin()];    
      }
}


//###########################Euclid dist################################/
/*Get euclid dist between pixels*/

float euclidDist(PixRGB<float> a, PixRGB<float> b)
{
    float rDiff, gDiff, bDiff;
    rDiff= abs(a.red()-b.red());
    gDiff= abs(a.green()-b.green());
    bDiff= abs(a.blue()-b.blue()); 

    return sqrt(rDiff*rDiff + gDiff*gDiff + bDiff*bDiff);
}





         

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPMEDIA_APP_MOUSE_CHASER_COLORHIST_C_DEFINED
