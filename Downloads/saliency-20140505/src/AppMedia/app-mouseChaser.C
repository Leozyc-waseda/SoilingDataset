/*!@file AppMedia/app-mouseChaser.C get estimate of mouse position in temperature preference assay */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-mouseChaser.C $
// $Id:   $
//

#ifndef APPMEDIA_APP_MOUSE_CHASER_C_DEFINED
#define APPMEDIA_APP_MOUSE_CHASER_C_DEFINED

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
#include <cstdio>
#include <numeric>
#include <iostream>

Point2D<int> getMeanMotion(const Image<float>& itsImg,float thresh);


int main(int argc,const char**argv)
{

     //Instantiate a ModelManager:
  ModelManager *mgr = new ModelManager("locust model frame series style");

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(*mgr));
  mgr->addSubComponent(ifs);
  mgr->setOptionValString(&OPT_FrameGrabberFPS, "30");
  mgr->exportOptions(MC_RECURSE);

 // Parse command-line:
  if (mgr->parseCommandLine(argc, argv, "<threshold>", 1, 1) == false) return(1);

  int itsMoThresh = mgr->getExtraArgAs<int>(0);
  
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
 
  Image <PixRGB<byte> > prev,crnt;
   
  std::vector<Image<PixRGB<byte> > > frames;
  std::vector<Image<float> > motion;

  prev = ifs->readRGB();
  frames.push_back(prev);

  if(!prev.initialized())
     LFATAL("frame killed");

      //figure out plate region threshold
  bool happy=false,happyBlob=false;
  int crit = imageDims.w()/2;
  while(!happy && !happyBlob)
  {
      Image<PixRGB<byte> > testImg = prev;
      int choice,x,y;

      LINFO("Input a threshold image center is at %d: ",imageDims.w()/2);
      std::cin >> choice;
      if(choice == 0)
          happy =true;
      else
      {
          
          drawLine(testImg,Point2D<int>(crit,0),
                   Point2D<int>(crit,imageDims.h()),PixRGB<byte>(255,0,0),2);
          layers.drawImage(rescale(testImg,320,240),0,0);
          crit = choice;
          
       }

      LINFO("Input an x pos: ");
      std::cin >> x;
      LINFO("Input an y pos: ");
      std::cin >> y;

      if(x == -1)
          happyBlob =true;
      else
      {
          Rectangle r1(Point2D<int>(x,y),Dims(30,30));
          drawRect(testImg,r1,PixRGB<byte>(255,0,0),1);
          layers.drawImage(rescale(testImg,320,240),0,0);
          crit = choice;
       }
              
     }
          
  
  LINFO("processing...");

  Timer T,frameTim;
  T.reset();
  int fTotal=0;
  
  frameTim.reset();
  while(is != FRAME_COMPLETE)
  {
      const FrameState is = ifs->updateNext();
      if(is == FRAME_COMPLETE)
          break;
      
          //grab the images
      crnt = ifs->readRGB();
      if(!crnt.initialized())
          break;
      frames.push_back(crnt);
      motion.push_back(absDiff(luminance(crnt),luminance(prev)));      
      prev= crnt;
      fTotal++;
  }
  
  LINFO("Total movie time %f: \ntime to compute motion: %f",frameTim.getSecs(),
        T.getSecs());
  T.reset();
  
  
  std::vector<Point2D<int> > posVec;
  std::vector<int> posX;
  std::vector<Image<float> >::iterator itM = motion.begin();
  Point2D<int> prevMean = getMeanMotion(*itM,itsMoThresh);
  int fLeft=0;
  posVec.push_back(prevMean);

  for(std::vector<Image<PixRGB<byte> > >::iterator it = frames.begin();
      it!=frames.end()-1;++it)
  {
      Image<PixRGB<byte> > vid= *it;
      Point2D<int> meanMotion = getMeanMotion(*itM,80);
      if(meanMotion.i == 0 && meanMotion.j == 0)
          meanMotion = prevMean;
      else
          prevMean = meanMotion;

      posVec.push_back(meanMotion);
      posX.push_back(meanMotion.i);
      if (meanMotion.i < crit)
          fLeft++;
      
         
      Image<PixRGB<byte> > mot= *itM;
      ++itM;
  }
  
  LINFO("time to compute motion position: %f",T.getSecs());
  LINFO("num frames on left %d, num frames on right %d",fLeft,fTotal-fLeft);
  LINFO("time on left %f, time on right %f",fLeft/25.0,(fTotal-fLeft)/25.0);
  
  
  Image<PixRGB<byte> > itsPlot = linePlot(posX, 400,500,0,0,
                                          "xPosition","position","time/s",
                                          PixRGB<byte>(0,0,0),
                                          PixRGB<byte>(255,255,255),5,0);
  
  itM = motion.begin();
  std::vector<Point2D<int> >::iterator posItr = posVec.begin();
  Point2D<int> meanMotion;
  layers.drawImage(itsPlot,0,240+10);
  for(std::vector<Image<PixRGB<byte> > >::iterator it = frames.begin();
      it!=frames.end()-1;++it)
  {
      
      Image<PixRGB<byte> > vid= *it;
      Image<PixRGB<byte> > mot= *itM++;
      mot= mot*5;
      meanMotion = *posItr++;
      drawCross(mot,meanMotion,PixRGB<byte>(255,255,0),3,2);
      drawLine(mot,Point2D<int>(crit,0),Point2D<int>(crit,imageDims.h()),
               PixRGB<byte>(255,0,0),2);
      layers.drawImage(rescale(mot,320,240),0,0);
          //layers.drawImage(rescale(vid,320,240),0,240 +10);
      
      
          // usleep(1000);
  }
  
  mgr->stop();  
  return 0;
}



//###########################getMotionVector############################//

std::vector<Point2D<int> > getMotionVector(const Image<float>& itsImg,
                                           float thresh)
{
    
    std::vector<Point2D<int> > motionVec;
    int cnt=0;
    
    for(Image<float>::const_iterator itr=itsImg.begin();itr!=itsImg.end();++itr)
    {
        if(*itr >= thresh)
            motionVec.push_back(Point2D<int>(cnt % itsImg.getWidth(), 
                                    (cnt-(cnt % itsImg.getWidth()))/itsImg.getWidth()));
        cnt++;
        
    }
    return motionVec;
    
    
}

//###########################getMotionVector############################//
Point2D<int> getMeanMotion(const Image<float>& itsImg,float thresh)

{
    
    std::vector<Point2D<int> > motionVec = getMotionVector(itsImg,thresh);    

    if(motionVec.size() == 0)
        return Point2D<int>(0,0);
    else
    {
        
        std::vector<int> x,y;
        std::vector<Point2D<int> >::const_iterator itr=motionVec.begin();
        
        while(itr!=motionVec.end())
        {
            Point2D<int> temp = *itr;
            x.push_back(temp.i);
            y.push_back(temp.j);
            ++itr;        
        }
        
        int meanX= std::accumulate(x.begin(),x.end(),0.0)/x.size();
        int meanY = std::accumulate(y.begin(),y.end(),0.0)/y.size();

        return Point2D<int>(meanX,meanY);
    }
    
}

         

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPMEDIA_APP_MOUSE_CHASER_C_DEFINED
