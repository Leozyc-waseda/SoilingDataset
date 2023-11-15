/*!@file AppDevices/locust_videoAndLaser.H record from high speed camera XC and
Hokuyo Laser Range finder and save to disk*/

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
// Primary maintainer for this file: farhan
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/locust_videoAndLaser.C $
// $Id: locust_videoAndLaser.C 12962 2010-03-06 02:13:53Z irock $
//
#include "Component/ModelManager.H"
#include "Devices/RangeFinder.H"
#include "Component/OptionManager.H"
#include "Devices/Serial.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/DeviceOpts.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/ShapeOps.H"
#include "Image/CutPaste.H"
#include "Image/ImageCache.H"
#include "Image/Pixels.H"
#include "Image/MathOps.H"
#include "Image/IO.H"
#include "Image/Layout.H"
#include "GUI/SDLdisplay.H"
#include "GUI/GUIOpts.H"
#include "Raster/Raster.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Transport/FrameInfo.H"
#include "Util/Timer.H"
#include "Util/log.H"
#include "Util/sformat.H"
#include "Video/RgbConversion.H" // for toVideoYUV422()
#include "Raster/DeBayer.H" // for debayer()
#include "Raster/PlaintextWriter.H"
#include "Robots/LoBot/io/LoLaserRangeFinder.H"
#include <pthread.h>
#include <algorithm>
#include <iterator>
#include <queue>
#include <fstream>
#include <iostream>

using namespace std;

#define MAXSAVETHREAD 4
std::vector<std::string> base;
ImageCache<byte> cacheVideo;
ImageCache<PixRGB<byte> > cacheLaserVid;
std::queue<Image<int> > cacheLaser;
pthread_mutex_t qmutex_cache,qmutex_timer, qmutex_laser;
lobot::LaserRangeFinder* lrf = new lobot::LaserRangeFinder();
bool haveLaser = false, haveFrames = false;
uint frameNum = 0,laserNum=0;
std::ofstream laserData, metaVideo, metaLaser;
Timer tim;


static void* saveLaser(void *arg)
{

  while(1)
    {
      Image<int> dists;
      Image<int>::iterator aptr, stop;
      float rad;
      int  min,max;
      long int startAng;
      Image<PixRGB<byte> > laserImg(512,512,ZEROS);
      uint64 t;

      lrf->update();
      dists = lrf->get_distances();
      pthread_mutex_lock(&qmutex_laser);
      cacheLaser.push(dists);
      haveLaser = true;
      laserNum++;

      pthread_mutex_unlock(&qmutex_laser);

      //write to file
      const int w = dists.getWidth();
      const int h = dists.getHeight();

      Image<int>::const_iterator itr = dists.begin();
      for (int y = 0; y < h; ++y)
        {
          for (int x = 0; x < w; ++x)
            {
              laserData<<" "<<*itr;
              ++itr;
            }
          laserData<<"\n";
        }

      laserData<<"\n";

      t = tim.get();
      metaLaser<<t<<" \t"<<laserNum<<std::endl;

      //try drawing here to see if its faster that way

      if(!cacheLaser.empty())
        {
          dists = cacheLaser.front();
          cacheLaser.pop();
          haveLaser = false;
          aptr = dists.beginw();
          stop = dists.endw();
          startAng = -141;

          while(aptr!=stop)
            {
              rad = *aptr++;
              //some scaling
              getMinMax(dists, min, max);
              rad = ((rad - min)/(max-min))*256;

              if (rad < 0)
                rad =1;

              Point2D<int> pt;
              pt.i = 256 - (int) (rad*sin(startAng*M_PI/180.0));
              pt.j = 256 - (int) (rad*cos(startAng*M_PI/180.0));

              drawCircle(laserImg, pt, 3, PixRGB<byte>(255,0,0));
              drawLine(laserImg, Point2D<int>(256,256),pt,
                       PixRGB<byte>(0,255,0));
              startAng = startAng+1;
              if(startAng > 140)
                startAng = -140;
            }
          pthread_mutex_lock(&qmutex_laser);
          cacheLaserVid.push_back(laserImg);
          pthread_mutex_unlock(&qmutex_laser);
        }
    }

  return NULL;
}

static void* saveFrames(void *arg)
{

  while(1)
  {

    Image<byte> savImg;
    uint64 t;
    haveFrames = false;
    pthread_mutex_lock(&qmutex_cache);
    if(cacheVideo.size())
      {
        savImg = cacheVideo.pop_front();
        frameNum++;

        if(cacheVideo.size()) haveFrames = true;

        if (savImg.initialized())
          {
            const char *b = base[frameNum % (base.size()-3)].c_str();
            Raster::WriteGray(savImg, sformat("%s%06u.pgm", b, frameNum));
          }
        t = tim.get();
        metaVideo<<t<<"\t"<<frameNum<<std::endl;
      }
    pthread_mutex_unlock(&qmutex_cache);
    if(haveFrames == false) usleep(1000);
  }

  return NULL;
}

static int submain(const int argc, char** argv)
{
  // Image<PixRGB<byte> > dispImg(640,1024,ZEROS);
  Image<byte> ima(640,480,ZEROS);
  pthread_t frameStreams[MAXSAVETHREAD], laserStream;
  uint64 t;

  // instantiate a model manager:
  ModelManager manager("locust recording camera + laser");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::soft_ref<FrameGrabberConfigurator>
  gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  manager.setOptionValString(&OPT_FrameGrabberType, "XC");
  manager.setOptionValString(&OPT_FrameGrabberDims, "640x478");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<basename1> <basename2> <basename3> <basename 4>..<laserdatafile> <metaDataVideo> <metaDataLaser>",4, MAXSAVETHREAD+3) == false)
    return(1);

 // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();

  if (gb.isInvalid())
    LFATAL("You need to have XC camera and XClibrary");

  // get the basename:
  for (uint i = 0; i < manager.numExtraArgs(); i ++)
    base.push_back(manager.getExtraArg(i));

  manager.start();

  //start streaming camera
  gb->startStream();

  laserData.open(base[4].c_str(), ios::out | ios::ate);
  metaVideo.open(base[5].c_str(), ios::out | ios::ate);
  metaLaser.open(base[6].c_str(), ios::out | ios::ate);

  for(int ii = 0; ii<(int)base.size()-3; ii++)
    pthread_create(frameStreams+ii, NULL, &saveFrames, (void *)NULL);

  //Laser Range finder setup
  //declare a new laserRangeFinder object
  int rc2 = pthread_create(&laserStream, NULL, &saveLaser, NULL);
  if(rc2)
    {
      LFATAL("cannot create thread");
      exit(-1);
    }

  Image<PixRGB<byte> > laserImg(512,512,ZEROS);

  while(1)
  {
      //read from grabber
      ima = gb->readGray();
      laserImg.clear();

      pthread_mutex_lock(&qmutex_cache);
      cacheVideo.push_back(ima);
      pthread_mutex_unlock(&qmutex_cache);

      pthread_mutex_lock(&qmutex_laser);
      if(cacheLaserVid.size())
        laserImg = cacheLaserVid.pop_front();
      pthread_mutex_unlock(&qmutex_laser);

      Image<PixRGB<byte> > imgByte = ima;
      Layout<PixRGB<byte> >  dispLayout(imgByte);
      dispLayout = vcat(dispLayout,laserImg);
      //inplacePaste(dispImg, imgByte, Point2D<int>(0,0));
      //inplacePaste(dispImg, laserImg, Point2D<int>(0,490));

      ofs->writeRgbLayout(dispLayout,"Output",FrameInfo("output",SRC_POS));

      t = tim.get();
      if(t >5000)
        {
          laserData.close();
          metaVideo.close();
          metaLaser.close();
          manager.stop();
          exit(1);
        }
  }

        laserData.close();
        metaVideo.close();
        metaLaser.close();
        manager.stop();
}



extern "C" int main(const int argc, char** argv)
{
  try
    {
      return submain(argc, argv);
    }
  catch (...)
    {
      REPORT_CURRENT_EXCEPTION;
    }

  return 1;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */


