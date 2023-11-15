/*!@file Beobot/test-followColor-master.C color segment following - master */

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
// Primary maintainer for this file:  T. Nathan Mundhenk <mundhenk@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/test-followColor-master.C $
// $Id: test-followColor-master.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Devices/CameraControl.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/KeyBoard.H"
#include "Image/Image.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "VFAT/segmentImageMerge.H"
#include "rutz/shared_ptr.h"
#include <cstdio>
#include <cstdlib>

// number of frames over which framerate info is averaged:
#define NAVG 20

// ######################################################################
//! The main routine. Grab frames, process, send commands to slave node.
int main(const int argc, const char **argv)
{
  // instantiate a model manager
  ModelManager manager( "Following Color Segments - Master" );

  nub::soft_ref<FrameGrabberConfigurator>
    gbc( new FrameGrabberConfigurator( manager ) );
  manager.addSubComponent( gbc );
  nub::soft_ref<CameraControl>
    camera( new CameraControl( manager, "Camera Controller", "CameraControl",
                               0, true, 0) );
  manager.addSubComponent( camera );
  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  // parse command-line
  if( manager.parseCommandLine( argc, argv, "", 0, 0 ) == false )
    {
      LFATAL( "Command-line parse error\n" );
    }

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
          "to be useful");

  // variables for segmenting and tracking
  int width = gb->getWidth(), height = gb->getHeight();
  float delay = 0;
  float H,S,V,Hs,Ss,Vs;
  float LOTcount = 0;

  // wait for key press 'A'
  KeyBoard myKB;
  while( myKB.getKey( true ) != KBD_KEY1 )
    printf("\a"); //beep

  // let's get all our ModelComponent instances started:
  manager.start();

  // timer initialization
  Timer tim; Image< PixRGB<byte> > ima; Image< PixRGB<float> > fima;
  Timer camPause;       // to pause the move command
  camPause.reset();
  uint64 t[NAVG]; int frame = 0;

  // configure segmenter and tracker
  segmentImage segment(HSV);
  segmentImageTrack track(1000, &segment);
  H = 200; Hs = 20;
  S = .70; Ss = .20;
  V = 150; Vs = 150;
  segment.setHue(H,Hs,0);
  segment.setSat(S,Ss,0);
  segment.setVal(V,Vs,0);
  segment.setHSVavg(15);
  segment.setFrame(0,0,width/4,height/4,width/4,height/4);

  while(1) {
    // start timing
    tim.reset();

    ima = gb->readRGB();

    uint64 t0 = tim.get();  // to measure display time

    // decimate image to 1/4 size
    fima = decXY(ima);
    fima = decXY(fima);

    // segment image
    segment.segment(fima);
    Image<byte> outputI = segment.returnNormalizedCandidates();
    segment.calcMassCenter();
    track.track(0);

    for(int i = 0; i < segment.numberBlobs(); i++)
    {
      if(track.isCandidate(i) == true)
      {
        segment.getHSVvalueMean(i,&H,&S,&V,&Hs,&Ss,&Vs);
        int tt = segment.getYmin(i); int bb = segment.getYmax(i);
        int ll = segment.getXmin(i); int rr = segment.getXmax(i);
        if((bb != tt) && (ll != rr))
          drawRect(ima, Rectangle::tlbrI(tt*4,ll*4,bb*4,rr*4),
                   PixRGB<byte>(255,255,0),1);
        drawCircle(ima, Point2D<int>((int)segment.getCenterX(i)*4
                                ,(int)segment.getCenterY(i)*4)
                   ,(int)sqrt((double)segment.getMass(i)),
                   PixRGB<byte>(0,0,255),2);
        drawCircle(ima, Point2D<int>((int)segment.getCenterX(i)*4
                                ,(int)segment.getCenterY(i)*4)
                   ,2,PixRGB<byte>(255,0,0),2);
      }
      if(track.returnLOT() == true)
      {
        if(LOTcount == 2)
        {
          H = 200; Hs = 20;
          S = .70; Ss = .20;
          V = 150; Vs = 150;
          LOTcount = 0;
        }
        else
        {
          LOTcount++;
        }
      }
      segment.setHue(H,(Hs*3),0);
      segment.setSat(S,(Ss*3),0);
      segment.setVal(V,(Vs*3),0);
    }
    drawCircle(ima, Point2D<int>((int)track.getObjectX()*4
                            ,(int)track.getObjectY()*4)
               ,2,PixRGB<byte>(0,255,0));

    if(camPause.get() > delay)
    {
      LINFO( "Object mass: %d", track.getMass() );
      int modi = (int)track.getObjectX()*8;
      int modj = 480-((int)track.getObjectY()*8);
      if(modi > 0 && modi < 640 && modj > 0 && modj < 480)
      {
        if(!track.returnLOT() &&
           track.getMass() < 2400 && track.getMass() > 30 )
        {
          /* // send speed and steer command to Board B
          if( car->getSpeed() < 0.18 )
            car->setSpeed( car->getSpeed() + 0.01 );
          car->setSteering( 1.0f * 1/320 * ( modi - 320 ) );
          */
          LINFO( "Steering to %f", 1.0f * 1/320 * ( modi - 320 ) );
        }
        else
          {
            /* // send speed and steer command to Board B
            car->setSpeed( 0.0 );
            car->setSteering( 0.0 );
            */
            LINFO("Loss of Track, stopping");
          }
      }
    }

    // display segment image if option was specified
    ofs->writeRGB(ima, "input");
    ofs->writeGray(outputI, "normalizedCandidates");

    // compute and show framerate over the last NAVG frames:
    t[frame % NAVG] = tim.get();
    t0 = t[frame % NAVG] - t0;
    if (t0 > 28) LINFO("Display took %llums", t0);

    if (frame % NAVG == 0 && frame > 0)
      {
        uint64 avg = 0; for (int i = 0; i < NAVG; i ++) avg += t[i];
        float avg2 = 1000.0 / (float)avg * NAVG;
        printf("Framerate: %.1f fps\n", avg2);
      }
    frame ++;
  }


  /************************ MERGING CHANGES??? ******************

  // allocate X-Windows if we are using them
  XWindow *wini = NULL;
  XWindow *wino1 = NULL;
  XWindow *wino2 = NULL;
  if( usingX )
    {
      wini = new XWindow( Dims(width, height), 0, 0, "test-input window" );
      wino1 = new XWindow( Dims(width/4, height/4), 0, 0, "test-output window 1" );
      wino2 = new XWindow( Dims(width/4, height/4), 0, 0, "test-output window 2" );
    }

  Timer tim;
  Image< PixRGB<byte> > ima;
  Timer camPause;       // to pause the move command
  camPause.reset();

  uint64 t[NAVG]; int frame = 0;
  segmentImageMerge segmenter(2);
  // set up tracking parameters
  segmenter.setTrackColor(10,10,0.15,0.20,150,150,0,true,15);
  //segmenter.setTrackColor(10,10,0.15,0.20,150,150,1,false,15);
  segmenter.setTrackColor(270,10,0.18,0.25,60,60,1,true,15);
  segmenter.setAdaptBound(15,5,.30,.25,140,100,0);
  segmenter.setAdaptBound(285,265,.25,.15,80,40,1);
  segmenter.setFrame(0,0,width/4,height/4,width/4,height/4,0);
  segmenter.setFrame(0,0,width/4,height/4,width/4,height/4,1);
  segmenter.setCircleColor(0,255,0,0);
  segmenter.setCircleColor(0,0,255,1);
  segmenter.setBoxColor(255,255,0,0);
  segmenter.setBoxColor(255,0,255,1);

  segmenter.setAdapt(3,true,3,true,3,true,0);
  segmenter.setAdapt(3,true,3,true,3,true,1);

  while(1) {
    tim.reset();
    ima = gb->readRGB();
    uint64 t0 = tim.get();  // to measure display time

    Image<PixRGB<byte> > Aux1;
    Image<PixRGB<byte> > Aux2;
    Aux1.resize(100,450,true);
    Aux2.resize(100,450,true);

    Image<byte> outputI1;
    Image<byte> outputI2;

    display = ima;
    segmenter.trackImage(ima,&display,&outputI1,0,&Aux1);
    segmenter.trackImage(ima,&display,&outputI2,1,&Aux2);
    segmenter.mergeImages(&display);

    if(camPause.get() > delay)
    {
      int modi,modj;
      segmenter.getImageTrackXY(&modi,&modj,0);
      //segmenter.getImageTrackXYMerge(&modi,&modj);
      modi = modi*8;
      modj = 480-modj*8;
      if(modi > 0 && modi < 640 && modj > 0 && modj < 480)
      {
        if(segmenter.returnLOT(0) == false)
        {
          camPause.reset();

          // send the speed and steer command to Board B
          //car->setSpeed( 0.18f );
          //car->setSteering( 0.5f * 1/320 * (modi-320) );


          // delay = camera->moveCamXYFrame(modi,modj);
        }
 else
        {
          LINFO("Loss of track, stopping");

          // send speed and steer command to Board B
          //car->setSpeed( 0.0f );
          //car->setSteering( 0.0f );
        }
      }
    }

    // display segment image in XWindow if option was specified
    if( usingX )
    {
      wini->drawImage(ima);
      wino1->drawImage(outputI1);
      wino2->drawImage(outputI2);
    }
    t[frame % NAVG] = tim.get();
    t0 = t[frame % NAVG] - t0;
    if (t0 > 28) LINFO("Display took %llums", t0);

    // compute and show framerate over the last NAVG frames:
    if (frame % NAVG == 0 && frame > 0)
      {
        uint64 avg = 0; for (int i = 0; i < NAVG; i ++) avg += t[i];
        float avg2 = 1000.0 / (float)avg * NAVG;
        printf("Framerate: %.1f fps\n", avg2);
      }
    frame++;
  }
  ********** END MERGING CHANGES????*********************************/

  manager.stop();

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
