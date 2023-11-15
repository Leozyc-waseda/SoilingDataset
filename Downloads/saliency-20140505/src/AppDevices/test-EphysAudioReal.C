/*!@file AppDevices/test-EphysAudio.C Test ephys board programmed as USB input audio device */

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
// Primary maintainer for this file: Farhan Baluch <fbaluch@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-EphysAudio.C $
// $Id: $
//

#include "Audio/AudioWavFile.H"
#include "Component/ModelManager.H"
#include "Devices/AudioGrabber.H"
#include "Devices/AudioMixer.H"
#include "GUI/XWindow.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Util/log.H"
#include "Util/sformat.H"
#include "Util/Timer.H"
#include <cstdio>
#include <signal.h>
#include <queue>
#include <fstream>

static bool goforever = true;  //!< Will turn false on interrupt signal
#define SCALE 4

std::queue<AudioBuffer<uint16> > rec;
std::queue<AudioBuffer<uint16> > dispQ;
pthread_mutex_t qmutex_data;
pthread_mutex_t qmutex_disp;

std::vector<AudioBuffer<uint16> > rec2;

uint nsamples;
int totalSamps=0;
int numBuffs=0;
int nChans =0;
Timer tim; 

//! Signal handler (e.g., for control-C)
void terminate(int s) { LERROR("*** INTERRUPT ***"); goforever = false; }


//##############################################################

//! main
int main(int argc, const char **argv)
{
  // setup signal handling:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);

  // Instantiate a ModelManager:
  ModelManager manager("Test Model for AudioGrabber Class");

  // Instantiate our various ModelComponents:
  nub::soft_ref<AudioMixer> mix(new AudioMixer(manager));
  manager.addSubComponent(mix);

  nub::soft_ref<AudioGrabber> agb(new AudioGrabber(manager));
  manager.addSubComponent(agb);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);
  
  //int recTime = atoi(manager.getExtraArg(0).c_str());
  // Let's get some of the option values, to configure our window:
  nsamples = agb->getModelParamVal<uint>("AudioGrabberBufSamples");
  nChans = agb->getModelParamVal<uint>("AudioGrabberChans");
  
 
  // let's get all our ModelComponent instances started:
  manager.start();

  
  tim.reset();
  AudioBuffer<uint16> buffer;


  int plotHeight = 128;
  XWindow win(Dims(nsamples, plotHeight * nChans),
                 -1, -1, "test-audioGrab window");
  Image<byte> ima(nsamples, plotHeight*nChans, NO_INIT);
  Image<byte> imaPrev(nsamples, plotHeight*nChans, NO_INIT);
    
  Point2D<int> p1, p2;
  
  double a1,a2,b0,b1,b2;
  a1 = -1.9923;
  b1 = a1;
  a2 = 0.9937;
  b0 = 0.9969;
  b2 = b0;

  double yt,xt,xt2,xtm1,xtm2,ytm1,ytm2;
  xtm1 = 0;
  xtm2 = xtm1;
  ytm1 = xtm1;
  ytm2 = xtm1;

  // main loop:
  while(goforever) {

    // grab a buffer: 
    agb->grab(buffer);
    // LINFO("Buff size: %d",buffer.size());
    
    ima.clear();
    //display the stuff
    // draw the buffer into the image:
    for (int c=0; c < nChans; c++)
      for (uint i = 1; i < nsamples; i ++)
        {
          
          xt = (buffer.getVal(i-1,c))*(2.5F/65536);
          yt = b2*xtm2 + b1*xtm1 + b0*xt - a2*ytm2 - a1*ytm1;
          
          float val = xt; 
          xtm2 = xtm1;
          xtm1 = xt;
          
          ytm2 = ytm1;
          ytm1 = yt;
          
          xt2 = (buffer.getVal(i,c))*(2.5F/65536);
          yt = b2*xtm2 + b1*xtm1 + b0*xt2 - a2*ytm2 - a1*ytm1;
          
          float val1 = xt2;
          
          float factor=plotHeight/2.5F;
          p1.i = i - 1;
          p1.j = plotHeight*(c+1) - val1*factor;
          p2.i = i;
          p2.j = plotHeight*(c+1) - val*factor;
          
          writeText(ima,Point2D<int>(4,plotHeight*(c+1)-plotHeight-5),
                    sformat("%f %f",val,tim.getSecs()).c_str());
          drawLine(ima, p1, p2, byte(255));
        } 

    
    // display the image:
    win.drawImage(ima);
    
 
  }

  LINFO("writing out audio file: testAudio.wav");
  LINFO("Num buffs processed %d",numBuffs);
  // save our current records
  writeAudioWavFile("testAudio.wav", rec2);

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
