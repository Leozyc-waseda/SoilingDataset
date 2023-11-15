/*!@file AppDevices/test-HSV.C Test HSV histogramming of framegrabber input */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-HSV.C $
// $Id: test-HSV.C 8145 2007-03-20 19:48:53Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "GUI/XWindow.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/log.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define NAVG 20

/*! This simple executable tests video frame grabbing through the
  video4linux driver (see V4Lgrabber.H) or the IEEE1394 (firewire)
  grabber (see IEEE1394grabber.H). Selection of the grabber type is
  made via the --fg-type=XX command-line option. */
int main(const int argc, const char **argv)
{
  // instantiate a model manager:
  ModelManager manager("Frame Grabber Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful");

  // let's get all our ModelComponent instances started:
  manager.start();

  // get ready for main loop:
  XWindow win(gb->peekDims(), -1, -1, "test-HSV window");
  XWindow histoH(Dims(425,210), -1, -1, "Histogram Hue");
  XWindow histoS(Dims(425,210), -1, -1, "Histogram Satutation");
  XWindow histoV(Dims(425,210), -1, -1, "Histogram Value");

  int sizeX, sizeY;
  int winXleft, winXright, winYtop, winYbottom;
  long pixels;
  float H,S,V;
  //float Hs,Ss,Vs;
  float Hmax,Smax,Vmax;
  int Hhist[200];
  int Shist[200];
  int Vhist[200];

  PixRGB<float> pix;
  Image<PixRGB<byte> > HistoH;
  Image<PixRGB<byte> > HistoS;
  Image<PixRGB<byte> > HistoV;
  while(1) {
    HistoH.resize(425,210,true);
    HistoS.resize(425,210,true);
    HistoV.resize(425,210,true);

    for(int i = 0; i < 200 ; i++)
    {
      Hhist[i] = 0;
      Shist[i] = 0;
      Vhist[i] = 0;
    }
    H = 0; S = 0; V = 0;
    Hmax = 0; Smax = 0; Vmax = 0;
    pixels = 0;
    Image< PixRGB<byte> > ima = gb->readRGB();
    sizeX = ima.getWidth();
    sizeY = ima.getHeight();
    winXleft = sizeX/3;
    winXright = sizeX - sizeX/3;
    winYtop = sizeY/3;
    winYbottom = sizeY - sizeY/3;
    for(int i = winXleft; i < winXright; i++)
    {
      for(int j = winYtop; j < winYbottom; j++)
      {
        float pixH,pixS,pixV;
        int doH,doS,doV;
        pix = PixRGB<float>(ima.getVal(i,j));
        PixHSV<float>(pix).getHSV(pixH,pixS,pixV);
        doH = (int)((pixH/360)*200);
        doS = (int)(pixS*200);
        doV = (int)((pixV/255)*200);
        Hhist[doH]++;
        Shist[doS]++;
        Vhist[doV]++;
        H += pixH;
        S += pixS;
        V += pixV;
        pixels++;
      }
    }
    H = H/pixels;
    S = S/pixels;
    V = V/pixels;
    for(int i = 0; i < 200 ; i++)
    {
      if(Hhist[i] > Hmax) Hmax = Hhist[i];
      if(Shist[i] > Smax) Smax = Shist[i];
      if(Vhist[i] > Vmax) Vmax = Vhist[i];
    }
    for(int i = 0; i < 200 ; i++)
    {
      drawRect(HistoH, Rectangle::tlbrI(1,((i+1)*2-1),(int)((Hhist[i]/Hmax)*200.0F)+2,((i+1)*2)),
               PixRGB<byte>(255,0,0),1);
      drawRect(HistoS, Rectangle::tlbrI(1,((i+1)*2-1),(int)((Shist[i]/Smax)*200.0F)+2,((i+1)*2)),
               PixRGB<byte>(0,255,0),1);
      drawRect(HistoV, Rectangle::tlbrI(1,((i+1)*2-1),(int)((Vhist[i]/Vmax)*200.0F)+2,((i+1)*2)),
               PixRGB<byte>(0,0,255),1);
    }
    drawRect(ima, Rectangle::tlbrI(winYtop,winXleft,winYbottom,winXright),
             PixRGB<byte>(255,255,0),1);
    LINFO("H %f S %f V %f",H,S,V);
    win.drawImage(ima);
    histoH.drawImage(HistoH);
    histoS.drawImage(HistoS);
    histoV.drawImage(HistoV);
  }
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
