/*!@file BeoSub/test-sampleH2SV2.C Grab image from camra and test for H2SV1 pixel data */

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
// Primary maintainer for this file: Zack Gossman <gossman@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-sampleH2SV2.C $
// $Id: test-sampleH2SV2.C 8145 2007-03-20 19:48:53Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "GUI/XWindow.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define NAVG 20

/*! This test grabs from a camera and then finds the average RGB and H2SV1 pixel values acros a predetermined area */
int main(const int argc, const char **argv)
{
  // instantiate a model manager:
  ModelManager manager("H2SV2 Tester");

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

                gb->setModelParamVal("FrameGrabberSubChan", 0);
                gb->setModelParamVal("FrameGrabberBrightness", 128);
gb->setModelParamVal("FrameGrabberHue", 180);
  //Load in config file for camera FIX: put in a check whether config file exists!
  manager.loadConfig("camconfig.pmap");

  // let's get all our ModelComponent instances started:
  manager.start();

  //int imgWidth = gb->getWidth();
 // int imgHeight = gb->getHeight();

  float avgR = 0.0;
  float avgG = 0.0;
  float avgB = 0.0;
  float avgH1 = 0.0;
  float avgH2 = 0.0;
  float avgS = 0.0;
  float avgV = 0.0;

  int count = 0;

        int sampleWidth = 20;
        int sampleHeight = 20;
        int centerX = gb->getWidth()/2;// gb->getWidth()/2; //sampleWidth + 1;
         int centerY = gb->getHeight()/2; //70; // gb->getHeight()/2; //imgHeight/2;

  // get ready for main loop:
  XWindow win(gb->peekDims(), 0, 0, "test-sample window");
  Timer tim; uint64 t[NAVG]; int frame = 0;

  // get the frame grabber to start streaming:
  gb->startStream();

  while(1) {
    count++;
    tim.reset();

    Image< PixRGB<byte> > ima = gb->readRGB();

    //draw rectangle from which the sample will be taken


   /* uint64 t0 = tim.get();  // to measure display time
    t[frame % NAVG] = tim.get();
    t0 = t[frame % NAVG] - t0;
    if (t0 > 20000ULL) LINFO("Display took %lluus", t0);*/


    // compute and show framerate over the last NAVG frames:
    if (frame % NAVG == 0 && frame > 0)
      {
        uint64 avg = 0ULL; for (int i = 0; i < NAVG; i ++) avg += t[i];
        // NOTE: commented out this unused variable to avoid compiler warnings
        // float avg2 = 1000.0F / float(avg) * float(NAVG);
        //printf("Framerate: %.1f fps\n", avg2);
      }
    frame ++;

        float h1[2];
        float h2[2];
        float s[2];
        float v[2];
        h1[0] = h1[1] = h2[0] = h2[1] = s[0] = s[1] = v[0] = v[1] = -1;

        float r[2], g[2], b[2];
        r[0] = r[1] = g[0] = g[1] = b[0] = b[1] = -1;

        float temp = 0;
    for(int i = -sampleWidth/2; i< sampleWidth/2; i++){
      for(int j = -sampleHeight/2; j < sampleHeight/2; j++){



        avgR += (ima.getVal((centerX)+i, (centerY)+j)).red();
        avgG += (ima.getVal((centerX)+i, (centerY)+j)).green();
        avgB += (ima.getVal((centerX)+i, (centerY)+j)).blue();

        avgH1 += (PixH2SV2 <float> (ima.getVal((centerX)+i, (centerY)+j))).H1();
        temp = (PixH2SV2 <float> (ima.getVal((centerX)+i, (centerY)+j))).H1();
        if (temp < h1[0] || h1[0] == -1) h1[0] = temp;
        if (temp > h1[1] || h1[1] == -1) h1[1] = temp;

        avgH2 += (PixH2SV2 <float> (ima.getVal((centerX)+i, (centerY)+j))).H2();
        temp = (PixH2SV2 <float> (ima.getVal((centerX)+i, (centerY)+j))).H2();
         if (temp < h2[0] || h2[0] == -1) h2[0] = temp;
        if (temp > h2[1] || h2[1] == -1) h2[1] = temp;

       avgS += (PixH2SV2 <float> (ima.getVal((centerX)+i, (centerY)+j))).S();
        temp = (PixH2SV2 <float> (ima.getVal((centerX)+i, (centerY)+j))).S();
        if (temp < s[0] || s[0] == -1) s[0] = temp;
        if (temp > s[1] || s[1] == -1) s[1] = temp;


        avgV += (PixH2SV2 <float> (ima.getVal((centerX)+i, (centerY)+j))).V();
        temp = (PixH2SV2 <float> (ima.getVal((centerX)+i, (centerY)+j))).V();
        if (temp < v[0]|| v[0] == -1) v[0] = temp;
        if (temp > v[1]|| v[1] == -1) v[1] = temp;


        //ima.setVal(centerX+i, centerY+j, PixRGB <byte> (0, 0, 255));
      }
    }

    drawRectEZ(ima, Rectangle::tlbrI((centerY)-sampleHeight/2,(centerX)-sampleWidth/2,(centerY)+sampleHeight/2,(centerX)+sampleWidth/2),
               PixRGB<byte>(225,
                            20,
                            20),2);
    win.drawImage(ima);

    if(count == 20){
      count = 0;
      avgR /= 8000; avgG /= 8000; avgB /= 8000;
      printf("\nR: %f G: %f B: %f \n", avgR, avgG, avgB);
      avgH1 /= 8000; avgH2 /= 8000; avgS /= 8000; avgV /= 8000;
      printf("H1: %f H2: %f S: %f V: %f\n", avgH1, avgH2, avgS, avgV);
      avgR = 0; avgG = 0; avgB = 0; avgH1 = 0; avgH2 = 0; avgS = 0; avgV = 0;
      printf("Max H1: %f H2: %f S: %f V: %f\n", h1[1], h2[1], s[1], v[1]);

      printf("Min H1: %f H2: %f S: %f V: %f\n", h1[0], h2[0], s[0], v[0]);

   }

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
