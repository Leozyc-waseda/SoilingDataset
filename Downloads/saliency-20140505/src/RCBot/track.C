/*!@file RCBot/track.C track featuers */

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
// Primary maintainer for this file: Lior Elazary <lelazary@yahoo.com>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/track.C $
// $Id: track.C 15310 2012-06-01 02:29:24Z itti $
//

#include "Component/ModelManager.H"
#include "Component/OptionManager.H"
#include "Controllers/PID.H"
#include "Corba/Objects/CMap.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/sc8000.H"
#include "GUI/XWinManaged.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"
#include "Image/fancynorm.H"
#include "Neuro/SaliencyMap.H"
#include "Neuro/VisualCortex.H"
#include "Neuro/WTAwinner.H"
#include "Neuro/WinnerTakeAll.H"
#include "RCBot/Motion/MotionEnergy.H"
#include "Raster/Raster.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"
#include <math.h>

#define UP_KEY 98
#define DOWN_KEY 104
#define LEFT_KEY 100
#define RIGHT_KEY 102

XWinManaged window(Dims(256, 256), -1, -1, "Test Output 1");
XWinManaged window1(Dims(256, 256), -1, -1, "S Map");
XWinManaged window2(Dims(256, 256), -1, -1, "S Map");

#define sml        0
#define delta_min  3
#define delta_max  4
#define level_min  0
#define level_max  2
#define maxdepth   (level_max + delta_max + 1)
#define normtyp    (VCXNORM_MAXNORM)

// ######################################################################
int main(int argc, char **argv)
{
  MYLOGVERB = LOG_DEBUG;

  CORBA::ORB_ptr orb = CORBA::ORB_init(argc,argv,"omniORB4");

  // instantiate a model manager:
  ModelManager manager("Track Features");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  // Parse command-line:
  if (manager.parseCommandLine(argc, (const char **)argv, "", 0, 0) == false) return(1);

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.get() == NULL)
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful -- ABORT");
 // int w = gb->getWidth(), h = gb->getHeight();

  // let's get all our ModelComponent instances started:
  manager.start();


  // get the frame grabber to start streaming:
  gb->startStream();


  // ########## MAIN LOOP: grab, process, display:
  int key = 0;

  double time = 0;


  int fr = 0;
  Image<float> cmap[8];

  CMapThreads cmap_th(orb);

  int cmapFrames[8];
  for(int i=0; i<8; i++)
          cmapFrames[i]=0;

  std::vector<float> bias[8];
  while(key != 24){
      // receive conspicuity maps:
      // grab an image:

      Image< PixRGB<byte> > ima = gb->readRGB();


                Image<byte> lum = luminance(ima);
                Image<byte> rImg, gImg, bImg, yImg;
      getRGBY(ima, rImg, gImg, bImg, yImg, (byte)25);


                Point2D<int> location = window.getLastMouseClick();
                if (location.isValid()){
                        LINFO("Click at at %i %i", location.i, location.j);
                        Dims WindowDims = window.getDims();

                        float newi = (float)location.i * (float)ima.getWidth()/(float)WindowDims.w();
                        float newj = (float)location.j * (float)ima.getHeight()/(float)WindowDims.h();
                        location.i = (int)newi;
                        location.j = (int)newj;

                        LINFO("Getting bias from at at %i %i", location.i, location.j);
                        bias[0] = cmap_th.getBias(cmap[0], lum,  Gaussian5, 0.0, 1.0, cmapFrames[0], location);
                        bias[1] = cmap_th.getBias(cmap[1], rImg, Gaussian5, 0.0, 1.0, cmapFrames[1], location);
                        bias[2] = cmap_th.getBias(cmap[2], gImg, Gaussian5, 0.0, 1.0, cmapFrames[2], location);
                        bias[3] = cmap_th.getBias(cmap[3], bImg, Gaussian5, 0.0, 1.0, cmapFrames[3], location);

                        bias[4] = cmap_th.getBias(cmap[4], lum, Oriented5, 0.0, 1.0, cmapFrames[4], location);
                        bias[5] = cmap_th.getBias(cmap[5], lum, Oriented5, 45.0, 1.0, cmapFrames[5], location);
                        bias[6] = cmap_th.getBias(cmap[6], lum, Oriented5, 90.0, 1.0, cmapFrames[6], location);
                        bias[7] = cmap_th.getBias(cmap[7], lum, Oriented5, 135.0, 1.0, cmapFrames[7], location);

                }


                // CMapWorkerThread* workth[8];
                LINFO("Bias size %" ZU , bias[0].size());
                if (bias[0].size() == 6){
/*
                        workth[0] = cmap_th.newth(cmap[0], lum,  Gaussian5, 0.0, 1.0, cmapFrames[0], bias[0]);
                        workth[1] = cmap_th.newth(cmap[1], rImg, Gaussian5, 0.0, 1.0, cmapFrames[1], bias[1]);
                        workth[2] = cmap_th.newth(cmap[2], gImg, Gaussian5, 0.0, 1.0, cmapFrames[2], bias[2]);
                        workth[3] = cmap_th.newth(cmap[3], bImg, Gaussian5, 0.0, 1.0, cmapFrames[3], bias[3]);
                        workth[4] = cmap_th.newth(cmap[4], lum, Oriented5, 0.0, 1.0, cmapFrames[4],  bias[4]);
                        workth[5] = cmap_th.newth(cmap[5], lum, Oriented5, 45.0, 1.0, cmapFrames[5], bias[5]);
                        workth[6] = cmap_th.newth(cmap[6], lum, Oriented5, 90.0, 1.0, cmapFrames[6], bias[6]);
                        workth[7] = cmap_th.newth(cmap[7], lum, Oriented5, 135.0, 1.0, cmapFrames[7],bias[7]);
*/
                        /*//wait for threads
                        for (int i=0; i<1; i++){
                                workth[i]->join(NULL);
                        }*/
                }

                /*cmap_th.newth(cmap[0], lum,  Gaussian5, 0.0, 1.0, cmapFrames[0]);
                cmap_th.newth(cmap[1], rImg, Gaussian5, 0.0, 1.0, cmapFrames[1]);
                cmap_th.newth(cmap[2], gImg, Gaussian5, 0.0, 1.0, cmapFrames[2]);
                cmap_th.newth(cmap[3], bImg, Gaussian5, 0.0, 1.0, cmapFrames[3]);

                cmap_th.newth(cmap[4], lum, Oriented5, 0.0, 1.0, cmapFrames[4]);
                cmap_th.newth(cmap[5], lum, Oriented5, 45.0, 1.0, cmapFrames[5]);
                cmap_th.newth(cmap[6], lum, Oriented5, 90.0, 1.0, cmapFrames[6]);
                cmap_th.newth(cmap[7], lum, Oriented5, 135.0, 1.0, cmapFrames[7]);*/




                Image<float> sminput;

                for (int i=0; i< 8; i++){
                        if (cmap[i].initialized()){
                                if (sminput.initialized())
                                        sminput += cmap[i];
                                else
                                        sminput = cmap[i];
                        }
                }


                LINFO("Frames %i: %i %i %i %i %i %i %i %i\n",
                                fr, cmapFrames[0], cmapFrames[1],cmapFrames[2],
                                cmapFrames[3],cmapFrames[4],cmapFrames[5],
                                cmapFrames[6],cmapFrames[7]);

                Point2D<int> winner(-1, -1); float maxval;
                if (sminput.initialized()){
                        findMax(sminput, winner, maxval);
                        LINFO("Max %f at %i %i\n", maxval, winner.i, winner.j);
                        if (winner.isValid())
                                drawDisk(ima, winner, 10, PixRGB<byte>(255, 0, 0));

                        window1.drawImage(rescale(sminput, 256, 256));
                }


                window.drawImage(rescale(ima, 256, 256));

                time += 0.1;
                fr++;
  }



  // got interrupted; let's cleanup and exit:
  manager.stop();
  return 0;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
