/*!@file RCBot/trackCMap.C track featuers */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/trackCMap.C $
// $Id: trackCMap.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "Component/ModelManager.H"
#include "Component/OptionManager.H"
#include "Controllers/PID.H"
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

Image<float> ComputeCMAP(const Image<float>& fimg, const PyramidType ptyp, const float ori, const float coeff, Point2D<int>& p, float *bias, bool find_bias)
{
  // compute pyramid:
  ImageSet<float> pyr = buildPyrGeneric(fimg, 0, maxdepth, ptyp, ori);

  // alloc conspicuity map and clear it:
  Image<float> *cmap = new Image<float>(pyr[sml].getDims(), ZEROS);


  // intensities is the max-normalized weighted sum of IntensCS:
  int ii=0;
  for (int delta = delta_min; delta <= delta_max; delta ++)
    for (int lev = level_min; lev <= level_max; lev ++)
      {
        Image<float> tmp = centerSurround(pyr, lev, lev + delta, true);
                  inplaceNormalize(tmp, 0.0F, 255.0F);
                  if (find_bias){
                          //bias the saliency map
                          for (Image<float>::iterator itr = tmp.beginw(), stop = tmp.endw();
                                itr != stop; ++itr) {
                                      *itr = 255.0F - fabs((*itr) - bias[ii]); //corelate the bias
                          }

                  } else if (p.isValid()){
                          //get the bias points
                          Point2D<int> newp(p.i / (cmap->getWidth()/tmp.getWidth()),
                                                                  p.j / (cmap->getHeight()/tmp.getHeight()));

                          bias[ii] = tmp.getVal(newp);
                  }

        //tmp = downSize(tmp, cmap->getWidth(), cmap->getHeight());
        tmp = rescale(tmp, cmap->getWidth(), cmap->getHeight());
        //inplaceAddBGnoise(tmp, 255.0);
        tmp = maxNormalize(tmp, MAXNORMMIN, MAXNORMMAX, normtyp);
        *cmap += tmp;
                  ii++;
      }
  if (normtyp == VCXNORM_MAXNORM)
    *cmap = maxNormalize(*cmap, MAXNORMMIN, MAXNORMMAX, normtyp);
  else
    *cmap = maxNormalize(*cmap, 0.0f, 0.0f, normtyp);

  // multiply by conspicuity coefficient:
  *cmap *= coeff;

  return *cmap;
}

// ######################################################################
int main(int argc, char **argv)
{
  MYLOGVERB = LOG_INFO;

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

  float *lum_bias = new float[6];
  float *r_bias = new float[6];
  float *b_bias = new float[6];
  float *g_bias = new float[6];

  float *o0_bias = new float[6];
  float *o45_bias = new float[6];
  float *o90_bias = new float[6];
  float *o135_bias = new float[6];

  float *oldlum_bias = new float[6];
  float *oldr_bias = new float[6];
  float *oldb_bias = new float[6];
  float *oldg_bias = new float[6];

  float *oldo0_bias = new float[6];
  float *oldo45_bias = new float[6];
  float *oldo90_bias = new float[6];
  float *oldo135_bias = new float[6];
  bool track_f = false;

  int fr = 40;
  while(key != 24){
      // receive conspicuity maps:
      // grab an image:

      Image< PixRGB<byte> > ima = gb->readRGB();
           /*char filename[255];
                sprintf(filename, "/home/elazary/robot/car/video/frame%i.ppm", fr);
      Image< PixRGB<byte> > ima = Raster::ReadRGB(filename);*/



                Image<float> lum = luminance(ima);
                Image<byte> rImg, gImg, bImg, yImg;
      getRGBY(ima, rImg, gImg, bImg, yImg, (byte)25);

                Image<float> cmap;

                Point2D<int> location = window.getLastMouseClick();
                if (location.isValid()){
                        LINFO("Click at at %i %i", location.i, location.j);
                        Dims WindowDims = window.getDims();

                        float newi = (float)location.i * (float)ima.getWidth()/(float)WindowDims.w();
                        float newj = (float)location.j * (float)ima.getHeight()/(float)WindowDims.h();
                        location.i = (int)newi;
                        location.j = (int)newj;

                        //we got a click, show the featuers at that point
                        LINFO("Getting the features at %i %i", location.i, location.j);
                   ComputeCMAP(lum, Gaussian5, 0.0, 1.0F, location, lum_bias, false);
                   ComputeCMAP(rImg, Gaussian5, 0.0, 1.0F, location, r_bias, false);
                   ComputeCMAP(gImg, Gaussian5, 0.0, 1.0F, location, g_bias, false);
                   ComputeCMAP(bImg, Gaussian5, 0.0, 1.0F, location, b_bias, false);

                        ComputeCMAP(lum, Oriented5, 0.0, 1.0F, location, o0_bias, false); //ori thread
                        ComputeCMAP(lum, Oriented5, 45.0, 1.0F, location, o45_bias, false); //ori thread
                        ComputeCMAP(lum, Oriented5, 90.0, 1.0F, location, o90_bias, false); //ori thread
                        ComputeCMAP(lum, Oriented5, 135.0, 1.0F, location, o135_bias, false); //ori thread

                        track_f = true;
                }

                cmap = ComputeCMAP(lum, Gaussian5, 0.0, 1.0F, location, lum_bias, 1);
                cmap += ComputeCMAP(rImg, Gaussian5, 0.0, 1.0F, location, r_bias, 1);
                cmap += ComputeCMAP(gImg, Gaussian5, 0.0, 1.0F, location, g_bias, 1);
                cmap += ComputeCMAP(bImg, Gaussian5, 0.0, 1.0F, location, b_bias, 1);

                cmap += ComputeCMAP(lum, Oriented5, 0.0, 1.0F, location, o0_bias, 1); //ori thread
                cmap += ComputeCMAP(lum, Oriented5, 45.0, 1.0F, location, o45_bias, 1); //ori thread
                cmap += ComputeCMAP(lum, Oriented5, 90.0, 1.0F, location, o90_bias, 1); //ori thread
                cmap += ComputeCMAP(lum, Oriented5, 135.0, 1.0F, location, o135_bias, 1); //ori thread

                for (int i=0; i<6; i++){
                        oldlum_bias[i] = lum_bias[i];
                        oldr_bias[i] = r_bias[i];
                        oldg_bias[i] = g_bias[i];
                        oldb_bias[i] = b_bias[i];
                        oldo0_bias[i] = o0_bias[i];
                        oldo45_bias[i] = o45_bias[i];
                        oldo90_bias[i] = o90_bias[i];
                        oldo135_bias[i] = o135_bias[i];
                }


                Point2D<int> winner; float maxval;
                findMax(cmap, winner, maxval);
                drawDisk(ima, winner, 10, PixRGB<byte>(255, 0, 0));

                window.drawImage(rescale(ima, 256, 256));
                window1.drawImage(rescale(cmap, 256, 256));

                if (track_f){
                        LINFO("Getting the features from winner at %i %i", winner.i, winner.j);
                        ComputeCMAP(lum, Gaussian5, 0.0, 1.0F, winner, lum_bias, 0);
                        ComputeCMAP(rImg, Gaussian5, 0.0, 1.0F, winner, r_bias, 0);
                        ComputeCMAP(gImg, Gaussian5, 0.0, 1.0F, winner, g_bias, 0);
                        ComputeCMAP(bImg, Gaussian5, 0.0, 1.0F, winner, b_bias, 0);

                        ComputeCMAP(lum, Oriented5, 0.0, 1.0F, winner, o0_bias, 0); //ori thread
                        ComputeCMAP(lum, Oriented5, 45.0, 1.0F, winner, o45_bias, 0); //ori thread
                        ComputeCMAP(lum, Oriented5, 90.0, 1.0F, winner, o90_bias, 0); //ori thread
                        ComputeCMAP(lum, Oriented5, 135.0, 1.0F, winner, o135_bias, 0); //ori thread
                }

                //find how close is the featuere from the old one
                double fmag = 0;
                for(int i=0; i<6; i++){
                        fmag += squareOf(oldlum_bias[i] - lum_bias[i]);
                        fmag += squareOf(oldr_bias[i] - r_bias[i]);
                        fmag += squareOf(oldg_bias[i] - g_bias[i]);
                        fmag += squareOf(oldb_bias[i] - b_bias[i]);
                        fmag += squareOf(oldo0_bias[i] - o0_bias[i]);
                        fmag += squareOf(oldo45_bias[i] - o45_bias[i]);
                        fmag += squareOf(oldo90_bias[i] - o90_bias[i]);
                        fmag += squareOf(oldo135_bias[i] - o135_bias[i]);
                }
                fmag = sqrt(fmag);
                LINFO("Distance %f\n", fmag);
                if (fmag > 200){
                        //featuer are too far from what we are tracking
                        //restore to orignal features
                        LINFO("Restoring featuers");
                        for (int i=0; i<6; i++){
                                lum_bias[i] = oldlum_bias[i];
                                r_bias[i] = oldr_bias[i];
                                g_bias[i] = oldg_bias[i];
                                b_bias[i] = oldb_bias[i];
                                o0_bias[i] = oldo0_bias[i];
                                o45_bias[i] = oldo45_bias[i];
                                o90_bias[i] = oldo90_bias[i];
                                o135_bias[i] = oldo135_bias[i];
                        }
                }



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
