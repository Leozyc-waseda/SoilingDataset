/*!@file Beobot/locust_model2.C implement the locust model for collision detection with frame series*/

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/locust_model.C $
// $Id: locust_model.C 14376 2011-01-11 02:44:34Z pez $//


#include "Component/ModelManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/DeviceOpts.H"
#include "GUI/XWindow.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Image/MathOps.H"
#include "Image/ColorOps.H"
#include "Image/Convolver.H"
#include "Image/DrawOps.H"
#include "Media/FrameSeries.H"
#include "Media/MediaOpts.H"
#include "Raster/GenericFrame.H"


#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, const char **argv)
{
  //Instantiate a ModelManager:
  ModelManager *mgr = new ModelManager("locust model frame series style");

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(*mgr));
  mgr->addSubComponent(ifs);

  mgr->setOptionValString(&OPT_FrameGrabberMode, "RGB24");
  mgr->setOptionValString(&OPT_FrameGrabberDims, "320x240");
  mgr->setOptionValString(&OPT_FrameGrabberFPS, "30");

  mgr->exportOptions(MC_RECURSE);

 // Parse command-line:
  if (mgr->parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  //XWindow xwin(Dims(1050,490),-1, -1, "locust model");
  //Image<PixRGB<byte> > inputImg = Image<PixRGB<byte> >(imageDims.w(),imageDims.h()+20, ZEROS);

  Dims layer_screen(1150,490);
  XWindow layers(layer_screen, 0, 0, "layers"); //preview window

  // let's get all our ModelComponent instances started:
  mgr->start();

  //layers.drawImage(bg, 0, 0);

  Image<float> p_layer_image[4];
  Image<float> i_layer_image[3];
  Image<float> s_layer_image[2];

  int maxHistory = 50;
  std::vector<float> lgmdPotential(1);
  float total_s_layer = 0;
  int framecnt=1;
  std::vector<Image< PixRGB <byte> >  > frames(5);

  const FrameState is = ifs->updateNext();
  if(is == FRAME_COMPLETE)
      LFATAL("frames completed!");

   //grab the images
  frames[framecnt] = ifs->readRGB();
   if(!frames[framecnt].initialized())
     LFATAL("frame killed");


  Image<PixRGB<byte> > temp_p, temp_i, temp_s;
  Image<float>::iterator aptr;
  int potentialCnt = 1;


  while(1){

    if(framecnt!=1)
      frames[1] = frames[4]; //make last frame of pervious batch the 1st frame for this one

    framecnt=2;


    while(framecnt<=4)
      {

        const FrameState is = ifs->updateNext();
        if(is == FRAME_COMPLETE)
          break;

        //grab the images
        frames[framecnt] = ifs->readRGB();
        if(!frames[framecnt].initialized())
          break;


        //LINFO("drawing frame %d",framecnt);
        layers.drawImage(frames[framecnt],0,0);
        framecnt++;
      }
    framecnt--;



  //p_layer processing
    p_layer_image[1] = absDiff(luminance(frames[framecnt-2]),luminance(frames[framecnt-3]));//p(t-2)
    p_layer_image[2] = absDiff(luminance(frames[framecnt-1]),luminance(frames[framecnt-2]));//p(t-1)
    p_layer_image[3] = absDiff(luminance(frames[framecnt]),luminance(frames[framecnt-1]));//p(t)

    temp_p = p_layer_image[1];
    writeText(temp_p,Point2D<int>(2,0),"P-layer",
              PixRGB<byte>(255,0,0),PixRGB<byte>(0,0,0), SimpleFont::FIXED(9));
    layers.drawImage(temp_p,321,0);


  //i_layer processing

  //define kernel and  convolver for inhibition layer
    Image<float> kernel(3,3,NO_INIT);
    std::fill(kernel.beginw(),kernel.endw(),1.0F/9.0F);


    i_layer_image[1] = ((p_layer_image[1]) + (p_layer_image[2]))*0.25;
    Convolver c1(kernel,i_layer_image[1].getDims());
    i_layer_image[1] = c1.spatialConvolve(i_layer_image[1]);          //i(t-1)

    i_layer_image[2] = ((p_layer_image[2]) + (p_layer_image[3]))*0.25;
    Convolver c2(kernel,i_layer_image[2].getDims());
    i_layer_image[2] = c2.spatialConvolve(i_layer_image[2]);           //i(t)

    temp_i = i_layer_image[1];
    writeText(temp_i,Point2D<int>(2,0),"I-layer",
              PixRGB<byte>(255,0,0),PixRGB<byte>(0,0,0), SimpleFont::FIXED(9));
    layers.drawImage(temp_i,0,241);


  //s_layer processing
    s_layer_image[1] = p_layer_image[3] - ((i_layer_image[1])*2);
    temp_s = s_layer_image[1];
    writeText(temp_s,Point2D<int>(2,0),"S-layer",
              PixRGB<byte>(255,0,0),PixRGB<byte>(0,0,0), SimpleFont::FIXED(9));
    layers.drawImage(temp_s, 321,241 );


    aptr  = s_layer_image[1].beginw();

    for (int w = 0; w < s_layer_image[1].getDims().w(); w++)
      for(int h = 0; h < s_layer_image[1].getDims().h(); h++)
        total_s_layer += *aptr++;

    LINFO("%f", total_s_layer);

    float tempPot = 1/(1 + exp(-total_s_layer/(320*240)));
    //float tempPot = total_s_layer;
    if(potentialCnt < maxHistory)
      {
        lgmdPotential.push_back(tempPot);
        potentialCnt++;
      }
    else
      {
        lgmdPotential.erase(lgmdPotential.begin(), lgmdPotential.begin()+1);
        lgmdPotential.push_back(tempPot);
        potentialCnt++;
      }

    Image<PixRGB<byte> > temp_grid = linePlot(lgmdPotential, 400, 400,
                                              0.0f, 5.0f, "time");
    char str[256];
    sprintf(str, "lgmd potential = %f", tempPot);

    writeText(temp_grid,Point2D<int>(2,0),str,
              PixRGB<byte>(255,0,0),PixRGB<byte>(0,0,0), SimpleFont::FIXED(9));
    layers.drawImage(temp_grid, 675, 25);

  }


  //stop all our modelcomponents
  mgr->stop();

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */





