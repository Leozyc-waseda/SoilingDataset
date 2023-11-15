/*! @file ObjRec/test-handRec.C test for robot arm hand rec */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/test-handRec.C $
// $Id: test-handRec.C 10794 2009-02-08 06:21:09Z itti $
//


#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/ShapeOps.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/ColorOps.H"
#include "Image/Transforms.H"
#include "Image/MathOps.H"
#include "Image/Kernels.H"
#include "Image/fancynorm.H"
#include "Image/Layout.H"
#include "Transport/FrameInfo.H"
#include "Raster/Raster.H"
#include "Raster/GenericFrame.H"
#include "GUI/DebugWin.H"
#include "Neuro/EnvVisualCortex.H"
#include "Neuro/NeoBrain.H"
#include "Media/FrameSeries.H"
#include "Util/Timer.H"
#include "Image/OpenCVUtil.H"
#include "Devices/Scorbot.H"
#include "GUI/XWinManaged.H"
#include "RCBot/Motion/MotionEnergy.H"

#define UP  98
#define DOWN 104
#define LEFT 100
#define RIGHT 102


float w (float *p, int k)
{
        int i;
        float x=0.0;

        for (i=1; i<=k; i++) x += p[i];
        return x;
}

float u (float *p, int k)
{
        int i;
        float x=0.0;

        for (i=1; i<=k; i++) x += (float)i*p[i];
        return x;
}

float nu (float *p, int k, float ut, float vt)
{
        float x, y;

        y = w(p,k);
        x = ut*y - u(p,k);
        x = x*x;
        y = y*(1.0-y);
        if (y>0) x = x/y;
         else x = 0.0;
        return x/vt;
}


Image<float> segmentProb(const Image<float> &img)
{
  Image<float> retImg = img;
  int hist[260];
  float p[260];

  inplaceNormalize(retImg, 0.0F, 255.0F);

  for(int i=0; i<260; i++)
  {
    hist[i] = 0; //set histogram to 0
    p[i] = 0; //set prob to 0
  }

  for (int y=0; y<retImg.getHeight(); y++)
    for(int x=0; x<retImg.getWidth(); x++)
    {
      int val = (int)retImg.getVal(x,y);
      hist[val+1]++;
    }

  //nomalize into a distribution
  for (int i=1; i<=256; i++)
    p[i] = (float)hist[i]/(float)retImg.getSize();

  //find the global mean
  float ut = 0.0;
  for(int i=1; i<=256; i++)
    ut += (float)i*p[i];

  //find the global variance
  float vt = 0.0;
  for(int i=1; i<=256; i++)
    vt += (i-ut)*(i-ut)*p[i];

  int j=-1, k=-1;
  for(int i=1; i<=256; i++)
  {
    if ((j<0) && (p[i] > 0.0)) j = i; //first index
    if (p[i] > 0.0) k = i; //last index
  }

  float z = -1.0;
  int m = -1;
  for (int i=j; i<=k; i++)
  {
    float y = nu(p,i,ut,vt); //compute nu

    if (y>=z)
    {
      z = y;
      m = i;
    }
  }

  int t = m;

  if (t < 0)
    LINFO("ERROR");

  //threshold
  for (int y=0; y<retImg.getHeight(); y++)
    for(int x=0; x<retImg.getWidth(); x++)
    {
      int val = (int)retImg.getVal(x,y);
      if (val < t)
        retImg.setVal(x,y,0);
      else
        retImg.setVal(x,y,255.0);
    }


  return retImg;
}

Image<float> integralImage(const Image<float> &img)
{

  Image<float> integImg(img.getDims(), ZEROS);

  int xDim = integImg.getWidth();
  int yDim = integImg.getHeight();


  float s[xDim];
  for (int i=0; i<xDim; i++) s[i] = 0;

  for(int y=0; y<yDim; y++)
    for(int x=0; x<xDim; x++)
    {
      float ii = x > 0 ? integImg.getVal(x-1, y) : 0;
      s[x] += img.getVal(x,y);
      integImg.setVal(x,y, ii+s[x]);
    }

  return integImg;


}

Image<float> getHaarFeature(Image<float> &integImg, int i)
{

    Image<float> fImg(integImg.getDims(), ZEROS);

    /*int w = 2, h = 4;

    for(int y=0; y<fImg.getHeight()-2*w; y++)
      for(int x=0; x<fImg.getWidth()-h; x++)
      {
        float left  = integImg.getVal(x+w,y+h) + integImg.getVal(x,y) -
          (integImg.getVal(x+w,y) + integImg.getVal(x,y+h));
        float right = integImg.getVal(x+(2*w),y+h) + integImg.getVal(x+w,y) -
          (integImg.getVal(x+(2*w),y) + integImg.getVal(x+w,y+h));

        float top  = integImg.getVal(x,y) + integImg.getVal(x+h,y+w) -
          (integImg.getVal(x+h,y) + integImg.getVal(x,y+w));
        float bottom = integImg.getVal(x,y+w) + integImg.getVal(x+h,y+(2*w)) -
          (integImg.getVal(x+h,y+w) + integImg.getVal(x,y+(2*w)));


        fImg.setVal(x,y, fabs(left-right) + fabs(top-bottom));
      }*/

    int c = 6+i, s = 8+i;

    int x = 320/2, y=240/2;

    Rectangle rect(Point2D<int>(x,y),Dims(s,s));
    drawRect(fImg, rect, float(255.0));
    //for(int y=0; y<fImg.getHeight()-s; y++)
    //  for(int x=0; x<fImg.getWidth()-s; x++)
      {
        int d = (s-c)/2;
        float center  = integImg.getVal(x+d,y+d) + integImg.getVal(x+d+c,y+d+c) -
          (integImg.getVal(x+d,y+d+c) + integImg.getVal(x+d+c,y+d));
        float surround  = integImg.getVal(x,y) + integImg.getVal(x+s,y+s) -
          (integImg.getVal(x+s,y) + integImg.getVal(x,y+s));

        center /= c*2;
        surround /= s*2;
        //fImg.setVal(x,y, center-surround);
        //printf("%i %f\n", i, center-surround);
      }


    return fImg;

}

Image<float> getObj(Image<float> &integImg)
{

    //Image<float> fImg(integImg.getDims(), ZEROS);
    Image<float> objImg(integImg.getDims(), ZEROS);

    /*int w = 2, h = 4;

    for(int y=0; y<fImg.getHeight()-2*w; y++)
      for(int x=0; x<fImg.getWidth()-h; x++)
      {
        float left  = integImg.getVal(x+w,y+h) + integImg.getVal(x,y) -
          (integImg.getVal(x+w,y) + integImg.getVal(x,y+h));
        float right = integImg.getVal(x+(2*w),y+h) + integImg.getVal(x+w,y) -
          (integImg.getVal(x+(2*w),y) + integImg.getVal(x+w,y+h));

        float top  = integImg.getVal(x,y) + integImg.getVal(x+h,y+w) -
          (integImg.getVal(x+h,y) + integImg.getVal(x,y+w));
        float bottom = integImg.getVal(x,y+w) + integImg.getVal(x+h,y+(2*w)) -
          (integImg.getVal(x+h,y+w) + integImg.getVal(x,y+(2*w)));


        fImg.setVal(x,y, fabs(left-right) + fabs(top-bottom));
      }*/

    for(int y=6; y<objImg.getHeight()-60; y++)
      for(int x=6; x<objImg.getWidth()-60; x++)
      {
        int  c, s;
        float center, surround;
        for (int i=0; i<30; i++)
        {
          c = 3+(i/2);
          s = 6+i;

          int d = (s-c)/2;
          center  = integImg.getVal(x+d,y+d) + integImg.getVal(x+d+c,y+d+c) -
            (integImg.getVal(x+d,y+d+c) + integImg.getVal(x+d+c,y+d));
          surround  = integImg.getVal(x,y) + integImg.getVal(x+s,y+s) -
            (integImg.getVal(x+s,y) + integImg.getVal(x,y+s)) - center;

          center /= (c*c);
          surround /= ((s*s) - (c*c));
          if (fabs(surround-center) > 1) break;
        }

        Rectangle rectC(Point2D<int>(x+((s-c)/2),y+((s-c)/2)),Dims(c,c));
        Rectangle rectS(Point2D<int>(x,y),Dims(s,s));

        drawFilledRect(objImg, rectC, center);
      }


    return objImg;

}

float centerSurround(Image<byte> &img, Point2D<int> &p)
{
  int s = 10;
  int c = 5;

  float sum = 0;
  for(int y=p.j-s; y<p.j+s; y++)
    for(int x=p.i-s; x<p.i+s; x++)
    {
      if (p.i-c < x && x > p.i+c &&
          p.j-c < y && y > p.j+c )
      {
        sum += img.getVal(x,y);
        img.setVal(x,y,0);
      }
    }



  return sum;
}


int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager *mgr = new ModelManager("Test Brain");

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(*mgr));
  mgr->addSubComponent(ifs);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
  mgr->addSubComponent(ofs);

  nub::soft_ref<Scorbot> scorbot(new Scorbot(*mgr));
  mgr->addSubComponent(scorbot);

  nub::soft_ref<NeoBrain> neoBrain(new NeoBrain(*mgr));
  mgr->addSubComponent(neoBrain);

  mgr->exportOptions(MC_RECURSE);

  if (mgr->parseCommandLine(
        (const int)argc, (const char**)argv, "", 0, 0) == false)
    return 1;

  mgr->start();

  Dims imageDims = ifs->peekDims();
  neoBrain->init(imageDims, 50, 30);

  XWinManaged* xwin = new XWinManaged(Dims(512,512), -1, -1, "Bot Control");

  // do post-command-line configs:

  //start streaming
  //ifs->startStream();

  MotionEnergyPyrBuilder<byte> motionPyr(Gaussian5, 0.0f, 10.0f, 5,
                                         1500.0f);

  Image< PixRGB<byte> > inputImg;
  FrameState is = ifs->updateNext();
  //grab the images
  GenericFrame input = ifs->readFrame();
  inputImg = input.asRgb();

  Image<byte> lum = luminance(inputImg);

  while(1)
  {




    Point2D<int> clickLoc = xwin->getLastMouseClick();
    int key = xwin->getLastKeyPress();

    if (clickLoc.isValid())
    {
      float val = centerSurround(lum, clickLoc);
      printf("Val %f %i\n", val, lum.getVal(clickLoc));


    }
    switch (key)
    {
      case UP:
        is = ifs->updateNext();
        if (is == FRAME_COMPLETE)
          break;

        //grab the images
        input = ifs->readFrame();
        if (!input.initialized())
          break;
        inputImg = input.asRgb();
        lum = luminance(inputImg);
        break;
      case DOWN:
        break;
      case LEFT:
        break;
      case RIGHT:
        break;
      case 65:  // space
        break;
      case 38: // a
        break;
      case 52: //z
        break;
      case 39: //s
        break;
      case 53: //x
        break;
      case 40: //d
        break;
      case 54: //c
        break;
      case 41: //f
        break;
      case 55: //v
        break;

      default:
        if (key != -1)
          printf("Unknown Key %i\n", key);
        break;

    }


    ofs->writeRGB(inputImg, "input", FrameInfo("input", SRC_POS));

    //mag = segmentProb(mag);

    xwin->drawImage(lum);

  }


  // stop all our ModelComponents
  mgr->stop();

  return 0;

}
