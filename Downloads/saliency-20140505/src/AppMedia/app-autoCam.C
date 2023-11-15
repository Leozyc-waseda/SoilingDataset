/*!@file AppMedia/app-autoCam.C */

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
// Primary maintainer for this file: Lior Elazary
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-autoCam.C $
// $Id: app-autoCam.C 10794 2009-02-08 06:21:09Z itti $
//

#include "Component/ModelManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/PixelsTypes.H"
#include "Image/DrawOps.H"
#include "Image/ShapeOps.H"
#include "Image/ColorOps.H"
#include "Raster/Raster.H"
#include "Transport/FrameIstream.H"
#include "Neuro/EnvVisualCortex.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define NAVG 20

void findMinMax(const std::vector<int> &vec, double &min, double &max)
{
  max = vec[0];
  min = max;
  for (uint n = 1 ; n < vec.size() ; n++)
  {
    if (vec[n] > max) max = vec[n];
    if (vec[n] < min) min = vec[n];
  }
}

std::vector<int> getHistogram(const Image<byte> &lum)
{
  std::vector<int> hist(256,0);
  for(int i=0; i<lum.getSize(); i++)
    hist[lum[i]]++;

  return hist;
}

Image<PixRGB<byte> > showHist(const std::vector<int> &hist)
{
  int w = 256, h = 256;
  if (hist.size() > (uint)w) w = hist.size();

  if (hist.size() == 0) return Image<PixRGB<byte> >();

  int dw = w / hist.size();
  Image<byte> res(w, h, ZEROS);

  // draw lines for 10% marks:
  for (int j = 0; j < 10; j++)
    drawLine(res, Point2D<int>(0, int(j * 0.1F * h)),
             Point2D<int>(w-1, int(j * 0.1F * h)), byte(64));
  drawLine(res, Point2D<int>(0, h-1), Point2D<int>(w-1, h-1), byte(64));

  double minii, maxii;
  findMinMax(hist, minii, maxii);

   // uniform histogram
  if (maxii == minii) minii = maxii - 1.0F;

  double range = maxii - minii;

  for (uint i = 0; i < hist.size(); i++)
    {
      int t = abs(h - int((hist[i] - minii) / range * double(h)));

      // if we have at least 1 pixel worth to draw
      if (t < h-1)
        {
          for (int j = 0; j < dw; j++)
            drawLine(res,
                     Point2D<int>(dw * i + j, t),
                     Point2D<int>(dw * i + j, h - 1),
                     byte(255));
          //drawRect(res, Rectangle::tlbrI(t,dw*i,h-1,dw*i+dw-1), byte(255));
        }
    }
  return res;
}

double getMSV(const std::vector<int>& hist)
{

  double topSum=0, sum=0;

  double h[4];
  for(int i=0; i<4; i++)
    h[i] = 0;

  for(uint i=0; i<hist.size(); i++)
    h[i/(256/4)] += hist[i];

  for(int i=0; i<4; i++)
  {
    topSum += (i+1)*h[i];
    sum += h[i];
  }

  return topSum/sum;
}

int contrastCtrl(double cVal, double dVal)
{
  double k=0;
  return (int)((dVal-cVal)*k);
}

int brightCtrl(double cVal, double dVal)
{
  double k=100;
  return (int)((dVal-cVal)*k);
}

/*! This simple executable tests video frame grabbing through the
  video4linux driver (see V4Lgrabber.H) or the IEEE1394 (firewire)
  grabber (see IEEE1394grabber.H). Selection of the grabber type is
  made via the --fg-type=XX command-line option. */
int main(const int argc, const char **argv)
{
  // instantiate a model manager:
  ModelManager manager("Frame Grabber Tester");

  nub::ref<EnvVisualCortex> evc(new EnvVisualCortex(manager));
  manager.addSubComponent(evc);

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);


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
  Timer tim; uint64 t[NAVG]; int frame = 0;

  // get the frame grabber to start streaming:
  gb->startStream();


  int count = 0;
  char* buf = new char[9999];

  int cont = 20000;
  int bright = 20000;
  while(1) {
    count++;
    tim.reset();

    Image< PixRGB<byte> > ima = gb->readRGB();
    sprintf(buf, "frame%d.png", count);

    uint64 t0 = tim.get();  // to measure display time
    ofs->writeRGB(ima, "image", FrameInfo("image", SRC_POS));
    t[frame % NAVG] = tim.get();
    t0 = t[frame % NAVG] - t0;
    if (t0 > 20000ULL) LINFO("Display took %lluus", t0);

    //evc->input(ima);
    //Image<byte> smap = evc->getVCXmap();

    Image<byte> smap = luminance(ima);

    std::vector<int> hist = getHistogram(smap);
    Image<PixRGB<byte> > histImg = showHist(hist);
    ofs->writeRGB(histImg, "hist", FrameInfo("hist", SRC_POS));
    ofs->writeRGB(rescale(smap, ima.getDims()), "smap", FrameInfo("smap", SRC_POS));

    double histVal = getMSV(hist);

    cont += contrastCtrl(histVal, 2.5);
    bright += brightCtrl(histVal, 2.5);

    LINFO("MSV %f cont %i bright %i", histVal, cont, bright);

    gb->setModelParamVal("FrameGrabberBrightness", bright, MC_RECURSE);
    //gb->setModelParamVal("FrameGrabberContrast", cont, MC_RECURSE);

    // compute and show framerate over the last NAVG frames:
    if (frame % NAVG == 0 && frame > 0)
      {
        uint64 avg = 0ULL; for (int i = 0; i < NAVG; i ++) avg += t[i];
        float avg2 = 1000.0F / float(avg) * float(NAVG);
        printf("Framerate: %.1f fps\n", avg2);
      }
    frame ++;
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
