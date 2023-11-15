/*!@file NeovisionII/objRec-Server.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/NeovisionII/objRec-server.C $
// $Id: objRec-server.C 13901 2010-09-09 15:12:26Z lior $
//

#ifndef OBJREC_SERVER_C_DEFINED
#define OBJREC_SERVER_C_DEFINED
#include <stdlib.h>
#include <stdio.h>
#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Media/FrameSeries.H"
#include "NeovisionII/nv2_common.h"
#include "NeovisionII/nv2_label_server.h"
#include "Util/sformat.H"
#include "Image/FilterOps.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/ShapeOps.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Learn/SOFM.H"
#include "GUI/DebugWin.H"

#include <iostream> // for std::cin
#include <signal.h>

bool debug = 0;
bool terminate = false;
struct nv2_label_server* server;

void terminateProc(int s)
{
  LINFO("Ending application\n");
  nv2_label_server_destroy(server);
  terminate = true;
  exit(0);
}

void findMinMax(const std::vector<double> &vec, double &min, double &max)
{
  max = vec[0];
  min = max;
  for (uint n = 1 ; n < vec.size() ; n++)
  {
    if (vec[n] > max) max = vec[n];
    if (vec[n] < min) min = vec[n];
  }
}

Image<PixRGB<byte> > showHist(const std::vector<double> &hist, int loc)
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

void smoothHist(std::vector<double> &hist)
{
  const uint siz = hist.size();
  float vect[siz];

  for (uint n = 0 ; n < siz ; n++)
  {
    float val0 = hist[ (n-1+siz) % siz ];
    float val1 = hist[ (n  +siz) % siz ];
    float val2 = hist[ (n+1+siz) % siz ];

    vect[n] = 0.25F * (val0 + 2.0F*val1 + val2);
  }

  for (uint n = 0 ; n < siz ; n++) hist[n] = vect[n];
}

void normalizeHist(std::vector<double> &hist, double high, double low)
{

  double oldmin, oldmax;
  findMinMax(hist, oldmin, oldmax);

   float scale = float(oldmax) - float(oldmin);
   //if (fabs(scale) < 1.0e-10F) scale = 1.0; // image is uniform
   const float nscale = (float(high) - float(low)) / scale;

   for(uint i=0; i<hist.size(); i++)
   {
     hist[i] = low + (float(hist[i]) - float(oldmin)) * nscale ;
   }


}


//get the max orientation of pixel given
//its neighbors

float getOriProb(
    const Image<float> &mag,
    const Image<float> &ori,
    int x, int y)
{

   float eMag = mag.getVal(x,y);
   float eOri = ori.getVal(x,y);

   //TODO: remove, should be only 0 if neighbors are zero
   if (eMag == 0) return 0;


   //look at the neigbors to determin the prob

   float maxProb = 0;
   for (int i=-1; i<=1; i++)
     for(int j=-1; j<=1; j++)
     {
       if (i==0 && j==0) continue; //don't compare ourself
       float nMag = mag.getVal(x+i, y+j);
       float nOri = ori.getVal(x+i, y+j);

       float prob = (1/fabs(eMag-nMag));
       //float prob = 1/(fabs(eMag-nMag)*fabs(eOri-nOri));
       if (prob > maxProb)
       {
         maxProb = prob;
       }


       LDEBUG("%ix%i,%ix%i| E: %f:%f  N:%f:%f %f",
           i,j,
           x,y,
           eMag, eOri,
           nMag, nOri,
           prob);
     }

   float prob = eMag * maxProb;
   if (prob > 1000) prob = 1000;
   LDEBUG("Max prob %f\n", prob);



   return prob;


}

Image<float> getEdgeProb(
    const Image<float> &mag,
    const Image<float> &ori)
{

  Image<float> retEdge(mag.getDims(), NO_INIT);
  for(int y=1; y<mag.getHeight()-1; y++)
    for(int x=1; x<mag.getWidth()-1; x++)
    {
      float pOri = getOriProb(mag, ori, x, y);

      retEdge.setVal(x, y, pOri);
    }

  inplaceNormalize(retEdge, 0.0F, 255.0F);

  SHOWIMG(mag, true);
  SHOWIMG(ori, true);
  //for(int i=0; i<retEdge.getSize(); i++)
  //  LINFO("%f", retEdge[i]);
  return retEdge;
}


int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager *mgr = new ModelManager("Test ObjRec");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
  mgr->addSubComponent(ofs);


  if (mgr->parseCommandLine(
        (const int)argc, (const char**)argv, "<Network file> <server ip>", 2, 2) == false)
    return 1;

  //// catch signals and redirect them to terminate for clean exit:
  //signal(SIGHUP, terminateProc); signal(SIGINT, terminateProc);
  //signal(SIGQUIT, terminateProc); signal(SIGTERM, terminateProc);
  //signal(SIGALRM, terminateProc);

  mgr->start();

  //get command line options
  //const char *bayesNetFile = mgr->getExtraArg(0).c_str();
  const char *server_ip = mgr->getExtraArg(1).c_str();

  server = nv2_label_server_create(9930,
        server_ip,
        9931);

  nv2_label_server_set_verbosity(server,1); //allow warnings


  while(!terminate)
  {

    struct nv2_image_patch p;
    const enum nv2_image_patch_result res =
      nv2_label_server_get_current_patch(server, &p);

    std::string objName = "nomatch";
    if (res == NV2_IMAGE_PATCH_END)
    {
      fprintf(stdout, "ok, quitting\n");
      break;
    }
    else if (res == NV2_IMAGE_PATCH_NONE)
    {
      usleep(10000);
      continue;
    }
    else if (res == NV2_IMAGE_PATCH_VALID &&
       p.type == NV2_PIXEL_TYPE_RGB24)
    {

      const Image<PixRGB<byte> > im((const PixRGB<byte>*) p.data,
          p.width, p.height);

      Image<byte> lum = luminance(im);


      Image<float> mag, ori;
      gradientSobel(lum, mag, ori, 3);

      Image<float> edgeProb = getEdgeProb(mag,ori);

      ofs->writeRGB(im, "headInput");
      ofs->writeGray(mag, "Sobel");
      ofs->writeGray(edgeProb, "Edges");

    }

    nv2_image_patch_destroy(&p);

    sleep(1);
  }

  nv2_label_server_destroy(server);

}

#endif
