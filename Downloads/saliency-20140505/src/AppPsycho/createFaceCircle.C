/*!@file AppPsycho/createFaceCircle.C                       */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/createFaceCircle.C $
// $Id: createFaceCircle.C 10794 2009-02-08 06:21:09Z itti $
//
//loads a set of files with faces and arranges them in a circle with each face appearing an even number of times.

#include "Component/ModelManager.H"
#include "GUI/XWinManaged.H"
#include "Image/Image.H"
#include "Image/CutPaste.H"
#include "Image/Kernels.H"     // for gaborFilter()
#include "Image/MathOps.H"
#include "Image/Transforms.H"
#include "Image/ShapeOps.H"
#include "Image/DrawOps.H"
#include "Util/log.H"
#include "Raster/Raster.H"
#include "Raster/PngWriter.H"
#include <stdio.h>
#include <algorithm>
#include <time.h>

int main(int argc, char** argv)
{

  //instantiate a model manager
  ModelManager manager("AppPsycho: gabor search stimuli");

  //dimensions of window
  Dims dims(1920,1080);
  char filename[255],fname[255];
  if (manager.parseCommandLine(argc, argv,"<name>", 1, 1) == false)
    return(1);

  manager.start();

  // get command line parameters
  sscanf(argv[1],"%s",fname);

  int numFaces =16;

  XWinManaged *imgWin;
  CloseButtonListener wList;
  imgWin  = new XWinManaged(dims,    0,    0, manager.getExtraArg(0).c_str());
  wList.add(imgWin);

  //lets create some positions in circular fashion;

  std::vector<Point2D> pos(360);
  int posCnt =0,radius=300;
  int centerX = 1960/2;
  int centerY = 1080/2;


  pos[0] = Point2D((int)(radius*cos(0) + centerX), (int)(radius*sin(0) + centerY));
  pos[1] = Point2D((int)(radius*cos(M_PI/4) + centerX), (int)(radius*sin(M_PI/4) + centerY));
  pos[2] = Point2D((int)(radius*cos(M_PI/2) + centerX), (int)(radius*sin(M_PI/2) + centerY));
  pos[3] = Point2D((int)(radius*cos(3*M_PI/4) + centerX), (int)(radius*sin(3*M_PI/4) + centerY));
  pos[4] = Point2D((int)(radius*cos(M_PI) + centerX), (int)(radius*sin(M_PI) + centerY));
  pos[5] = Point2D((int)(radius*cos(5*M_PI/4) + centerX), (int)(radius*sin(5*M_PI/4) + centerY));
  pos[6] = Point2D((int)(radius*cos(3*M_PI/2) + centerX), (int)(radius*sin(3*M_PI/2) + centerY));
  pos[7] = Point2D((int)(radius*cos(7*M_PI/4) + centerX), (int)(radius*sin(7*M_PI/4) + centerY));

  posCnt = 7;

  std::vector< Image<PixRGB<byte> > > ivector(numFaces);
  Image<PixRGB<byte> > dispImg(dims,NO_INIT);
  Dims standardDims(200,200);

   Image<PixRGB<byte> >::iterator aptr= dispImg.beginw();

  while(aptr!=dispImg.endw())
    *aptr++ = PixRGB<byte>(127,127,127);

  imgWin->drawImage(dispImg);

  for(int i=1;i<=posCnt+1;i++)
    {
      sprintf(filename, "%s/slide%d.jpg",fname,i);
      ivector[i] = Raster::ReadRGB(filename);
       imgWin->drawImage(rescale(ivector[i],standardDims),pos[i-1].i,pos[i-1].j);
    //imgWin->drawImage(rescale(ivector[i],standardDims),i*10,i*10);
    }

  Raster::waitForKey();
  //finish all
  manager.stop();

  return 0;

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */





