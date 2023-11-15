/*!@file AppMedia/enhanceImageGamma.C program to enhance salience of image at given locations  */

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
// Primary maintainer for this file: Farhan Baluch <fbaluch at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/enhanceImageGamma.C $
// $Id: enhanceImageGamma.C 10794 2009-02-08 06:21:09Z itti $
//

#ifndef APPMEDIA_ENHANCEIMAGEGAMMA_C_DEFINED
#define APPMEDIA_ENHANCEIMAGEGAMMA_C_DEFINED

#include "Component/ModelManager.H"
#include "GUI/XWinManaged.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H" // for rescale()
#include "Image/FilterOps.H" //for correlation();
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/ColorOps.H"
#include "Neuro/getSaliency.H"
#include "Raster/Raster.H"
#include "Util/log.H"
#include "Psycho/EyeTrace.H"
#include "Image/MathOps.H"
#include "Util/StringConversions.H"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>


//this code will take as input an image file and increase the salience of a region of an image by using the toPow function (gamma)


int main(int argc, char** argv)
{
  ModelManager manager("Enhance gamma on image");

  nub::ref<GetSaliency> saliency(new GetSaliency(manager));
  manager.addSubComponent(saliency);
  if (manager.parseCommandLine(argc, argv, "<imagefile> <specDirectory> <outputfile> gamma", 4, 4) == false)
    return -1;


  manager.start();

  Dims dims(1000,1000);
  double gam = atof(manager.getExtraArg(3).c_str());


  Image<PixRGB<float> > inputImg(1920,1080,NO_INIT);
  inputImg = Raster::ReadRGB(manager.getExtraArg(0).c_str());

  Image<float> r, g, b;
  getComponents(inputImg, r, g, b);

  Image<float> r2 = toPower(r, gam);
  Image<float> g2 = toPower(g, gam);
  Image<float> b2 = toPower(b, gam);

  inplaceNormalize(r2, 0.0f, 255.0f);
  inplaceNormalize(g2, 0.0f, 255.0f);
  inplaceNormalize(b2, 0.0f, 255.0f);


  Image<PixRGB<byte> > dispImg(640,360,NO_INIT);
  Image<PixRGB<byte> > enhancedImg(1920,1080,NO_INIT);

  enhancedImg = makeRGB(Image<byte>(r2), Image<byte>(g2),
                       Image<byte>(b2));


  dispImg = rescale(enhancedImg,dispImg.getDims());

  const int numSalientSpotsTr = saliency->compute(enhancedImg, SimTime::SECS(1));

  std::string temp ="found"  + convertToString(numSalientSpotsTr) + " salient spots";

  const Image<float> salmap = saliency->getSalmap();
  writeText(dispImg,Point2D<int>(5,5),temp.c_str());

  XWinManaged *imgWin,*modImg;
  CloseButtonListener wList;
  modImg =new XWinManaged(Dims(1920,1080),0,0, manager.getExtraArg(0).c_str());
  imgWin = new XWinManaged(dims,0,0, manager.getExtraArg(0).c_str());
  wList.add(imgWin);
  wList.add(modImg);


  imgWin->drawImage(dispImg,0,0);
  imgWin->drawImage(rescale(salmap,dispImg.getDims()),0,360);

  modImg->drawImage(enhancedImg);

  char filename[255];
  sprintf(filename,"/lab/farhan/research/exp1/stimMod/%fgamma.png",gam);

  Raster::WriteRGB(enhancedImg,filename);


  Raster::waitForKey();

  manager.stop();

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPMEDIA_ENHANCEIMAGEGAMMA_C_DEFINED
