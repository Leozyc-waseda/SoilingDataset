/*!@file AppPsycho/createBarStimuli.C program to generate bar stimuli for pulvinar experiment  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/createBarStimuli.C $
// $Id: createBarStimuli.C 10794 2009-02-08 06:21:09Z itti $
//

#ifndef APPNEURO_CREATEBARSTIMULI_C_DEFINED
#define APPNEURO_CREATEBARSTIMULI_C_DEFINED

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

//this code will generate bar stimuli for pulvinar experiment.

int main(int argc, char** argv)
{
  ModelManager manager("create stimuli for pulvinar experiment");

  nub::ref<GetSaliency> saliency(new GetSaliency(manager));
  manager.addSubComponent(saliency);

  if (manager.parseCommandLine(argc, argv, "<orientation1> <orientation2>", 2, 2) == false)
    return -1;
  manager.start();


  Dims dims(700,700);
  Image<PixRGB<byte> > inputImg(600,600,NO_INIT);

  Image<PixRGB<byte> >::iterator aptr=inputImg.beginw();

  while(aptr!= inputImg.end())
      *aptr++ = PixRGB<byte>(127.0,127.0,127.0);

  int numBars=8;
  int hSpacing=50,vSpacing=50,beginX=100,beginY = 100;
  double or1, or2;
  or1 = atof(manager.getExtraArg(0).c_str());
  or2 = atof(manager.getExtraArg(1).c_str());

for(int i=0; i<numBars;i++)
      for(int j=0;j<numBars;j++)
      {
          if(i==2 && j==3)
              drawLine(inputImg, Point2D<int>(beginX+j*hSpacing,beginY+i*vSpacing),or1*M_PI/180,25.0,PixRGB<byte>(255.0,255.0,0.0),2);
          else if (i==5 && j==6)
          drawLine(inputImg, Point2D<int>(beginX+j*hSpacing,beginY+i*vSpacing),or2*M_PI/180,25.0,PixRGB<byte>(255.0,255.0,0.0),2);
          else
          drawLine(inputImg, Point2D<int>(beginX+j*hSpacing,beginY+i*vSpacing),20*M_PI/180,25.0,PixRGB<byte>(255.0,255.0,0.0),2);

      }



    const int numSalientSpotsTr = saliency->compute(inputImg, SimTime::SECS(1));

  std::string temp ="found"  + convertToString(numSalientSpotsTr) + " salient spots";

  LINFO("%s",temp.c_str());

  const Image<float> OrigSalmap = saliency->getSalmap();
//  writeText(OrigSalMap,Point2D<int>(5,5),temp.c_str());


  XWinManaged *imgWin;
  CloseButtonListener wList;
  imgWin = new XWinManaged(dims,0,0, manager.getExtraArg(0).c_str());
  wList.add(imgWin);
  imgWin->drawImage(inputImg);

  char filename[255],salname[255];
  sprintf(filename,"/lab/farhan/research/pulvinar/sampleDisp.png");
  sprintf(salname,"/lab/farhan/research/pulvinar/sampleDispSal.png");

  Raster::WriteRGB(inputImg,filename);
  Raster::WriteFloat(rescale(OrigSalmap,inputImg.getDims()),FLOAT_NORM_0_255,salname);

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

#endif // APPPSYCHO_CREATEBARSTIMULI_C_DEFINED
