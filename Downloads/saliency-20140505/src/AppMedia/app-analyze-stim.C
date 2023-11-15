/*!@file AppMedia/app-analyze-stim.C make different kind of visual test stimuli
*/

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
// Primary maintainer for this file: T. Nathan Mundhenk <mundhenk@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-analyze-stim.C $
// $Id: app-analyze-stim.C 6795 2006-06-29 20:45:32Z rjpeters $

#include <cfloat>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include "Util/log.H"
#include "Raster/Raster.H"
#include "Image/Image.H"
#include "Psycho/StimAnalyzer.H"

using namespace std;

int main(const int argc, const char **argv)
{
  if(argc < 4)
  {
    std::cerr << "image-prefix      - prefix#.png for salmap\n";
    std::cerr << "gt-prefix         - prefix#.png for ground truth images\n";
    std::cerr << "output-stats-file - file to append frame stats to\n";
    std::cerr << "output-stats-file - file to append total stats to\n";
    std::cerr << "frames            - number of frames to run\n";
    LFATAL("Usage: app-analyze-stim image-prefix gt-prefix output-stats-file output-stats-file frames");
  }
  const string inFilePrefix   = argv[1];
  const string inFileGTPrefix = argv[2];
  const string outStatsFileF  = argv[3];
  const string outStatsFileT  = argv[4];
  const unsigned int frames   = atoi(argv[5]);

  const string dot            = ".";
  const string type           = "png";

  StimAnalyzer SA(frames,1);

  for(unsigned int i = 0; i < frames; i++)
  {
    char c[100];
    if(i < 10)
      sprintf(c,"00000%d",i);
    else if(i < 100)
      sprintf(c,"0000%d",i);
    else if(i < 1000)
      sprintf(c,"000%d",i);
    else if(i < 10000)
      sprintf(c,"00%d",i);
    else if(i < 100000)
      sprintf(c,"0%d",i);
    else
      sprintf(c,"%d",i);

    const string imageName = inFilePrefix   + c + dot + type;
    const string gtName    = inFileGTPrefix + c + dot + type;

    const Image<byte> inImage       = Raster::ReadGray(imageName);
    const Image<PixRGB<byte> > inGT = Raster::ReadRGB(gtName);

    const Image<float> finImage       = inImage;
    const Image<PixRGB<float> > finGT = inGT;

    SA.SAinputImages(finImage,finGT,i,0);
    SA.SAcompImages();
  }

  SA.SAfinalStats();
  SA.SAdumpFrameStats(inFilePrefix,outStatsFileF,false);
  SA.SAdumpConditionStats(inFilePrefix,outStatsFileT,false);
}







