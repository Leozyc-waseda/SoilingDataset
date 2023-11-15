/*!@file CINNIC/pointAndFlood.C find N most salient points in some image for CINNIC */

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
// Primary maintainer for this file: Nathan Mundhenk <nathan@mundhenk.com>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/pointAndFlood.C $
// $Id: pointAndFlood.C 6191 2006-02-01 23:56:12Z rjpeters $
//


// ############################################################
// ############################################################
// ##### ---CINNIC---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

#include "CINNIC/CINNICstatsRun.H"

//! this contains the name of the config file
char* configFile;

//! this file contains the input RAW image
char* rawImage;

//! this file contains a salincy map
char* salImage;

//! This is the configFile object
readConfig ReadConfig(25);

//! main object for stats
CINNICstatsRun runStats;

//! This is a package to run tests on the output from CINNIC
int main(int argc, char* argv[])
{
  LINFO("STARTING pointAndFlood");
  LINFO("Copyright 2002 ACME Vision Systems, Wallawalla, WA");

  if(argc < 1)
  {
    LFATAL("USAGE: paintAndFlood rawImage salImage configFile");
  }

  rawImage = argv[1];
  salImage = argv[2];
  configFile = argv[3];

  ReadConfig.openFile(configFile);
  runStats.setStuff();
  runStats.setConfig(ReadConfig,ReadConfig);

  Image<float> raw = Raster::ReadGray(rawImage, RASFMT_PNM);
  Image<PixRGB<float> > bsal = Raster::ReadRGB(salImage, RASFMT_PNM);
  Image<float> sal = luminance(bsal);
  LINFO("RUNNING");
  runStats.pointAndFloodImage(raw,sal,salImage);
  LINFO("DONE");
}
