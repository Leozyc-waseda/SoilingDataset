/*!@file CINNIC/CINNICstats.C do statistical tests on CINNIC outputs */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/CINNICstats.C $
// $Id: CINNICstats.C 6410 2006-04-01 22:12:24Z rjpeters $
//


// ############################################################
// ############################################################
// ##### ---CINNIC---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

#include "Util/Assert.H"
#include "CINNIC/CINNICstatsRun.H"

//! the file to save stats out to
const char *savefilename;
//! this string contains the name of the list file to open of images
const char* listfile;
//! this contains the name of the config file
const char* configFile;
//! this contains the name of the config file
const char* configFile2;
//! this is the parameter special for this analysis
const char* specialParam;
//! This is the configFile object
readConfig fileList(25);
//! this is the main config file
readConfig ReadConfigMain(25);
//! this is the main config file
readConfig ReadConfigMain2(25);
//! main object for stats
CINNICstatsRun runStats;

//! This is a package to run tests on the output from CINNIC
int main(int argc, char* argv[])
{
  LINFO("STARTING CINNICstats");
  LINFO("Copyright 2002 ACME Vision Systems, Wallawalla, WA");
  //you must input a list of files
  ASSERT(argc >= 1);
  listfile = argv[1];

  if(argc > 2)
  {
    savefilename = argv[2];
  }
  else
  {
    savefilename = "stats.noname";
  }

  if(argc > 3)
  {
    configFile = argv[3];
  }
  else
  {
    configFile = "stats.conf";
  }

  if(argc > 4)
  {
    specialParam = argv[4];
  }
  else
  {
    specialParam = "noSpecial";
  }

  if(argc > 5)
  {
    configFile2 = argv[5];
  }
  else
  {
    configFile2 = "contour.conf";
  }

  int do1,do2;
  ReadConfigMain.openFile(configFile);
  ReadConfigMain2.openFile(configFile2);
  do1 = (int)ReadConfigMain.getItemValueF("runStandard");
  do2 = (int)ReadConfigMain.getItemValueF("runPointAndFlood");

  fileList.openFile(listfile);
  //! main object for stats
  LINFO("SETTING stuff");
  runStats.setStuff(fileList);
  LINFO("SETTING config");
  runStats.setConfig(ReadConfigMain,ReadConfigMain2);

  if(do1 == 1)
    runStats.runStandardStats(fileList);
  if((do1 == 1) && (do2 == 1))
    runStats.setStuff(fileList);
  if(do2 == 1)
    runStats.runPointAndFlood(fileList,specialParam);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
