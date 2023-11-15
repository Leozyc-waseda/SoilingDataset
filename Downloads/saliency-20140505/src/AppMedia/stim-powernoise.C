/*!@file AppPsycho/inception.C                       */
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
// Primary maintainer for this file: John Shen <shenjohn@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/inception.C $
// $Id: gaborSearch.C 10794 2009-02-08 06:21:09Z itti $
//

//generate a pseudo-real textured target 

#include "Component/ModelManager.H"
#include "Image/MathOps.H"
#include "Raster/PngWriter.H"
#include "Raster/Raster.H"
#include "Util/log.H"
#include <stdio.h>
#include <time.h>

int main(int argc, char** argv)
{

  clock_t t1 = clock();
  //instantiate a model manager
  ModelManager manager("AppMedia: powerlaw stimuli");

  // get dimensions of window
  if (manager.parseCommandLine(argc, argv,"<name> <width> <height> <powlaw>", 4,4) == false)
    return(1);

  uint width = fromStr<uint>(manager.getExtraArg(1));
  uint height = fromStr<uint>(manager.getExtraArg(2));
  double powlaw = fromStr<double>(manager.getExtraArg(3));
  Dims dims(width, height);
  Image<float> myFloatImg(dims.w(), dims.h(), ZEROS);
  Image<double> myDoubleImg(dims.w(), dims.h(), ZEROS);
  char stem[255], filename[255];

  manager.start();

  // get command line parameters
  sscanf(argv[1],"%s",stem);

  //tests
  myDoubleImg = addPowerNoise(myDoubleImg,-powlaw);

  sprintf(filename, "%s.png",stem);
  LINFO("writing image to %s", filename);

  int flags = FLOAT_NORM_0_255;
  Raster::WriteFloat(myDoubleImg,flags,filename);

  //finish all
  clock_t t2=  clock();
  LINFO("generated power noise in %fs", double(t2-t1)/CLOCKS_PER_SEC);

  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
