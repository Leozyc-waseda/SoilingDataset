/*!@file AppPsycho/singleGaborTest.C                       */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/singleGaborTest_elno.C $
// $Id: singleGaborTest_elno.C 12962 2010-03-06 02:13:53Z irock $
//


//test funky color gabor function by displaying a single patch.

#include "Component/ModelManager.H"
#include "GUI/XWinManaged.H"
#include "Image/Image.H"
#include "Image/CutPaste.H"
#include "Image/Kernels.H"     // for gaborFilter()
#include "Image/MathOps.H"
#include "Image/Transforms.H"
#include "Image/ShapeOps.H"
#include "Util/log.H"
#include "Raster/Raster.H"
#include "Raster/PngWriter.H"

#include <stdio.h>
#include <vector>


int main(int argc, char** argv)
{

  //instantiate a model manager
  ModelManager manager("AppPsycho: gabor search stimuli");

  LINFO("Gabor search the new version");

  //dimensions of window
  Dims dims(1000,1000);

   float stddev = 50.0;
   float freq = 5.0;
   float theta = 10;
   float hueShift = 50;

   Image<PixRGB<byte> > finalImage;

 // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<stddev>" "<freq>" "<theta>" "<hueShift>", 4, 4) == false)
    return(1);


  manager.start();


  LINFO("hello locust");

  // get command line parameters

  sscanf(argv[1],"%g",&stddev);
  sscanf(argv[2],"%g",&freq);
  sscanf(argv[3],"%g",&theta);
  sscanf(argv[4],"%g",&hueShift);
  LINFO("stddev = %g; freq = %g; theta = %g; hueShift = %g",stddev,freq,theta,hueShift);


  XWinManaged *imgWin;
  CloseButtonListener wList;
  imgWin  = new XWinManaged(dims,    0,    0, manager.getExtraArg(0).c_str());
  wList.add(imgWin);


  Dims standardDims(500,500);
 finalImage = rescale(gaborFilterRGB(stddev, freq, theta,hueShift),standardDims);

 //finalImage = gaborFilterRGB(stddev, freq, theta, hueShift);

 LINFO("final image size width  %d, height %d", finalImage.getDims().w(), finalImage.getDims().h());
  imgWin->drawImage(finalImage,0,0);


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
