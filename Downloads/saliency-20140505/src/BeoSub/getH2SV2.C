/*!@file BeoSub/getH2SV2.C */

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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/getH2SV2.C $
// $Id: getH2SV2.C 5567 2005-09-21 15:24:23Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/DrawOps.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(const int argc, const char **argv)
{

  // instantiate a model manager:
  ModelManager manager("H2SV2 Tester");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 3, 3) == false) return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  int R = strtol(argv[1], NULL, 10);
  int G = strtol(argv[2], NULL, 10);
  int B = strtol(argv[3], NULL, 10);
  PixRGB<float> P1(R,G,B);
  PixH2SV2<float> Clr(P1);
  std::vector<float> color(4,0.0F);
  color[0] = Clr.H1(); color[1] = Clr.H2(); color[2] = Clr.S(); color[3] = Clr.V();
  printf("H1: %f H2: %f S: %f V: %f\n", color[0], color[1], color[2], color[3]);
  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}
