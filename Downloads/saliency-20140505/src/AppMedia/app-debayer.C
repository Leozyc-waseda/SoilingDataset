/*!@file AppMedia/app-debayer.C */

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
// Primary maintainer for this file: Zhicheng Li <zhicheng@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-debayer.C $
// $Id: app-debayer.C 12962 2010-03-06 02:13:53Z irock $
//

#ifndef APPMEDIA_APP_DEBAYER_C_DEFINED
#define APPMEDIA_APP_DEBAYER_C_DEFINED

#include "Image/Image.H"
#include "Raster/Raster.H"
#include "Raster/DeBayer.H"

int main(int argc, char** argv)
{
  if (argc < 3 || argc > 4)
    fprintf(stderr, "USAGE: app-debayer <input bayer image> "
            "<output debayered image name> [bayer type]");

  BayerFormat fmt=BAYER_GBRG;
  if(argc == 4) {
    if(strcmp(argv[3], "BAYER_GBRG") == 0) fmt = BAYER_GBRG;
    else if(strcmp(argv[3], "BAYER_GRBG") == 0) fmt = BAYER_GRBG;
    else if(strcmp(argv[3], "BAYER_RGGB") == 0) fmt = BAYER_RGGB;
    else if(strcmp(argv[3], "BAYER_BGGR") == 0) fmt = BAYER_BGGR;
    else
      LFATAL("please choose correct bayer format");
  }

  Image<byte> inputImg = Raster::ReadGray(argv[1]);

  Image<PixRGB<byte> > outImg = deBayer(inputImg, fmt);

  Raster::WriteRGB(outImg, std::string(argv[2]));

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
#endif //APPMEDIA_APP_DEBAYER_C_DEFINED
