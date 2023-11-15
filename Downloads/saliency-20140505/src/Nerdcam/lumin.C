/*!@file Nerdcam/lumin.C NERD CAM program sweet */

//Nathan Mundhenk
//mundhenk@usc.edu

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
// Primary maintainer for this file: T Nathan Mundhenk <mundhenk@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Nerdcam/lumin.C $
// $Id: lumin.C 6191 2006-02-01 23:56:12Z rjpeters $
//

// ############################################################
// ############################################################
// ##### ---NERD CAM---
// ##### Nerd Cam Binary:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

#include <stdlib.h>
#include <iostream>
#include "Util/log.H"
#include "Image/Image.H"
#include "Image/ColorOps.H"
#include "Raster/Raster.H"
#include "Image/Pixels.H"

//! this is a simple program for determining the average luminocity of an image

//!what file to open
char* filename;
//! image object
Image<PixRGB <float> > image;
//! byte based image
Image<PixRGB <byte> > bimage;
//! float image B/W
Image<float> fimage;
//! luminance of image
float lum;

int main(int argc, char* argv[])
{
  filename = argv[1];
  bimage = Raster::ReadRGB(filename, RASFMT_PNM);
  image = bimage;
  fimage = luminance(image);
  int N = 0;
  float sum = 0.0F;
  for(int x = 0; x < fimage.getWidth(); x++)
  {
    for(int y = 0; y < fimage.getHeight(); y++)
    {
      sum += fimage.getVal(x,y);
      N++;
    }
  }
  lum = sum/N;
  std::cout << lum << "\n";
  return (int)lum;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
