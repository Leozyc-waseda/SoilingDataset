/*!@file AppMedia/test-binMorph.C [put description here] */

// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/test-binMorph.C $
// $Id: test-binMorph.C 12074 2009-11-24 07:51:51Z itti $
//

// test functions for binary morphology (dilate, erode, open, close)

#include "GUI/XWinManaged.H"
#include "Image/ColorOps.H"
#include "Image/MorphOps.H"
#include "Image/Image.H"
#include "Image/Kernels.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Util/log.H"

#include <iostream>
#include <cstdio>

// ######################################################################
// ##### Main Program:
// ######################################################################
int main (int argc, char **argv)
{
  if (argc != 3)
    {
      std::cerr << "usage: " << argv[0] << " file.[ppm|pgm] kernel_size\n\n";
      exit (2);
    }

  //retrieve kernel size
  int ksize;
  sscanf(argv[2],"%d",&ksize);

  // read the image
  Image<byte> img = luminance(Raster::ReadRGB(argv[1]));
  Image<byte> se = twofiftyfives(ksize);
  if (!img.initialized())
    {
      std::cerr << "error loading image: " << argv[1] << "\n\n";
      exit (2);
    }

  CloseButtonListener l;

  l.add(new XWinManaged(img,"original Image"));
  l.add(new XWinManaged(dilateImg(img,se),"Dilation result"));
  l.add(new XWinManaged(erodeImg(img,se),"Erosion result"));
  l.add(new XWinManaged(openImg(img,se),"Opening result"));
  l.add(new XWinManaged(closeImg(img,se),"Closure result"));

  l.waitForClose();
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
