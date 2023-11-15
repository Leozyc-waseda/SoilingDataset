/*!@file TestSuite/test-ImageEqual.C A small program to help testing the Raster
  class by loading two images and checking if they are equal. */

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
// Primary maintainer for this file: Rob Peters <rjpeters@klab.caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/test-ImageEqual.C $
// $Id: test-ImageEqual.C 7725 2007-01-18 20:00:51Z rjpeters $
//

#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Util/log.H"

#include <cstdio>

int main(int argc, const char** argv)
{
  if (argc != 4)
    {
      fprintf(stderr, "usage: %s <imgfile-1> <imgfile-2> <outname>",
              argv[0]);
      return 1;
    }

  Image<PixRGB<byte> > img1 = Raster::ReadRGB(argv[1]);
  Image<PixRGB<byte> > img2 = Raster::ReadRGB(argv[2]);
  const char* const fname = argv[3];

  FILE* f = fopen(fname, "w");
  if (f == 0)
    {
      fprintf(stderr, "couldn't open '%s' for writing\n", fname);
      return -1;
    }

  if (img1 == img2)
    {
      fprintf(f, "1 (equal)\n");
    }
  else
    {
      fprintf(f, "0 (not equal)\n");
    }

  fclose(f);

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
