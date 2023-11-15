/*!@file AppMedia/app-pfmtopgm.C simple program to convert PFM images */

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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-pfmtopgm.C $
// $Id: app-pfmtopgm.C 8267 2007-04-18 18:24:24Z rjpeters $
//

#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Raster/Raster.H"
#include <cstdio>

/* This is a trivial program bin/pfmtopgm to convert PFM images */
int main(const int argc, const char **argv)
{
  if (argc != 3 && argc != 4)
    {
      fprintf(stderr, "USAGE: %s <in.pfm> <out.pgm> [factor]\n", argv[0]);
      return 1;
    }

  Image<float> img = Raster::ReadFloat(argv[1]);

  float imi, ima; getMinMax(img, imi, ima);

  if (argc == 4)
    {
      const float factor = atof(argv[3]);
      img *= factor;
    }
  else
    inplaceNormalize(img, 0.0F, 255.0F);

  Image<byte> out = img; // clamp to 0..255. NOTE: We are rounding down.

  byte omi, oma; getMinMax(out, omi, oma);

  fprintf(stderr, "%s [%f .. %f] -> %s [%d .. %d]\n",
          argv[1], imi, ima, argv[2], omi, oma);

  Raster::WriteGray(out, argv[2]);

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
