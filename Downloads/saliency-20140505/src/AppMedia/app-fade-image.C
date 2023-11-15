/*!@file AppMedia/app-fade-image.C video fade effects */

//////////////////////////////////////////////////////////////////////////
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
// See http://iLab.usc.edu for information about this project.          //
//////////////////////////////////////////////////////////////////////////
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
//////////////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////////////
//
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-fade-image.C $
// $Id: app-fade-image.C 6191 2006-02-01 23:56:12Z rjpeters $
//

#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"

int main(int argc, char** argv)
{
  if (argc < 4 || argc > 6)
    LFATAL("USAGE: %s <in|out|none|mix> <image.ppm> <first_frame#> "
           "<last_frame#> [mix_frame.ppm]\n"
           "       %s fac <image.ppm> <factor>\n", argv[0], argv[0]);

  Image< PixRGB<byte> > input1 = Raster::ReadRGB(argv[2]);
  Image< PixRGB<byte> > input2;

  if (strcmp(argv[1], "mix") == 0)
    {
      if (argc == 6) input2 = Raster::ReadRGB(argv[5]);
      else LFATAL("You need to provide a mixing image when using mix mode");
    }
  else if (strcmp(argv[1], "none") == 0)
    {
      input2 = input1;
    }
  else if (strcmp(argv[1], "in") == 0)
    {
      input2 = input1; input1.clear();
    }
  else if (strcmp(argv[1], "out") == 0)
    {
      input2.resize(input1.getDims(), true);
    }
  else if (strcmp(argv[1], "fac") == 0)
    {
      input2 = input1 * atof(argv[3]);
      Raster::WriteRGB(input2, argv[2], RASFMT_PNM);
    }
  else LFATAL("Incorrect arguments. Run with no parameter for usage.");

  int first_frame = atoi(argv[3]), last_frame = atoi(argv[4]);
  if (last_frame <= first_frame) LFATAL("last_frame must be > first_frame");
  float coeff = 1.0F, coeff_incr = 1.0F / (float)(last_frame - first_frame);

  for (int i = first_frame; i <= last_frame; i ++)
    {
      Image< PixRGB<byte> > result = input1 * coeff + input2 * (1.0F - coeff);
      Raster::WriteRGB(result, sformat("frame%06d.ppm", i));
      coeff -= coeff_incr;
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
