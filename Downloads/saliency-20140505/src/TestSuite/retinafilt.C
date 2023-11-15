/*!@file TestSuite/retinafilt.C Apply a retinalike transformation to an image */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/retinafilt.C $
// $Id: retinafilt.C 12074 2009-11-24 07:51:51Z itti $
//

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/PyramidOps.H"
#include "Raster/Raster.H"

#include <algorithm> // for std::min
#include <cmath>
#include <cstdio>
#include <typeinfo>

//! depth of the pyramid used to obtain the eccentricity-dependent resolution
#define PYR_DEPTH 10

//! blind spot center, as a factor of image width and height, from center
#define BLIND_X 0.15f
#define BLIND_Y 0.00f

//! blind spot 2*sigma^2, in pixels
#define BLIND_2S2 600.0f

//! fovea 2*sigma^2, in pixels:
#define FOVEA_2S2 150.0f


template <class T>
void showtypeof(T t, const char* expr)
{
  LINFO("type of %s is %s", expr, typeid(T).name());
}

#define SHOWTYPEOF(x) showtypeof(x, #x)

/* ###################################################################### */
int main(int argc, char **argv)
{
  bool fovea_blue = false, blind_spot = false;
  if (argc < 3) {
    fprintf(stderr,
            "USAGE: %s [opts] <input.ppm> <output.ppm> [fov_x fov_y]\n"
            "options:\n"
            "      -b implement blind spot\n"
            "      -f no blue cones in fovea\n", argv[0]);
    exit(1);
  }
  int ar = 1;
  if (argv[ar][0] == '-') {
    for (unsigned int ii = 1; ii < strlen(argv[ar]); ii ++) {
      switch (argv[ar][ii]) {
      case 'b': blind_spot = true; break;
      case 'f': fovea_blue = true; break;
      default: LFATAL("Unknown option '%c'.", argv[ar][ii]);
      }
    }
    ar ++;
  }

  // read input image:
  Image< PixRGB<byte> > input = Raster::ReadRGB(argv[ar++]);

  // create gaussian pyramid and destination image:
  ImageSet< PixRGB<byte> > pyr = buildPyrGaussian(input, 0, PYR_DEPTH, 5);

  // apply radial smoothing:
  int w = input.getWidth(), h = input.getHeight(), ci = w / 2, cj = h / 2;

  // use given fovea center if present in params:
  if (argc - ar > 1)
    {
      ci = atoi(argv[ar + 1]); ci = std::max(0, std::min(w-1, ci));
      cj = atoi(argv[ar + 2]); cj = std::max(0, std::min(h-1, cj));
      LINFO("Using (%d, %d) for fovea center", ci, cj);
    }

  Image< PixRGB<byte> > output(w, h, NO_INIT);
  float rstep = float(std::max(w, h) / PYR_DEPTH);
  int bi = ci + int(w * BLIND_X), bj = cj + int(h * BLIND_Y);

  for (int i = 0; i < w; i ++)
    for (int j = 0; j < h; j ++)
      {
        float dist = sqrtf((i-ci)*(i-ci) + (j-cj)*(j-cj));

        // determine resolution from which this pixel will be taken:
        float d = dist / rstep;

        // get the pixel value from depth d:
        int di = int(d);
        float dd = d - float(di);

        // uncomment these 2 lines for radial pixelization:
        //int ii=(i/(1<<di))*(1<<di), jj=(j/(1<<di))*(1<<di);
        //if (ii-i<(1<<di) && jj-j<(1<<di)) d=0.0f; else d=1.0f;

        Point2D<int> loc(i, j);
        PixRGB<byte> pix0 = getPyrPixel(pyr, loc, di);
        PixRGB<byte> pix1 = getPyrPixel(pyr, loc, std::min(di+1, PYR_DEPTH-1));

        float blind = 1.0f;
        if (blind_spot)
          blind = 1.0f - expf(-((i-bi)*(i-bi) + (j-bj)*(j-bj)) / BLIND_2S2);

        static bool didit = false;

        if (!didit)
          {
            SHOWTYPEOF(pix0);
            SHOWTYPEOF(pix1);
            SHOWTYPEOF(pix1-pix0);
            SHOWTYPEOF((pix1-pix0)*dd);
            SHOWTYPEOF(pix0 + (pix1-pix0)*dd);
            SHOWTYPEOF((pix0 + (pix1-pix0)*dd) * blind);
            didit = true;
          }

        PixRGB<byte> pix( (pix0 + (pix1 - pix0) * dd) * blind );
        if (fovea_blue)
          pix.setBlue(pix.blue() * (1.0f - expf(-dist * dist / FOVEA_2S2)));

        // for Nature Reviews Neuroscience paper: neuron-cam mode
        //if (dist > 20.0) {
        //pix /= exp((dist - 20.0f) / 10.0f);
        //}

        // store:
        output.setVal(loc, pix);
      }

  // write out result:
  Raster::WriteRGB(output, argv[ar], RASFMT_PNM);
  return 0;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
