/*!@file AppMedia/app-font2c.C Takes a .pgm image with ASCII chars written in it, and
  transforms it into C source code
*/

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-font2c.C $
// $Id: app-font2c.C 8077 2007-03-08 19:38:18Z rjpeters $
//

#include "Image/Image.H"
#include "Util/log.H"
#include "Raster/Raster.H"

#include <cstdio>
#include <cstdlib>

// number of characters in image:
#define NBCHARS 95
// image should be:
//  !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~
// in one row of 95 chars

/*! font2c.C -- takes a .pgm image with ASCII chars written in it, and
  transforms it into C source code
*/
int main(int argc, char **argv)
{
  if (argc != 3)
    { printf("USAGE: %s <image.pgm> <fontarrayname>\n", argv[0]); exit(0); }

  Image<byte> img = Raster::ReadGray(argv[1]);
  const char* const fontarrayname = argv[2];

  printf("// created by font2c; source image: %s\n", argv[1]);
  int w = img.getWidth(), h = img.getHeight();
  printf("// Image Width = %d, Height = %d\n", w, h);
  int w2 = w / NBCHARS;  // NBCHARS chars in image
  printf("// Letter size = %d x %d\n", w2, h);
  printf("#define WW 0x00\n");
  printf("#define o  0xff\n");
  printf("static unsigned char %s[%d][%d] = {\n",
         fontarrayname, NBCHARS, w2 * h);

  for (int i = 0; i < NBCHARS; i ++) // loop over chars
    {
      printf("  {\n    ");
      for (int j = 0; j < h; j ++)  // scan y one char
        {
          const bool lastrow = (j == h - 1);
          for (int ii = i * w2; ii < (i + 1) * w2; ii ++)  // scan x one char
            {
              const bool lastcol = (ii == (i + 1) * w2 - 1);

              if (img.getVal(ii, j) != 0) printf(lastrow && lastcol ? "o" : "o ");
              else                        printf("WW");

              if (!lastcol) printf(",");
            }
          if (!lastrow) printf(",\n    "); else printf("\n");
        }
      if (i != NBCHARS - 1) printf("  },\n"); else printf("  }\n");
    }
  printf("};\n");
  printf("#undef WW\n");
  printf("#undef o\n");
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
