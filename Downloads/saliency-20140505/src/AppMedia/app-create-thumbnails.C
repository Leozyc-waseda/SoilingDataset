/*!@file AppMedia/app-create-thumbnails.C Create small images from originals */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-create-thumbnails.C $
// $Id: app-create-thumbnails.C 13712 2010-07-28 21:00:40Z itti $
//

#include "Image/Image.H"
#include "Image/FilterOps.H"
#include "Image/ShapeOps.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"

#include <cstdio>

#define TSIZEX 64
#define TBORDER 2
#define TSIZEY (TSIZEX + TBORDER)
#define TNAME "thumbnails.ppm"

int main(int argc, char** argv)
{
  PixRGB<byte> cpix(255, 0, 0);

  if (argc == 1)
    {
      LERROR("USAGE: %s <image.ppm> .. <image.ppm>\n"
             "       puts the thumbnails in 'thumbnails.ppm'\n", argv[0]);
      return 1;
    }

  Image< PixRGB<byte> > thumb(TSIZEX * (argc - 1) + TBORDER * (argc - 2),
                              TSIZEY, ZEROS);

  for (int i = 1; i < argc; i ++)
    {
      bool compressed = false;
      if (argv[i][strlen(argv[i])-1] == 'z') {
        char t[1000]; sprintf(t, "gunzip %s",argv[i]);
        if (system(t)) LERROR("Error in system()");
        argv[i][strlen(argv[i])-3]='\0'; compressed = true;
      }
      Image< PixRGB<byte> > tmp = Raster::ReadRGB(argv[i]);
      if (compressed) {
        char t[1000]; sprintf(t, "gzip -9 %s", argv[i]);
        if (system(t)) LERROR("Error in system()");
      }

      // low-pass filter
      tmp = lowPass9(lowPass9(lowPass9(tmp)));

      // preserve aspect ratio
      float sx = ((float)TSIZEX) / ((float)tmp.getWidth());
      float sy = ((float)TSIZEX) / ((float)tmp.getHeight());
      float scale = (sx <= sy ? sx : sy);
      int w = (int)(scale * ((float)tmp.getWidth()));
      int h = (int)(scale * ((float)tmp.getHeight()));
      if (w > TSIZEX) w = TSIZEX;
      if (h > TSIZEY) h = TSIZEY;

      int xoffset = (TSIZEX + TBORDER) * (i - 1) + (TSIZEX - w) / 2;
      int yoffset = (TSIZEY - h) / 2;

      LDEBUG("%s: (%d, %d) at (%d, %d)", argv[i], w, h, xoffset, yoffset);
      tmp = rescale(tmp, w, h);

      PixRGB<byte> pix;
      for (int yy = 0; yy < h; yy ++)
        for (int xx = 0; xx < w; xx ++)
          {
            tmp.getVal(xx, yy, pix);
            thumb.setVal(xx + xoffset, yy + yoffset, pix);
          }

      // put border
      if (i < argc - 1)
        for (int yy = 0; yy < TSIZEY; yy ++)
          for (int xx = TSIZEX; xx < TSIZEX+TBORDER; xx ++)
            thumb.setVal(xx + (TSIZEX+TBORDER)*(i-1), yy, cpix);

    }
  Raster::WriteRGB(thumb, TNAME, RASFMT_PNM);
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
