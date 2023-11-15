/*!@file AppMedia/app-iiv.C Bare-bones image viewer (a minimal replacement for xv) */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-iiv.C $
// $Id: app-iiv.C 12074 2009-11-24 07:51:51Z itti $
//

#ifndef APPMEDIA_APP_IIV_C_DEFINED
#define APPMEDIA_APP_IIV_C_DEFINED

#include "GUI/XWinManaged.H"
#include "Image/ColorMap.H"
#include "Image/ColorOps.H" // for colorize()
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Util/log.H"
#include "Util/sformat.H"
#include "rutz/shared_ptr.h"

#include <X11/keysym.h>
#include <iostream>
#include <vector>
#include <cstdio>

namespace
{
  void usage(const char* argv0)
  {
    fprintf(stderr, "USAGE: %s [-d] [-geom WxH] [-w width] [-h height] [-crop l,t,r,b] [-cmapfile filename] imgfile1 [imgfile2 [imgfile3 ...]]\n", argv0);
    fprintf(stderr, "   or: %s [-d] [-geom WxH] [-w width] [-h height] [-crop l,t,r,b] [-cmapfile filename] [-wait] -stdin-filenames\n", argv0);
    exit(1);
  }

  Image<PixRGB<byte> > doRescale(const GenericFrame& in,
                                 const Rectangle& cropbox,
                                 const Dims& geom,
                                 const ColorMap& cmap)
  {
    Image<PixRGB<byte> > img =
      (cmap.initialized()
       && (in.nativeType() == GenericFrame::GRAY_U8
           || (in.nativeType() == GenericFrame::VIDEO
               && in.asVideo().getMode() == VIDFMT_GREY)))
      ? colorize(in.asGrayU8(), cmap)
      : in.asRgbU8();

    if (cropbox.isValid())
      img = crop(img, cropbox.getOverlap(img.getBounds()));

    const int w = img.getWidth();
    const int h = img.getHeight();

    if (geom.w() != 0 && geom.h() != 0)
      return rescale(img, geom);
    else if (geom.w() != 0)
      return rescale(img,
                     geom.w(),
                     int(0.5 + geom.w() * double(h) / double(w)));
    else if (geom.h() != 0)
      return rescale(img,
                     int(0.5 + geom.h() * double(w) / double(h)),
                     geom.h());
    // else ...
    return img;
  }
}

int main(int argc, char** argv)
{
  LOG_FLAGS &= ~LFATAL_PRINTS_ABORT;
  LOG_FLAGS &= ~LFATAL_THROWS;

  MYLOGVERB = LOG_INFO;

  if (argc == 1)
    {
      usage(argv[0]);
    }

  std::vector<rutz::shared_ptr<XWinManaged> > windows;

  bool wait = false;
  bool allowopts = true;
  Rectangle cropbox;
  Dims geom(0,0);
  ColorMap cmap; // empty

  for (int i = 1; i < argc; ++i)
    {
      if (allowopts && strcmp(argv[i], "-d") == 0)
        {
          MYLOGVERB = LOG_DEBUG;
        }
      else if (allowopts && strcmp(argv[i], "-crop") == 0)
        {
          ++i; if (i >= argc) usage(argv[0]);
          convertFromString(argv[i], cropbox);
        }
      else if (allowopts && strcmp(argv[i], "-geom") == 0)
        {
          ++i; if (i >= argc) usage(argv[0]);
          convertFromString(argv[i], geom);
        }
      else if (allowopts && strcmp(argv[i], "-w") == 0)
        {
          ++i; if (i >= argc) usage(argv[0]);
          geom = Dims(atoi(argv[i]), geom.h());
        }
      else if (allowopts && strcmp(argv[i], "-h") == 0)
        {
          ++i; if (i >= argc) usage(argv[0]);
          geom = Dims(geom.w(), atoi(argv[i]));
        }
      else if (allowopts && strcmp(argv[i], "-cmapfile") == 0)
        {
          ++i; if (i >= argc) usage(argv[0]);
          cmap = Raster::ReadRGB(argv[i]);
        }
      else if (allowopts && strcmp(argv[i], "-wait") == 0)
        {
          // whether to wait between images when showing images
          // serially, as with -stdin-filenames
          wait = true;
        }
      else if (allowopts && strcmp(argv[i], "--") == 0)
        {
          allowopts = false;
        }
      else if (allowopts && strcmp(argv[i], "-stdin-filenames") == 0)
        {
          std::string line;
          while (getline(std::cin, line))
            {
              if (line.empty())
                break;

              const GenericFrame img = Raster::ReadFrame(line);
              const Image<PixRGB<byte> > rescaled =
                doRescale(img, cropbox, geom, cmap);

              const int w = rescaled.getWidth();
              const int h = rescaled.getHeight();
              const std::string title = sformat("[%dx%d] %s", w, h, line.c_str());

              if (windows.size() == 0)
                {
                  rutz::shared_ptr<XWinManaged> win
                    (new XWinManaged(rescaled, title.c_str()));
                  windows.push_back(win);
                }
              else
                {

                  windows.back()->setDims(rescaled.getDims());
                  windows.back()->drawImage(rescaled);
                  windows.back()->setTitle(title.c_str());
                }

              if (wait)
                {
                  fprintf(stderr, "%s\n", line.c_str());

                  // wait until the user presses any key in the window
                  while (windows.back()->getLastKeyPress() == -1)
                    usleep(10000);
                }
            }


          return 0;
        }
      else
        {
          if (!Raster::fileExists(argv[i]))
            {
              if (allowopts)
                fprintf(stderr, "%s is neither a valid option "
                        "nor a valid image filename\n", argv[i]);
              else
                fprintf(stderr, "%s is not a valid image filename\n",
                        argv[i]);
              usage(argv[0]);
            }

          const GenericFrame img = Raster::ReadFrame(argv[i]);
          const Image<PixRGB<byte> > rescaled =
            doRescale(img, cropbox, geom, cmap);
          const int w = rescaled.getWidth();
          const int h = rescaled.getHeight();
          const std::string title = sformat("[%dx%d] %s", w, h, argv[i]);
          rutz::shared_ptr<XWinManaged> win
            (new XWinManaged(rescaled, title.c_str()));
          windows.push_back(win);
        }
    }

  if (windows.size() == 0)
    usage(argv[0]);

  bool keepgoing = true;

  while (keepgoing)
    {
      for (uint i = 0; i < windows.size(); ++i)
        {
          if (windows[i]->pressedCloseButton())
            { keepgoing = false; break; }

          const KeySym sym = windows[i]->getLastKeySym();

          if (sym == XK_Escape || sym == XK_Q || sym == XK_q)
            { keepgoing = false; break; }
        }

      usleep(10000);
    }

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPMEDIA_APP_IIV_C_DEFINED
