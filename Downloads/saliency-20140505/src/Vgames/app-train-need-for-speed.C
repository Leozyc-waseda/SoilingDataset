/*!@file AppMedia/app-train-need-for-speed.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Vgames/app-train-need-for-speed.C $
// $Id: app-train-need-for-speed.C 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef APPMEDIA_APP_TRAIN_NEED_FOR_SPEED_C_DEFINED
#define APPMEDIA_APP_TRAIN_NEED_FOR_SPEED_C_DEFINED

#include "GUI/XWinManaged.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/Point2D.H"
#include "Image/ShapeOps.H"
#include "Raster/Raster.H"
#include "Util/log.H"
#include "Util/sformat.H"
#include "Video/VideoFrame.H"
#include "rutz/rand.h"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

int main(const int argc, const char** argv)
{
  if (argc != 3)
    {
      LFATAL("usage: %s <imagelist> <dataout>", argv[0]);
    }

  std::vector<std::string> fnames;
  std::vector<unsigned char> done;

  {
    std::ifstream ifs(argv[1]);

    std::string fname;
    while (ifs >> fname)
      fnames.push_back(fname);

    ifs.close();
  }

  done.resize(fnames.size(), (unsigned char)0);

  rutz::urand_irange rnd(0, int(fnames.size()), int(getpid()));

  const Dims zoomdims(121,69);

  XWinManaged fullwin(Dims(640,480), -1, -1, "full");
  XWinManaged zoomwin(zoomdims * 2, -1, -1, "zoom");

  std::ofstream ofs(argv[2], std::ios::app);

  while (1)
    {
      int p = 0;
      do {
        p = rnd.draw();
      } while (done[p]);

      const char* imname = fnames[p].c_str();
      done[p] = 1;

      const VideoFrame vf = VideoFrame::fromFile(imname, Dims(640,480),
                                                 VIDFMT_UYVY, false);

      const int bottom = rnd.draw() % 2;

      const VideoFrame vf2 = vf.makeBobDeinterlaced(bottom);

      const Image<PixRGB<byte> > rgb = vf2.toRgb();

      const std::string rgbname = sformat("%s-%d.png", imname, bottom);

      if (Raster::fileExists(rgbname))
        {
          std::cout << "already exists: " << rgbname << std::endl;
          continue;
        }

      const Image<PixRGB<byte> > im2 = crop(rgb, Point2D<int>(453, 86), zoomdims);

      fullwin.drawImage(rgb, 0, 0);
      zoomwin.drawImage(intXY(im2, true), 0, 0);

      std::cout << imname << "? " << std::flush;

      std::string line;
      std::getline(std::cin, line);

      if (line == "end")
        break;

      if (line.size() > 0)
        {
          std::istringstream iss(line);

          std::string s[4];

          iss >> s[0] >> s[1] >> s[2] >> s[3];

          bool ok = true;
          bool gotilab = false;

          for (size_t i = 0; i < 4; ++i)
            {
              if (s[i].length() == 0) ok = false;
              else if (s[i] == "ilab") gotilab = true;
            }

          if (ok && gotilab)
            {
              for (size_t i = 0; i < 4; ++i)
                std::cout << s[i] << std::endl;

              ofs << sformat("%-50s\t%8s\t%8s\t%8s\t%8s",
                             rgbname.c_str(),
                             s[0].c_str(), s[1].c_str(),
                             s[2].c_str(), s[3].c_str())
                  << std::endl;

              Raster::WriteRGB(rgb, rgbname);
            }
          else
            {
              std::cout << "skipped" << std::endl;
            }
        }
    }

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPMEDIA_APP_TRAIN_NEED_FOR_SPEED_C_DEFINED
