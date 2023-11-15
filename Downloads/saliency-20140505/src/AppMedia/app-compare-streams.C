/*!@file AppMedia/app-compare-streams.C compute correlation coefficients between two series of images */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-compare-streams.C $
// $Id: app-compare-streams.C 15310 2012-06-01 02:29:24Z itti $
//

#ifndef APPMEDIA_APP_COMPARE_STREAMS_C_DEFINED
#define APPMEDIA_APP_COMPARE_STREAMS_C_DEFINED

#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Range.H"
#include "Raster/Raster.H"
#include "Util/log.H"
#include <cstdio>

#ifdef HAVE_FENV_H
#include <fenv.h>
#endif

#include <string>
#include <fstream>
#include <vector>

namespace
{
  void readfile(const char* fname,
                std::vector<std::string>& filelist)
  {
    filelist.resize(0);

    std::ifstream ifs(fname);

    if (!ifs.is_open())
      {
        LFATAL("couldn't open %s for reading", fname);
      }

    std::string line;
    while (std::getline(ifs, line))
      {
        filelist.push_back(line);
      }

    LINFO("read %" ZU " filenames from %s", filelist.size(), fname);
  }
}

int main(int argc, char** argv)
{
  if (argc != 3)
    {
      printf("usage: %s filelist1 filelist2\n", argv[0]);
      return 1;
    }

#ifdef HAVE_FEENABLEEXCEPT
  feenableexcept(FE_DIVBYZERO|FE_INVALID);
#endif

  std::vector<std::string> filelist1;
  std::vector<std::string> filelist2;

  readfile(argv[1], filelist1);
  readfile(argv[2], filelist2);

  const size_t chunk_size = 256;

  for (size_t i = 0; i < filelist1.size(); i += chunk_size)
    {
      std::vector<Image<float> > images1;

      for (size_t ii = 0; ii < chunk_size; ++ii)
        {
          if (i+ii >= filelist1.size())
            break;

          images1.push_back(Raster::ReadFloat(filelist1[i+ii]));
        }

      for (size_t j = 0; j < filelist2.size(); j += chunk_size)
        {
          std::vector<Image<float> > images2;

          for (size_t jj = 0; jj < chunk_size; ++jj)
            {
              if (j+jj >= filelist2.size())
                break;

              images2.push_back(Raster::ReadFloat(filelist2[j+jj]));
            }

          for (size_t iii = 0; iii < images1.size(); ++iii)
            for (size_t jjj = 0; jjj < images2.size(); ++jjj)
              {
                const double rsq = corrcoef(images1[iii], images2[jjj]);

                fprintf(stdout, "%4" ZU "  %4" ZU "  %15e\n",
                        i + iii, j + jjj, rsq);
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

#endif // APPMEDIA_APP_COMPARE_STREAMS_C_DEFINED
