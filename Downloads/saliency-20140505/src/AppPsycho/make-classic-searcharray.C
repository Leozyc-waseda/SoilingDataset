/*!@file AppPsycho/make-classic-searcharray.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/make-classic-searcharray.C $
// $Id: make-classic-searcharray.C 9084 2007-12-13 01:00:11Z rjpeters $
//

#ifndef APPPSYCHO_MAKE_CLASSIC_SEARCHARRAY_C_DEFINED
#define APPPSYCHO_MAKE_CLASSIC_SEARCHARRAY_C_DEFINED

#include "Image/geom.h"
#include "Psycho/ClassicSearchItem.H"
#include "Psycho/SearchArray.H"
#include "Raster/Raster.H"
#include "Util/StringConversions.H"
#include "Util/sformat.H"
#include "rutz/rand.h"

#include <stdio.h>

static double c2r(const char* arg)
{
  const double deg = fromStr<double>(arg);
  return geom::rad_0_2pi(geom::deg2rad(deg));
}

int main(int argc, char** argv)
{
  if (argc != 11)
    {
      fprintf(stderr, "usage: %s arrayWxarrayH itemsize "
              "fgitemlist fg_min_angle fg_max_angle "
              "bgitemlist bg_min_angle bg_max_angle "
              "rand_seed outimagestem\n\n"
              "available item types: C, O, Q, +, L, T, -\n",
              argv[0]);
      return -1;
    }

  const Dims arraydims = fromStr<Dims>(argv[1]);
  const int itemsize = fromStr<int>(argv[2]);

  const uint seed = fromStr<uint>(argv[9]);

  ClassicSearchItemFactory fg(SearchItem::FOREGROUND, argv[3], itemsize,
                              Range<double>(c2r(argv[4]), c2r(argv[5])),
                              3, seed + 12345U);
  ClassicSearchItemFactory bg(SearchItem::BACKGROUND, argv[6], itemsize,
                              Range<double>(c2r(argv[7]), c2r(argv[8])),
                              3, seed + 23456U);

  const std::string outimagestem = argv[10];

  rutz::urand fgpos(seed + 34567U);

  SearchArray g(arraydims, 85.0, 70.0, itemsize);
  g.generateBackground(bg, 2, false, 10000, true, seed + 45678U);
  g.replaceElementAtSamePos(fgpos.idraw(g.numElements()),
                            fg.make(geom::vec2d()));

  Raster::WriteGray(g.getImage(0.0, 1.0, 1.0),
                    sformat("%s.png", outimagestem.c_str()));

  g.saveCoords(sformat("%s.txt", outimagestem.c_str()));
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPPSYCHO_MAKE_CLASSIC_SEARCHARRAY_C_DEFINED
