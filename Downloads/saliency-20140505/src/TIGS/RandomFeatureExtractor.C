/*!@file TIGS/RandomFeatureExtractor.C Random control for topdown feature extractors */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/RandomFeatureExtractor.C $
// $Id: RandomFeatureExtractor.C 15465 2013-04-18 01:45:18Z itti $
//

#ifndef TIGS_RANDOMFEATUREEXTRACTOR_C_DEFINED
#define TIGS_RANDOMFEATUREEXTRACTOR_C_DEFINED

#include "TIGS/RandomFeatureExtractor.H"

#include "rutz/trace.h"
#include <sys/types.h>
#include <unistd.h>

RandomFeatureExtractor::RandomFeatureExtractor(OptionManager& mgr) :
  FeatureExtractor(mgr, "rfx"),
  itsN(4*112), // FIXME make this a command-line param
  itsGenerator(time((time_t*)0)+getpid()) {}

RandomFeatureExtractor::~RandomFeatureExtractor() {}

Image<float> RandomFeatureExtractor::doExtract(const TigsInputFrame&)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  Image<float> result(1, itsN, NO_INIT);

  for (int i = 0; i < itsN; ++i)
    result[i] = 255.0 * itsGenerator.fdraw();

  return result;
}

bool RandomFeatureExtractor::isCacheable() const
{
  return false;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_RANDOMFEATUREEXTRACTOR_C_DEFINED
