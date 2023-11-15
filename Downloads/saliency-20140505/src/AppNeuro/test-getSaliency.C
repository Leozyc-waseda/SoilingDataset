/*!@file AppNeuro/test-getSaliency.C Sample program illustrating the use of GetSaliency */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppNeuro/test-getSaliency.C $
// $Id: test-getSaliency.C 14286 2010-12-01 17:46:34Z sophie $
//

#ifndef APPNEURO_TEST_GETSALIENCY_C_DEFINED
#define APPNEURO_TEST_GETSALIENCY_C_DEFINED

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H" // for rescale()
#include "Neuro/getSaliency.H"
#include "Raster/Raster.H"
#include "Util/log.H"

// Here is a full working program that may be a useful simple starting
// point for anybody who wants to access saliency maps
// programmatically from within their own code, rather than via the
// command line (in which case ezvision is the way to go)

int main(int argc, char** argv)
{
  ModelManager manager("test");

  nub::ref<GetSaliency> saliency(new GetSaliency(manager));
  manager.addSubComponent(saliency);
  if (manager.parseCommandLine(argc, argv, "input-image", 1, 1) == false)
    return -1;
  manager.start();

  const Image<PixRGB<byte> > img = Raster::ReadRGB(manager.getExtraArg(0));

  const int num_salient_spots = saliency->compute(img, SimTime::SECS(1));

  LINFO("found %d salient spots", num_salient_spots);

  const Image<float> salmap = saliency->getSalmap();

  const Image<float> resized_salmap = rescale(salmap, img.getDims());
  const std::vector<subMap> itsSubMaps = saliency->getSubMaps();

  Raster::WriteFloat(resized_salmap, FLOAT_NORM_0_255, "salmap.png");
  Raster::WriteFloat(rescale(itsSubMaps[1].itsSubMap,
                           img.getDims()), FLOAT_NORM_0_255,"subMap1.png");

  manager.stop();

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPNEURO_TEST_GETSALIENCY_C_DEFINED
