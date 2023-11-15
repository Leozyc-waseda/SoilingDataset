/*!@file AppNeuro/test-SoxChannel.C Test the SoxChannel class */

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
// Primary maintainer for this file: Rob Peters <rjpeters@klab.caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppNeuro/test-SoxChannel.C $
// $Id: test-SoxChannel.C 9412 2008-03-10 23:10:15Z farhan $
//


#include "Channels/SoxChannel.H"
#include "Component/ModelManager.H"
#include "Component/ParamMap.H"
#include "Image/ColorOps.H"
#include "Image/DrawOps.H"
#include "Image/LevelSpec.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/Range.H"
#include "Image/ShapeOps.H"
#include "Image/fancynorm.H"
#include "Raster/Raster.H"
#include "Util/log.H"
#include "rutz/compat_snprintf.h"

#include <algorithm>
#include <vector>

int main(const int argc, const char** argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("SoxChannel Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<SoxChannel> lc(new SoxChannel(manager));
  manager.addSubComponent(lc);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<image.ppm> [scale]", 1, 2) == false)
    return(1);

  // do post-command-line configs:
  int SCALE = 1;
  if (manager.numExtraArgs() > 1) SCALE = manager.getExtraArgAs<int>(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  // read the input image:
  const Image<PixRGB<byte> > input =
    Raster::ReadRGB(manager.getExtraArg(0));

  lc->input(InputFrame::fromRgb(&input));

  std::vector<Image<float> > lin(lc->numChans());
  std::vector<Image<float> > nonlin(lc->numChans());

  Range<float> lin_rng;
  Range<float> nonlin_rng;

  for (uint ori = 0; ori < lc->numChans(); ++ori)
    {
      lin[ori] = lc->getLinearResponse(ori, SCALE);
      nonlin[ori] = lc->getNonlinearResponse(ori, SCALE);

      lin_rng.merge(rangeOf(lin[ori]));
      nonlin_rng.merge(rangeOf(nonlin[ori]));
    }

  LINFO("lin_rng: [%g, %g]", lin_rng.min(), lin_rng.max());
  LINFO("nonlin_rng: [%g, %g]", nonlin_rng.min(), nonlin_rng.max());

  std::vector<Image<PixRGB<float> > > resps;

  // This is a "backwards" range so that we in effect do a binaryReverse()
  // when we call remapRange()
  Range<float> stdrange(1.0f, 0.0f);

  for (uint ori = 0; ori < lc->numChans(); ++ori)
    {
      lin[ori] = remapRange(lin[ori], lin_rng, stdrange);
      nonlin[ori] = remapRange(nonlin[ori], nonlin_rng, stdrange);

      char text[256]; text[0] = 0;

      if (lin[ori].getWidth() > 50)
        snprintf(text, 256, "%d", ori);

      int border_width = lin[ori].getWidth() > 25 ? 1 : 0;

      resps.push_back(stain(lin[ori], PixRGB<float>(245, 255, 245)));
      writeText(resps.back(), Point2D<int>(0,0), text);
      inplaceSetBorders(resps.back(), border_width, PixRGB<float>(128, 255, 128));

      resps.push_back(stain(nonlin[ori], PixRGB<float>(245, 245, 255)));
      inplaceSetBorders(resps.back(), border_width, PixRGB<float>(128, 128, 255));

      Image<float> diff = (lin[ori] - nonlin[ori]);

      Image<PixRGB<float> > cdiff = normalizeRGPolar(diff, 2.0, -2.0);
      normalizeC(cdiff, 0, 255);
      resps.push_back(cdiff);
      inplaceSetBorders(resps.back(), border_width, PixRGB<float>(255, 128, 128));
    }

  Image<PixRGB<float> > arr = concatArray(&resps[0], resps.size(), 3);
  Raster::VisuRGB(arr, "sox.ppm");

  // get ready for a clean exit:
  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
