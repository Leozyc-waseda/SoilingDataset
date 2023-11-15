/*!@file TIGS/VisualCortexFeatureExtractor.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/VisualCortexFeatureExtractor.C $
// $Id: VisualCortexFeatureExtractor.C 10982 2009-03-05 05:11:22Z itti $
//

#ifndef TIGS_VISUALCORTEXFEATUREEXTRACTOR_C_DEFINED
#define TIGS_VISUALCORTEXFEATUREEXTRACTOR_C_DEFINED

#include "TIGS/VisualCortexFeatureExtractor.H"

#include "Channels/ChannelBase.H"
#include "Image/ShapeOps.H"
#include "Neuro/VisualCortex.H"
#include "Util/Assert.H"
#include "rutz/trace.h"

VisualCortexFeatureExtractor::
VisualCortexFeatureExtractor(OptionManager& mgr) :
  FeatureExtractor(mgr, "vfx"),
  itsVC(new VisualCortex(mgr))
{
  this->addSubComponent(itsVC);

  this->setCheckFrequency(0);

  LFATAL("FIXME");

  /*
  VisualCortexWeights wts = VisualCortexWeights::zeros();
  wts.chanIw = 1.0;
  wts.chanCw = 1.0;
  wts.chanOw = 1.0;
  wts.chanFw = 1.0;
  wts.chanMw = 1.0;
  wts.chanCow = 1.0;
  wts.chanLw = 1.0;
  wts.chanTw = 1.0;
  wts.chanXw = 1.0;
  wts.chanEw = 1.0;
  wts.chanGw = 1.0;
  itsVC->addDefaultChannels(wts);
  */
}

VisualCortexFeatureExtractor::
~VisualCortexFeatureExtractor() {}

Image<float> VisualCortexFeatureExtractor::
doExtract(const TigsInputFrame& fin)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  if (fin.isGhost())
    LFATAL("VisualCortexFeatureExtractor needs non-ghost frames");

  LFATAL("FIXME");

  /*
  itsVC->input(InputFrame::fromRgb(&fin.rgb(), fin.t()));

  const int sml = 4;

  const int mapsize =
    (fin.rgb().getWidth() >> sml) * (fin.rgb().getHeight() >> sml);

  const int numFeatures =
    (mapsize/4) // for visual cortex itself
    + itsVC->numChans() * (mapsize/64);

  Image<float> result(numFeatures, 1, ZEROS);

  Image<float>::iterator dptr = result.beginw();

  ASSERT(fin.rgb().getDims() == Dims(512,512));

  const Image<float> vco = downSize(itsVC->getOutput(), 16, 16, 3);

  for (int i = 0; i < vco.getSize(); ++i)
    {
      *dptr++ = 5e9f * vco[i];
    }

  for (uint m = 0; m < itsVC->numChans(); ++m)
    {
      nub::ref<ChannelBase> chan = itsVC->subChan(m);
      const Image<float> chanout = downSize(chan->getOutput(), 4, 4, 3);

      for (int i = 0; i < chanout.getSize(); ++i)
        *dptr++ = chanout[i];
    }

  ASSERT(dptr == result.endw());

  return result * 50.0f;
  */

  return Image<float>();
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_VISUALCORTEXFEATUREEXTRACTOR_C_DEFINED
