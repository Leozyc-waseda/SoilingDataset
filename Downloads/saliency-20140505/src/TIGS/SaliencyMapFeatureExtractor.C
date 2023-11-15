/*!@file TIGS/SaliencyMapFeatureExtractor.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/SaliencyMapFeatureExtractor.C $
// $Id: SaliencyMapFeatureExtractor.C 10845 2009-02-13 08:49:12Z itti $
//

#ifndef TIGS_SALIENCYMAPFEATUREEXTRACTOR_C_DEFINED
#define TIGS_SALIENCYMAPFEATUREEXTRACTOR_C_DEFINED

#include "TIGS/SaliencyMapFeatureExtractor.H"

#include "Component/ModelOptionDef.H"
#include "Image/fancynorm.H"
#include "Neuro/VisualCortex.H"
#include "Neuro/VisualCortexConfigurator.H"
#include "TIGS/TigsOpts.H"
#include "rutz/trace.h"

// Used by: VisualCortexFeatureExtractor
static const ModelOptionDef OPT_SmfxVcType =
  { MODOPT_ARG_STRING, "SmfxVcType", &MOC_TIGS, OPTEXP_CORE,
    "VisualCortex type for the saliency-map feature extractor",
    "smfx-vc-type", '\0', "<string>", "Std" };

// Used by: SaliencyMapFeatureExtractor
static const ModelOptionDef OPT_SmfxNormType =
  { MODOPT_ARG(MaxNormType), "SmfxNormType", &MOC_TIGS, OPTEXP_CORE,
    "Normalization type for the saliency-map feature extractor",
    "smfx-norm-type", '\0', "<string>", "Fancy" };

// Used by: SaliencyMapFeatureExtractor
static const ModelOptionDef OPT_SmfxRescale512 =
  { MODOPT_FLAG, "SmfxRescale512", &MOC_TIGS, OPTEXP_CORE,
    "Whether to rescale smfx to 512x512 before computing the saliency map",
    "smfx-rescale-512", '\0', "", "true" };

// ######################################################################
SaliencyMapFeatureExtractor::
SaliencyMapFeatureExtractor(OptionManager& mgr)
  :
  FeatureExtractor(mgr, "uninitialized"),
  itsNormType(&OPT_SmfxNormType, this),
  itsVcType(&OPT_SmfxVcType, this),
  itsRescale512(&OPT_SmfxRescale512, this),
  itsVCC(new VisualCortexConfigurator(mgr))
{
  this->addSubComponent(itsVCC);

  this->setCheckFrequency(0);
}

// ######################################################################
SaliencyMapFeatureExtractor::
~SaliencyMapFeatureExtractor() {}

// ######################################################################
void SaliencyMapFeatureExtractor::
paramChanged(ModelParamBase* const param,
             const bool valueChanged,
             ParamClient::ChangeStatus* status)
{
  FeatureExtractor::paramChanged(param, valueChanged, status);

  if (param == &itsVcType)
    {
      itsVCC->setModelParamString("VisualCortexType",
                                  itsVcType.getVal());

      itsVCC->getVC()->setModelParamVal
        ("MaxNormType", itsNormType.getVal(), MC_RECURSE);

      LINFO("set SaliencyMapFeatureExtractor to vctype=%s",
            itsVcType.getVal().c_str());
    }

  const char* pfx = itsRescale512.getVal() ? "smfx" : "smofx";

  if (itsVcType.getValString() == "Std")
    this->changeFxName(pfx + itsNormType.getValString());
  else
    this->changeFxName(pfx + itsNormType.getValString()
                       + itsVcType.getValString());
}

// ######################################################################
Dims SaliencyMapFeatureExtractor::smDims() const
{
  if (itsRescale512.getVal())
    {
      return Dims(512,512);
    }
  else
    {
      return Dims(640,480);
    }
}

// ######################################################################
Image<float> SaliencyMapFeatureExtractor::
doExtract(const TigsInputFrame& fin)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  if (fin.isGhost())
    LFATAL("SaliencyMapFeatureExtractor needs non-ghost frames");

  LFATAL("FIXME");
  /*
  if (itsRescale512.getVal())
    {
      ASSERT(fin.rgb().getDims() == Dims(512, 512));
      itsVCC->getVC()->input(InputFrame::fromRgb(&fin.rgb(), fin.t()));
    }
  else
    {
      ASSERT(fin.origframe().getDims() == Dims(640, 480));
      itsVCC->getVC()->input(InputFrame::fromRgb(&fin.origframe(), fin.t()));
    }

  Image<float> result = itsVCC->getVC()->getOutput();
  result *= 5e9f;

  return result;
  */
  return Image<float>();
}

// ######################################################################
void SaliencyMapFeatureExtractor::start1()
{
  itsVCC->getVC()->setModelParamVal
    ("MaxNormType", itsNormType.getVal(), MC_RECURSE);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_SALIENCYMAPFEATUREEXTRACTOR_C_DEFINED
