/*!@file TIGS/TrainingSet.C Manage a paired set of eye position data and input feature vectors */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/TrainingSet.C $
// $Id: TrainingSet.C 15310 2012-06-01 02:29:24Z itti $
//

#ifndef TIGS_TRAININGSET_C_DEFINED
#define TIGS_TRAININGSET_C_DEFINED

#include "TIGS/TrainingSet.H"

#include "Component/ModelOptionDef.H"
#include "Image/ShapeOps.H"
#include "Media/MediaOpts.H"
#include "Raster/Raster.H"
#include "TIGS/TigsOpts.H"
#include "Util/AllocAux.H"
#include "rutz/trace.h"

// Used by: TrainingSet
static const ModelOptionDef OPT_TrainingSetDecimation =
  { MODOPT_ARG(int), "TrainingSetDecimation", &MOC_TIGS, OPTEXP_CORE,
    "Factor by which to decimate the number of samples in "
    "topdown context training sets",
    "tdata-decimation", '\0', "<int>", "1" };

// Used by: TrainingSet
static const ModelOptionDef OPT_TrainingSetRebalance =
  { MODOPT_FLAG, "TrainingSetRebalance", &MOC_TIGS, OPTEXP_CORE,
    "Whether to rebalance the training set so that the distribution "
    "of eye positions is as flat as possible",
    "tdata-rebalance", '\0', "", "false" };

// Used by: TrainingSet
static const ModelOptionDef OPT_TrainingSetRebalanceThresh =
  { MODOPT_ARG(uint), "TrainingSetRebalanceThresh", &MOC_TIGS, OPTEXP_CORE,
    "When rebalancing the training set's distribution of eye "
    "positions, only include positions for which at least this many "
    "samples are available",
    "tdata-rebalance-thresh", '\0', "<int>", "10" };

// Used by: TrainingSet
static const ModelOptionDef OPT_TrainingSetRebalanceGroupSize =
  { MODOPT_ARG(uint), "TrainingSetRebalanceGroupSize", &MOC_TIGS, OPTEXP_CORE,
    "When rebalancing the training set's distribution of eye "
    "positions, pool the samples into this many samples per eye position",
    "tdata-rebalance-group-size", '\0', "<int>", "10" };

TrainingSet::TrainingSet(OptionManager& mgr, const std::string& fx_type)
  :
  ModelComponent(mgr, "TrainingSet", "TrainingSet"),
  itsRawInputDims(&OPT_InputFrameDims, this),
  itsDoRebalance(&OPT_TrainingSetRebalance, this),
  itsRebalanceThresh(&OPT_TrainingSetRebalanceThresh, this),
  itsRebalanceGroupSize(&OPT_TrainingSetRebalanceGroupSize, this),
  itsFxType(fx_type),
  itsReduction(32),
  itsNumFeatures(0),
  itsLocked(false),
  itsFeatureVec(),
  itsPositionVec(),
  itsPosGroups(),
  itsNumTraining(0),
  itsNumLoaded(0),
  itsFeatures(),
  itsPositions(),
  itsDecimationFactor(&OPT_TrainingSetDecimation, this)
{}

Dims TrainingSet::scaledInputDims() const
{
  ASSERT(itsRawInputDims.getVal().isNonEmpty());
  ASSERT(itsReduction > 0);

  return itsRawInputDims.getVal() / int(itsReduction);
}

size_t TrainingSet::numPositions() const
{
  ASSERT(scaledInputDims().isNonEmpty());
  return scaledInputDims().sz();
}

int TrainingSet::p2p(const int i, const int j) const
{
  ASSERT(scaledInputDims().isNonEmpty());
  ASSERT(itsReduction > 0);
  return (j / itsReduction) * scaledInputDims().w() + (i / itsReduction);
}

int TrainingSet::p2p(const Point2D<int>& p) const
{
  return p2p(p.i, p.j);
}

Image<float> TrainingSet::recordSample(const Point2D<int>& loc,
                                       const Image<float>& features)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  ASSERT(!itsLocked);

  ASSERT(scaledInputDims().isNonEmpty());

  if (itsNumFeatures == 0)
    {
      // ok, it's the first time, so let's pick up our number of
      // features from the size of the input vector
      itsNumFeatures = features.getSize();
      LINFO("%s TrainingSet with %" ZU " features",
            itsFxType.c_str(), itsNumFeatures);
    }

  ASSERT(itsNumFeatures > 0);
  ASSERT(size_t(features.getSize()) == itsNumFeatures);

  ASSERT(loc.i >= 0);
  ASSERT(loc.j >= 0);

  ASSERT(itsReduction > 0);

  const Point2D<int> locr(loc.i / itsReduction, loc.j / itsReduction);

  const size_t i1 = locr.i;
  const size_t i0 = locr.i > 0 ? (locr.i-1) : locr.i;
  const size_t i2 = locr.i < (scaledInputDims().w() - 1) ? (locr.i+1) : locr.i;

  const size_t j1 = locr.j;
  const size_t j0 = locr.j > 0 ? (locr.j-1) : locr.j;
  const size_t j2 = locr.j < (scaledInputDims().h() - 1) ? (locr.j+1) : locr.j;

  const size_t p00 = j0 * scaledInputDims().w() + i0;
  const size_t p01 = j1 * scaledInputDims().w() + i0;
  const size_t p02 = j2 * scaledInputDims().w() + i0;

  const size_t p10 = j0 * scaledInputDims().w() + i1;
  const size_t p11 = j1 * scaledInputDims().w() + i1;
  const size_t p12 = j2 * scaledInputDims().w() + i1;

  const size_t p20 = j0 * scaledInputDims().w() + i2;
  const size_t p21 = j1 * scaledInputDims().w() + i2;
  const size_t p22 = j2 * scaledInputDims().w() + i2;

  const size_t np = this->numPositions();

  for (size_t x = 0; x < np; ++x)
    {
      itsPositionVec.push_back(0.0f);

      float& v = itsPositionVec.back();

      // note these are not a series of "else if", since it's
      // possible that more than one of the p00, p01, etc. point to
      // the same position, if we are at an edge of the image
      if (x == p00) v += 0.25;
      if (x == p01) v += 0.5;
      if (x == p02) v += 0.25;

      if (x == p10) v += 0.5;
      if (x == p11) v += 1.0;
      if (x == p12) v += 0.5;

      if (x == p20) v += 0.25;
      if (x == p21) v += 0.5;
      if (x == p22) v += 0.25;
    }

  for (size_t x = 0; x < itsNumFeatures; ++x)
    {
      itsFeatureVec.push_back(features[x]);
    }

  ++itsNumTraining;

  // return an image showing the eye position array after being
  // subjected to our little 3x3 blurring
  return Image<float>(&*itsPositionVec.end() - this->numPositions(),
                      scaledInputDims());
}

void TrainingSet::load(const std::string& pfx)
{
  if (itsDoRebalance.getVal())
    {
      this->loadRebalanced(pfx);
      return;
    }

  GVX_TRACE(__PRETTY_FUNCTION__);

  const std::string ffile = pfx+"-features.pfm";
  const std::string pfile = pfx+"-positions.pfm";

  Image<float> feat = Raster::ReadFloat(ffile, RASFMT_PFM);
  Image<float> pos = Raster::ReadFloat(pfile, RASFMT_PFM);

  ASSERT(feat.getHeight() == pos.getHeight());

  if (itsNumFeatures == 0)
    {
      // ok, it's the first time, so let's pick up our number of
      // features from the size of the input vector
      itsNumFeatures = feat.getWidth();
      LINFO("%s TrainingSet with %" ZU " features",
            itsFxType.c_str(), itsNumFeatures);
    }

  ASSERT(size_t(feat.getWidth()) == itsNumFeatures);

  if (itsDecimationFactor.getVal() > 1)
    {
      feat = blurAndDecY(feat, itsDecimationFactor.getVal());
      pos = blurAndDecY(pos, itsDecimationFactor.getVal());

      ASSERT(feat.getHeight() == pos.getHeight());
    }

  itsFeatureVec.insert(itsFeatureVec.end(), feat.begin(), feat.end());
  itsPositionVec.insert(itsPositionVec.end(), pos.begin(), pos.end());

  itsNumTraining += feat.getHeight();

  ++itsNumLoaded;

  // we've loaded external data, so we don't want to allow any more
  // internal training samples to come in through recordSample();
  // however, we could still accept more external samples through
  // additional load() calls
  itsLocked = true;

  LINFO("loaded %d samples from training set %s, %d total training samples from %d files",
        feat.getHeight(), pfx.c_str(), itsNumTraining, itsNumLoaded);

  // we have loaded some huge .pfm files here which are likely to be
  // of unusual sizes, so we will only waste memory by trying to cache
  // those memory blocks, so let's just release all free memory now:
  invt_allocation_release_free_mem();
}

void TrainingSet::loadRebalanced(const std::string& pfx)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  const std::string ffile = pfx+"-features.pfm";
  const std::string pfile = pfx+"-positions.pfm";

  Image<float> feat = Raster::ReadFloat(ffile, RASFMT_PFM);
  Image<float> pos = Raster::ReadFloat(pfile, RASFMT_PFM);

  ASSERT(feat.getHeight() == pos.getHeight());

  if (itsNumFeatures == 0)
    {
      // ok, it's the first time, so let's pick up our number of
      // features from the size of the input vector
      itsNumFeatures = feat.getWidth();
      LINFO("%s TrainingSet with %" ZU " features",
            itsFxType.c_str(), itsNumFeatures);

      std::vector<PosGroup>().swap(itsPosGroups);
      itsPosGroups.resize(pos.getWidth(),
                          PosGroup(itsRebalanceGroupSize.getVal(),
                                   feat.getWidth(), pos.getWidth()));

      ASSERT(itsRebalanceThresh.getVal() >= itsRebalanceGroupSize.getVal());
    }

  ASSERT(size_t(feat.getWidth()) == itsNumFeatures);

  for (int y = 0; y < pos.getHeight(); ++y)
    {
      int nmax = 0;
      for (int x = 0; x < pos.getWidth(); ++x)
        {
          const float v = pos.getVal(x, y);
          if (v >= 1.0f)
            {
              ++nmax;
              itsPosGroups[x].add(feat.getArrayPtr() + y * feat.getWidth(),
                                  pos.getArrayPtr() + y * pos.getWidth());
            }
        }

      if (nmax != 1)
        LFATAL("nmax = %d (expected nmax = 1) in row %d", nmax, y);
    }

  std::vector<float>().swap(itsFeatureVec);
  std::vector<float>().swap(itsPositionVec);

  uint nzero = 0;
  uint naccept = 0;
  uint nsamp = 0;
  Image<byte> bb(20, 15, ZEROS);
  itsNumTraining = 0;
  for (uint i = 0; i < itsPosGroups.size(); ++i)
    {
      if (itsPosGroups[i].totalcount == 0)
        ++nzero;
      if (itsPosGroups[i].totalcount >= itsRebalanceThresh.getVal())
        {
          ++naccept;

          for (uint k = 0; k < itsPosGroups[i].counts.size(); ++k)
            {
              const Image<float> f =
                itsPosGroups[i].features[k] / itsPosGroups[i].counts[k];

              const Image<float> p =
                itsPosGroups[i].positions[k] / itsPosGroups[i].counts[k];

              itsFeatureVec.insert(itsFeatureVec.end(),
                                   f.begin(), f.end());
              itsPositionVec.insert(itsPositionVec.end(),
                                    p.begin(), p.end());

              ++itsNumTraining;
            }
          bb[i] = 255;
        }
      nsamp += itsPosGroups[i].totalcount;
    }

  LINFO("ngroups = %" ZU ", nsamp = %u, naccept = %u, nzero = %u",
        itsPosGroups.size(), nsamp, naccept, nzero);

  ++itsNumLoaded;

  // we've loaded external data, so we don't want to allow any more
  // internal training samples to come in through recordSample();
  // however, we could still accept more external samples through
  // additional load() calls
  itsLocked = true;

  LINFO("loaded %d samples from training set %s, %d total training samples from %d files",
        feat.getHeight(), pfx.c_str(), itsNumTraining, itsNumLoaded);

  // we have loaded some huge .pfm files here which are likely to be
  // of unusual sizes, so we will only waste memory by trying to cache
  // those memory blocks, so let's just release all free memory now:
  invt_allocation_release_free_mem();
}

void TrainingSet::save(const std::string& pfx)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  const std::string ffile = pfx+"-features.pfm";
  const std::string pfile = pfx+"-positions.pfm";

  if (Raster::fileExists(ffile))
    LINFO("save skipped; file already exists: %s", ffile.c_str());
  else
    Raster::WriteFloat(this->getFeatures(), FLOAT_NORM_PRESERVE, ffile, RASFMT_PFM);

  if (Raster::fileExists(pfile))
    LINFO("save skipped; file already exists: %s", pfile.c_str());
  else
    Raster::WriteFloat(this->getPositions(), FLOAT_NORM_PRESERVE, pfile, RASFMT_PFM);

  LINFO("saved training set %s", pfx.c_str());
}

Image<float> TrainingSet::getFeatures() const
{
  ASSERT(itsNumFeatures > 0);

  if (itsFeatures.getHeight() != itsNumTraining)
    {
      itsFeatures = Image<float>(&itsFeatureVec[0],
                                 itsNumFeatures, itsNumTraining);
    }

  return itsFeatures;
}

Image<float> TrainingSet::getPositions() const
{
  if (itsPositions.getHeight() != itsNumTraining)
    {
      itsPositions = Image<float>(&itsPositionVec[0],
                                  this->numPositions(), itsNumTraining);
    }

  return itsPositions;
}

uint TrainingSet::inputReduction() const
{
  return itsReduction;
}

const std::string& TrainingSet::fxType() const
{
  return itsFxType;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_TRAININGSET_C_DEFINED
