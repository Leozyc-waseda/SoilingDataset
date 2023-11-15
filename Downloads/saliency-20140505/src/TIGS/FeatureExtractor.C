/*!@file TIGS/FeatureExtractor.C Base class for topdown feature extractors. */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/FeatureExtractor.C $
// $Id: FeatureExtractor.C 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef TIGS_FEATUREEXTRACTOR_C_DEFINED
#define TIGS_FEATUREEXTRACTOR_C_DEFINED

#include "TIGS/FeatureExtractor.H"

#include "Component/ModelOptionDef.H"
#include "Image/CutPaste.H"
#include "Image/MathOps.H"
#include "Raster/Raster.H"
#include "TIGS/TigsOpts.H"
#include "Util/log.H"
#include "rutz/error_context.h"
#include "rutz/sfmt.h"
#include "rutz/trace.h"

// Used by: FeatureExtractor
static const ModelOptionDef OPT_CacheSavePrefix =
  { MODOPT_ARG_STRING, "CacheSavePrefix", &MOC_TIGS, OPTEXP_CORE,
    "Filename stem name for feature-extractor caches",
    "cache-save-prefix", '\0', "<string>", "" };

namespace
{
  template <class T>
  Image<T> asRow(const Image<T>& in)
  {
  GVX_TRACE(__PRETTY_FUNCTION__);
    return Image<T>(in.getArrayPtr(), in.getSize(), 1);
  }
}

FeatureExtractor::FeatureExtractor(OptionManager& mgr,
                                   const std::string& name)
  :
  ModelComponent(mgr, name, name),
  itsCacheSavePrefix(&OPT_CacheSavePrefix, this),
  itsName(name),
  itsCache(),
  itsNumHits(0),
  itsCheckFrequency(100),
  itsPrevTime(SimTime::ZERO())
{}

FeatureExtractor::~FeatureExtractor() {}

void FeatureExtractor::start2()
{
  this->load(itsCacheSavePrefix.getVal().c_str());
}

void FeatureExtractor::stop1()
{
  this->save(itsCacheSavePrefix.getVal().c_str());
}

Image<float> FeatureExtractor::extract(const TigsInputFrame& fin)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  GVX_ERR_CONTEXT(rutz::sfmt("extracting features in FeatureExtractor %s",
                             itsName.c_str()));

  const bool isnewtime = (fin.t() > itsPrevTime);
  itsPrevTime = fin.t();

  if (!this->isCacheable())
    return doExtract(fin);

  const Digest<16> digest = fin.getHash();;

  typedef std::map<Digest<16>, Image<float> > CacheType;

  CacheType::iterator itr = itsCache.find(digest);

  if (itr != itsCache.end())
    {
      LINFO("cache hit in %s on md5 digest %s",
            itsName.c_str(), digest.asString().c_str());

      ++itsNumHits;

      // periodically check things to make sure the cache is valid:
      if (isnewtime
          &&
          !fin.isGhost()
          &&
          (itsNumHits == 1 ||
           (itsCheckFrequency > 0
            && itsNumHits % itsCheckFrequency == 0)))
        {
          const Image<float> f = doExtract(fin);

          const Image<float> actual = asRow(f);
          const Image<float> expected = asRow((*itr).second);

          if (!(actual == expected))
            {
              LINFO("RMSerr(actual,expected)=%g", RMSerr(actual,expected));
              LINFO("corrcoef(actual,expected)=%g", corrcoef(actual,expected));

              LFATAL("cache integrity check failed "
                     "in %s after %d hits",
                     itsName.c_str(), itsNumHits);
            }
          else
            LINFO("cache integrity check OK in %s after %d hits",
                  itsName.c_str(), itsNumHits);
        }

      return (*itr).second;
    }

  // else...
  const Image<float> f = doExtract(fin);
  itsCache.insert(CacheType::value_type(digest, f));
  return f;
}

void FeatureExtractor::save(const char* pfx) const
{
  GVX_TRACE(__PRETTY_FUNCTION__);
  if (!this->isCacheable())
    {
      LINFO("%s not cacheable; save() skipped", itsName.c_str());
      return;
    }

  if (itsCache.size() == 0)
    {
      LINFO("%s has no cache entries; save() skipped", itsName.c_str());
      return;
    }

  const std::string dfile = sformat("%s-%s-digests.pgm",
                                    pfx, itsName.c_str());
  const std::string ffile = sformat("%s-%s-features.pfm",
                                    pfx, itsName.c_str());

  if (Raster::fileExists(dfile))
    {
      LINFO("%s cache file %s already exists; save() skipped",
            itsName.c_str(), dfile.c_str());
      return;
    }

  if (Raster::fileExists(ffile))
    {
      LINFO("%s cache file %s already exists; save() skipped",
            itsName.c_str(), ffile.c_str());
      return;
    }

  ASSERT(itsCache.size() > 0);

  Image<byte> digests(16, itsCache.size(), NO_INIT);
  Image<float> features((*(itsCache.begin())).second.getSize(),
                        itsCache.size(), NO_INIT);

  typedef std::map<Digest<16>, Image<float> > CacheType;

  int row = 0;

  for (CacheType::const_iterator
         itr = itsCache.begin(), stop = itsCache.end();
       itr != stop; ++itr)
    {
      inplacePaste(digests, (*itr).first.asImage(),
                   Point2D<int>(0, row));

      inplacePaste(features, asRow((*itr).second),
                   Point2D<int>(0, row));

      ++row;
    }

  ASSERT(size_t(row) == itsCache.size());

  Raster::WriteGray(digests, dfile, RASFMT_PNM);
  Raster::WriteFloat(features, FLOAT_NORM_PRESERVE, ffile, RASFMT_PFM);

  LINFO("saved %d cache entries to %s", row, dfile.c_str());
  LINFO("saved %d cache entries to %s", row, ffile.c_str());
}

void FeatureExtractor::load(const char* pfx)
{
  GVX_TRACE(__PRETTY_FUNCTION__);
  if (!this->isCacheable())
    {
      LINFO("%s not cacheable; load() skipped", itsName.c_str());
      return;
    }

  const std::string dfile = sformat("%s-%s-digests.pgm",
                                    pfx, itsName.c_str());
  const std::string ffile = sformat("%s-%s-features.pfm",
                                    pfx, itsName.c_str());

  if (!Raster::fileExists(dfile))
    {
      LINFO("%s cache file %s not found; load() skipped",
            itsName.c_str(), dfile.c_str());
      return;
    }

  if (!Raster::fileExists(ffile))
    {
      LINFO("%s cache file %s not found; load() skipped",
            itsName.c_str(), ffile.c_str());
      return;
    }

  const Image<byte> digests = Raster::ReadGray(dfile, RASFMT_PNM);
  const Image<float> features = Raster::ReadFloat(ffile, RASFMT_PFM);

  ASSERT(digests.getHeight() == features.getHeight());

  const int nrows = digests.getHeight();

  typedef std::map<Digest<16>, Image<float> > CacheType;

  itsCache.clear();

  LINFO("digests=%dx%d", digests.getWidth(), digests.getHeight());

  for (int i = 0; i < nrows; ++i)
    {
      Digest<16> d = Digest<16>::asDigest(&digests[Point2D<int>(0, i)],
                                          digests.getWidth());

      Image<float> f(&features[Point2D<int>(0, i)],
                     features.getWidth(), 1);

      itsCache.insert(CacheType::value_type(d, f));
    }

  ASSERT(itsCache.size() == size_t(nrows));

  LINFO("loaded %d cache entries from %s", nrows, dfile.c_str());
  LINFO("loaded %d cache entries from %s", nrows, ffile.c_str());
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_FEATUREEXTRACTOR_C_DEFINED
