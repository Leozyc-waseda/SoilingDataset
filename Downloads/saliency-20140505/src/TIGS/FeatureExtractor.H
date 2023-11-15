/*!@file TIGS/FeatureExtractor.H Base class for topdown feature extractors. */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/FeatureExtractor.H $
// $Id: FeatureExtractor.H 8297 2007-04-25 00:26:17Z rjpeters $
//

#ifndef TIGS_FEATUREEXTRACTOR_H_DEFINED
#define TIGS_FEATUREEXTRACTOR_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Image/Hash.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "TIGS/TigsInputFrame.H"
#include "Util/Assert.H"
#include "Util/SimTime.H"

#include <map>

/// Base class for topdown feature extractors.
class FeatureExtractor : public ModelComponent
{
public:
  FeatureExtractor(OptionManager& mgr, const std::string& name);

  virtual ~FeatureExtractor();

  virtual void start2();

  virtual void stop1();

  Image<float> extract(const TigsInputFrame& fin);

  void save(const char* pfx) const;

  void load(const char* pfx);

protected:
  void setCheckFrequency(int f) { itsCheckFrequency = f; }

  void changeFxName(const std::string& s)
  {
    ASSERT(!this->started());
    itsName = s;
    this->setTagName(s);
    this->setDescriptiveName(s);
  }

private:
  virtual Image<float> doExtract(const TigsInputFrame& fin) = 0;

  virtual bool isCacheable() const { return true; }

  OModelParam<std::string> itsCacheSavePrefix;
  std::string itsName;
  std::map<Digest<16>, Image<float> > itsCache;
  int itsNumHits;
  int itsCheckFrequency;
  SimTime itsPrevTime;
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_FEATUREEXTRACTOR_H_DEFINED
