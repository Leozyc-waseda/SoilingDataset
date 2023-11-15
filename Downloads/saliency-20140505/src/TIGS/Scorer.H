/*!@file TIGS/Scorer.H Score the fit between predicted and actual eye positions */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/Scorer.H $
// $Id: Scorer.H 9132 2008-01-15 21:45:31Z rjpeters $
//

#ifndef TIGS_SCORER_H_DEFINED
#define TIGS_SCORER_H_DEFINED

#include "Image/Image.H"
#include "Util/log.H"
#include "rutz/rand.h"

#include <iosfwd>
#include <string>

/// Score the fit between predicted and actual eye positions
class Scorer
{
public:
  virtual ~Scorer();

  virtual void accum(const Image<float>& eyeposmap, int pos) = 0;

  virtual std::string getScoreString(const std::string& name) = 0;
};

struct KLScorer : public Scorer
{
  KLScorer(int nbins, int nrand);

  virtual ~KLScorer();

  virtual void accum(const Image<float>& eyeposmap, int pos);

  virtual std::string getScoreString(const std::string& name);

private:
  static rutz::urand theirGenerator;

  int itsNbins;
  int itsNrand;
  Image<int> itsObservedBins;
  Image<int> itsRandomBins;
  int itsNtrials;
};

class NssScorer : public Scorer
{
public:
  NssScorer();

  virtual void accum(const Image<float>& eyeposmap, int pos);

  virtual std::string getScoreString(const std::string& name);

  double getCurrentZscore() const
  {
    if (observedCount <= 0)
      LFATAL("no zscore observations");

    return currentZscore;
  }

  double getOverallZscore() const
  {
    if (observedCount <= 0)
      LFATAL("no zscore observations");

    return observedZscore / observedCount;
  }

private:
  double currentZscore;
  double observedZscore;
  double maxZscore;
  int observedCount;
};

class PercentileScorer : public Scorer
{
public:
  PercentileScorer();

  virtual void accum(const Image<float>& eyeposmap, int pos);

  virtual std::string getScoreString(const std::string& name);

  double getCurrentPrctile() const
  {
    if (observedCount <= 0)
      LFATAL("no prctile observations");

    return currentPrctile;
  }

  double getOverallPrctile() const
  {
    if (observedCount <= 0)
      LFATAL("no prctile observations");

    return observedPrctile / observedCount;
  }

private:
  double currentPrctile;
  double observedPrctile;
  int observedCount;
};

/// Saliency-weighted prediction error scorer
class SwpeScorer : public Scorer
{
public:
  SwpeScorer();

  virtual void accum(const Image<float>& eyeposmap, int pos);

  virtual std::string getScoreString(const std::string& name);

private:
  Dims itsDims;
  double itsEyeScore;
  double itsRandEyeScore;
  double itsRandMapScore;
  double itsFlatMapScore;
  int itsObservedCount;

  rutz::urand itsGenerator;
};

struct MulticastScorer
{
  MulticastScorer();

  void score(const std::string& name,
             const Image<float>& eyeposmap, int pos);

  void showScore(const std::string& name);

  void writeScore(const std::string& name, std::ostream& os);

  int observedCount;
  PercentileScorer itsPrctileScorer;
  NssScorer itsNssScorer;
  KLScorer itsKLScorer;
  SwpeScorer itsSwpeScorer;
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_SCORER_H_DEFINED
