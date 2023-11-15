/*!@file TIGS/Scorer.C Score the fit between predicted and actual eye positions */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/Scorer.C $
// $Id: Scorer.C 15465 2013-04-18 01:45:18Z itti $
//

#ifndef TIGS_SCORER_C_DEFINED
#define TIGS_SCORER_C_DEFINED

#include "TIGS/Scorer.H"

#include "Image/MathOps.H"
#include "Util/MathFunctions.H" // for clampValue()
#include "Util/sformat.H"
#include "rutz/trace.h"

#include <ostream>
#include <unistd.h>

Scorer::~Scorer() {}

KLScorer::KLScorer(int nbins, int nrand)
  :
  itsNbins(nbins),
  itsNrand(nrand),
  itsObservedBins(1, nbins, ZEROS),
  itsRandomBins(nrand, nbins, ZEROS),
  itsNtrials(0)
{}

KLScorer::~KLScorer() {}

void KLScorer::accum(const Image<float>& eyeposmap, int pos)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  float mi, ma; getMinMax(eyeposmap, mi, ma);

  // rescale into the range [0..nbins]
  Image<float> scaledmap = eyeposmap - mi;
  if ((ma-mi) > 0.0f)
    scaledmap /= ((ma-mi) / itsNbins);

  ASSERT(pos >= 0);
  ASSERT(pos < scaledmap.getSize());
  const int humanval = clampValue(int(scaledmap[pos]), 0, itsNbins - 1);

  ASSERT(humanval >= 0);
  ASSERT(humanval < itsObservedBins.getSize());
  itsObservedBins[humanval] += 1;

  for (int i = 0; i < itsNrand; ++i)
    {
      const int randpos = theirGenerator.idraw(scaledmap.getSize());

      ASSERT(randpos >= 0);
      ASSERT(randpos < scaledmap.getSize());
      const int randval = clampValue(int(scaledmap[randpos]), 0, itsNbins - 1);

      ASSERT(i >= 0);
      ASSERT(i < itsRandomBins.getWidth());
      ASSERT(randval >= 0);
      ASSERT(randval < itsRandomBins.getHeight());
      itsRandomBins[Point2D<int>(i, randval)] += 1;
    }

  ++itsNtrials;
}

std::string KLScorer::getScoreString(const std::string& name)
{
  if (itsNtrials == 0)
    {
      return sformat("[%s] no klscore observations", name.c_str());
    }

  Image<double> scores(itsNrand, 1, ZEROS);

  // count how many zeros we encounter in the histograms; we only
  // have a "non-tainted" kl score if there are no zeros
  int tainted = 0;

  for (int i = 0; i < itsNrand; ++i)
    {
      double currentscore = 0.0;

      for (int j = 0; j < itsNbins; ++j)
        {
          ASSERT(j < itsObservedBins.getSize());
          const double aa = itsObservedBins[j] / double(itsNtrials);
          ASSERT(i < itsRandomBins.getWidth());
          ASSERT(j < itsRandomBins.getHeight());
          const double bb = itsRandomBins[Point2D<int>(i,j)] / double(itsNtrials);

          if (aa != 0.0 && bb != 0.0)
            {
              currentscore += 0.5 * (aa*log(aa/bb) + bb*log(bb/aa));
            }
          else
            {
              ++tainted;
            }
        }

      ASSERT(i < scores.getSize());
      scores[i] = currentscore;
    }

  const double klmean = mean(scores);
  const double klstdev = stdev(scores);

  return sformat("[%s] %s (nt=%d) klscore: %10.5f +/- %10.5f",
                 name.c_str(), tainted > 0 ? "tainted" : "notaint",
                 tainted, klmean, klstdev);
}

rutz::urand KLScorer::theirGenerator(time((time_t*)0)+getpid());

NssScorer::NssScorer()
  :
  currentZscore(0.0),
  observedZscore(0.0),
  maxZscore(0.0),
  observedCount(0)
{}

void NssScorer::accum(const Image<float>& eyeposmap, int pos)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  float mi, ma, me; getMinMaxAvg(eyeposmap, mi, ma, me);

  float std = stdev(eyeposmap);

  if (std > 0.0)
    {
      currentZscore   = (eyeposmap[pos] - me) / std;
      observedZscore += currentZscore;
      maxZscore      += (ma - me) / std;
    }
  else
    {
      currentZscore = 0.0;
    }

  ++observedCount;
}

std::string NssScorer::getScoreString(const std::string& name)
{
  if (observedCount > 0)
    return sformat("[%s] observed zscore: %10.5f max zscore: %10.5f",
                   name.c_str(), observedZscore / observedCount,
                   maxZscore / observedCount);
  // else...
  return sformat("[%s] no zscore observations", name.c_str());
}

SwpeScorer::SwpeScorer()
  :
  itsDims(),
  itsEyeScore(0.0),
  itsRandEyeScore(0.0),
  itsRandMapScore(0.0),
  itsObservedCount(0),
  itsGenerator(time((time_t*)0)+getpid())
{}

namespace
{
  double getSwpe(const Image<float>& smap, const Point2D<int>& pos)
  {
    const int w = smap.getWidth();
    const int h = smap.getHeight();

    double result = 0.0;
    double mapsum = 0.0;

    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        {
          const double dist =
            sqrt((double(x-pos.i) * double(x-pos.i)) +
                 (double(y-pos.j) * double(y-pos.j)));

          const double mapval = smap.getVal(x,y);

          result += mapval * dist;
          mapsum += mapval;
        }

    return (result / mapsum);
  }
}

void SwpeScorer::accum(const Image<float>& eyeposmap, int pos)
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  if (itsDims.isEmpty())
    {
      itsDims = eyeposmap.getDims();
    }
  else
    {
      if (itsDims != eyeposmap.getDims())
        LFATAL("wrong eyeposmap dims (expected %dx%d, got %dx%d)",
               itsDims.w(), itsDims.h(),
               eyeposmap.getWidth(), eyeposmap.getHeight());
    }

  const Point2D<int> eye(pos % eyeposmap.getWidth(),
                    pos / eyeposmap.getWidth());

  const Point2D<int> randpos(itsGenerator.idraw_range(0, eyeposmap.getWidth()),
                        itsGenerator.idraw_range(0, eyeposmap.getHeight()));

  Image<float> randmap(eyeposmap.getDims(), NO_INIT);
  for (Image<float>::iterator itr = randmap.beginw(), stop = randmap.endw();
       itr != stop; ++itr)
    *itr = itsGenerator.fdraw();

  Image<float> flatmap(eyeposmap.getDims(), NO_INIT);
  flatmap.clear(1.0f);

  itsEyeScore += getSwpe(eyeposmap, eye);
  itsRandEyeScore += getSwpe(eyeposmap, randpos);
  itsRandMapScore += getSwpe(randmap, eye);
  itsFlatMapScore += getSwpe(flatmap, eye);

  ++itsObservedCount;
}

std::string SwpeScorer::getScoreString(const std::string& name)
{
  if (itsObservedCount > 0)
    return sformat("[%s] swpe (dims %dx%d) "
                   "eye@smap: %10.5f "
                   "eye@randmap: %10.5f "
                   "eye@flatmap: %10.5f "
                   "rand@smap: %10.5f ",
                   name.c_str(),
                   itsDims.w(), itsDims.h(),
                   itsEyeScore / itsObservedCount,
                   itsRandMapScore / itsObservedCount,
                   itsFlatMapScore / itsObservedCount,
                   itsRandEyeScore / itsObservedCount);

  // else...
  return sformat("[%s] swpe no data", name.c_str());
}

PercentileScorer::PercentileScorer()
  :
  currentPrctile(0.0),
  observedPrctile(0.0),
  observedCount(0)
{}

void PercentileScorer::accum(const Image<float>& eyeposmap, int pos)
{
  GVX_TRACE(__PRETTY_FUNCTION__);
  const float observed = eyeposmap[pos];

  int nless = 0, nequal = 0;
  for (int i = 0; i < eyeposmap.getSize(); ++i)
    {
      if (eyeposmap[i] < observed) ++nless;
      else if (eyeposmap[i] == observed) ++nequal;
    }

  ASSERT(eyeposmap.getSize() > 0);

  currentPrctile =
    (nless + nequal/2.0)/double(eyeposmap.getSize());

  observedPrctile += currentPrctile;

  ++observedCount;
}

std::string PercentileScorer::getScoreString(const std::string& name)
{
  if (observedCount > 0)
    return sformat("[%s] current prctile: %10.5f overall prctile: %10.5f",
                   name.c_str(), currentPrctile,
                   observedPrctile / observedCount);
  // else...
  return sformat("[%s] no prctile observations", name.c_str());
}

MulticastScorer::MulticastScorer()
  :
  observedCount(0),
  itsPrctileScorer(),
  itsNssScorer(),
  itsKLScorer(10, 100),
  itsSwpeScorer()
{}

void MulticastScorer::score(const std::string& name,
                            const Image<float>& eyeposmap, int pos)
{
  ++observedCount;

  itsPrctileScorer.accum(eyeposmap, pos);
  itsNssScorer.accum(eyeposmap, pos);
  itsKLScorer.accum(eyeposmap, pos);
  itsSwpeScorer.accum(eyeposmap, pos);

  if (observedCount % 100 == 1)
    this->showScore(name);
}

void MulticastScorer::showScore(const std::string& name)
{
  if (observedCount > 0)
    {
      LINFO("%s", itsPrctileScorer.getScoreString(name).c_str());
      LINFO("%s", itsNssScorer.getScoreString(name).c_str());
      LINFO("%s", itsKLScorer.getScoreString(name).c_str());
      LINFO("%s", itsSwpeScorer.getScoreString(name).c_str());
    }
}

void MulticastScorer::writeScore(const std::string& name,
                                 std::ostream& os)
{
  if (observedCount > 0)
    {
      os << itsPrctileScorer.getScoreString(name) << std::endl;
      os << itsNssScorer.getScoreString(name) << std::endl;
      os << itsKLScorer.getScoreString(name) << std::endl;
      os << itsSwpeScorer.getScoreString(name) << std::endl;
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_SCORER_C_DEFINED
