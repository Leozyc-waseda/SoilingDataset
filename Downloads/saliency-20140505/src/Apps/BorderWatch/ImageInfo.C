/*!@file ImageInfo.C A class to maintain chip state */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Apps/BorderWatch/ImageInfo.C $
// $Id: ImageInfo.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Apps/BorderWatch/ImageInfo.H"
#include "Image/DrawOps.H"
#include "GUI/DebugWin.H"
#include "Util/StringUtil.H"
#include "Channels/ChannelMaps.H"
#include "Channels/ChannelOpts.H"
#include "Neuro/NeuroSimEvents.H"
#include "Simulation/SimEventQueue.H"
#include "Transport/FrameInfo.H"
#include <cmath>

const ModelOptionCateg MOC_ImageInfo = {
MOC_SORTPRI_3, "Image Info-related options" };

const ModelOptionDef OPT_RandGain =
{ MODOPT_ARG(float), "RandGain", &MOC_ImageInfo, OPTEXP_CORE,
  "The multiplier for the rand when computing the chip score",
  "k-rand", '\0', "<float>", "0.0" };

const ModelOptionDef OPT_EntropyGain =
{ MODOPT_ARG(float), "EntropyGain", &MOC_ImageInfo, OPTEXP_CORE,
  "The multiplier for the entropy when computing the chip score",
  "k-entropy", '\0', "<float>", "0.0" };

const ModelOptionDef OPT_SaliencyGain =
{ MODOPT_ARG(float), "SaliencyGain", &MOC_ImageInfo, OPTEXP_CORE,
  "The multiplier for the saliency when computing the chip score",
  "k-saliency", '\0', "<float>", "0.0" };

const ModelOptionDef OPT_UniquenessGain =
{ MODOPT_ARG(float), "UniquenessGain", &MOC_ImageInfo, OPTEXP_CORE,
  "The multiplier for the uniqueness when computing the chip score",
  "k-uniqueness", '\0', "<float>", "0.0" };

const ModelOptionDef OPT_MSDSurpriseGain =
{ MODOPT_ARG(float), "MSDSurpriseGain", &MOC_ImageInfo, OPTEXP_CORE,
  "The multiplier for the msd surprise when computing the chip score",
  "k-msdsurprise", '\0', "<float>", "0.0" };

const ModelOptionDef OPT_KLSurpriseGain =
{ MODOPT_ARG(float), "KLSurpriseGain", &MOC_ImageInfo, OPTEXP_CORE,
  "The multiplier for the kl surprise when computing the chip score",
  "k-klsurprise", '\0', "<float>", "1.0" };


// ######################################################################
ImageInfo::ImageInfo(OptionManager& mgr, const std::string& descrName, const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  itsVcc(new VisualCortexConfigurator(mgr)),
  itsRandGain(&OPT_RandGain, this, 0),
  itsEntropyGain(&OPT_EntropyGain, this, 0),
  itsSaliencyGain(&OPT_SaliencyGain, this, 0),
  itsUniquenessGain(&OPT_UniquenessGain, this, 0),
  itsMSDSurpriseGain(&OPT_MSDSurpriseGain, this, 0),
  itsKLSurpriseGain(&OPT_KLSurpriseGain, this, 0)
{
  addSubComponent(itsVcc);
}

// ######################################################################
ImageInfo::~ImageInfo()
{ }

// ######################################################################
ImageInfo::ImageStats ImageInfo::update(nub::ref<SimEventQueue>& q,
                                        const Image<PixRGB<byte> >& img, const int frameID)
{
  ImageStats stats;

  // Post the image to the queue:
  q->post(rutz::make_shared(new SimEventRetinaImage(this, InputFrame(InputFrame::fromRgb(&img, q->now())),
                                                    Rectangle(Point2D<int>(0,0), img.getDims()),
                                                    Point2D<int>(0,0))));
  // Get the visual cortex output:
  if (SeC<SimEventVisualCortexOutput> e = q->check<SimEventVisualCortexOutput>(this, SEQ_ANY))
    stats.smap = e->vco(1.0F);
  else LFATAL("Can not get the Visual cortex output");

  // Find the most salient point in the saliency map
  findMax(stats.smap, stats.salpoint, stats.saliency);

  // Compute the 'energy' (sum of saliency) of the saliency map
  if (itsUniquenessGain.getVal() != 0.0F || itsRandGain.getVal() != 0.0F || itsEntropyGain.getVal() != 0.0F)
    stats.energy = sum(stats.smap); // needed by uniqueness, rand, and entropy
  else stats.energy = 0.0F;

  // Find the uniqueness of the salient point: Which is the difference
  //    between the most salient value, and the average of the rest of the values:
  const uint sz = stats.smap.getSize();
  if (itsUniquenessGain.getVal() != 0.0F)
    stats.uniqueness = stats.saliency - ( (stats.energy - stats.saliency) / (sz - 1) );
  else stats.uniqueness = 0.0F;

  // Compute the entropy and the 'rand' (an unnormalized version of the entropy which somehow
  //    seems to give much better ROC results)
  if (itsRandGain.getVal() != 0.0F || itsEntropyGain.getVal() != 0.0F) {
    stats.entropy = 0.0F; stats.rand = 0.0F;
    for (uint i = 0; i < sz; ++i)
      if (stats.smap[i] > 0.0F)
        {
          stats.entropy += (stats.smap[i] / stats.energy) * logf(stats.smap[i] / stats.energy);
          stats.rand += stats.smap[i] * logf(stats.smap[i]) / sz;
        }
  } else { stats.entropy = 0.0F; stats.rand = 0.0F; }

  // Compute the surprise using Lior's Kalman Filter method
  if (itsKLSurpriseGain.getVal() != 0.0F) {
    //const float smean = float(mean(smap)) + 1.0e-20F;
    (void)integrateData(stats.smap,
                        5, //smean * 0.001F,
                        0.1, //smean * 0.01F,
                        itsBelief1Mu, itsBelief1Sig);
    //const float smean2 = float(mean(itsBelief1Mu)) + 1.0e-20F;
    stats.KLsurprise = integrateData(itsBelief1Mu,
                                     5, //smean2 * 0.001F,
                                     0.01, //smean2 * 0.01F,
                                     itsBelief2Mu, itsBelief2Sig);
    stats.belief1 = itsBelief1Mu;
    stats.belief2 = itsBelief2Mu;
  } else stats.KLsurprise = 0.0F; // belief1 and belief2 have a default constructor and are ok

  // Compute the surprise using Bruce's Mean Square Difference method
  if (itsMSDSurpriseGain.getVal() != 0.0F) stats.MSDsurprise = updateMSDiff(img, stats.smap);
  else stats.MSDsurprise = 0.0F;

  // compute the final score:
  stats.score = itsRandGain.getVal() * stats.rand + itsEntropyGain.getVal() * stats.entropy +
    itsSaliencyGain.getVal() * stats.saliency + itsUniquenessGain.getVal() * stats.uniqueness +
    itsMSDSurpriseGain.getVal() * stats.MSDsurprise + itsKLSurpriseGain.getVal() * stats.KLsurprise;

  return stats;
}

// ######################################################################
// Update surprise value based on a simple mean squared difference of
// the current and previous saliency maps.  Both saliency maps are
// normalized to have unity integrals.
float ImageInfo::updateMSDiff(const Image<PixRGB<byte> >& img, Image<float> salMap)
{
  // Just an excess of caution -- this condition should never be true.
  if (!salMap.initialized()) { LINFO("Current smap not set"); return 0.0; }

  // Always exit here on the first frame, since there is no previous map.
  if (!itsPrevSmap.initialized()) { itsPrevSmap = salMap; return 0.0; }

  // Compute the normalizing constant for both maps.
  const float sCurr = sum(salMap);
  const float sPrev = sum(itsPrevSmap);

  // Compute the sum of the squared differences of each map.
  float diffSum = 0.0F;
  for (int i = 0; i < salMap.getSize(); ++i)
    {
      const float cVal = salMap.getVal(i);
      const float pVal = itsPrevSmap.getVal(i);
      const float diff = cVal / sCurr - pVal / sPrev;
      diffSum += diff * diff;
    }

  // Surprise is sum squared normalized by pixel count
  float surprise = diffSum / float(salMap.getSize());

  // Use logistic fit to map surprise to (0,1) interval
  const float m = 2.8900e-05F, s = 9.6030e-06F;
  surprise = 1.0F / (1.0F + expf(-(surprise - m) / s));

  // Update prior map
  itsPrevSmap = salMap;

  return surprise;
}

// ######################################################################
float ImageInfo::integrateData(const Image<float> &data, const float R, const float Q,
                              Image<float>& bel_mu, Image<float>& bel_sig)
{
  if (!bel_mu.initialized()) { bel_mu = data; bel_sig.resize(bel_mu.getDims(), true); }

  Image<float>::const_iterator inPtr = data.begin(), inStop = data.end();
  Image<float>::iterator muPtr = bel_mu.beginw(), sigPtr = bel_sig.beginw();
  float surprise = 0.0F;

  // Kalman filtering for each pixel:
  while (inPtr != inStop)
    {
      // Predict
      const float mu_hat = *muPtr;
      const float sig_hat = *sigPtr + Q;

      // update
      const float K = sig_hat / (sig_hat + R);
      *muPtr = mu_hat + K * (*inPtr - mu_hat);
      *sigPtr = (1.0F - K) * sig_hat;

      // Calculate surprise KL(P(M|D),P(M))
      // P(M|D) = N(*muPtr, *sigPtr);
      // P(M) = N(mu_hat, sig_hat);
      //float localSurprise = (((*muPtr - mu_hat)*(*muPtr - mu_hat)) + (*sigPtr * *sigPtr) + (sig_hat * sig_hat));
      //localSurprise = localSurprise / (2.0F * sig_hat * sig_hat);
      //localSurprise += log(sig_hat / *sigPtr);

      float localSurprise = fabs(*muPtr - mu_hat);
      surprise += localSurprise;

      ++inPtr; ++muPtr; ++sigPtr;
    }

  return surprise;
}

