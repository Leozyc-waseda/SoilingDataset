/*!@file TIGS/BackpropLearner.C Learn feature/position pairings with a backprop-training neural network */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/BackpropLearner.C $
// $Id: BackpropLearner.C 5883 2005-11-07 18:55:58Z rjpeters $
//

#ifndef TIGS_BACKPROPLEARNER_C_DEFINED
#define TIGS_BACKPROPLEARNER_C_DEFINED

#include "TIGS/BackpropLearner.H"

#include "Image/MathOps.H"
#include "Image/MatrixOps.H" // for transpose()
#include "Learn/BackpropNetwork.H"
#include "TIGS/LeastSquaresLearner.H"
#include "TIGS/TrainingSet.H"
#include "rutz/trace.h"

BackpropLearner::BackpropLearner(OptionManager& mgr)
  :
  TopdownLearner(mgr, "BackpropLearner", "BackpropLearner"),
  itsLsq(new LeastSquaresLearner(mgr)),
  itsNetwork(0),
  itsInRange(0.0f, 1.0f),
  itsOutRange(0.0f, 1.0f)
{
  itsLsq->dontSave();
}

BackpropLearner::~BackpropLearner()
{
  delete itsNetwork;
}

Image<float> BackpropLearner::getBiasMap(const TrainingSet& tdata,
                                         const Image<float>& features) const
{
  GVX_TRACE(__PRETTY_FUNCTION__);

  if (itsNetwork == 0)
    {
      itsNetwork = new BackpropNetwork;

      const int nhidden = 100;
      const float eta = 0.5f;
      const float alph = 0.5f;
      const int iters = 3000;

      Image<float> XX = itsLsq->getBiasMap(tdata,
                                           tdata.getFeatures());

      double preE = RMSerr(XX, tdata.getPositions());
      double preC = corrcoef(XX, tdata.getPositions());
      LINFO("preE=%f, preC=%f", preE, preC);

      Image<float> X = transpose(XX);
      Image<float> D = transpose(tdata.getPositions());

      itsInRange = rangeOf(X);
      itsOutRange = rangeOf(D);

      X = remapRange(X, itsInRange, Range<float>(0.0f, 1.0f));
      D = remapRange(D, itsOutRange, Range<float>(0.0f, 1.0f));

      double E, C;
      itsNetwork->train(X, D, nhidden, eta, alph, iters, &E, &C);

      LINFO("E=%f, C=%f", E, C);
    }

  Image<float> ff = transpose(itsLsq->getBiasMap(tdata, features));
  ff = remapRange(ff, itsInRange, Range<float>(0.0f, 1.0f));
  Image<float> bb = transpose(itsNetwork->compute(ff));
  bb = remapRange(bb, Range<float>(0.0f, 1.0f), itsOutRange);
  return bb;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_BACKPROPLEARNER_C_DEFINED
