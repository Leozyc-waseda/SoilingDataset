/*!@file TIGS/MovingAvgLearner.C Transform another learner with a moving average */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/MovingAvgLearner.C $
// $Id: MovingAvgLearner.C 5969 2005-11-21 23:17:54Z rjpeters $
//

#ifndef TIGS_MOVINGAVGLEARNER_C_DEFINED
#define TIGS_MOVINGAVGLEARNER_C_DEFINED

#include "TIGS/MovingAvgLearner.H"

#include "Component/ModelOptionDef.H"
#include "TIGS/TrainingSet.H"
#include "TIGS/TigsOpts.H"
#include "rutz/trace.h"

// Used by: MovingAvgLearner
static const ModelOptionDef OPT_MovingAvgFactor =
  { MODOPT_ARG(float), "MovingAvgFactor", &MOC_TIGS, OPTEXP_CORE,
    "Compute a moving average with 'image = factor*old + (1-factor)*new'",
    "moving-avg-factor", '\0', "<float>", "0.9" };

MovingAvgLearner::MovingAvgLearner(OptionManager& mgr,
                                   nub::ref<TopdownLearner> child)
  :
  TopdownLearner(mgr, "movingavg", "movingavg"),
  itsFactor(&OPT_MovingAvgFactor, this),
  itsChild(child), itsAvgMap()
{
  this->addSubComponent(itsChild);
}

MovingAvgLearner::~MovingAvgLearner() {}

Image<float> MovingAvgLearner::getBiasMap(const TrainingSet& tdata,
                                          const Image<float>& features) const
{
  GVX_TRACE(__PRETTY_FUNCTION__);
  const Image<float> childmap = itsChild->getBiasMap(tdata, features);

  if (!itsAvgMap.initialized())
    {
      itsAvgMap = Image<float>(childmap.getDims(), ZEROS);
    }
  else
    {
      itsAvgMap *= itsFactor.getVal();
    }

  itsAvgMap += childmap * (1.0-itsFactor.getVal());

  return itsAvgMap;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_MOVINGAVGLEARNER_C_DEFINED
