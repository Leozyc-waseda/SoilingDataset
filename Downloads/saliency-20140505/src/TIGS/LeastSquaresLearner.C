/*!@file TIGS/LeastSquaresLearner.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/LeastSquaresLearner.C $
// $Id: LeastSquaresLearner.C 6191 2006-02-01 23:56:12Z rjpeters $
//

#ifndef TIGS_LEASTSQUARESLEARNER_C_DEFINED
#define TIGS_LEASTSQUARESLEARNER_C_DEFINED

#include "TIGS/LeastSquaresLearner.H"

#include "Component/ModelOptionDef.H"
#include "GUI/XWinManaged.H"
#include "Image/LinearAlgebra.H"
#include "Image/MatrixOps.H"
#include "Image/MathOps.H"
#include "Image/Range.H"
#include "Raster/Raster.H"
#include "TIGS/TigsOpts.H"
#include "TIGS/TrainingSet.H"
#include "Util/CpuTimer.H"
#include "Util/log.H"
#include "rutz/trace.h"

// Used by: LeastSquaresLearner
static const ModelOptionDef OPT_LsqSvdThresholdFactor =
  { MODOPT_ARG(float), "LsqSvdThresholdFactor", &MOC_TIGS, OPTEXP_CORE,
    "Multiple of the largest eigenvalue below which eigenvectors "
    "with small eigenvalues will be thrown out",
    "lsq-svd-thresh", '\0', "<float>", "1.0e-8f" };

// Used by: LeastSquaresLearner
static const ModelOptionDef OPT_LsqUseWeightsFile =
  { MODOPT_FLAG, "LsqUseWeightsFile", &MOC_TIGS, OPTEXP_CORE,
    "Whether to write/read least-squares weights file(s)",
    "lsq-use-weights-files", '\0', "", "false" };

namespace
{
  void inspect(const Image<float>& img, const char* name)
  {
    float m = mean(img);
    Range<float> r = rangeOf(img);
    LINFO("%s: (w,h)=(%d,%d), range=[%f..%f], mean=%f",
          name, img.getWidth(), img.getHeight(), r.min(), r.max(), m);
  }
}

LeastSquaresLearner::LeastSquaresLearner(OptionManager& mgr)
  :
  TopdownLearner(mgr, "LeastSquaresLearner", "LeastSquaresLearner"),
  itsSvdThresh(&OPT_LsqSvdThresholdFactor, this),
  itsXptSavePrefix(&OPT_XptSavePrefix, this),
  itsUseWeightsFile(&OPT_LsqUseWeightsFile, this),
  itsWeights() // don't initialize until we're done training
{}

void LeastSquaresLearner::dontSave()
{
  itsUseWeightsFile.setVal(false);
}

Image<float> LeastSquaresLearner::getBiasMap(const TrainingSet& tdata,
                                             const Image<float>& features) const
{
  GVX_TRACE(__PRETTY_FUNCTION__);
  if (!itsWeights.initialized())
    {
      const Image<float> rawTrainFeatures = tdata.getFeatures();
      inspect(rawTrainFeatures, "rawTrainFeatures");

      itsMeanFeatures = meanRow(tdata.getFeatures());
      inspect(itsMeanFeatures, "itsMeanFeatures");

      Image<float> trainFeatures =
        subtractRow(rawTrainFeatures, itsMeanFeatures);
      inspect(trainFeatures, "trainFeatures");

      itsStdevFeatures = stdevRow(trainFeatures);
      inspect(itsStdevFeatures, "itsStdevFeatures");

      trainFeatures = divideRow(trainFeatures, itsStdevFeatures);
      inspect(trainFeatures, "trainFeatures");

      const std::string name =
        itsXptSavePrefix.getVal() + "-" + tdata.fxType() + "-lsq";

      const std::string weightsfile = name + "-weights.pfm";

      if (itsUseWeightsFile.getVal() &&
          Raster::fileExists(weightsfile))
        {
          itsWeights = Raster::ReadFloat(weightsfile.c_str(), RASFMT_PFM);

          LINFO("loaded weights (%s) from %s",
                name.c_str(), weightsfile.c_str());
        }
      else
        {

          try {
            CpuTimer t;

            int rank = 0;

            LINFO("svd threshold factor is %e",
                  double(itsSvdThresh.getVal()));

            const Image<float> pinvFeatures =
              svdPseudoInvf(trainFeatures, SVD_LAPACK, &rank,
                            itsSvdThresh.getVal());

            t.mark();
            t.report(sformat("pinvFeatures (%s)", name.c_str()).c_str());

            LINFO("svd rank=%d, fullrank=%d",
                  rank, trainFeatures.getWidth());

            LINFO("trainFeatures size %dx%d, pinvFeatures size %dx%d",
                  trainFeatures.getWidth(), trainFeatures.getHeight(),
                  pinvFeatures.getWidth(), pinvFeatures.getHeight());

            const bool do_precisioncheck = false;

            if (do_precisioncheck) {
              const Image<float> precisioncheck =
                matrixMult(trainFeatures, pinvFeatures);

              t.mark();
              t.report(sformat("precisioncheck (%s)", name.c_str()).c_str());

              const Image<float> diff =
                precisioncheck - eye<float>(pinvFeatures.getWidth());

              t.mark();
              t.report(sformat("diff (%s)", name.c_str()).c_str());

              LINFO("rms error after inversion: %f",
                    RMSerr(precisioncheck, eye<float>(pinvFeatures.getWidth())));
            }

            const Image<float> rawTrainPositions = tdata.getPositions();

            itsMeanPositions = meanRow(rawTrainPositions);

            itsWeights =
              matrixMult(pinvFeatures,
                         subtractRow(rawTrainPositions, itsMeanPositions));

            t.mark();
            t.report(sformat("itsWeights (%s)", name.c_str()).c_str());

            if (itsUseWeightsFile.getVal())
              {
                Raster::WriteFloat(itsWeights, FLOAT_NORM_PRESERVE,
                                   weightsfile.c_str(), RASFMT_PFM);

                LINFO("saved weights (%s) to %s",
                      name.c_str(), weightsfile.c_str());
              }
          }
          catch (SingularMatrixException& e) {
            XWinManaged win(e.mtx, "singular matrix", true);

            int c = 0;
            while (!win.pressedCloseButton() && ++c < 100)
              usleep(10000);

            exit(1);
          }
        }
    }

  ASSERT(itsWeights.getHeight() == features.getWidth());
  ASSERT(itsWeights.getWidth() == tdata.scaledInputDims().sz());

  const Image<float> featureVec =
    divideRow(subtractRow(features, itsMeanFeatures),
              itsStdevFeatures);

  const Image<float> result =
    addRow(matrixMult(featureVec, itsWeights),
           itsMeanPositions);

  ASSERT(result.getWidth() == tdata.scaledInputDims().sz());
  ASSERT(result.getHeight() == features.getHeight());

  return result;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_LEASTSQUARESLEARNER_C_DEFINED
