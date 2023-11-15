/*!@file TIGS/test-TopdownContext.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/test-TopdownContext.C $
// $Id: test-TopdownContext.C 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef APPNEURO_TEST_TOPDOWNCONTEXT_C_UTC20050726230120_DEFINED
#define APPNEURO_TEST_TOPDOWNCONTEXT_C_UTC20050726230120_DEFINED

#include "Component/GlobalOpts.H"
#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Media/FrameSeries.H"
#include "Media/MediaOpts.H"
#include "Psycho/EyeSFile.H"
#include "TIGS/FeatureExtractorFactory.H"
#include "TIGS/Figures.H"
#include "TIGS/SaliencyMapFeatureExtractor.H"
#include "TIGS/Scorer.H"
#include "TIGS/TigsOpts.H"
#include "TIGS/TopdownLearnerFactory.H"
#include "TIGS/TrainingSet.H"
#include "Util/Assert.H"
#include "Util/FileUtil.H"
#include "Util/Pause.H"
#include "Util/SimTime.H"
#include "Util/StringConversions.H"
#include "Util/csignals.H"
#include "Util/fpe.H"
#include "rutz/error_context.h"
#include "rutz/sfmt.h"
#include "rutz/shared_ptr.h"
#include "rutz/trace.h"

#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <vector>

// Used by: Context
static const ModelOptionDef OPT_DoBottomUpContext =
  { MODOPT_FLAG, "DoBottomUpContext", &MOC_TIGS, OPTEXP_CORE,
    "Whether to scale the top-down prediction by a bottom-up map",
    "bottom-up-context", '\0', "", "false" };

// Used by: Context
static const ModelOptionDef OPT_TdcSaveSumo =
  { MODOPT_FLAG, "TdcSaveSumo", &MOC_TIGS, OPTEXP_CORE,
    "Whether to save the sumo display",
    "tdc-save-sumo", '\0', "", "false" };

// Used by: Context
static const ModelOptionDef OPT_TdcSaveSumo2 =
  { MODOPT_FLAG, "TdcSaveSumo2", &MOC_TIGS, OPTEXP_CORE,
    "Whether to save the sumo2 display",
    "tdc-save-sumo2", '\0', "", "false" };

// Used by: Context
static const ModelOptionDef OPT_TdcSaveMaps =
  { MODOPT_FLAG, "TdcSaveMaps", &MOC_TIGS, OPTEXP_CORE,
    "Whether to save the individual topdown context maps",
    "tdc-save-maps", '\0', "", "false" };

// Used by: Context
static const ModelOptionDef OPT_TdcSaveMapsNormalized =
  { MODOPT_FLAG, "TdcSaveMapsNormalized", &MOC_TIGS, OPTEXP_CORE,
    "Whether to rescale maps to [0,255] when saving with --tdc-save-maps",
    "tdc-save-maps-normalized", '\0', "", "true" };

// Used by: Context
static const ModelOptionDef OPT_TdcLocalMax =
  { MODOPT_ARG(float), "TdcLocalMax", &MOC_TIGS, OPTEXP_CORE,
    "Diameter of local max region to be applied to bias maps before scoring",
    "tdc-local-max", '\0', "<float>", "1" };

// Used by: Context
static const ModelOptionDef OPT_TdcTemporalMax =
  { MODOPT_ARG(unsigned int), "TdcTemporalMax", &MOC_TIGS, OPTEXP_CORE,
    "Number of frames across which to apply a temporal max to "
    "bias maps before scoring",
    "tdc-temporal-max", '\0', "<integer>", "1" };

// Used by: Context
static const ModelOptionDef OPT_TdcSaveRawData =
  { MODOPT_FLAG, "TdcSaveRawData", &MOC_TIGS, OPTEXP_CORE,
    "Whether to save a raw binary file containing the "
    "bottom-up and top-down maps",
    "tdc-save-raw-data", '\0', "", "false" };

// Used by: Context
static const ModelOptionDef OPT_TdcRectifyTd =
  { MODOPT_FLAG, "TdcRectifyTd", &MOC_TIGS, OPTEXP_CORE,
    "Whether to rectify the top-down maps",
    "tdc-rectify-td", '\0', "", "false" };

// Used by: TigsJob
static const ModelOptionDef OPT_TopdownContextSpec =
  { MODOPT_ARG_STRING, "TopdownContextSpec", &MOC_TIGS, OPTEXP_CORE,
    "Specification string for a topdown context",
    "context", '\0', "<string>", "" };

// Used by: TigsJob
static const ModelOptionDef OPT_MoviePeriod =
  { MODOPT_ARG(SimTime), "MoviePeriod", &MOC_TIGS, OPTEXP_CORE,
    "Inter-frame period (or rate) of input movie",
    "movie-period", '\0', "<float>{s|ms|us|ns|Hz}", "0.0s" };

// Obsolete
static const ModelOptionDef OPT_MovieHertzObsolete =
  { MODOPT_OBSOLETE, "MovieHertzObsolete", &MOC_TIGS, OPTEXP_CORE,
    "Obsolete; use --movie-period instead with a SimTime value",
    "movie-hertz", '\0', "<float>", "0.0" };

// Used by: TigsJob
static const ModelOptionDef OPT_NumSkipFrames =
  { MODOPT_ARG(int), "NumSkipFrames", &MOC_TIGS, OPTEXP_CORE,
    "Number of frames to skip over at beginning of input movie",
    "num-skip-frames", '\0', "<int>", "0" };

// Used by: TigsJob
static const ModelOptionDef OPT_NumTrainingFrames =
  { MODOPT_ARG(int), "NumTrainingFrames", &MOC_TIGS, OPTEXP_CORE,
    "Number of input movie frames to use as training data",
    "num-training-frames", '\0', "<int>", "0" };

// Used by: TigsJob
static const ModelOptionDef OPT_NumTestingFrames =
  { MODOPT_ARG(int), "NumTestingFrames", &MOC_TIGS, OPTEXP_CORE,
    "Number of input movie frames to use as testing data",
    "num-testing-frames", '\0', "<int>", "0" };

// Used by: TigsJob
static const ModelOptionDef OPT_SaveGhostFrames =
  { MODOPT_ARG_STRING, "SaveGhostFrames", &MOC_TIGS, OPTEXP_CORE,
    "Name of a file in which to save ghost frame info that can "
    "be used to accelerate processing in a subsequent run",
    "save-ghost-frames", '\0', "<filanem>", "" };

// Used by: TigsInputFrameSeries
static const ModelOptionDef OPT_GhostInput =
  { MODOPT_ARG_STRING, "GhostInput", &MOC_TIGS, OPTEXP_CORE,
    "Read ghost frame info from this file",
    "ghost-input", '\0', "<filename>", "" };

namespace
{
  template <class T>
  Image<T> asRow(const Image<T>& in)
  {
  GVX_TRACE(__PRETTY_FUNCTION__);
    return Image<T>(in.getArrayPtr(), in.getSize(), 1);
  }
}

class Context : public ModelComponent
{
public:
  Context(OptionManager& mgr,
          const std::string& fx_type_,
          const std::string& learner_type_)
    :
    ModelComponent(mgr, "Context", "Context"),
    itsXptSavePrefix(&OPT_XptSavePrefix, this),
    itsDoBottomUp(&OPT_DoBottomUpContext, this),
    itsSaveSumo(&OPT_TdcSaveSumo, this),
    itsSaveSumo2(&OPT_TdcSaveSumo2, this),
    itsSaveMaps(&OPT_TdcSaveMaps, this),
    itsSaveMapsNormalized(&OPT_TdcSaveMapsNormalized, this),
    itsLocalMaxSize(&OPT_TdcLocalMax, this),
    itsTemporalMax(&OPT_TdcTemporalMax, this),
    itsSaveRawData(&OPT_TdcSaveRawData, this),
    itsRectifyTd(&OPT_TdcRectifyTd, this),
    itsTdata(new TrainingSet(this->getManager(), fx_type_)),
    itsFxType(fx_type_),
    itsFx(makeFeatureExtractor(mgr, fx_type_)),
    itsLearnerType(learner_type_),
    itsLearner(makeTopdownLearner(mgr, learner_type_)),
    itsCtxName("-" + itsFxType + "-" + itsLearnerType),
    itsScorer(),
    itsBottomUpScorer(),
    itsComboScorer(),
    itsRawDataFile(0)
  {
    if (theirBottomUp.is_invalid())
      {
        GVX_ERR_CONTEXT(rutz::sfmt
                        ("constructing SaliencyMapFeatureExtractor "
                         "on behalf of %s", itsCtxName.c_str()));

        theirBottomUp.reset
          (new SaliencyMapFeatureExtractor(this->getManager()));
        this->addSubComponent(theirBottomUp);
      }

    this->addSubComponent(itsTdata);
    this->addSubComponent(itsFx);
    this->addSubComponent(itsLearner);
  }

  virtual void start2()
  {
    ASSERT(itsRawDataFile == 0);

    if (itsSaveRawData.getVal())
      {
        std::string rawdatfname = this->contextName()+".rawdat";
        itsRawDataFile = fopen(rawdatfname.c_str(), "w");
        if (itsRawDataFile == 0)
          LFATAL("couldn't open %s for writing", rawdatfname.c_str());
      }
  }

  virtual void stop1()
  {
    itsScorer.showScore("finalscore:" + this->contextName());
    if (itsDoBottomUp.getVal())
      {
        itsBottomUpScorer.showScore("finalscore:" + this->contextName()
                                    + "...bu-only");
        itsComboScorer.showScore("finalscore:" + this->contextName()
                                 + "+bu");
      }

    std::ofstream ofs((this->contextName() + ".score").c_str());
    if (ofs.is_open())
      {
        itsScorer.writeScore(this->contextName(), ofs);
        if (itsDoBottomUp.getVal())
          {
            itsBottomUpScorer.writeScore(this->contextName() + "...bu-only", ofs);
            itsComboScorer.writeScore(this->contextName() + "+bu", ofs);
          }
      }
    ofs.close();

    if (itsRawDataFile != 0)
      {
        fclose(itsRawDataFile);
        itsRawDataFile = 0;
      }
  }

  std::string contextName() const
  {
    if (itsXptSavePrefix.getVal() == "")
      LFATAL("no xpt name specified!");

    return itsXptSavePrefix.getVal() + itsCtxName;
  }

  void loadTrainingSet(const std::string& xpt)
  {
    itsTdata->load(xpt + itsCtxName);
  }

  void trainingFrame(const TigsInputFrame& fin,
                     const Point2D<int>& eyepos, bool lastone,
                     OutputFrameSeries& ofs)
  {
    GVX_ERR_CONTEXT(rutz::sfmt("handling training frame in Context %s",
                               this->contextName().c_str()));

    const Image<float> features = itsFx->extract(fin);

    if (fin.origbounds().contains(eyepos))
      {
        const Image<float> biasmap =
          itsTdata->recordSample(eyepos, features);

        if (!ofs.isVoid() && itsSaveSumo.getVal())
          ofs.writeRGB(makeSumoDisplay(fin, biasmap, *itsTdata,
                                       eyepos, features),
                       this->contextName());
      }

    if (lastone)
      itsTdata->save(this->contextName());

    if (itsDoBottomUp.getVal())
      {
        // this is a no-op, except that we want to force the features
        // to be computed so that they can be cached and saved
        (void) theirBottomUp->extract(fin);
      }
  }

  Image<float> localMax(const Image<float>& in) const
  {
    if (itsLocalMaxSize.getVal() < 2.0f)
      return in;

    const double rad = itsLocalMaxSize.getVal() / 2.0;
    const double rad2 = rad*rad;
    const int bound = int(rad+1.0);

    Image<float> result(in.getDims(), ZEROS);

    const int w = in.getWidth();
    const int h = in.getHeight();

    for (int x = 0; x < w; ++x)
      for (int y = 0; y < h; ++y)
        {
          float maxv = in.getVal(x, y);

          for (int i = -bound; i <= bound; ++i)
            for (int j = -bound; j <= bound; ++j)
              {
                if (i*i + j*j <= rad2 && result.coordsOk(x+i,y+j))
                  maxv = std::max(maxv, 0.999f*in.getVal(x+i,y+j));
              }

          result.setVal(x, y, maxv);
        }

    return result;
  }

  Image<float> combineBuTd(const Image<float>& bu,
                           const Image<float>& td) const
  {
    Image<float> rtd = td;
    inplaceRectify(rtd);

    return bu * rtd;
  }

  Image<float> temporalMax(const Image<float>& img,
                           std::deque<Image<float> >& q) const
  {
    q.push_front(img);

    ASSERT(itsTemporalMax.getVal() > 0);

    while (q.size() > itsTemporalMax.getVal())
      q.pop_back();
    ASSERT(q.size() <= itsTemporalMax.getVal());

    Image<float> result = q[0];

    for (uint i = 1; i < q.size(); ++i)
      result = takeMax(result, q[i]);

    return result;
  }

  void testFrame(const TigsInputFrame& fin,
                 const Point2D<int>& eyepos,
                 OutputFrameSeries& ofs)
  {
    GVX_ERR_CONTEXT(rutz::sfmt("handling training frame in Context %s",
                               this->contextName().c_str()));

    LINFO("context %s", this->contextName().c_str());

    if (!fin.origbounds().contains(eyepos))
      return;

    const Image<float> features = itsFx->extract(fin);

    const Image<float> rawtdmap =
      reshape(itsLearner->getBiasMap(*itsTdata, asRow(features)),
              itsTdata->scaledInputDims());

    Image<float> tdmap =
      this->temporalMax(this->localMax(rawtdmap), itsTdQ);

    if (itsRectifyTd.getVal())
      inplaceRectify(tdmap);

    const int pos = itsTdata->p2p(eyepos);
    itsScorer.score(this->contextName(), tdmap, pos);

    if (itsDoBottomUp.getVal())
      {
        const Image<float> rawbumap =
          rescale(reshape(theirBottomUp->extract(fin),
                          Dims(512 >> 4, 512 >> 4)),
                  tdmap.getDims());

        const Image<float> bumap =
          this->temporalMax(this->localMax(rawbumap), itsBuQ);

        itsBottomUpScorer.score(this->contextName() + "...bu-only",
                                bumap, pos);

        const Image<float> rawcombomap =
          this->combineBuTd(rawbumap, rawtdmap);

        const Image<float> combomap =
          this->temporalMax(this->localMax(rawcombomap), itsComboQ);

        itsComboScorer.score(this->contextName() + "+bu",
                             combomap, pos);

        if (!ofs.isVoid() && itsSaveMaps.getVal())
          {
            const int flags =
              itsSaveMapsNormalized.getVal()
              ? (FLOAT_NORM_0_255 | FLOAT_NORM_WITH_SCALE)
              : FLOAT_NORM_PRESERVE;

            makeDirectory(this->contextName() + "-maps");

            ofs.writeFloat(tdmap, flags,
                           this->contextName() + "-maps/td");

            ofs.writeFloat(bumap, flags,
                           this->contextName() + "-maps/bu");

            ofs.writeFloat(combomap, flags,
                           this->contextName() + "-maps/combo");
          }

        if (!ofs.isVoid() && itsSaveSumo2.getVal())
          {
            ofs.writeRGB(makeSumoDisplay2(fin,
                                          tdmap,
                                          bumap,
                                          combomap,
                                          *itsTdata,
                                          eyepos),
                         this->contextName() + "-sumo2");
          }

        if (itsSaveRawData.getVal())
          {
            ASSERT(itsRawDataFile != 0);

            ASSERT(bumap.getSize() == tdmap.getSize());

            const float sz = float(bumap.getSize());
            fwrite(&sz, sizeof(float), 1, itsRawDataFile);

            const float fpos = float(pos);
            fwrite(&fpos, sizeof(float), 1, itsRawDataFile);

            const float buval = bumap[pos];
            fwrite(&buval, sizeof(float), 1, itsRawDataFile);

            const float tdval = tdmap[pos];
            fwrite(&tdval, sizeof(float), 1, itsRawDataFile);

            const float bumean = mean(bumap);
            fwrite(&bumean, sizeof(float), 1, itsRawDataFile);

            const float tdmean = mean(tdmap);
            fwrite(&tdmean, sizeof(float), 1, itsRawDataFile);

            const float bustd = stdev(bumap);
            fwrite(&bustd, sizeof(float), 1, itsRawDataFile);

            const float tdstd = stdev(tdmap);
            fwrite(&tdstd, sizeof(float), 1, itsRawDataFile);

            const float buz = (buval - bumean) / bustd;
            fwrite(&buz, sizeof(float), 1, itsRawDataFile);

            const float tdz = (tdval - tdmean) / tdstd;
            fwrite(&tdz, sizeof(float), 1, itsRawDataFile);

            fwrite(bumap.getArrayPtr(), sizeof(float),
                   bumap.getSize(), itsRawDataFile);

            fwrite(tdmap.getArrayPtr(), sizeof(float),
                   tdmap.getSize(), itsRawDataFile);

            fflush(itsRawDataFile);
          }
      }

    if (!ofs.isVoid() && itsSaveSumo.getVal())
      ofs.writeRGB(makeSumoDisplay(fin, tdmap, *itsTdata,
                                   eyepos, features),
                   this->contextName());
  }

private:
  OModelParam<std::string> itsXptSavePrefix;
  OModelParam<bool> itsDoBottomUp;
  OModelParam<bool> itsSaveSumo;
  OModelParam<bool> itsSaveSumo2;
  OModelParam<bool> itsSaveMaps;
  OModelParam<bool> itsSaveMapsNormalized;
  OModelParam<float> itsLocalMaxSize;
  OModelParam<uint> itsTemporalMax;
  OModelParam<bool> itsSaveRawData;
  OModelParam<bool> itsRectifyTd;

  const nub::ref<TrainingSet> itsTdata;
  const std::string itsFxType;
  const nub::ref<FeatureExtractor> itsFx;
  const std::string itsLearnerType;
  const nub::ref<TopdownLearner> itsLearner;
  static nub::soft_ref<FeatureExtractor> theirBottomUp;
  std::string itsCtxName;
  MulticastScorer itsScorer;
  MulticastScorer itsBottomUpScorer;
  MulticastScorer itsComboScorer;
  std::deque<Image<float> > itsBuQ;
  std::deque<Image<float> > itsTdQ;
  std::deque<Image<float> > itsComboQ;
  FILE* itsRawDataFile;
};

nub::soft_ref<FeatureExtractor> Context::theirBottomUp;

class TigsJob : public ModelComponent
{
public:
  TigsJob(OptionManager& mgr)
    :
    ModelComponent(mgr, "TigsJob", "TigsJob"),
    itsXptSavePrefix(&OPT_XptSavePrefix, this),
    itsContextSpec(&OPT_TopdownContextSpec, this),
    itsMoviePeriod(&OPT_MoviePeriod, this),
    itsNumSkipFrames(&OPT_NumSkipFrames, this),
    itsNumTrainingFrames(&OPT_NumTrainingFrames, this),
    itsNumTestingFrames(&OPT_NumTestingFrames, this),
    itsSaveGhostFrames(&OPT_SaveGhostFrames, this),
    itsObsolete1(&OPT_MovieHertzObsolete, this)
  {}

  virtual void start2()
  {
    rutz::prof::prof_summary_file_name
      ((itsXptSavePrefix.getVal() + "-prof.out").c_str());

    if (itsSaveGhostFrames.getVal().length() > 0)
      {
        itsGhostOutput =
          rutz::shared_ptr<std::ofstream>
          (new std::ofstream(itsSaveGhostFrames.getVal().c_str()));

        if (!itsGhostOutput->is_open())
          LFATAL("couldn't open '%s' for writing",
                 itsSaveGhostFrames.getVal().c_str());
      }
  }

  virtual void paramChanged(ModelParamBase* const param,
                            const bool valueChanged,
                            ParamClient::ChangeStatus* status)
  {
    ModelComponent::paramChanged(param, valueChanged, status);

    if (param == &itsContextSpec)
      {
        if (itsContextSpec.getVal() != "")
          this->addContext(itsContextSpec.getVal());
      }
  }

  void addContext(const std::string& spec)
  {
    GVX_ERR_CONTEXT(rutz::sfmt
                    ("adding context for spec %s", spec.c_str()));

    std::string::size_type comma = spec.find_first_of(',');
    if (comma == spec.npos)
      LFATAL("missing comma in context spec '%s'", spec.c_str());
    if (comma+1 >= spec.length())
      LFATAL("bogus context spec '%s'", spec.c_str());

    std::string fx_type = spec.substr(0, comma);
    std::string learner_type = spec.substr(comma+1);

    LINFO("fxtype=%s, learnertype=%s",
          fx_type.c_str(), learner_type.c_str());

    nub::ref<Context> ctx(new Context(this->getManager(),
                                      fx_type, learner_type));

    this->addSubComponent(ctx);
    itsContexts.push_back(ctx);

    ctx->exportOptions(MC_RECURSE);
  }

  void loadTrainingSet(const std::string& xpt)
  {
    ASSERT(itsNumTrainingFrames.getVal() == 0);

    for (size_t c = 0; c < itsContexts.size(); ++c)
      itsContexts[c]->loadTrainingSet(xpt);
  }

  // returns true to continue looping, false to quit main loop
  bool handleFrame(int nframe,
                   const TigsInputFrame& fin,
                   const Point2D<int>& eyepos,
                   const bool islast,
                   OutputFrameSeries& ofs)
  {
    GVX_ERR_CONTEXT(rutz::sfmt("handling input frame %d in TigsJob",
                               nframe));

    if (nframe < itsNumSkipFrames.getVal())
      return true;

    if (nframe >= (itsNumSkipFrames.getVal()
                   +itsNumTrainingFrames.getVal()
                   +itsNumTestingFrames.getVal()))
      {
        LINFO("exceeded skip+train+test frames");
        return false;
      }

    if (itsGhostOutput.is_valid())
      {
        *itsGhostOutput << fin.toGhostString() << std::endl;
      }

    for (size_t c = 0; c < itsContexts.size(); ++c)
      {
        if (nframe < itsNumSkipFrames.getVal()+itsNumTrainingFrames.getVal())
          {
            const bool lasttraining =
              islast
              ||
              (nframe+1 == (itsNumSkipFrames.getVal()
                            +itsNumTrainingFrames.getVal()));

            itsContexts[c]->trainingFrame(fin, eyepos,
                                          lasttraining, ofs);
          }
        else if (nframe < itsNumSkipFrames.getVal()+itsNumTrainingFrames.getVal()+itsNumTestingFrames.getVal())
          {
            itsContexts[c]->testFrame(fin, eyepos, ofs);
          }
      }

    return true;
  }

  SimTime movieFrameLength() const
  {
    ASSERT(itsMoviePeriod.getVal() > SimTime::ZERO());
    return itsMoviePeriod.getVal();
  }

  std::string getSavePrefix() const { return itsXptSavePrefix.getVal(); }

private:
  OModelParam<std::string> itsXptSavePrefix;
  OModelParam<std::string> itsContextSpec;
  OModelParam<SimTime> itsMoviePeriod;
  OModelParam<int> itsNumSkipFrames;
  OModelParam<int> itsNumTrainingFrames;
  OModelParam<int> itsNumTestingFrames;
  OModelParam<std::string> itsSaveGhostFrames;
  OModelParam<bool> itsObsolete1;
  std::vector<nub::ref<Context> > itsContexts;
  rutz::shared_ptr<std::ofstream> itsGhostOutput;
};

class TigsInputFrameSeries : public ModelComponent
{
public:
  TigsInputFrameSeries(OptionManager& mgr)
    :
    ModelComponent(mgr, "TigsInputFrameSeries", "TigsInputFrameSeries"),
    itsGhostInput(&OPT_GhostInput, this),
    itsIfs(new InputFrameSeries(mgr)),
    itsFirst(true)
  {
    this->addSubComponent(itsIfs);
  }

  virtual void paramChanged(ModelParamBase* param,
                            const bool valueChanged,
                            ChangeStatus* status)
  {
    if (param == &itsGhostInput && valueChanged)
      {
        if (itsGhostInput.getVal().length() == 0)
          {
            ASSERT(itsIfs.is_valid() == false);

            // close our ghost file;
            itsGhostFile.close();

            // make a new regular InputFrameSeries:
            itsIfs.reset(new InputFrameSeries(getManager()));
            this->addSubComponent(itsIfs);
            itsIfs->exportOptions(MC_RECURSE);
            itsFirst = true;
          }
        else
          {
            ASSERT(itsIfs.is_valid() == true);
            this->removeSubComponent(*itsIfs);
            itsIfs.reset(0);

            itsGhostFile.open(itsGhostInput.getVal().c_str());
            if (!itsGhostFile.is_open())
              LFATAL("couldn't open file '%s' for reading",
                     itsGhostInput.getVal().c_str());

            LINFO("reading ghost frames from '%s'",
                  itsGhostInput.getVal().c_str());

            // ok, let's read the first line from the file so that we
            // can set the input dims:
            if (!std::getline(itsGhostFile, itsNextLine))
              itsNextLine = "";
            itsFirst = false;

            rutz::shared_ptr<TigsInputFrame> f =
              TigsInputFrame::fromGhostString(itsNextLine);

            getManager().setOptionValString
              (&OPT_InputFrameDims,
               convertToString(f->origbounds().dims()));
          }

        // OK, now one way or the other we should have either a valid
        // InputFrameSeries or an open std::ifstream, but not both:
        ASSERT(itsIfs.is_valid() != itsGhostFile.is_open());
      }
  }

  rutz::shared_ptr<TigsInputFrame> getFrame(const SimTime stime,
                                            bool* islast)
  {
    if (itsIfs.is_valid())
      {
        if (itsFirst)
          {
            itsIfs->updateNext();
            itsNextFrame = itsIfs->readRGB();
            itsFirst = false;
          }

        if (!itsNextFrame.initialized())
          return rutz::shared_ptr<TigsInputFrame>();

        // get a new frame and swap it with itsNextFrame
        itsIfs->updateNext();
        Image<PixRGB<byte> > frame = itsIfs->readRGB();
        frame.swap(itsNextFrame);

        ASSERT(frame.initialized());

        *islast = (itsNextFrame.initialized() == false);

        return rutz::shared_ptr<TigsInputFrame>
          (new TigsInputFrame(frame, stime));
      }
    else
      {
        ASSERT(itsGhostFile.is_open());

        if (itsNextLine.length() == 0)
          return rutz::shared_ptr<TigsInputFrame>();

        // get a new line and swap it with itsNextLine
        std::string line;
        if (!std::getline(itsGhostFile, line))
          line = "";
        line.swap(itsNextLine);

        ASSERT(line.length() > 0);

        *islast = (itsNextLine.length() == 0);

        rutz::shared_ptr<TigsInputFrame> result =
          TigsInputFrame::fromGhostString(line);

        if (result->t() != stime)
          LFATAL("wrong time in ghost frame: expected %.2fms "
                 "but got %.2fms", stime.msecs(), result->t().msecs());

        LINFO("got ghost frame: %s", line.c_str());

        return result;
      }
  }

private:
  OModelParam<std::string> itsGhostInput;

  nub::soft_ref<InputFrameSeries> itsIfs;
  std::ifstream itsGhostFile;

  bool itsFirst;
  Image<PixRGB<byte> > itsNextFrame;
  std::string itsNextLine;
};

int submain(int argc, const char** argv)
{
  GVX_TRACE("test-TopdownContext-main");

  volatile int signum = 0;
  catchsignals(&signum);

  rutz::prof::print_at_exit(true);

  fpExceptionsUnlock();
  fpExceptionsOff();
  fpExceptionsLock();

  ModelManager mgr("topdown context tester");

  nub::ref<TigsInputFrameSeries> ifs(new TigsInputFrameSeries(mgr));
  mgr.addSubComponent(ifs);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(mgr));
  mgr.addSubComponent(ofs);

  nub::ref<TigsJob> job(new TigsJob(mgr));
  mgr.addSubComponent(job);

  nub::ref<EyeSFile> eyeS(new EyeSFile(mgr));
  mgr.addSubComponent(eyeS);

  mgr.exportOptions(MC_RECURSE);

  mgr.setOptionValString(&OPT_UseRandom, "false");

  if (mgr.parseCommandLine(argc, argv,
                           "[load-pfx1 [load-pfx2 [...]]]",
                           0, -1) == false)
    return 1;

  ofs->setModelParamVal("OutputMPEGStreamFrameRate", int(24),
                        MC_RECURSE | MC_IGNORE_MISSING);
  ofs->setModelParamVal("OutputMPEGStreamBitRate", int(2500000),
                        MC_RECURSE | MC_IGNORE_MISSING);

  mgr.start();

  {
    std::ofstream ofs((job->getSavePrefix() + ".model").c_str());
    if (ofs.is_open())
      {
        mgr.printout(ofs);
      }
    ofs.close();
  }

  mgr.printout(std::cout);

  for (uint e = 0; e < mgr.numExtraArgs(); ++e)
    job->loadTrainingSet(mgr.getExtraArg(e));

  PauseWaiter p;

  int nframes = 0;

  while (1)
    {
      GVX_TRACE("frames loop");

      if (signum != 0)
        {
          LINFO("caught signal %s; quitting", signame(signum));
          break;
        }

      if (p.checkPause())
        continue;

      LINFO("trying frame %d", nframes);

      const SimTime stime = job->movieFrameLength() * (nframes+1);

      bool islast;
      rutz::shared_ptr<TigsInputFrame> fin =
        ifs->getFrame(stime, &islast);

      if (fin.get() == 0)
        {
          LINFO("input exhausted; quitting");
          break;
        }

      const Point2D<int> eyepos = eyeS->readUpTo(stime);

      LINFO("simtime %.6fs, movie frame %d, eye sample %d, ratio %f, "
            "eyepos (x=%d, y=%d)",
            stime.secs(), nframes+1, eyeS->lineNumber(),
            double(eyeS->lineNumber())/double(nframes+1),
            eyepos.i, eyepos.j);

      if (!job->handleFrame(nframes, *fin, eyepos, islast, *ofs))
        break;

      ofs->updateNext();

      ++nframes;
    }

  mgr.stop();

  return 0;
}

int main(int argc, const char** argv)
{
  try {
    submain(argc, argv);
  } catch(...) {
    REPORT_CURRENT_EXCEPTION;
    std::terminate();
  }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // !APPNEURO_TEST_TOPDOWNCONTEXT_C_UTC20050726230120DEFINED
