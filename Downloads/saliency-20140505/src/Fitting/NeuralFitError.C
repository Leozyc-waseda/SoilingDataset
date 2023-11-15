//!For fitting neural data to a model

//////////////////////////////////////////////////////////////////////////
// University of Southern California (USC) and the iLab at USC.         //
// See http://iLab.usc.edu for information about this project.          //
//////////////////////////////////////////////////////////////////////////
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
//////////////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////////////
//
// Primary maintainer for this file: David J. Berg <dberg@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ModelNeuron/NeuralFitError.C $

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Util/StringUtil.H"
#include "Util/StringConversions.H"
#include "Util/SimTime.H"
#include "Fitting/NeuralFitError.H"
#include "Fitting/NeuralFitOpts.H"
#include "Raster/Raster.H"
#include "Image/Transforms.H"
#include "Image/DrawOps.H"

#include <iostream>
#include <fstream>
#include <algorithm>

#define EPSILON 0.001

// ######################################################################
NeuralFitError::NeuralFitError(OptionManager& mgr, std::string const & descrName, 
                               std::string const & tagName) :
    ModelComponent(mgr, descrName, tagName), 
    itsSamplingRate("SamplingRate", this, SimTime::ZERO()),
    itsShowDiagnostic(&OPT_ShowDiagnostic, this),
    itsDiagnosticPlotDims(&OPT_DiagnosticPlotDims, this), 
    itsNormalizeType(&OPT_NormalizeType, this), 
    itsSmoothParam(&OPT_SmoothParam, this), 
    itsErrorType(&OPT_ErrorType, this), 
    itsTimeStep(&OPT_SimTime, this),
    itsStimFileNames(&OPT_StimFileNames, this), 
    itsNeuralDataFileNames(&OPT_NeuralDataFileNames, this),
    itsSimplexSize(&OPT_SimplexSize, this),
    itsSimplexErrorTol(&OPT_SimplexErrorTol, this),
    itsSimplexIterTol(&OPT_SimplexIterTol, this),
    itsSimplexDeltaErrorTol(&OPT_SimplexDeltaErrorTol, this),
    itsOutputFileNames(&OPT_ModelOutputNames, this),
    itsOfs(new OutputFrameSeries(mgr)),
    itsNeuralData(), itsSmoothKernel(), 
    itsNumFiles(0), itsHasDiagnostic(false), itsWriteToDisk(false),
    itsSmooth(0.0)
{ 
  addSubComponent(itsOfs);
}

// ######################################################################
nub::ref<OutputFrameSeries> NeuralFitError::getOutputFrameSeries()
{
  return itsOfs;
}

// ######################################################################
NeuralFitError::NormalizeType NeuralFitError::getNormType() const
{
  return itsNormalizeType.getVal();
}

// ######################################################################
void NeuralFitError::setSamplingRate(SimTime const & samplingrate)
{
  itsSamplingRate.setVal(samplingrate);
}

// ######################################################################
bool const NeuralFitError::hasDiagnostic() const
{
  return itsHasDiagnostic;
}

// ######################################################################
void NeuralFitError::start1()
{
  ModelComponent::start1();

  if (itsSamplingRate.getVal() == SimTime::ZERO())
    LFATAL("Classes which derive from NeuralFitError must set the NModelParam itsSamplingRate "
           "to the default sampling rate of the data and stimulus source. This should be done "
           "in the derived classes constructor");

  const double ratio = itsSamplingRate.getVal().hertz() / itsTimeStep.getVal().hertz();
  if (ratio != double(int(ratio)))
    LFATAL("The timestep in hertz must divide evenly into the sampling rate in hertz");

  itsHasDiagnostic = itsShowDiagnostic.getVal();
  itsSmooth = itsSmoothParam.getVal();
  itsSubSample = int(ratio);
  
  std::vector<std::string> stimFiles, neuralDataFiles;
  split(itsStimFileNames.getVal(), ",", back_inserter(stimFiles));
  split(itsNeuralDataFileNames.getVal(), ",", back_inserter(neuralDataFiles));
  
  if (stimFiles.size() != neuralDataFiles.size())
    LFATAL("usage: must use pairs of stimfile and neuraldatafile");
  
  itsNumFiles = stimFiles.size();
  loadData(stimFiles, neuralDataFiles);

  if (~itsOutputFileNames.getVal().empty())
    itsWriteToDisk = true;
}

// ######################################################################
NeuralFitError::ErrorType NeuralFitError::stringToErrorType(std::string const & errortype)
{
  if (errortype.compare("SUMSQUARE") == 0)
    return SUMSQUARE;
  
  if (errortype.compare("SUMABS") == 0)
    return SUMABS;
  
  if (errortype.compare("MEDABS") == 0)
    return MEDABS;
  
  if (errortype.compare("MEDSQUARE") == 0)
    return MEDSQUARE;
  
  if (errortype.compare("ROOTMEANSQUARE") == 0)
    return ROOTMEANSQUARE;

  LFATAL("string cannot be converted to ErrorType");
  return SUMSQUARE;
}

// ######################################################################
NeuralFitError::NormalizeType NeuralFitError::stringToNormalizeType(std::string const & normtype)
{
  if (normtype.compare("MAX") == 0)
    return MAX;
  if (normtype.compare("MAXCOND") == 0)
    return MAXCOND;
  else if (normtype.compare("NONE") == 0)
    return NONE;
  else
  {
    LFATAL("string cannot be converted to NormalizeType");
    return NONE;
  }
}

// ######################################################################
double const NeuralFitError::operator()(std::vector<double> const & params)
{
  DataSet modelResponse;
  
  if (itsSmooth == 0.0)
  { 
    modelResponse = computeModelResponse(std::vector<double>(params.begin()+1, params.end()));
    
    std::vector<double> max;
    DataSet neuralResponse = smooth(itsNeuralData, params[0], max);
    subSample(neuralResponse, itsSubSample, max);
    normalize(neuralResponse, max);
    
    if (hasDiagnostic())
    {
      getOutputFrameSeries()->updateNext();
      showDiagnostic(modelResponse, neuralResponse);
      if (getOutputFrameSeries()->shouldWait())
        Raster::waitForKey();
    }

    return computeError(modelResponse, neuralResponse, itsErrorType.getVal());
  }
  else
  {
    modelResponse = computeModelResponse(params);
    
    if (hasDiagnostic())
    {
      getOutputFrameSeries()->updateNext();
      showDiagnostic(modelResponse, itsNeuralData);
      if (getOutputFrameSeries()->shouldWait())
        Raster::waitForKey();
    }

    // write model response to disk
    if(itsWriteToDisk)
    {
      std::vector<std::string> outputNames;
      split(itsOutputFileNames.getVal(), ",", back_inserter(outputNames));

      for (uint i=0; i < modelResponse.size(); ++i)//loop over stimuli
      {
        std::string fname = outputNames[i];

        LINFO("Writing model response to '%s'", fname.c_str());
     
        std::ofstream modelRespOutFile(fname); 
        if (!modelRespOutFile.is_open())
          LFATAL("Error: cannot write to file %s", fname.c_str());
        
        for (uint j=0; j< modelResponse[i].size(); ++j)//loopo over the samples
          modelRespOutFile<< modelResponse[i][j] <<std::endl;

        modelRespOutFile.close();
      }
    }
    
    return computeError(modelResponse, itsNeuralData, itsErrorType.getVal());
  }
}

// ######################################################################
void NeuralFitError::loadData(std::vector<std::string> const & stimFiles, std::vector<std::string> const & neuralDataFiles)
{
  if (stimFiles.size() != neuralDataFiles.size())
    LFATAL("you must have the same number of stimulus files and neural data files");

  for (uint i=0; i < stimFiles.size(); ++i)
  {
    readNeuralDataFile(neuralDataFiles[i]);	//read each neural data file
    readStimulusFile(stimFiles[i], itsSubSample);//read each stimulus file
  }

  //smooth if its not a param
  if (itsSmooth > 0.0)
  {
    std::vector<double> max;
    itsNeuralData = smooth(itsNeuralData, itsSmooth, max);
    subSample(itsNeuralData, itsSubSample, max);
    normalize(itsNeuralData, max);
  }
  else if (itsSmooth < 0.0)
  {
    std::vector<double> max;
    normalize(itsNeuralData, max);
  }
}

// ######################################################################
SimTime const NeuralFitError::getTimeStep()
{
  return itsTimeStep.getVal();
}

// ######################################################################
std::vector<double> NeuralFitError::getStartingParams()
{
  std::vector<double> p;
  if (itsSmooth == 0.0)
    p.push_back(15.0);

  std::vector<double> mp = startingParams();
  p.insert(p.end(), mp.begin(), mp.end());

  return p;
}

// ######################################################################
void NeuralFitError::getParamRange(std::vector<double> & min, std::vector<double> & max)
{
  min = std::vector<double>();
  max = std::vector<double>();
  if (itsSmooth == 0.0)
  {
    min.push_back(5.0);
    max.push_back(25.0);
  }

  std::vector<double> minp, maxp;
  paramRange(minp, maxp);
  min.insert(min.end(), minp.begin(), minp.end());
  max.insert(max.end(), maxp.begin(), maxp.end());
}

// ######################################################################
uint const NeuralFitError::getNumFiles() const
{
  return itsNumFiles;
}

// ######################################################################
ParamList<double, double, uint, double> NeuralFitError::getNelderMeadParams()
{
  return ParamListHelper::make_paramlist(itsSimplexSize.getVal(), itsSimplexErrorTol.getVal(), itsSimplexIterTol.getVal(), itsSimplexDeltaErrorTol.getVal());
}

// ######################################################################
NeuralFitError::DataSet const & NeuralFitError::getNeuralResponse() 
{ 
  return itsNeuralData;
}

// ######################################################################
void NeuralFitError::readNeuralDataFile(std::string const & fileName)
{
  std::ifstream neuralDataFile(fileName);
  std::vector<double> spikeTrain;
  
  if (!neuralDataFile.is_open())
    LFATAL("cannot locate neural data file %s", fileName.c_str());
      
  std::string line;
  while (getline(neuralDataFile, line))
  {
    std::vector<std::string> toks;
    split(line, "", back_inserter(toks));
    if (toks.size() != 1)
      LFATAL("Invalid line while reading the neural data file. The reader expects one floating point value per line.");

    double spikeCount = fromStr<double>(toks[0]);          
    spikeTrain.push_back(spikeCount);
  }

  //close stream
  neuralDataFile.close();
  
  itsNeuralData.push_back(spikeTrain);

  LINFO("neural data file '%s' loaded. %d samples ", fileName.c_str(), (uint)spikeTrain.size());;
}

// ######################################################################
double const NeuralFitError::computeError(DataSet const & model, DataSet const & data, NeuralFitError::ErrorType errortype)
{

  if (model.size() != data.size())
    LFATAL("model and data must have the same number of conditions");
  
  double error = 0.0;
  int cnt = 0;
  std::vector<double> err;
  double e = 0;

  DataSet::const_iterator mCondition(model.begin()), end(model.end()), dCondition(data.begin());

  //loop through the elements and compute error
  while (mCondition != end)
  {
    std::vector<double>::const_iterator modSamp(mCondition->begin()), dataSamp(dCondition->begin()), tend(mCondition->end());
    while (modSamp != tend)
    {
      double const m = *modSamp++;
      double const d = *dataSamp++;
      
      switch (errortype)
      {
      case SUMSQUARE :
        e = d - m;
        error += e*e;
        break;
        
      case SUMABS :
        e = std::abs(d - m); 
        error += e;
        break;

      case MEDABS :
        e = std::abs(d - m); 
        err.push_back(e);
        break;

       case MEDSQUARE :
        e = d - m; 
        err.push_back(e * e);
        break;
        
      case ROOTMEANSQUARE:
        e = d - m;
        error += e*e;
        cnt++;
        break;

      default:
        LFATAL("Not an error type");
        break;
      }
    }
    ++mCondition; ++dCondition;
  }

  //for computations that need to be done outside the loop
  switch (errortype)
  {
  case MEDABS:
    break;
    
  case MEDSQUARE:
    if (err.size() > 1)
    {
      std::sort(err.begin(), err.end());
      if (err.size()%2 ==0)
        error = 0.5 * (err[err.size()/2] + err[err.size()/2 - 1]);
      else      
        error = err[err.size()/2];
    }
    else
      error = err[0];
    break;
    
  case ROOTMEANSQUARE:
    error = sqrt(error / cnt);
    break;
    
  default:
    break;
  }

  return error;
}

// ######################################################################
NeuralFitError::DataSet NeuralFitError::smooth(NeuralFitError::DataSet const & data, double const & smoothParam, std::vector<double> & max)
{
  if (smoothParam <= 0.0)
    return data;

  DataSet outSet;

  // recompute kernel if necessary
  if ((itsSmoothKernel.size() < 1) || (itsSmooth != smoothParam))
    itsSmoothKernel = computeKernel(smoothParam);
  
  DataSet::const_iterator condition(data.begin()), end(data.end());
  while (condition != end)
  {
    double mx = 0.0;
  
    //convolve kernel with data
    std::vector<double> sData(condition->size());
    
    std::vector<double>::const_iterator begin(condition->begin()-1), elem(condition->begin()), end(condition->end());
    std::vector<double>::iterator out(sData.begin());
    while (elem != end)
    {
      std::vector<double>::const_iterator e = elem++;
      std::vector<double>::iterator k = itsSmoothKernel.begin();
      std::vector<double>::iterator kend = itsSmoothKernel.end();
      double sum = 0.0;
      while ((e != begin) && (k != kend))
        sum += *e-- * *k++;
      
      *out++ = sum;

      if (sum > mx)
        mx = sum;
    }

    outSet.push_back(sData);
    max.push_back(mx);
    ++condition;
  }
  return outSet;
}

// ######################################################################
std::vector<double> const NeuralFitError::findMax(NeuralFitError::DataSet const & data)
{
  DataSet::const_iterator resp(data.begin()), end(data.end());
  std::vector<double> maxs;
  while (resp != end)
  {
    double const mx = *std::max_element(resp->begin(), resp->end());
    maxs.push_back(mx);
    ++resp;
  }
  
  return maxs;
}

// ######################################################################
double const NeuralFitError::findMax(std::vector<double> const & data)
{
  return *std::max_element(data.begin(), data.end());
}

// ######################################################################
void NeuralFitError::normalize(NeuralFitError::DataSet & data, std::vector<double> & max)
{
  //normalize by max if desired
  if (itsNormalizeType.getVal() == MAX)
  {
    if (!max.size())
      max = findMax(itsNeuralData);
    
    std::vector<double>::iterator mx(max.begin());
    DataSet::iterator resp(data.begin()), end(data.end());
    while (resp != end)
    {
      if (*mx > 0.0)
        std::for_each(resp->begin(), resp->end(), [&mx](double& value){ value /= *mx; });

      ++resp; ++mx;
    }
  }

  //normalize by max condition if desired
  else if (itsNormalizeType.getVal() == MAXCOND) 
  {
    if (!max.size())
      max = findMax(itsNeuralData);

    double const mx = *std::max_element(max.begin(), max.end());
    if (mx > 0.0)
    {
      DataSet::iterator resp(data.begin()), end(data.end());
      while (resp != end)
      {
        std::for_each(resp->begin(), resp->end(), [&mx](double& value){ value /= mx; });
        ++resp;
      }
    }
  }
}

// ######################################################################
void NeuralFitError::subSample(NeuralFitError::DataSet & data, const uint subsample, std::vector<double> & max)
{
  if (subsample < 2)
    return;

  max = std::vector<double>(data.size(), 0.0);
  DataSet sdata(data.size());

  //loop over all conditions
  std::vector<double>::iterator mx(max.begin());
  DataSet::iterator sd(sdata.begin()), d(data.begin()), end(data.end());
  while (d != end)
  {
    //loop through data points
    std::vector<double>::const_iterator ii(d->begin()), stop(d->end());
    uint c = 0;
    while ((c < d->size()) && (ii != stop))
    {
      if (*ii > *mx)
        *mx = *ii;
      
      sd->push_back(*ii);
      ii += subsample;
      c += subsample;
    }
    ++d; ++sd; ++mx;
  }
  data = sdata;
}

// ######################################################################
std::vector<double> NeuralFitError::computeKernel(double const & smoothParam)
{
  //put the smooth param in inverse seconds
  const double a = 1000.0 / smoothParam;
  SimTime t = SimTime::ZERO();
  const SimTime timestep = itsSamplingRate.getVal();

  double y = 0.0;
  bool keepgoing = true;
  std::vector<double> filter;
  while ((y > EPSILON) || keepgoing)
  {
    t += timestep;
    y = a * a * t.secs() * exp(-1.0 * a * t.secs());
    filter.push_back(y);
    if (y > EPSILON)
      keepgoing = false;
  }
  return filter;
}

// ######################################################################
std::string convertToString(const NeuralFitError::ErrorType val)
{
    return errorTypeNames(val);
}

// ######################################################################
void convertFromString(const std::string& str, NeuralFitError::ErrorType& val)
{
  for (int i = 0; i < ERRORTYPECOUNT; i ++)
  {
    if (str.compare(errorTypeNames(NeuralFitError::ErrorType(i))) == 0)
    { 
      val = NeuralFitError::ErrorType(i); 
      return;
    }
  }
  conversion_error::raise<NeuralFitError::ErrorType>(str);
}

// ######################################################################
std::string convertToString(const NeuralFitError::NormalizeType val)
{
    return normalizeTypeNames(val);
}

// ######################################################################
void convertFromString(const std::string& str, NeuralFitError::NormalizeType& val)
{
  for (int i = 0; i < NORMALIZETYPECOUNT; i ++)
  {
    if (str.compare(normalizeTypeNames(NeuralFitError::NormalizeType(i))) == 0)
    { 
      val = NeuralFitError::NormalizeType(i); 
      return;
    }
  }
  conversion_error::raise<NeuralFitError::NormalizeType>(str);
}

// ######################################################################
void NeuralFitError::showDiagnostic(NeuralFitError::DataSet const & neural, NeuralFitError::DataSet const & model)
{
  uint w = itsDiagnosticPlotDims.getVal().w();
  uint h = itsDiagnosticPlotDims.getVal().h() / (uint)neural.size();
  
  Layout<PixRGB<byte> > layout;
  
  if (!neural[0].size())
    LFATAL("No Neural data is available");
  
  std::vector<std::vector<double> >::const_iterator iter(neural.begin()), end(neural.end());
  std::vector<std::vector<double> >::const_iterator miter(model.begin());
  
  Image<PixRGB<byte> > nplot, mplot;
  while (iter != end)
  { 
    if ((itsNormalizeType.getVal() == MAX) || (itsNormalizeType.getVal() == MAXCOND))
    {
      nplot = linePlot(*iter,  w, h, 0.0, 1.0, " ", " ", "samples");
      mplot = linePlot(*miter, w, h, 0.0, 1.0, " ", " ", " ",PixRGB<byte>(0,255,0), PixRGB<byte>(255,255,255), 0, true);
    }
    else
    {
      nplot = linePlot(*iter,  w, h, 0.0, 0.0, " ", " ", "samples");
      mplot = linePlot(*miter, w, h, 0.0, 0.0, " ", " ", " ",PixRGB<byte>(0,255,0), PixRGB<byte>(255,255,255), 0, true);
    }
    
    layout = vcat(layout, composite(nplot, mplot, PixRGB<byte>(255,255,255)));
    ++iter; ++miter;
  }
  getOutputFrameSeries()->writeRgbLayout(layout, "Neural Fitting Display");
}

// ######################################################################
uint const NeuralFitError::getSubSampleRatio() const
{
  return itsSubSample;
}

// ######################################################################
SimTime const NeuralFitError::getSamplingRate() const
{
  return itsSamplingRate.getVal();
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
