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
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ModelNeuron/NeuralFitError.H $
#ifndef FITTING_NEURALFITERROR_H_DEFINED
#define FITTING_NEURALFITERROR_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Util/SimTime.H"
#include "Image/Layout.H"
#include "Image/Pixels.H"
#include "Media/FrameSeries.H"
#include <GenericUtils/ParamList.H>

//////////////////////////////////////////////////////////////////////////
/* A base class that can serve as an objective function for fitting a model
   response to neural data */
//////////////////////////////////////////////////////////////////////////
class NeuralFitError: public ModelComponent
{	
  public:
    //!different error types
    enum ErrorType {SUMSQUARE, SUMABS, MEDABS, MEDSQUARE, ROOTMEANSQUARE};
    
    //!different normalizing types
    enum NormalizeType {MAX, MAXCOND, NONE};

    //!convert errortype to string
    static ErrorType stringToErrorType(std::string const & errortype);

    //!convert normtype to string
    static NormalizeType stringToNormalizeType(std::string const & normtype);

    //!convenient type def
    typedef std::vector<std::vector<double> > DataSet;

    //!constructor
    NeuralFitError(OptionManager& mgr, std::string const & descrName, std::string const & tagName);
    
    //! get model output and return error between model and neural data
    double const operator()(std::vector<double> const & params);

    //!load in stimulus and neural response from text files
    void loadData(std::vector<std::string> const & stimFiles, std::vector<std::string>const & neuralDataFiles);

    //!get time step of simulations
    SimTime const getTimeStep();

    //!get some starting parameters
    std::vector<double> getStartingParams();

    //!get some starting parameters
    void getParamRange(std::vector<double> & min, std::vector<double> & max);

    //!get the number of files (stimulus/neural data)
    uint const getNumFiles() const;

    //!return true if we are in diagnostic mode
    bool const hasDiagnostic() const;

    //!get some paramters for the nelder mead method
    virtual ParamList<double, double, uint, double> getNelderMeadParams();

    //!get the subsampling ratio
    SimTime const getSamplingRate() const;

  protected:

    void start1();

    //! get the output stream
    nub::ref<OutputFrameSeries> getOutputFrameSeries();

    //! get neural data as a DataSet
    DataSet const & getNeuralResponse();

    //!set the sampling rate of the data 
    void setSamplingRate(SimTime const & samplingrate);

    //!get the subsampling ratio
    uint const getSubSampleRatio() const;

    //!get the normalizatoin type
    NormalizeType getNormType() const;
    
  private:    
    //! read a neural data file
    virtual void readNeuralDataFile(std::string const & fileName);

    //! read a stimulus file, subsampling by this factor
    virtual void readStimulusFile(std::string const & fileName, uint subsample) = 0;
    
    //! get starting params
    virtual std::vector<double> startingParams() = 0;
    
    //! get starting params range
    virtual void paramRange(std::vector<double> & min, std::vector<double> & max) = 0;

    //! compute model response to stimulus
    virtual DataSet computeModelResponse(std::vector<double> const & params) = 0;
    
    //! compute error between model and data
    double const computeError(NeuralFitError::DataSet const & model, NeuralFitError::DataSet const & data, ErrorType errortype);

    //! smooth the data
    DataSet smooth(DataSet const & data, double const & smoothParam, std::vector<double> & max);

    //! normalize data
    void normalize(NeuralFitError::DataSet & data, std::vector<double> & max);

    //! subsample the data
    void subSample(NeuralFitError::DataSet & data, const uint subsample, std::vector<double> & max);

    //! find the max in the data set
    std::vector<double> const findMax(NeuralFitError::DataSet const & data);

    //! find the max in the data set
    double const findMax(std::vector<double> const & data);

    //! compute the smoothing kernel
    std::vector<double> computeKernel(double const & smoothParam);

    //! show a diagnostic plot
    void showDiagnostic(NeuralFitError::DataSet const & neural, NeuralFitError::DataSet const & model);
    
    NModelParam<SimTime> itsSamplingRate;
    OModelParam<bool> itsShowDiagnostic;
    OModelParam<Dims> itsDiagnosticPlotDims;
    OModelParam<NormalizeType> itsNormalizeType;
    OModelParam<double> itsSmoothParam;
    OModelParam<ErrorType> itsErrorType;
    OModelParam<SimTime> itsTimeStep;
    OModelParam<std::string> itsStimFileNames;
    OModelParam<std::string> itsNeuralDataFileNames;
    OModelParam<double> itsSimplexSize;
    OModelParam<double> itsSimplexErrorTol;
    OModelParam<uint> itsSimplexIterTol; 
    OModelParam<double> itsSimplexDeltaErrorTol;
    OModelParam<std::string> itsOutputFileNames;

    nub::ref<OutputFrameSeries> itsOfs;    
    DataSet itsNeuralData;
    std::vector<double> itsSmoothKernel;
    uint itsNumFiles;
    bool itsHasDiagnostic;
    bool itsWriteToDisk;
    double itsSmooth;
    uint itsSubSample;
};

// #####################################################################
std::string convertToString(const NeuralFitError::ErrorType val);
void convertFromString(const std::string& str,  NeuralFitError::ErrorType& val);

std::string convertToString(const  NeuralFitError::NormalizeType val);
void convertFromString(const std::string& str,  NeuralFitError::NormalizeType& val);

//! Get a name in clear for a given type
//! need to update this value if you are adding a new ErrorType this code
#define ERRORTYPECOUNT 5
inline const char* errorTypeNames(const  NeuralFitError::ErrorType m)
{
  static const char err[ERRORTYPECOUNT][20] = {"SUMSQUARE", "SUMABS", "MEDABS", "MEDSQUARE", "ROOTMEANSQUARE"};
  if (int(m) < 0 || int(m) >= ERRORTYPECOUNT)
          LFATAL("Invalid Error Type value %d (valid range is 0-%d inclusive)",
                         int(m), ERRORTYPECOUNT);
  return err[int(m)];
}

//! need to update this value if you are adding a new ErrorType this code
#define NORMALIZETYPECOUNT 3 
inline const char* normalizeTypeNames(const  NeuralFitError::NormalizeType m)
{
  static const char nor[NORMALIZETYPECOUNT][10] = {"MAX", "MAXCOND", "NONE"};
  if (int(m) < 0 || int(m) >= NORMALIZETYPECOUNT)
          LFATAL("Invalid Normalize Type value %d (valid range is 0-%d inclusive)",
                         int(m), NORMALIZETYPECOUNT);
  return nor[int(m)];
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

