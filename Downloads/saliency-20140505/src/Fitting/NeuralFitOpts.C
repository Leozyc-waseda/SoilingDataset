/*!@file ModelNeuron/NeuralFitOpts.C */

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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Fitting/NeuralFitOpts.C $

#include "Fitting/NeuralFitOpts.H"
#include "Fitting/NeuralFitError.H"
#include "Component/ModelOptionDef.H"
#include "Image/Dims.H"

// #################### SC Fitting options:
const ModelOptionCateg MOC_NEURALFIT = { MOC_SORTPRI_3, "Neural fitting options" };

const ModelOptionDef OPT_ShowDiagnostic = { MODOPT_FLAG, "ShowDiagnostic", & MOC_NEURALFIT, OPTEXP_CORE, 
                                            "Show diagnostic plots", 
                                            "show-diagnostic", '\0', "<bool>", "false" };  

const ModelOptionDef OPT_DiagnosticPlotDims = { MODOPT_ARG(Dims), "DiagnosticPlotDims", & MOC_NEURALFIT, OPTEXP_CORE, "Dims of diagnostic plot", "diagnostic-dims", '\0', "<Dims>", "800x600" };  

const ModelOptionDef OPT_SmoothParam = { MODOPT_ARG(double), "SmoothParam", & MOC_NEURALFIT, OPTEXP_CORE, "smooth paramter in ms, 0.0 will make this parameter part of the fitting procedure", "data-smooth", '\0', "<double>", "15.0" };  

const ModelOptionDef OPT_NormalizeType = { MODOPT_ARG(NeuralFitError::NormalizeType), "NormalizeType", & MOC_NEURALFIT, OPTEXP_CORE, "normalization type", "data-normtype", '\0', "<MAX, NONE>", "MAX" };  

const ModelOptionDef OPT_ErrorType = { MODOPT_ARG(NeuralFitError::ErrorType), "ErrorType", & MOC_NEURALFIT, OPTEXP_CORE, "error function type", "fit-errortype", '\0', "<SUMSQUARE, SUMABS, MEDABS, MEDSQUARE, ROOTMEANSQUARE>", "SUMSQUARE" };  

const ModelOptionDef OPT_SimTime = { MODOPT_ARG(SimTime), "SimTime", & MOC_NEURALFIT, OPTEXP_CORE, "time step to process data and stimulus file, which may be a number that divides evenly into the sampling rate of the data", "sim-timestep", '\0', "<SimTime>", "1ms" };

const ModelOptionDef OPT_StimFileNames = { MODOPT_ARG(std::string), "StimFileNames",  &MOC_NEURALFIT, OPTEXP_CORE, "stimulus file names" , "stim-file-names", '\0', "<file1, file2, fileN>", ""};

const ModelOptionDef OPT_NeuralDataFileNames = { MODOPT_ARG(std::string), "NeuralDaraFileNames", & MOC_NEURALFIT, OPTEXP_CORE, "neural data file names" , "neural-data-file-names", '\0', "<file1, file2, fileN>", ""};

const ModelOptionDef OPT_SimplexSize = { MODOPT_ARG(double), "SimplexSize", & MOC_NEURALFIT, OPTEXP_CORE, "the size of the initial simplex", "simplex-size", '\0', "<double>", "5.0" };  

const ModelOptionDef OPT_SimplexErrorTol = { MODOPT_ARG(double), "SimplexErrorTol", & MOC_NEURALFIT, OPTEXP_CORE, "stop when we get this close to the objective", "simplex-errortol", '\0', "<double>", "1.0e-5" };  

const ModelOptionDef OPT_SimplexDeltaErrorTol = { MODOPT_ARG(double), "SimplexDeltaErrorTol", & MOC_NEURALFIT, OPTEXP_CORE, "stop when the average error of the simplex verticies is changing by less than this tolerance", "simplex-deltaerrortol", '\0', "<double>", "1.0e-4" };  

const ModelOptionDef OPT_SimplexIterTol = { MODOPT_ARG(uint), "SimplexIterTol", & MOC_NEURALFIT, OPTEXP_CORE, "stop after this many iterations", "simplex-itertol", '\0', "<uint>", "500" };  

const ModelOptionDef OPT_UseMultiThread = { MODOPT_FLAG, "UseMultiThread", & MOC_NEURALFIT, OPTEXP_CORE, 
                                            "Use multi-threading when evaluating the initial simplex verticies. Your functor must be copyable, and ensure that no state is shared between copies of the functor.", "use-multithread", '\0', "<bool>", "false" };

const ModelOptionDef OPT_ModelOutputNames = { MODOPT_ARG(std::string), "ModelOutputNames", & MOC_NEURALFIT, OPTEXP_CORE, 
                                              "A comma separated list of output filenames", "model-output-files", '\0', "<filename>", "" };

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
