/*!@file Learn/LocalBinaryPatterns.H Multiclass LocalBinaryPatterns Classifier module */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
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
// Primary maintainer for this file: Dan Parks <danielfp@usc.edu>
// $HeadURL$
// $Id$
//

#ifndef LOCALBINARYPATTERNS_H_DEFINED
#define LOCALBINARYPATTERNS_H_DEFINED

#include <deque>
#include <map>

#include "Image/Image.H"
#include "Image/Pixels.H"

// ######################################################################
//! Multiclass LocalBinaryPatterns Class
class LocalBinaryPatterns
{

public:
  // Map of model histograms (multiple per model id)
  typedef std::map<int, std::vector<std::vector<float> > > MapModelVector;

  //! Constructor
  LocalBinaryPatterns(int LBPRadii=1, int LBPPixels=8, int varBins=128, bool useColor=false, bool normalize=true);

  //! Destructor
  ~LocalBinaryPatterns();

  //! Add histograms using color texture
  void addModel(const Image<PixRGB<byte> >& texture, const int id);

  //! Add a LBP and VAR histogram explicitly as a model to the system
  void addModel(const std::vector<float>& lbp, const std::vector<float>&col, const std::vector<float>& var, const int id);

  //! Return the final histogram 
  std::vector<float> createHistogram(const Image<PixRGB<byte> >& texture);
 
  //! Determine the LBP histogram and variance values
  void createRawHistogram(const Image<PixRGB<byte> >& colTexture, std::vector<float>& lbps, std::vector<float>& col, std::vector<float>& vars);

  //! Sum the colors within the rectangle
  void colorSum(Image<PixRGB<byte> >img, Rectangle rec, float& rg, float& by);

  //! Convert list of variances to a variance histogram
  std::vector<float> convertVariance(const std::vector<float>& vars);

  //! Set the map of built models
  void setModels(const MapModelVector& models);

  //! Return a map of the built models
  MapModelVector getModels();

  //! Return a separate data vector and label vector from the MapModelVector
  void getLabeledData(const MapModelVector& models, std::vector<std::vector<float> >& data, std::vector<float>& labels);

  //! Merge two ordered vectors
  std::vector<float> merge(const std::vector<float>& left, const std::vector<float>& right);

  //! Load the incomplete models from a combined format
  void setIncompleteModels(const MapModelVector& incompleteModels);

  std::vector<float> getVarThresholds();

  void setVarThresholds(std::vector<float> thresholds);

  void combineModels(const std::vector< MapModelVector >& allModels, MapModelVector& combined);

  void appendMap(MapModelVector& dst, const MapModelVector& src);

  //! Return the incomplete models in a combined format
  MapModelVector getIncompleteModels();

  //! After all model exemplars have been given, build final model histograms and set up variance bins
  void buildModels();

  //! Convert incomplete models to complete models
  void convertIncompleteModels();

  //! Predict using classifier
  int predict(const Image<float>& texture);

  //! Get total number of model exemplars
  uint getTotalModelExemplars(MapModelVector models);

  //! Read file of models into system
  MapModelVector readModelsFile(std::string modelsFile);

  //! Write file of models out
  void writeModelsFile(std::string modelsFile, MapModelVector models);

  // Read bin thresholds in
  std::vector<float> readThresholdsFile(std::string thresholdsFile);

  // Write bin thresholds out
  void writeThresholdsFile(std::string thresholdsFile, std::vector<float> thresholds);
  

protected:
  template<class T> int getVarIndex(const T&thresh, float var, int binHint=-1);

  float itsLBPRadius;
  int itsLBPPixels;
  int itsVarBins;
  bool itsUseColor;
  bool itsNormalize;
  int itsColorBins;
  std::vector<float> itsVarBinThresholds;
  //! Store each model
  MapModelVector itsModels;
  //! Temporary LBP models (multiple per class)
  MapModelVector itsTempLBP;
  //! List of actual variances, kept until training is complete so that a full histogram can be built
  MapModelVector itsTempVariance;
  //! List of color histogram models
  MapModelVector itsTempColor;
};

#endif


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

