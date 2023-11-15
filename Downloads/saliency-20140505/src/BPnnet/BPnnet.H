/*!@file BPnnet/BPnnet.H header for Back Prop Neural Net */

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
// Primary maintainer for this file: Philip Williams <plw@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BPnnet/BPnnet.H $
// $Id: BPnnet.H 4786 2005-07-04 02:18:56Z itti $
//
#ifndef BPNNET_H_DEFINED
#define BPNNET_H_DEFINED

#include "BPnnet/BPneuron.H"
#include "Image/Image.H"
#include "BPnnet/KnowledgeBase.H"
#include "BPnnet/SimpleVisualObject.H"

template <class T> class Jet;

//! describes structure of a 3 layer back prop neural net
class BPnnet {
public:
  //! Constructor
  BPnnet(const int numInput, const int numHidden, const KnowledgeBase *kb);

  //! Destructor
  ~BPnnet();

  //! Initialize all weights to (small) random values
  void randomizeWeights();

  //! Normalize weights
  void normalizeWeights();

  //! Do one training iteration
  /*! returns rms1 = sum of (target output - actual output)^2/n_outputs
    for all output layer neurons */
  double train(const Image<float> &in, const SimpleVisualObject& target,
               const double learnRate);

  //! Attempt to recognize given jet as a certain visual object
  bool recognize(const Image<float> &in, SimpleVisualObject &vo);

  //! Store 2 matrices of weights to the file "filename"
  /*! use this after training a net */
  bool save(const char* filename) const;

  //! Assign all weights based on data stored in the file "filename"
  /*! returns false if the # of units in each layer do not match the
    matrix sizes in the file
  */
  bool load(const char* filename);

private:

  // Performs forward propogation tasks common to recognition and training
  // Used to avoid repetition of code within recognize() and train() methods
  void forwardProp(const Image<float> &in);

  // layers - 1 dimensional, should these be vectors instead of images?
  std::vector<BPneuron> inputLayer;
  std::vector<BPneuron> hiddenLayer;
  std::vector<BPneuron> outputLayer;

  // Weight (i,j) from input neuron i to hidden neuron j
  Image<double> weightFromInput;
  // Weight (i,j) from hidden neuron i to output neuron j
  Image<double> weightToOutput;

  int numInputUnits;        // number of features in input
  int numHiddenUnits;        // variable
  int numOutputUnits;        // number of known visual objects

  const KnowledgeBase *itsKb;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
