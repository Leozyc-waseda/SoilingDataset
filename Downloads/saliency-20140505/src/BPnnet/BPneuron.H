/*!@file BPnnet/BPneuron.H header for a back prop neuron */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BPnnet/BPneuron.H $
// $Id: BPneuron.H 4786 2005-07-04 02:18:56Z itti $
//

#ifndef BPNEURON_H_DEFINED
#define BPNEURON_H_DEFINED

//! neuron for a back prop neural net (BPnnet)
class BPneuron {
public:
  //! Constructor
  BPneuron();

  //! Destructor
  ~BPneuron();

  //! Assign given weighted input sum [S_j] and activation level
  /*! Activation Level a_j = f(S_j)*/
  void assignInput(double input);

  //! Calculate and set delta for output neurons
  /*! For output layer neurons; finds difference between output and
    target and multiplies by derivative of sigmoid function [delta_j =
    (t_j - a_j)*f'(S_j)]
    returns the difference [t_j - a_j] for RMS error calculation */
  double calcOutputDelta(const double target);

  //! Calculate and set delta for hidden neurons
  /*! For hidden layer neurons; compute hidden delta from weighted sum
    of output deltas [delta_j = sum_k(delta_k * w_kj)*f'(S_j)]
    returns delta_j */
  void calcHiddenDelta(double weightedDeltaSum);

  //! Return delta (error level)
  double getDelta();

  //! Return activation level
  double getActivationLevel();

private:
  // Perform sigmoid function on x
  double sigmoidFunction(const double x);

  // Perform first derivative of sigmoid function on x
  double derivSigmoidFunction(const double x);

  double delta;            // d_j
  double weightedInputSum; // a_j
  double activationLevel;  // S_j
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
