/*!@file BPnnet/BPneuron.C Back Prop Neuron class */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BPnnet/BPneuron.C $
// $Id: BPneuron.C 4786 2005-07-04 02:18:56Z itti $
//

#include "BPnnet/BPneuron.H"
#include <math.h>

// ######################################################################
BPneuron::BPneuron( void )
{
  delta = 0.0;
  weightedInputSum = 0.0;
  activationLevel = 0.0;
}

// ######################################################################
BPneuron::~BPneuron()
{ }

// ######################################################################
void BPneuron::assignInput( double input )
{
  weightedInputSum = input;
  activationLevel = sigmoidFunction(weightedInputSum);
}

// ######################################################################
double BPneuron::calcOutputDelta( double target )
{
  double difference = target - activationLevel;
  double f1 = derivSigmoidFunction(weightedInputSum);
  delta = difference * f1;
  return difference;
}

// ######################################################################
void BPneuron::calcHiddenDelta( double weightedDeltaSum )
{
  double f1 = derivSigmoidFunction(weightedInputSum);
  delta = weightedDeltaSum * f1;
}

// ######################################################################
double BPneuron::getDelta( void )
{
  return delta;
}

// ######################################################################
double BPneuron::getActivationLevel( void )
{
  return activationLevel;
}

// ######################################################################
double BPneuron::sigmoidFunction( const double x )
{ return 1.0 / (1.0 + exp(- x)); }

// ######################################################################
double BPneuron::derivSigmoidFunction( const double x )
{
  double eS = exp( - x );
  double dF = 1.0 + eS;
  return eS / (dF * dF);
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
