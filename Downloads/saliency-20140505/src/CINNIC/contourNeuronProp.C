/*!@file CINNIC/contourNeuronProp.C CINNIC classes */

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
// Primary maintainer for this file:  T Nathan Mundhenk <mundhenk@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/contourNeuronProp.C $
// $Id: contourNeuronProp.C 4663 2005-06-23 17:47:28Z rjpeters $
//

#include "CINNIC/contourNeuronProp.H"
#include "Util/log.H"

// ############################################################
// ############################################################
// ##### ---CINNIC---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################



//#################################################################
/*!This method when called will place a charge in this neuron telling it
  where the charge came from and the polarity properties of the placing
  neuron
*/

template <class CH, class X, class Y>
ContourNeuronProp<CH,X,Y>::ContourNeuronProp()
{
}

template <class CH, class X, class Y>
ContourNeuronProp<CH,X,Y>::~ContourNeuronProp()
{
}

template <class CH, class X, class Y>
CH& ContourNeuronProp<CH,X,Y>::getCharge()
{
  return charge;
}

template <class CH, class X, class Y>
CH& ContourNeuronProp<CH,X,Y>::getThreshold()
{
  return threshold;
}

template <class CH, class X, class Y>
CH& ContourNeuronProp<CH,X,Y>::getSupressionMod()
{
  return supressionMod;
}

template <class CH, class X, class Y>
void ContourNeuronProp<CH,X,Y>::setSupressionMod(CH& sup)
{
  supressionMod = sup;
}

template <class CH, class X, class Y>
void ContourNeuronProp<CH,X,Y>::setSupressionBool()
{
  supress = false;
}

template <class CH, class X, class Y>
void ContourNeuronProp<CH,X,Y>::setSupressionThresh(CH& thresh, CH& changeVal, CH& max)
{
  supressionModThresh = thresh;
  supressionModChange = changeVal;
  supressionCeiling = max;
}

template <class CH, class X, class Y>
void ContourNeuronProp<CH,X,Y>::Charge(CH& _charge, bool _pol, bool _sender)
{
  if(_pol == true)
  {
    if(_charge > 0)
    {
      chargePos+=(_charge*(float)(1/resistance));
      //add charges which will go to pos polarity synapses
    }
  }
  else
  {
    if(_charge > 0)
    {
      chargeNeg+=(_charge*(float)(1/resistance));
      //add charges which will go to neg  polarity synapses
    }
  }
  charge+=_charge;
}

template <class CH, class X, class Y>
void ContourNeuronProp<CH,X,Y>::ChargeSimple(CH& _charge)
{
  charge+=_charge;
}

/*!This method when called will return all the charges that should go to a
  another neuron from this neuron given the other neurons polarity
*/
template <class CH, class X, class Y>
CH& ContourNeuronProp<CH,X,Y>::DisCharge(bool _pol, bool _sender, int a, int cascadeType)
{
  if(_sender == true)
  {
    discharge = 0;
  }
  else
  {
    if((a == 0) || (cascadeType == 2))// allow cross over for cascade at 0 degrees
    {
      discharge = chargePos + chargeNeg;
    }
    else
    {
      if(_pol == true)
      {
        discharge = chargePos;
      }
      else
      {
        discharge = chargeNeg;
      }
    }
  }
  return discharge;
}

/*!this method stores the neurons fireing threshold at which point
  it fires a salience charge
*/
template <class CH, class X, class Y>
void ContourNeuronProp<CH,X,Y>::setThreshold(CH& Threshold, CH& Resistance)
{
  threshold = Threshold;
  resistance = 1/Resistance;
}

/*!This method resets the place counter for the charge storage
  vector in effect putting its charge to 0;
*/
template <class CH, class X, class Y>
void ContourNeuronProp<CH,X,Y>::ResetTempCharge()
{
  chargePos = 0;
  chargeNeg = 0;
}

template <class CH, class X, class Y>
void ContourNeuronProp<CH,X,Y>::ResetCharge()
{
  charge = 0;
}

template <class CH, class X, class Y>
void ContourNeuronProp<CH,X,Y>::ResetChargeAll()
{
  charge = 0;
  chargePos = 0;
  chargeNeg = 0;
}

template <class CH, class X, class Y>
void ContourNeuronProp<CH,X,Y>::setUpperLimit()
{
  // FIXME had to comment this out because 'upperLimit' is not declared
  // anywhere

//   charge = upperLimit;
}

#if 0
// commented this section out; see explanation in contourNeuronProp.H

void ContourNeuronPropVec::setSize(int t, int i, int j, int k)
{
  T=t;I=i;J=j;K=k;
  LINFO("setting image size as %d,%d,%d,%d",T,I,J,K);
  ContourNeuronProp<float,int,int> Ptemp[T][I][J][K];
  NeuronMatrix[T][I][J][K] = ****Ptemp;
  LINFO("done");
}

int ContourNeuronPropVec::getSizeT()
{
  return T;
}

#endif

template class ContourNeuronProp<float, int, int>;

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
