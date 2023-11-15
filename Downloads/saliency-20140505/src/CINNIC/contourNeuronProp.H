/*!@file CINNIC/contourNeuronProp.H CINNIC classes */
//! Holds the property for each neuron
/*! this class is used to process and hold the membrane potential
  of each neuron. It keeps track of total potential as well as
  potential that is "up" or "down" that is, to simulate the
  effects of tranfering energy from neuron to another neuron it
  holds energy in either a positive or negative energy bank.
*/
//#include "cascadeHold.H"
#include "CINNIC/contourNeuron.H"
//#include "ImageMap.H"
#include "CINNIC/contourDefine.H"

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
// Primary maintainer for this file: T. Nathan Mundhenk <mundhenk@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/contourNeuronProp.H $
// $Id: contourNeuronProp.H 4663 2005-06-23 17:47:28Z rjpeters $
//

// ############################################################
// ############################################################
// ##### ---CINNIC---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

template <class CH, class X, class Y> class ContourNeuronProp
{
private:
  int Count;
  bool supress;
  CH charge; //charge
  CH discharge;
  CH chargePos;
  CH chargeNeg;
  CH totCount;
  CH resistance; // Viva La Resistance - Restance divided from energy transplorted
  CH supressionMod; //Modifier for adaptive supression
  CH supressionModThresh; //Threshold for adaptive supression
  CH supressionModChange; //The amount to increment supression
  CH supressionCeiling;
protected:
  //! Current threshold for this neuron
  float threshold; //current_threshold
  //float resistance; // Viva La Resistance - Restance divided from energy transplorted
public:

  ContourNeuronProp();
  ~ContourNeuronProp();
  //! return this neurons charge
  CH& getCharge();
  //! return this neurons threshold
  CH& getThreshold();
  //! return this neurons supression modifier
  CH& getSupressionMod();
  //! set up supression for this neuron
  void setSupressionMod(CH& sup);
  //! set holder to know if supression has been incremented
  void setSupressionBool();
  //! set up supression threshold for this neuron and its weight augementation
  /*!This method is used to set up adaptive supression for its threshold and the
    amount supression will increase every time threshold is reached
    @param thresh This is the threshold number
    @param changeVal The amount supression should be increased
  */
  void setSupressionThresh(CH& thresh, CH& changeVal, CH& max);
  //! reset the "temporary" charges of this neuron
  void ResetTempCharge();
  //! reset base charge
  void ResetCharge();
  //! reset all charges
  void ResetChargeAll();
  //! set neuron potential to upper limit
  void setUpperLimit();
  //! input a charge to this neuron, give your polarity so it can be sent on properly
  /*! @param charge The charge to give this neuron
    @param _pol the polarity you have
    @param _sender Is true if you are a sender of cascade
  */
  inline void Charge(CH& _charge,bool _pol, bool _sender);
  //! input a charge to this neuron (potential) without polarity
  /*! @param charge The charge to give this neuron
  */
  void ChargeSimple(CH& _charge);
  //! output a charge, make sure it goes to the correct polarity
  /*! @param _pol Your polarity
    @param _sender Are you a sender?
  */
  inline CH& DisCharge(bool _pol, bool _sender, int a, int cascadeType);
  //! Tell the neuron its saturation level
  void setThreshold(CH& Threshold, CH& Resistance);
  //when you are all done set the vector counter to 0 to clear all chrages
};

#if 0
/* commented this out because the array declaration:

   ContourNeuronProp<float,int,int> NeuronMatrix[][][][]

   fails to compile with gcc 3.4:

   contourNeuronProp.H:134: error: declaration of `NeuronMatrix' as
   multidimensional array must have bounds for all dimensions except the
   first
 */

//! a resizable vector to contain the neuron properties in ContourNeuronProp
/*! this is a container class used to allow ContourNeuronProp for multiple
  iterations.
*/

class ContourNeuronPropVec
{
public:
  //! container class for the properties of each neuron at each iteration
  /*! The vector class is used for multiple iterations */
  ContourNeuronProp<float,int,int> NeuronMatrix[][][][];
  void setSize(int  t,int i, int j, int k);
  int getSizeT();
  int T,I,J,K;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
