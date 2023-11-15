/*!@file contourNeuronProp2.H CINNIC classes */
//! Holds the property for each neuron
/*! this class is used to process and hold the membrane potential
  of each neuron. It keeps track of total potential as well as
  potential that is "up" or "down" that is, to simulate the
  effects of tranfering energy from neuron to another neuron it
  holds energy in either a positive or negative energy bank.
*/

#include "CINNIC/contourNeuron.H"
#include "CINNIC/contourDefine.H"
#include "CINNIC/staticContourNeuronProp.H"

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/contourNeuronProp2.H $
// $Id: contourNeuronProp2.H 6393 2006-03-26 00:57:36Z rjpeters $
//

// ############################################################
// ############################################################
// ##### ---CINNIC2---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

#ifndef CONTOURNEURONPROP2_H_DEFINED
#define CONTOURNEURONPROP2_H_DEFINED

#include "CINNIC/staticContourNeuronProp.H"

//! Holds generally temporal neuron properties
template <class FLOAT, class INT> class ContourNeuronProp2
{
private:
  //! This neurons charge
  FLOAT            CNP_charge;
  //! store fast plasticity term
  FLOAT            CNP_fastPlasticity;
  //! pointer to this neurons temporally static values
  staticContourNeuronProp<FLOAT,INT> *CNP_staticMap;
public:

  //! default constructor on this neuron
  ContourNeuronProp2();
  //! destructor
  ~ContourNeuronProp2();
  //! link this dynamic neuron to its static component
  inline void CNP_linkToStaticMap(staticContourNeuronProp<FLOAT,INT>
                                  *staticMap);
  //! return this neurons charge
  inline FLOAT CNP_getCharge() const;
  //! reset base charge
  inline void CNP_resetCharge();
  //! set the charge on this neuron
  inline void CNP_chargeSimple(const FLOAT _charge);
  //! set the fast plasticity of this neuron
  inline void CNP_setFastPlast(const FLOAT plast);
  //! get this neurons fast plasticity
  inline FLOAT CNP_getFastPlast() const;
  //! compute supression on this neuron
  inline FLOAT CNP_computeSupress(const FLOAT inputVal,
                                  const FLOAT groupMod) const;
  //! compute Excitation on this neuron
  inline FLOAT CNP_computeExcite(const FLOAT inputVal,
                                 const FLOAT groupMod) const;
};

//#################################################################
template <class FLOAT, class INT>
ContourNeuronProp2<FLOAT,INT>::ContourNeuronProp2()
  :
  CNP_charge(),
  CNP_fastPlasticity(),
  CNP_staticMap(0)
{}

//#################################################################
template <class FLOAT, class INT>
ContourNeuronProp2<FLOAT,INT>::~ContourNeuronProp2()
{}

//#################################################################
template <class FLOAT, class INT>
inline void ContourNeuronProp2<FLOAT,INT>::CNP_linkToStaticMap(
                                          staticContourNeuronProp<FLOAT,INT>
                                          *staticMap)
{
  CNP_staticMap = staticMap;
}

//#################################################################
template <class FLOAT, class INT>
inline FLOAT ContourNeuronProp2<FLOAT,INT>::CNP_getCharge() const
{
  return CNP_charge;
}

//#################################################################
template <class FLOAT, class INT>
inline void ContourNeuronProp2<FLOAT,INT>::CNP_resetCharge()
{
  CNP_charge = 0;
}

//#################################################################
template <class FLOAT, class INT>
inline void ContourNeuronProp2<FLOAT,INT>::CNP_chargeSimple(const FLOAT charge)
{
  CNP_charge += charge;
}

//#################################################################
template <class FLOAT, class INT>
inline void ContourNeuronProp2<FLOAT,INT>::CNP_setFastPlast(const FLOAT plast)
{
  CNP_fastPlasticity = plast;
}

//#################################################################
template <class FLOAT, class INT>
inline FLOAT ContourNeuronProp2<FLOAT,INT>::CNP_getFastPlast() const
{
  return CNP_fastPlasticity;
}


//#################################################################
template <class FLOAT, class INT>
inline FLOAT ContourNeuronProp2<FLOAT,INT>::CNP_computeSupress(
                                            const FLOAT inputVal,
                                            const FLOAT groupMod) const
{
  return ((groupMod/CNP_fastPlasticity) * inputVal);
  //return ((groupMod - CNP_fastPlasticity) * inputVal);
}

//#################################################################
template <class FLOAT, class INT>
inline FLOAT ContourNeuronProp2<FLOAT,INT>::CNP_computeExcite(
                                            const FLOAT inputVal,
                                            const FLOAT groupMod) const
{
  return ((CNP_fastPlasticity/groupMod) * inputVal);
  //return ((CNP_fastPlasticity - groupMod) * inputVal);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif
