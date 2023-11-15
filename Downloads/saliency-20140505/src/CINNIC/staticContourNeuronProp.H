/*!@file staticContourNeuronProp.H CINNIC classes */
//! Holds the property for each neuron
/*! this class is used to process and hold the membrane potential
  of each neuron. It keeps track of total potential as well as
  potential that is "up" or "down" that is, to simulate the
  effects of tranfering energy from neuron to another neuron it
  holds energy in either a positive or negative energy bank.
*/


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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/staticContourNeuronProp.H $
// $Id: staticContourNeuronProp.H 5270 2005-08-09 03:56:55Z mundhenk $

// ############################################################
// ############################################################
// ##### ---CINNIC2---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

#ifndef STATICCONTOURNEURONPROP2_H_DEFINED
#define STATICCONTOURNEURONPROP2_H_DEFINED

//! Holds generally non-temporal neuron properties
template <class FLOAT, class INT> class staticContourNeuronProp
{
private:
  //! this is how many other active neurons I connect to
  unsigned int     sCNP_activeNeuronCount;
  //! This neurons unique ID
  INT              sCNP_ID;
  //! group modifier, a pointer
  FLOAT            *sCNP_groupMod;
  //! list of whether a neuron interaction is excite or supress
  std::vector<bool>          sCNP_otherNeuronPol;
  //! list of coordinates to other active connected neurons
  std::vector<unsigned char> sCNP_otherNeuronAlpha;
  //! list of coordinates to other active connected neurons
  std::vector<unsigned char> sCNP_otherNeuron_i;
  //! list of coordinates to other active connected neurons
  std::vector<unsigned char> sCNP_otherNeuron_j;
  //! list of weight computed between neurons
  std::vector<FLOAT>         sCNP_weightStoreValue;

public:
  //! the amount to resize any vector when resize is called
  const static int sCNP_vecSize = 32;
  //! default constructor
  staticContourNeuronProp();
  //! destructor
  ~staticContourNeuronProp();
  //! set the ID of this neuron
  inline void sCNP_setID(const INT ID);
  //! get this neurons ID
  inline INT sCNP_getID() const;
  //! set a local copy of this neurons group modifer
  inline void sCNP_setGroupMod(FLOAT *gmod);
  //! get a local copy of this neurons group modifer
  inline FLOAT* sCNP_getGroupMod() const;
  //! insert another neuron as active to this one
  inline void sCNP_insertStoreList(const unsigned char a,
                                   const unsigned char i,
                                   const unsigned char j,
                                   const bool polarity,
                                   const FLOAT storeVal);

  //! get polarity to other neuron
  inline bool sCNP_getOtherNeuronPol(           unsigned int i) const;
  //! get the weight value to the other neuron
  inline FLOAT sCNP_getWeightStoreVal(          unsigned int i) const;
  //! get coords to the other neuron
  inline unsigned char sCNP_getOtherNeuronAlpha(unsigned int i) const;
  //! get coords to the other neuron
  inline unsigned char sCNP_getOtherNeuron_i(   unsigned int i) const;
  //! get coords to the other neuron
  inline unsigned char sCNP_getOtherNeuron_j(   unsigned int i) const;
  //! get how many active neurons there are
  inline unsigned int sCNP_getActiveNeuronCount() const;
};

//#################################################################
template <class FLOAT, class INT>
inline staticContourNeuronProp<FLOAT,INT>::staticContourNeuronProp()
{
  sCNP_activeNeuronCount = 0;

  sCNP_otherNeuronAlpha.resize(sCNP_activeNeuronCount + sCNP_vecSize,0);
  sCNP_otherNeuron_i.resize(sCNP_activeNeuronCount    + sCNP_vecSize,0);
  sCNP_otherNeuron_j.resize(sCNP_activeNeuronCount    + sCNP_vecSize,0);
  sCNP_otherNeuronPol.resize(sCNP_activeNeuronCount   + sCNP_vecSize,false);
  sCNP_weightStoreValue.resize(sCNP_activeNeuronCount + sCNP_vecSize,0.0F);
}

//#################################################################
template <class FLOAT, class INT>
inline staticContourNeuronProp<FLOAT,INT>::~staticContourNeuronProp()
{}

//#################################################################
template <class FLOAT, class INT>
inline void staticContourNeuronProp<FLOAT,INT>::sCNP_setID(const INT ID)
{
  sCNP_ID = ID;
}

//#################################################################
template <class FLOAT, class INT>
inline INT staticContourNeuronProp<FLOAT,INT>::sCNP_getID() const
{
  return sCNP_ID;
}

//#################################################################
template <class FLOAT, class INT>
inline void staticContourNeuronProp<FLOAT,INT>::sCNP_setGroupMod(FLOAT *gmod)
{
  sCNP_groupMod = gmod;
}

//#################################################################
template <class FLOAT, class INT>
inline FLOAT *staticContourNeuronProp<FLOAT,INT>::sCNP_getGroupMod() const
{
  return sCNP_groupMod;
}

//#################################################################
template <class FLOAT, class INT>
inline void staticContourNeuronProp<FLOAT,INT>::sCNP_insertStoreList(
                                           const unsigned char a,
                                           const unsigned char i,
                                           const unsigned char j,
                                           const bool polarity,
                                           const FLOAT storeVal)
{
  if(sCNP_activeNeuronCount == (sCNP_weightStoreValue.size() - 1))
  {
    sCNP_otherNeuronAlpha.resize(sCNP_activeNeuronCount + sCNP_vecSize);
    sCNP_otherNeuron_i.resize(sCNP_activeNeuronCount    + sCNP_vecSize);
    sCNP_otherNeuron_j.resize(sCNP_activeNeuronCount    + sCNP_vecSize);
    sCNP_otherNeuronPol.resize(sCNP_activeNeuronCount   + sCNP_vecSize);
    sCNP_weightStoreValue.resize(sCNP_activeNeuronCount + sCNP_vecSize);
  }

  sCNP_otherNeuronAlpha[sCNP_activeNeuronCount] = a;
  sCNP_otherNeuron_i[sCNP_activeNeuronCount]    = i;
  sCNP_otherNeuron_j[sCNP_activeNeuronCount]    = j;
  sCNP_otherNeuronPol[sCNP_activeNeuronCount]   = polarity;
  sCNP_weightStoreValue[sCNP_activeNeuronCount] = storeVal;

  sCNP_activeNeuronCount++;
}

//#################################################################
template <class FLOAT, class INT> inline bool
staticContourNeuronProp<FLOAT,INT>::sCNP_getOtherNeuronPol(unsigned int i) const
{
  return sCNP_otherNeuronPol[i];
}

//#################################################################
template <class FLOAT, class INT> inline FLOAT
staticContourNeuronProp<FLOAT,INT>::sCNP_getWeightStoreVal(unsigned int i) const
{
  return sCNP_weightStoreValue[i];
}

//#################################################################
template <class FLOAT, class INT> inline unsigned char
staticContourNeuronProp<FLOAT,INT>::sCNP_getOtherNeuronAlpha(unsigned int i) const
{
  return sCNP_otherNeuronAlpha[i];
}

//#################################################################
template <class FLOAT, class INT> inline unsigned char
staticContourNeuronProp<FLOAT,INT>::sCNP_getOtherNeuron_i(unsigned int i) const
{
  return sCNP_otherNeuron_i[i];
}

//#################################################################
template <class FLOAT, class INT> inline unsigned char
staticContourNeuronProp<FLOAT,INT>::sCNP_getOtherNeuron_j(unsigned int i) const
{
  return sCNP_otherNeuron_j[i];
}
//#################################################################
template <class FLOAT, class INT> inline unsigned int
staticContourNeuronProp<FLOAT,INT>::sCNP_getActiveNeuronCount() const
{
  return sCNP_activeNeuronCount;
}


#endif
