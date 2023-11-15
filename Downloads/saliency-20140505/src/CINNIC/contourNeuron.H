/*!@file CINNIC/contourNeuron.H CINNIC classes */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/contourNeuron.H $
// $Id: contourNeuron.H 4663 2005-06-23 17:47:28Z rjpeters $
//

#ifndef CONTOURNEURON_H_DEFINED
#define CONTOURNEURON_H_DEFINED

#include "Util/readConfig.H"
#include "CINNIC/contourDefine.H"

// ############################################################
// ############################################################
// ##### ---CINNIC---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################
//CLASSES:
//PropHold
//ContourNeuronCreate

extern float AngleDropOff,AngleSupress,SupMult,OrthA,OrthB,OrthMult,valueCutOff;
extern float coLinearCutOff;
extern float NeuronExSize,NeuronSupSize,NeuronSupStart,NeuronOrthSize;
extern float CoLinearDiff;
extern float NeuralAngles[AnglesUsed];
extern float XCenter, YCenter, ExReach, SupReach,OrthReach;

//#############################################################
//! this is a container class for the property at each hyper pixel for ContourNeuronCreate
/*! This class is used to hold the properties of each hyper pixel in in the 4D
  neural template. each of the floats used here are essentilly weights used
  when a pixel in the ImageMap is run against each of the other pixels. In
  essence it simulates the connection weights of each neuron to each other
  neuron. This is used by ContourNeuronCreate.
*/

class PropHold
{
public:
  //! misalignement modifier
  float mis;
  //! Beta angle modifier
  float ang;
  //! Alpha angle modifier
  float ang2;
  //! Aux angle modifer
  float ang3;
  //! Product of Alpha and Beta
  float angAB;
  //! Product of Alpha,Beta and distance
  float angABD;
  //! Generic distance modifier
  float dis;
  //! Excitation distance modifier
  float Edis;
  //! Supression distance modifier
  float Sdis;
  //! orthogonal distance modifier
  float Odis;
  //Mis-alignment of other neuron, angle to other neuron
  //(ang = 0-1-0,ang2 = 0-180), distance to other neuron
  //! polarity of hyper pixel
  bool pol;
  //! True if hyper pixel can pass on energy
  bool sender;
  //! true if this pixel is co-linear to the perfered angle
  bool coLinear;
  //! true if this pixel is fully orthogonoal to the perfered angle
  bool fullOrth;
  //! True if this hyper pixel is non-zero
  bool zero; //if true then neuron may recieve a charge, quad determins if
  //sender or reciever . zero is true if contains no data in this element
  //! This is true if the element is and orthogonal excitation connection
  bool orth;

};

// ############################################################

//! This class will create the neuron connection template
/*! This class creates the 4D neuron template. It processes
  the excitation and inhabition each neuron should have to
  another neuron given the property of the two neurons.
  The result is a set of weights. The weight for each
  hyperpixel is held in PropHold. The entire set is held
  in a 4D array called FourDNeuralMap.
*/

template <class TC2> class ContourNeuronCreate
{
private:
  float hold,theta,phi,OT;
  bool stop;
  float distanceX, distanceY,distance,AngleAlpha,AngleBeta;
  float Angle, AngleA;
  float ii,jj,kk,ll;
  float one;
  float factor,factor2,OFactorA,OFactorB;
  float foo;
  float bar;
  float FloatAngleCenter;
public:
  //! create the 4D neuron map, return that map
  PropHold FourDNeuralMap[AnglesUsed][AnglesUsed][XSize+1][YSize+1];
  //! find the ratio of XMax to X from 0 to 1
  float FindRatio(TC2& XMax,TC2& X);
  //! find the length from center to coord X,Y
  float FindDistance(TC2& PiX, TC2& PiY);
  //! Calculate the where the center pixel is
  void FindCenter();
  //! calculate the reach of excitation or inhabition
  void FindReach();
  //! find the angle to X,Y from the Center
  float FindAngleToRad(TC2& X, TC2& Y);
  //! Transform angle from radians to degrees
  float FindRadToDeg(TC2& r);
  //! Convert along linear function y=mx+b
  float FindLinear(TC2& m,TC2& n,TC2& o,TC2& x,TC2& b);
  //! common sigmoidal function
  float sigmoid(TC2& beta, TC2& v);
  //! Return the completed neuron
  PropHold GetCompleteNeuron();
  //! Create the neuron using config file parameters
  /*! @param &config This is the config file object contaning parameters
   */
  void CreateNeuron(readConfig &config);
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
