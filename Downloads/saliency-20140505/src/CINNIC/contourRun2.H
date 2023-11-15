/*!@file CINNIC/contourRun2.H CINNIC classes */

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
// Primary maintainer for this file: T Nathan Mundhenk <mundhenk@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/contourRun2.H $
// $Id: contourRun2.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef CONTOURRUN2_H_DEFINED
#define CONTOURRUN2_H_DEFINED

#include "CINNIC/contourNeuron.H"
#include "CINNIC/contourNeuronProp2.H"

#include "Raster/Raster.H"

#include "Image/Pixels.H"
#include "Image/Image.H"

#include "Util/Timer.H"

#include <vector>

// ############################################################
// ############################################################
// ##### ---CINNIC2---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################
//CLASSES:
//NeuronChargeHold
//ContourNeuronProp
//RunNeuron

//###############################################################
//Holder of the basic properties of each contour neuron
//HOW THE HECK DOES THIS CLASS WORK...
//this class stores and releises charges for the CINNIC neurons
//Other neurons will place a charge in into this neuron using the
//Charge method. These charges are stored in a resizable vector
//in a container class called NeuronChargeHold which is then
//stored in a vector called HistoryVector.
//
//When it is a neurons turn to run it does several things. This classes
//part is to take a charge from another neuron that is running and store it
//INT this neuron. When this neuron runs it then handles requests to hand
//back the charges stored in here so that this neuron can pass a charge
//to another neuron with an appropriate charge.

//##########################################################################

//This class will run the CINNIC neuron on a given Matrix of angle values
//to another node. This is factored with timestep.
//static ContourNeuronPropVec (*NeuronMatrix)[ImageSizeX][ImageSizeY];

#define CONTOUR_RUN2_DEC template <unsigned short kernelSize,   \
                                   unsigned short scales,       \
                                   unsigned short orientations, \
                                   unsigned short iterations,   \
                                   class FLOAT, class INT>

#define CONTOUR_RUN2_CLASS contourRun2<kernelSize,scales,orientations, \
                                       iterations,FLOAT,INT>

//! Run the hyper column on a given image with a given connection template
/*! This is the central iterative process to CINNIC. It takes as input
  a 3D ImageMap and a 4D PropHold made from ContourNeuronCreate. It matches
  each neuron in the hyper column against each other neuron in the
  hyper column. Energy is tranfered to the other neuron based on a product
  of the two neurons current exitation from the ImageMap and the weight of
  thier connections. A negative weight signifies inhabition. Energys are
  allowed to build iteratively in an upper layer SalMap built into this class.
*/
CONTOUR_RUN2_DEC
class contourRun2
{
private:
  //! this is set to true if we use a frame series, not a single image
  bool CONTuseFrameSeries;
  const char*    CONTimageOutDir;
  const char*    CONTimageSaveTo;
  const char*    CONTlogSaveTo;
  unsigned short CONTgroups;
  INT CONTadaptType, CONTdoFastPlast, CONTlastIterOnly, CONTdoTableOnly;
  INT CONTsetImageSizeX, CONTsetImageSizeY;
  INT CONTdoGroupSupression, CONTdoPassThrough;
  //! the current iteration frame in the CINNIC
  INT CONTcurrentIter;
  //! if we are using frames this is different than CONTcurrentIter
  INT CONTcurrentFrame;
  //! Store current iteration frame in the CINNIC
  INT CONTstoreCurrentIter;
  //! counts individual iterations over neurons
  unsigned long CONTiterCounter;
  FLOAT CONTtimestep, CONTmaxEnergy;
  FLOAT CONTupperLimit, CONTfastPlast;
  FLOAT CONTgroupTop, CONTgroupBottom, CONTsupressionAdd;
  FLOAT CONTsupressionSub, CONTadaptNeuronThresh, CONTorThresh;
  FLOAT CONTadaptNeuronMax, CONTleak, CONTinitialGroupVal;
  FLOAT CONTpassThroughGain;
  FLOAT CONTenergy, CONTexcMult;
  //! how quickly do we lose fast plasticity (decay term)
  FLOAT CONTplastDecay;
  const Point2D<int> *point;
  //! stores values for subsequent iterations
  Image<FLOAT> CONTsalMap;
  //! stores the map of group membership for hyper columns
  Image<FLOAT> CONTgroup;

  //! base activity for this group - supression
  std::vector<FLOAT> CONTgroupMod;
  //! base activity for this group - 1/supression
  std::vector<FLOAT> CONTgroupMod2;
  //! holds the finite difference for this group from t-1 to t
  std::vector<FLOAT> CONTgroupDelta;
  std::vector< Image<FLOAT> > CONTsalMapIterations;
  std::vector< Image<FLOAT> > CONTgroupMap;
  std::vector< Image<FLOAT> > CONTimageOpt;
  std::vector< Image<FLOAT> > CONTcombinedSalMap;

  ContourNeuronCreate<FLOAT> *CONTneuron; //copy the generic neuron
  PropHold *Neurons; //hold a property set for each neuron

  //! Hold the values for each neuron over time
  std::vector< std::vector< std::vector< std::vector<
  ContourNeuronProp2<FLOAT,INT> > > > > CONTneuronMatrix;

  //! Holds temporally static values for each neuron
  std::vector< std::vector< std::vector<
  staticContourNeuronProp<FLOAT,INT> > > > CONTstaticNeuronMatrix;
  //*********************************************************************
  // Private member function
  //*********************************************************************

  //! reset the charges at iteration iter
  void CONTresetCharge(const INT iter);
  //! Set up initial values and optimizations, run before runImage
  void CONTpreImage(const std::vector< Image<FLOAT> > &imageMap,
                    const ContourNeuronCreate<FLOAT> &N);
  //! sets up CONTimageOpt
  void CONTsetImageOpt(const std::vector< Image<FLOAT> > &imageMap, bool resize);

  //! Like runImage except that a sigmoid is used to approximate neuron fireing rate
  /*! Insted of using neuron firering potential is
    routed through a sigmoid which
    approximates the rate of firering.
    @param Image This is the ImageMap of orientations
    @param N This is the contour neuron template
  */
  void CONTrunImageSigmoid(const std::vector< Image<FLOAT> > &imageMap,
                           const ContourNeuronCreate<FLOAT> &N,
                           const INT iter, const INT lastIter,
                           const INT nextIter, const bool init);
  //! calculate top level saliency map from hyper-column
  void CONTcalcSalMap(const std::vector< Image<FLOAT> > &imageMap,
                      const INT iter);
  //! process saliency map incl. leak and storage
  void CONTprocessSalMap(const std::vector< Image<FLOAT> > &imageMap,
                         const INT iter);
  //! Calculate group suppression changes
  /*! find new group weights based on the delta of excitation
    @param Image This is the ImageMap of orientations
    @param N This is the contour neuron template
  */
  void CONTcalcGroups(const std::vector< Image<FLOAT> > &imageMap,
                      const INT iter, const INT lastIter,
                      const bool init);
  //! set pointers from static neurons to its group membership
  void CONTsetGroupPointers(const std::vector< Image<FLOAT> > &imageMap,
                            const INT a, const INT iter);
  //! compute fast plasticity on neurons
  inline void CONTfindFastPlasticity( const std::vector< Image<FLOAT> > &imageMap,
                                      const INT a, const INT iter);
  //! compute the pass through gain on this image
  inline void CONTfindPassThroughGain(const std::vector< Image<FLOAT> > &imageMap,
                               const INT a, const INT iter,
                               const INT nextIter);
   //! one iterative process, as exaple called from runImage. Math using convolution
  /*! This accounts for one iterations at a time of the hyper column
    it will take the ImageMap and template neuron from ContourNeuronCreate
    and calculate this iterations energy values
    @param iter This is the current iteration number
    @param Image The 3D image map of orientations
    @param N The emplate neuron
    @param node The node number of this process,
    defaults to -1 for singe process operations
  */
  inline void CONTiterateConvolve(const std::vector< Image<FLOAT> > &imageMap,
                                  const ContourNeuronCreate<FLOAT> &N,
                                  const INT node, const INT iter,
                                  const INT lastIter, const bool init);
  //! This is the second part of iterate convolve using the convolutoins
  /*! This accounts for one iterations at a time of the hyper column
    it will take the ImageMap and template neuron from ContourNeuronCreate
    and calculate this iterations energy values
    NOTE: This version excludes cascades
    @param iter This is the current iteration number
    @param Image The 3D image map of orientations
    @param N The emplate neuron
    @param node The node number of this process, defaults to -1
    for singe process operations
  */
  inline void CONTconvolveSimpleInit(
                                 const std::vector< Image<FLOAT> > &imageMap,
                                 const ContourNeuronCreate<FLOAT> &N,
                                 const INT a, const INT b, const INT node,
                                 const INT iter, const INT nextIter);
  //! This is the second part of iterate convolve using the convolutoins
  /*! This accounts for one iterations at a time of the hyper column
    it will take the ImageMap and template neuron from ContourNeuronCreate
    and calculate this iterations energy values
    NOTE: This version excludes cascades
    @param iter This is the current iteration number
    @param Image The 3D image map of orientations
    @param N The emplate neuron
    @param node The node number of this process, defaults to -1
    for singe process operations
  */
   inline void CONTconvolveSimpleOld(
                                 const std::vector< Image<FLOAT> > &imageMap,
                                 const ContourNeuronCreate<FLOAT> &N,
                                 const INT a, const INT b, const INT node,
                                 const INT iter, const INT nextIter);
  //! This is the second part of iterate convolve using the convolutoins
  /*! This accounts for one iterations at a time of the hyper column
    it will take the ImageMap and template neuron from ContourNeuronCreate
    and calculate this iterations energy values
    NOTE: This version excludes cascades
    @param iter This is the current iteration number
    @param Image The 3D image map of orientations
    @param N The emplate neuron
    @param node The node number of this process, defaults to -1
    for singe process operations
  */
  inline void CONTconvolveSimple(const std::vector< Image<FLOAT> > &imageMap,
                                 const ContourNeuronCreate<FLOAT> &N,
                                 const INT a, const INT node,
                                 const INT iter, const INT nextIter);
  //! This is the second part of iterate convolve using the convolutoins
  /*! This accounts for one iterations at a time of the hyper column
    it will take the ImageMap and template neuron from ContourNeuronCreate
    and calculate this iterations energy values
    NOTE: This version excludes cascades
    @param iter This is the current iteration number
    @param Image The 3D image map of orientations
    @param N The emplate neuron
    @param node The node number of this process, defaults to -1
    for singe process operations
  */
  inline void CONTconvolveSimpleFrames(
                                 const std::vector< Image<FLOAT> > &imageMap,
                                 const ContourNeuronCreate<FLOAT> &N,
                                 const INT a, const INT node,
                                 const INT iter, const INT nextIter);
  //! Calculate f with sigmoid described in Kock, Biophysics of Computation
  /*! @param beta This is a positive constant for the gain rate
    @param v This is the generator potential
    For beta higher numbers mean faster gain. a beta of 1
    will start at approx -1 and
    go to 1 with a v of 0 f will be .5 in all cases
  */
  FLOAT CONTsigmoid(const FLOAT beta, const FLOAT v) const;
   //! Another sigmoid function
  /* @param beta this is the ceiling for the function
     @param v this is the potential input
  */
  FLOAT CONTsigmoid2(const FLOAT beta, const FLOAT v) const;
  //! This method is used to pre-process for sigmoid based on a threshold
  /*! @param beta The beta param in the sigmoid
    @param v The generator potential at its current level
    @param thresh This is the max threshold v should reach
  */
  FLOAT CONTpreSigmoid(const FLOAT v, const FLOAT thresh,
                   const FLOAT beta = 1) const;
  //! reset the energy matrix
  void CONTresetMatrix();
  //! set the image size
  void CONTsetImageSize(const INT X, const INT Y);
  //! Set configurations for this run
  /*! @param readConfig This is the object that contains the config file*/
  void CONTsetConfig(readConfig &config);



public:
  // ######################################################################
  // Public members etc.
  // ######################################################################
  const static unsigned short CONTkernelSize   = kernelSize;
  const static unsigned short CONTscales       = scales;
  const static unsigned short CONTorientations = orientations;
  const static unsigned short CONTiterations   = iterations;

  // the values for these static floating-point constants have to be
  // given out of line (see note on
  // http://gcc.gnu.org/onlinedocs/gcc/Deprecated-Features.html: "G++
  // allows static data members of const floating-point type to be
  // declared with an initializer in a class definition. The standard
  // only allows initializers for static members of const integral
  // types and const enumeration types so this extension has been
  // deprecated and will be removed from a future version.").

  //! image values smaller than this are ignored to optimize
  const static FLOAT CONTsmallNumber;
  //! maxmimum value for group supression
  const static FLOAT CONTmaxGroupSupress;
  //! minimum value for group supression
  const static FLOAT CONTminGroupSupress;
  //! maxmimum value for fast plasticity
  const static FLOAT CONTmaxFastPlasticity;
  //! minimum value for fast plasticity
  const static FLOAT CONTminFastPlasticity;

  //! default constructor
  contourRun2();
  //! default destructor
  ~contourRun2();
  //! toggles whether to use frame series for movies
  void CONTtoggleFrameSeries(bool toggle);
  //! copy the combined sal map to this class for analysis or dumping
  void CONTcopyCombinedSalMap(const std::vector< Image<FLOAT> > &CSM);
  //! returned the processed salmap for this object at iter
  Image<FLOAT> CONTgetSMI(const INT iter);
  //! find the time to total energy ratio
  void CONTderiveEnergy();
  //! This is a short cut method to run runImage + other methods
  /*! This will run setConfig setImageSize, resetMatrix and runImage
    for you.
    @param Image the processed image map of orientations from ImageMap
    @param N This is the generic neuron template from ContourNeuronCreate
    @param config This is the config file object from readConfig.C
    @param sizeX The size of the image in X
    @param sizeY The size of the image in Y
  */
  void CONTcontourRunMain(const std::vector< Image<FLOAT> > &imageMap,
                          const ContourNeuronCreate<FLOAT> &N,
                          readConfig &config,
                          const Image<FLOAT> &group,
                          const INT groups,
                          const INT iter,
                          const FLOAT groupTop);

  void CONTcontourRunFrames(const std::vector< Image<FLOAT> > &imageMap,
                            const ContourNeuronCreate<FLOAT> &N,
                            readConfig &config,
                            const Image<FLOAT> &group,
                            const INT groups,
                            const INT frame,
                            const FLOAT groupTop);
  //! display and save fast plasticity activity for iteration
  void CONToutputFastPlasticity(INT iter);
  //! display and save group supression activity for iteration
  void CONToutputGroupSupression(INT iter);
  //! get the current iteration pointer for returning data
  INT CONTgetCurrentIter();
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif
