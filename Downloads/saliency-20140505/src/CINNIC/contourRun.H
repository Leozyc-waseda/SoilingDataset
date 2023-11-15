/*!@file CINNIC/contourRun.H CINNIC classes */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/contourRun.H $
// $Id: contourRun.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef CONTOURRUN_H_DEFINED
#define CONTOURRUN_H_DEFINED

#include "CINNIC/cascadeHold.H"
#include "CINNIC/contourNeuron.H"
#include "CINNIC/contourNeuronProp.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include <vector>

#define initSize 50 //This is the inital size of the vector

// ############################################################
// ############################################################
// ##### ---CINNIC---
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
//int this neuron. When this neuron runs it then handles requests to hand
//back the charges stored in here so that this neuron can pass a charge
//to another neuron with an appropriate charge.

//##########################################################################

//This class will run the CINNIC neuron on a given Matrix of angle values
//to another node. This is factored with timestep.
//static ContourNeuronPropVec (*NeuronMatrix)[ImageSizeX][ImageSizeY];

//! Run the hyper column on a given image with a given connection template
/*! This is the central iterative process to CINNIC. It takes as input
  a 3D ImageMap and a 4D PropHold made from ContourNeuronCreate. It matches
  each neuron in the hyper column against each other neuron in the
  hyper column. Energy is tranfered to the other neuron based on a product
  of the two neurons current exitation from the ImageMap and the weight of
  thier connections. A negative weight signifies inhabition. Energys are
  allowed to build iteratively in an upper layer SalMap built into this class.
*/
class contourRun
{
private:
  float timestep,maxEnergy,iterations,BaseThreshold;
  float upperLimit,iTrans,fastPlast;
  float GroupTop, GroupBottom, supressionAdd;
  float supressionSub, adaptNeuronThresh,orThresh;
  float adaptNeuronMax,leak,initialGroupVal;
  float passThroughGain;
  float passThroughTaper;
  int cascadeType,adaptType,doFastPlast,lastIterOnly,doTableOnly;
  const char* saveto;
  int dumpSwitchPos;
  const char* imageSaveTo;
  const char* logSaveTo;
  int setImageSizeX, setImageSizeY; //the size of the image
  float energy,cascadeStore,castemp,excMult;
  int cascadeChunk,posCount,negCount,totCount;
  int Groups;
  int InspX, InspY; //Neuron being inspected by current neuron
  int Insp2X, Insp2Y;
  bool pol,sender,Mypol,Mysender;
  int cascadeCounter;
  //! stores values for subsequent iterations
  float *storeVal;
  Image<float> overlay;
  Image<PixRGB<float> > overlayS;
  Image<float> Finput;
  Image<float> SM;
  Image<float> Group;
  Image<int> cascadeSize;
  Image<cascadeHold> ICH;
  std::vector<float> GroupMod;
  std::vector<float> GroupMod2;
  std::vector<float> GroupHold;
  std::vector< Image<float> > SMI;
  std::vector< Image<PixRGB<float> > >cascadeMap;
  std::vector< Image<float> > GroupMap;
  std::vector< Image<float> > imageOpt;
  std::vector< Image<float> > combinedSalMap;
  const Point2D<int> *point;
  std::vector<Image<cascadeHold> > cascadeImage;
  std::vector<cascadeHold> cascade;
  ContourNeuronCreate<float> *Neuron; //copy the generic neuron
  PropHold *Neurons; //hold a property set for each neuron
  ContourNeuronProp<float,int,int> prop;
  std::vector<  ContourNeuronProp<float,int,int> > mat1;
  std::vector< std::vector< ContourNeuronProp<float,int,int> > > mat2;
  std::vector< std::vector< std::vector< ContourNeuronProp<float,int,int> > > > mat3;
  std::vector< std::vector< std::vector< std::vector<
  ContourNeuronProp<float,int,int> > > > > NeuronMatrix;

  //*********************************************************************
  // Private member function
  //*********************************************************************

  //! Set up initial values and optimizations, run before runImage
  void preImage(std::vector< Image<float> > &imageMap,
                ContourNeuronCreate<float> &N);
  //! Like runImage except that a sigmoid is used to approximate neuron fireing rate
  /*! Insted of using neuron firering potential is
    routed through a sigmoid which
    approximates the rate of firering.
    @param Image This is the ImageMap of orientations
    @param N This is the contour neuron template
  */
  void runImageSigmoid(std::vector< Image<float> > &imageMap,
                       ContourNeuronCreate<float> &N, int iter);
  //! Calculate group suppression changes
  /*! find new group weights based on the delta of excitation
    @param Image This is the ImageMap of orientations
    @param N This is the contour neuron template
  */
  void calcGroups(std::vector< Image<float> > &imageMap,
                       ContourNeuronCreate<float> &N, int iter);

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
  inline void iterateConvolve(int iter,std::vector< Image<float> > &imageMap,
                              ContourNeuronCreate<float> &N, const int node = -1);
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
  inline void convolveSimple(int iter,std::vector< Image<float> > &imageMap,
                             ContourNeuronCreate<float> &N,
                             const int a, const int b, const int node = -1);
  //! Calculate f with sigmoid described in Kock, Biophysics of Computation
  /*! @param beta This is a positive constant for the gain rate
    @param v This is the generator potential
    For beta higher numbers mean faster gain. a beta of 1
    will start at approx -1 and
    go to 1 with a v of 0 f will be .5 in all cases
  */
  float sigmoid(float beta, float v);
   //! Another sigmoid function
  /* @param beta this is the ceiling for the function
     @param v this is the potential input
  */
  float sigmoid2(float beta, float v);
  //! This method is used to pre-process for sigmoid based on a threshold
  /*! @param beta The beta param in the sigmoid
    @param v The generator potential at its current level
    @param thresh This is the max threshold v should reach
  */
  float preSigmoid(float v, float thresh, float beta = 1);
  //! reset the energy matrix
  void resetMatrix();
  //! set the image size
  void setImageSize(int X, int Y);
  //! Set configurations for this run
  /*! @param readConfig This is the object that contains the config file*/
  void setConfig(readConfig &config);
public:
  // ######################################################################
  // Public members etc.
  // ######################################################################
  long iterCounter;
  contourRun();
  ~contourRun();
  //! copy the combined sal map to this class for analysis or dumping
  void copyCombinedSalMap(std::vector< Image<float> > &CSM);
  //! set the storage array size for linear components in subsq. iteration
  void setArraySize(long size);
  //! returned the processed salmap for this object at iter
  Image<float> getSMI(int iter);
  //! set number of iterations
  void setIterations(int iter);
  //! find the time to total energy ratio
  void deriveEnergy();
  //! Dump energy values computed with sigmoidal process
  /*!@param filename The base name to give this file, should include path
    @param savefile The tailend of the file name
    @param readConfig the config file object
  */
  void dumpEnergySigmoid(const char* filename, const char* savefile,
                         readConfig &config, Image<float> image,
                         int scaleNo, int scaleTot);
  //! This is a short cut method to run runImage + other methods
  /*! This will run setConfig setImageSize, resetMatrix and runImage
    for you.
    @param Image the processed image map of orientations from ImageMap
    @param N This is the generic neuron template from ContourNeuronCreate
    @param config This is the config file object from readConfig.C
    @param sizeX The size of the image in X
    @param sizeY The size of the image in Y
  */
  void contourRunMain(std::vector< Image<float> > &imageMap,
                      ContourNeuronCreate<float> &N
                      ,readConfig &config, Image<float> &group,
                      int groups,int iter, float groupTop);

};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif
