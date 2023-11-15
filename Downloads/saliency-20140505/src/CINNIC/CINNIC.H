/*!@file CINNIC/CINNIC.H Test binary for CINNIC neuron */
#include "CINNIC/contourRun.H"
#include "Util/readConfig.H"
#include "Util/log.H"
#include "CINNIC/contourDefine.H"
#include "CINNIC/CINNICstatsRun.H"

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/CINNIC.H $
// $Id: CINNIC.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef CINNIC_H_DEFINED
#define CINNIC_H_DEFINED

//this is a simple testing alg. for CINNIC


// ############################################################
// ############################################################
// ##### ---CINNIC---
// ##### CINNICtest:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################
//! Main class to run an image in CINNICtest for contours
/*! This is the central class at this point for running CINNIC.
  It works by creating a CINNIC template for neuron connections
  in contourNeuron. It then imports an image as defines from
  the command line. The image is processed for orientations
  using the orientedFilter in Image as well as a few other
  processes. The image is then reduced and a 3D map of the
  image created and stored in ImageMap. This map and the
  neuron are then fed into contourRun which runs the 3D image
  against the template neuron. Output is produced in many ways
  including several raw data dumps and image dumps.
*/

class CINNIC
{
private:
  const char* logto;
  const char* saveto;
  ContourNeuronCreate<float> *CINNICNeuronTemplate;
  int dumbchoice, SIZEX,SIZEY, lastIterOnly, doBias;
  float temp,min,max,avg,mean,std,thresh,GroupTop,compGain;
  //! use a gabor filter instead of oriented filter
  int doGaborFilter;
  //! Parameters for using the gabor filter
  float Gstddev,Gperiod,Gphase,Gtheta,GsigMod,Gamplitude;
  float scale1,scale2,scale3;
  contourRun* RN;
  CINNICstatsRun cinnicStats;
  std::vector<Image<float> > Finput;
  std::vector<Image<byte> > combinedSalMap;
  std::vector<Image<byte> > combinedSalMapMax;
  std::vector<Image<byte> > combinedSalMapMin;
  std::vector< std::vector< Image<float> > > VFinput;
  Image<float> Original;
  Image<float> floatImage;
  Image<float> orientComposite;
  Image<float> output[AnglesUsed][AnglesUsed];
  Image<PixRGB <float> >Poutput[AnglesUsed][AnglesUsed];
  PixRGB<byte> pixel;
  std::vector< Image<float> > group;
  std::vector<int> groupCount;
  std::vector<float> groupTopVec;
  const Point2D<int> *point;
  Point2D<int> *Toint;
  float dev;
  int SX,SY,edge,Amp,cheatVal,cheatNum;
  int dumpImage,redOrder,lPass,reduction,groupSize;
  int scalesNumber,iterations,addNoise;
  int doNerdCam,doTableOnly;
  int origX,origY; //The images original size
  long storeArraySize;
  float preOrientFilterNoise,Gnum;
  //! bias for angles (see polat,sagi Spat. Vis. 1994)
  float NeuralAnglesBias[AnglesUsed];
  //! The difference between the biases
  float BiasDiff;
  //! compensation for 45 degree angles on the grid
  float GridBias;
  //********************************************************
  // Private member functions
  //********************************************************
  //! Load config values for this class
  void configLoad(readConfig &config);
  //! Find bias towards angles (verticle/diagonal/horizontal)
  void findNeuralAnglesBias();
  //! pre-process image, do orientation filtereing
  void preProcessImage(ContourNeuronCreate<float> &NeuronTemplate
                      ,Image<byte> &input,readConfig &config);
  //! Set up to run a nested vector image in contourRun
  void preImage(std::vector< std::vector<Image<float> > > &input
                ,readConfig &config, int sizeX, int sizeY);
  //! run the nested vector image in contourRun
  void runImage(ContourNeuronCreate<float> &NeuronTemplate,
                std::vector< std::vector<Image<float> > > &input
                ,readConfig &config, std::vector<float> &GTV);
  //! send the results to disk
  void getResults(readConfig &config);
public:
  //********************************************************
  // Public stuff
  //********************************************************
  //! View the output visualy of the contour neuron templates
  void viewNeuronTemplate(ContourNeuronCreate<float> &NeuronTemplate
                          ,readConfig &config);
  //! Aux Image holder
  Image<byte> Ninput;
  //! Run an image (this is the important method)
  void RunSimpleImage(ContourNeuronCreate<float> &NeuronTemplate
                      ,Image<byte> &input,readConfig &config);
  //! test a simple image against itself to find the value of convolution
  void convolveTest(ContourNeuronCreate<float> &NeuronTemplate
                    ,readConfig &config, Image<float> &testImage);
  //! This is the file name as supplied by argv
  const char* filename;
  //! this is what to save output as
  const char* savefilename;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
