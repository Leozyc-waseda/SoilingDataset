/*!@file CINNIC/CINNIC2.H Test binary for CINNIC neuron */


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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CINNIC/CINNIC2.H $
// $Id: CINNIC2.H 15310 2012-06-01 02:29:24Z itti $
//

#ifndef CINNIC2_H_DEFINED
#define CINNIC2_H_DEFINED

#include "Util/readConfig.H"
#include "Util/log.H"

#include "VFAT/readMatrix.H"

#include "CINNIC/contourRun2.H"
#include "CINNIC/contourDefine.H"
#include "CINNIC/CINNICstatsRun.H"
#include "Image/ConvolutionMap.H"

//this is a simple testing alg. for CINNIC


// ############################################################
// ############################################################
// ##### ---CINNIC2---
// ##### Contour Integration:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

#define CINNIC2_DEC template <unsigned short kernelSize,   \
                              unsigned short scales,       \
                              unsigned short orientations, \
                              unsigned short iterations,   \
                              class FLOAT, class INT>

#define CINNIC2_CLASS CINNIC2<kernelSize,scales,orientations, \
                                       iterations,FLOAT,INT>

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
CINNIC2_DEC
class CINNIC2
{
private:
  //! set true if using frames
  bool CINuseFrameSeries;
  bool CINuseMaps;
  unsigned int CINframe;
  INT CINedgeAtten,CINGnum;
  INT CINlPass,CINgroupSize;
  INT CINstoreArraySize;
  INT CINbaseSize;
  INT CINorignalImageSizeX, CINorignalImageSizeY;
  FLOAT CINGroupTop;
  std::string CINlogto;
  std::string CINsaveto;
  INT CINscaleSize[scales];
  FLOAT CINscaleBias[scales];
  std::vector<Image<FLOAT> > CINFinput;
  std::vector<Image<FLOAT> > CINgroup;
  std::vector<Image<FLOAT> > CINresults;
  std::vector< std::vector< Image<FLOAT> > > CINIresults;
  std::vector< std::vector< Image<FLOAT> > > CINVFinput;
  std::vector<INT> CINgroupCount;
  std::vector<FLOAT> CINgroupTopVec;
  std::vector<
       contourRun2<kernelSize,scales,orientations,iterations,FLOAT,INT> >
       CINcontourRun;
  Image<byte>  CINcombinedSalMap;
  convolutionMap<FLOAT> CINcMap;
public:
  const static unsigned short CINkernelSize   = kernelSize;
  const static unsigned short CINscales       = scales;
  const static unsigned short CINorientations = orientations;
  const static unsigned short CINiterations   = iterations;

#ifdef INVT_USE_CPP11//we need c++ 0X features for this to work
  constexpr static FLOAT      CINsmallNumber  = 0.001F;
#else 
  const static FLOAT          CINsmallNumber  = 0.001F;
#endif

  //! default constructor
  CINNIC2();
  ~CINNIC2();
  //! turn on using movie frames
  void CINtoggleFrameSeries(bool toggle);
  //! load configuration file params
  void CINconfigLoad(readConfig &config);
  //! run an image in CINNIC2 with a neuron template
  void CINrunSimpleImage(const ContourNeuronCreate<FLOAT> &NeuronTemplate,
                         const char fileName[100], unsigned int frame,
                         Image<byte> &input,
                         readConfig &config);
  //! get the orientation filtered pre version of the input image
  void CINgetOrientFiltered(Image<byte> &input);
  //! get the orientation filtered version, but use convolution maps
  void CINgetOrientFilteredMap(Image<byte> &input);
  //! get the scaled representations of the image
  void CINgetScaled();
  //! initalize parameters to run CINNIC2
  void CINinitCINNIC(const std::vector< std::vector<Image<FLOAT> > > &input,
                     readConfig &config,
                     const INT sizeX,
                     const INT sizeY);
  //! initalize groups to run CINNIC2
  void CINcomputeGroups(const std::vector< std::vector<Image<FLOAT> > > &input);
  //! call contourRun2 for each iteration and scale
  void CINrunImage(const ContourNeuronCreate<FLOAT> &NeuronTemplate,
                   const std::vector< std::vector<Image<FLOAT> > > &input,
                   readConfig &config,
                   const std::vector<FLOAT> &GTV);
  //! call contourRun2 for each iteration and scale
  void CINrunImageFrames(const ContourNeuronCreate<FLOAT> &NeuronTemplate,
                         const std::vector< std::vector<Image<FLOAT> > > &input,
                         readConfig &config,
                         const std::vector<FLOAT> &GTV);

};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
