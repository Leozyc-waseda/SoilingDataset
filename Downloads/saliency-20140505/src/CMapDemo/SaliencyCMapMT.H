/*!@file CMapDemo/SaliencyCMapMT.H A class for quick-and-dirty
  saliency mapping integrated with Cmap corba object                    */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// Primary maintainer for this file: Zack Gossman <gossman@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CMapDemo/SaliencyCMapMT.H $
// $Id: SaliencyCMapMT.H 13993 2010-09-20 04:54:23Z itti $
//

#ifndef SALIENCYMT_H_DEFINED
#define SALIENCYMT_H_DEFINED

#include "Image/OpenCVUtil.H"
#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Image/ColorOps.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/MathOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Pixels.H"
#include "Image/PyramidOps.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"
#include "Image/fancynorm.H"
#include "Util/Types.H"
#include "Image/colorDefs.H"
#include "Corba/Objects/CMapSK.hh"
#include "Corba/ImageOrbUtil.H"

#include <pthread.h>
#include <cstdlib>
#include <unistd.h>
#include <list>

// are we using a local linked cmap server or a remote
// set this if we are running on only one computer
#ifndef LocalCMapServer
//#define LocalCMapServer // use a local linked CMapServer instead of remote
#endif

// ######################################################################
struct jobData
{
  jobData() { };

  jobData(const int jt, const PyramidType pt, const float w,
          const float ori = 0.0F) :
    jobType(jt), ptyp(pt), weight(w), orientation(ori) { }

  int jobType;
  PyramidType ptyp;
  float weight;
  float orientation;
};

struct biasData
{
  CMap::BiasSeq redgreen;
  CMap::BiasSeq blueyellow;
  CMap::BiasSeq skinhue;
  CMap::BiasSeq ori0;
  CMap::BiasSeq ori45;
  CMap::BiasSeq ori90;
  CMap::BiasSeq ori135;
  CMap::BiasSeq intensity;
  CMap::BiasSeq flicker;
};


// ######################################################################
class SaliencyMT : public ModelComponent
{
public:

// action definitions CHANGE TO ENUM? FIX?
#define RETINA     1
#define WINNER     2
#define LUMINANCE  3
#define REDGREEN   4
#define BLUEYELLOW 5
#define ORI0       6
#define ORI45      7
#define ORI90      8
#define ORI135     9
#define CMAP       10
#define FLICKER    11
#define INTENSITY  12
#define SKINHUE    13

  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  SaliencyMT(OptionManager& mgr, CORBA::ORB_ptr orb, short saliencyMapLevel,
             const std::string& descrName = "SaliencyMT",
             const std::string& tagName = "SaliencyMT");

  //! Destructor
  ~SaliencyMT();

  //@}

  // ######################################################################
  /*! @name member functions */
  //@{

  //! Setup up with image to be processed and build the job queue
  void newInput(const Image< PixRGB<byte> > img);

  //! Is the last given input done processing?
  bool outputReady();

  //! Get last computed output
  Image<float> getOutput();

  //! blocking call for getting the saliency map
  Image<float> getSMap(Image< PixRGB<byte> > img);

  //! For internal thread use: Compute a conspicuity map from an image
  void computeCMAP();
  //@}

  //! Set the bias values for the given cmap type
  void setBias(int type, std::vector<float> &bias);

  //! do we bias the saliency map or not
  void setBiasSM(bool val);

  //! get the bias values for the given cmap type at the specified location
  void getBias(Image<PixRGB<byte> > &img, std::vector<float> &bias, int type, Point2D<int> &loc);

  //!Number of cmap objects that we have running
  int nCmapObj;
  //TODO: watch for overflow
  CORBA::Object_ptr CMap_ref[100]; //100 is the limit for now

  //set the saliency Map Level
  void setSaliencyMapLevel(const short saliencyMapLevel);

  Image<float> cmaps[14];        //hold the cmaps in memory

  void setSMBias(ImageSet<float> &bias);

protected:
  OModelParam<uint> itsNumThreads; //!< Number of worker threads to use

  void start1(); //!< get started
  void stop2();  //!< get stopped

  //!The current cmap object to send the request to
  CMap_ptr getCmapRef();

private:
  Image< PixRGB<byte> > colima;     // current color image
  Image<float> lum;        //curent luminance image
  Image<byte> r, g, b, y;  //curent RGBY images
  Image<float> prev;       // previous float image
  Image<float> outmap;     // output saliency map
  Image<float> skinima;    //skin hue map
  bool gotLum, gotRGBY, gotSkin;

  std::list<jobData> jobQueue;
  uint jobsTodo;

  // thread stuff
  pthread_t *worker;
  pthread_mutex_t jobLock, mapLock, jobStatusLock;
  pthread_cond_t jobCond;
  pthread_cond_t jobDone;
  uint numWorkers;

  // bias stuff
  biasData cmapBias;
  bool biasSM;        // do we bias the saliency map
  ImageSet<float> SMBias;

};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
