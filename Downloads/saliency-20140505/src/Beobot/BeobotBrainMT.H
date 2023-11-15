/*!@file Beobot/BeobotBrainMT.H efficient implementation of the
  feature pyramids computation, Saliency, Gist, Shape Estimator,
  for Beobot                                                            */
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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotBrainMT.H $
// $Id: BeobotBrainMT.H 15426 2012-11-02 21:44:22Z siagian $
//

// ######################################################################

#ifndef BEOBOT_BEOBOTBRAINMT_H_DEFINED
#define BEOBOT_BEOBOTBRAINMT_H_DEFINED

// saliency map scale params
// we start at level_min = 1 (not 2)
// because the image is already scaled to 160x120
#define delta_min  3
#define delta_max  4
#define level_min  1
#define level_max  3
#define maxdepth   (level_max + delta_max + 1)
#define normtyp    (VCXNORM_MAXNORM)

 // designated pyramid level of saliency map
#define sml        2

// defaultimage size vars 
#define IMAGE_WIDTH  160
#define IMAGE_HEIGHT 120

// relative feature weights:
#define IWEIGHT 0.7
#define CWEIGHT 1.0
#define OWEIGHT 1.0

// channel action definitions
#define NUM_CHANNELS 7
#define REDGREEN   0
#define BLUEYELLOW 1
#define INTENSITY  2
#define ORI0       3
#define ORI45      4
#define ORI90      5
#define ORI135     6

#define numBBMTthreads 7

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Neuro/VisualCortex.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/ColorOps.H"
#include "Image/PyramidOps.H"
#include "Image/Pixels.H"
#include "GUI/XWinManaged.H"
#include "Raster/Raster.H"
#include "Util/Timer.H"

#include <list>
#include <pthread.h>


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

// ######################################################################
//! Brain of a Beobot
//! this is not derived from Brain
//! specifically used for the Beobot system
class BeobotBrainMT : public ModelComponent
{
public:

  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  BeobotBrainMT(OptionManager& mgr,
                const std::string& descrName = "Beobot Brain MT",
                const std::string& tagName = "BeobotBrainMT");

  //! Destructor
  virtual ~BeobotBrainMT();

  //@}

  // ######################################################################
  //! @name Access functions
  //@{

  //! get the winning channel
  inline int getWinChan(uint i);

  //! get the winning submap number
  inline int getWinSubmapNum(uint i);

  //! Get last inputted image
  inline Image< PixRGB<byte> > getCurrImage();

  //! Get current conspicuity image
  inline Image<float> getCurrCMap(uint cnum);

  //! get salient point for object i
  inline Point2D<int> getSalPoint(uint i);

  //! get the pre-attentive feature vector of the passed in coordinate
  void getFeatures(Point2D<int> p, std::vector<double>& v);

  //! get the object masks
  inline std::vector<Image<byte> > getObjectMask();

  //! get the mask for object i
  inline Image<byte> getObjectMask(uint i);

  //! get the bounding boxes
  inline std::vector<Rectangle> getObjRect();

  //! get the bounding box for object i
  inline Rectangle getObjRect(uint i);

  //! get the salient features for object i
  inline void getSalientFeatures( uint i, std::vector<double>& v);

  //! get the time it takes to produce the gist and saliency features
  inline float getProcessTime();

  //! Get last computed Gist Vector
  Image<double> getGist();

  //! Get last computed Saliency Map
  Image<float> getSalMap();

  //@}

  // ######################################################################
  /*! @name member functions */
  //@{

  //! Setup up with image to be processed and build the job queue
  void input(const Image< PixRGB<byte> > img);

  //! Is the last given input done processing?
  bool outputReady();

  //! For internal thread use: Compute a conspicuity map from an image
  void threadCompute();

  //! get number of objects found in current image
  int getNumSalPoint();

  //! get winning channel index 
  Point2D<int> getWinningChannelIndex(Point2D<int> winner);

  //@}

protected:
  void start1(); //!< get started
  void stop2();  //!< get stopped

private:

  // ######################################################################
  //! @name Compute functions (all the inner-working functions)
  //@{

  //! compute the Color, Intensity, and Orientation (CIO) pyramids
  void computeCIOpyr(Image< PixRGB<byte> > img);

  //! find all the objects in the image
  void findObjects();

  //! compute the salient feature vectors
  void computeSalientFeatures();

  //! scaling dimensions
  Point2D<int> downScaleCoords(Point2D<int> winner, Dims targetDims);

  //! Get the winning map for Shape eastimation
  bool setWinningMap(Point2D<int> winner);

  //! return a binary mask (0 and 255) of the attended object
  //! in saliency map scale
  Image<byte> getCurrObjectMask();

  //! Get the shape estimator mask
  Rectangle getSEBoundingBox();

  //! correct bounding box of object for SIFT recognition
  Rectangle correctBB(Rectangle r, Point2D<int> locn, int w, int h);

  //! display intermediate maps
  //void display();

  //@}

  Image< PixRGB<byte> > itsCurrImg;        //!< current color image
  int itsCurrImgWidth;
  int itsCurrImgHeight;
  Image<float> itsCurrLumImg;              //!< current luminance image
  Image<byte> r, g, b, y;                  //!< current RGBY images
  Image<float> rgimg;
  Image<float> byimg;
  Image<float> itsSalmap;                  //!< output saliency map
  bool gotLum, gotRGBY;

  //! the number of channels we emulate
  uint itsNumChannels;

  //! the channels weights
  std::vector<float> itsChanWeight;

  //! conspicuity maps
  std::vector<Image<float> > itsCMaps;

  //! center surround maps storage before normalization
  //! used for feature comparison
  std::vector<std::vector<Image<float> > > itsRawCSMaps;

  //! center surround maps storage after normalization
  //! used for saliency map
  std::vector<std::vector<Image<float> > > itsCSMaps;

  //! feature maps storage
  std::vector<ImageSet<float> > itsImgPyrs;

  //! gist feature vector
  Image<double> itsGistVector;

  //! total number of gist features
  uint itsGistFeatSize;

  //! the list of winning points
  std::vector<Point2D<int> > itsWinner;

  //! the current list of object segmented region
  std::vector<Rectangle> itsObjRect;

  //! salient feature vector
  std::vector<std::vector<double> > itsSalientFeatures;

  //! index of the winning channel
  std::vector<uint> itsWinningChan;

  //! index of the winning submap of the winning channel
  std::vector<uint> itsWinningSubmapNum;

  //! a copy of the winning map
  std::vector<Image<float> > itsWinningMap;

  //! store the last computed Shape Estimator mask
  //! a binary mask {0,255} in saliency map coordinates
  //! specify the extend of the object
  std::vector<Image<byte> > itsObjectMask;

  //! the structuring element for eroding and dilating
  //! for the chamfer smoothing method
  Image<byte> structEl;

  //! job queue and number of jobs to do
  std::list<jobData> jobQueue;
  uint jobsTodo;
  rutz::shared_ptr<Timer> itsTimer;
  float itsProcessTime;

  //! thread stuff
  pthread_t *worker;
  pthread_mutex_t jobLock;       //!< locking jobQueue
  pthread_mutex_t mapLock;       //!< locking related saliency maps
  pthread_mutex_t gistLock;      //!< locking gist vectors
  pthread_cond_t jobCond;
  uint numWorkers;

  //! debug window
  XWinManaged* itsWin;
};

// ######################################################################
// Implementation for BeobotBrainMT inline functions
// ######################################################################
inline Image< PixRGB<byte> > BeobotBrainMT::getCurrImage()
{ return itsCurrImg; }

inline int BeobotBrainMT::getWinChan(uint i)
{ ASSERT(i < itsWinningChan.size()); return itsWinningChan[i]; }

inline int BeobotBrainMT::getWinSubmapNum(uint i)
{ ASSERT(i < itsWinningSubmapNum.size()); return itsWinningSubmapNum[i]; }

inline Image<float> BeobotBrainMT::getCurrCMap(uint i)
{ ASSERT(i < itsCMaps.size()); return itsCMaps[i]; }

inline int BeobotBrainMT::getNumSalPoint()
{ return itsWinner.size(); }

inline Point2D<int> BeobotBrainMT::getSalPoint(uint i)
{ ASSERT(i < itsWinner.size()); return itsWinner[i]; }

inline std::vector<Image<byte> > BeobotBrainMT::getObjectMask()
{ return itsObjectMask; }

inline Image<byte> BeobotBrainMT::getObjectMask(uint i)
{ ASSERT(i < itsObjectMask.size()); return itsObjectMask[i]; }

inline std::vector<Rectangle> BeobotBrainMT::getObjRect()
{ return itsObjRect; }

inline Rectangle BeobotBrainMT::getObjRect(uint i)
{ ASSERT(i < itsObjRect.size()); return itsObjRect[i]; }

inline void BeobotBrainMT::getSalientFeatures( uint i, std::vector<double>& v)
{
  ASSERT(i < itsSalientFeatures.size());
  v.clear();
  for(uint j = 0; j < itsSalientFeatures[i].size(); j++)
    v.push_back(itsSalientFeatures[i][j]);
}

inline float BeobotBrainMT::getProcessTime()
{ return itsProcessTime; }

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
