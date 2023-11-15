/*!@file Beobot/BeobotVisualCortex.H Header for navigation class */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotVisualCortex.H $
// $Id: BeobotVisualCortex.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef BEOBOTVISUALCORTEX_H_DEFINED
#define BEOBOTVISUALCORTEX_H_DEFINED

#include "Beobot/ImageSpring.H"
#include "Beowulf/Beowulf.H"
#include "Beowulf/TCPmessage.H"
#include "Channels/Jet.H"
#include "Image/ImageSet.H"
#include "Image/PyramidTypes.H"
#include "Image/fancynorm.H"
#include "Util/Assert.H"
#include "Util/Types.H"
#include "rutz/shared_ptr.h"

//! Visual Cortex of a Beobot
/*! this is no longer derived from VisualCortex
 it can be used with one CPU or SMP systems */
class BeobotVisualCortex {
public:
  //! constructor
  BeobotVisualCortex();

  //! initialization
  /*! use beow=NULL for single-CPU processing */
  void init(const int imgw, const int imgh, const int lev_min,
            const int lev_max, const int delta_min, const int delta_max,
            const int smlev, const int nborient, const MaxNormType normtype,
            const int jlev, const int jdepth, const int nbneig,
            nub::soft_ref<Beowulf> beow);

  //! present a new visual input from an existing image
  void newVisualInput(Image< PixRGB<byte> >& scene);

  //! get pointer to scene image, so that we can directly grab it
  Image< PixRGB<byte> >* getScenePtr();

  //! call this to execute all the low-level processing on current scene
  /*! This method will automatically parallelize the processing if
    a non-NULL beowulf was passed at initialization. However, it will
    call processStart() and processEnd() in sequence, waiting for the
    results to be ready before returning. It may be a better idea
    to separately call processStart(), do something, then processEnd().
    For single-CPU operation, process() is the only option. */
  void process(const int frame);

  //! start parallel processing by sending off current frame
  void processStart(const int frame);

  //! end parallel process by receiving results
  void processEnd(const int frame);

  //! not normally used; use process() instead
  void singleCPUprocess(const int frame);

  //! not normally used; use process() instead
  void masterProcess(const int frame);

  //! Call this in an infinite loop on slaves to process & return incoming maps
  void slaveProcess();

  //! not normally used; use process() instead
  void masterCollect();

  //! get most salient location:
  void getWinner(Point2D<int>& win) const;

  //! initialize clustering with "spring method"
  void initSprings( bool initPosMasses );

  //! iterate clustering with "spring method"
  void iterateSprings(const float dt);

  //! returns the clustered image and the position of the track centroid
  void getClusteredImage(Image< PixRGB<byte> >  &clusteredImage,
                         Point2D<int> &supposedTrackCentroid,
                         const Point2D<int> &previousTrackCentroid);

  //! get size of input image
  inline void getInputSize(Point2D<int>& size);

  //! get mass positions
  void getPositions(Image< PixRGB<byte> > &img, const int zoom);

private:
  //! compute a feature for incoming image in message
  void computeFeature(TCPmessage &rmsg, const PyramidType ptyp,
                      const float ori, const int maptype);

  //! compute a feature for two incoming images in message
  void computeFeature2(TCPmessage &rmsg, const PyramidType ptyp,
                      const float ori, const int maptype);

  //! compute feature for given image
  void computeFeature(const Image<float>& fima, const PyramidType ptyp,
                      const float ori, const int id, const int maptype);

  //! compute conspicuity map given a pyramid
  void computeCmap(const ImageSet<float>& pyr, Image<float>& cmap);

  bool initialized;

  // the input scene (retinal image):
  Image< PixRGB<byte> > scene;
  Image<float> prevlum;   // luminance of previous scene, for flicker
  int currframe;

  // the segmented scene:
  ImageSpring< Jet<float> > jets;
  int nbjmap;

  // saliency map input:
  Image<float> sminput;
  int nbcmap;
  Point2D<int> winner;

  // for inter-CPU communication:
  nub::soft_ref<Beowulf> beo;

  // parameters:
  int iw, ih, lmin, lmax, dmin, dmax, sml, nori;
  MaxNormType nortyp;
  int jetlevel, jetdepth;
  int nbneigh;
  rutz::shared_ptr<JetSpec> jetSpec;
};

// ######################################################################
inline void BeobotVisualCortex::getInputSize(Point2D<int>& size)
{
  ASSERT(scene.initialized());
  size.i = scene.getWidth(); size.j = scene.getHeight();
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
