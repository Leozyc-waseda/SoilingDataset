/*!@file Beobot/Beobot.H Header file for the main Beobot class */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/Beobot.H $
// $Id: Beobot.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef BEOBOT_H_DEFINED
#define BEOBOT_H_DEFINED

#include "Beobot/BeobotEffectors.H"
#include "Beobot/BeobotMemory.H"
#include "Beobot/BeobotSensors.H"
#include "Beobot/BeobotVisualCortex.H"
#include "Util/Types.H"
#include <netinet/in.h>

//! this class is a complete Beobot
/*! i.e. Visual Cortex, Memory, Sensors and Effectors */
class Beobot {
public:
  //! Constructor
  /*! use NULL for slaves to use the single-CPU version */
  Beobot(const char *slaves, const int imgw, const int imgh,
         const int lev_min, const int lev_max,
         const int delta_min, const int delta_max,
         const int smlev, const int nborient, const MaxNormType normtype,
         const int jlev, const int jdepth, const int nbneig,
         const in_addr_t ip, const short int port);

  //! set new scene from an existing image
  void newVisualInput(Image< PixRGB<byte> >& scene);

  //! get pointer to internal retina, for example to directly framegrab it
  Image< PixRGB<byte> >* getRetinaPtr();

  //! low level pyramid business
  /*! When using parallel processing, this method will send off the current
    frame and wait until all results have come back. For better performance,
    use lowLevelStart() and lowLevelEnd() instead. */
  void lowLevel(const int frame);

  //! start low-level by sending off current frame
  void lowLevelStart(const int frame);

  //! receive results from slave for given frame
  void lowLevelEnd(const int frame);

  //! get most salient location:
  void getWinner(Point2D<int>& win) const;

  //! compute the new "sensation" and update "memory"
  /*! ( given a new scene i.e. after see()
    and someday the state of the sensors )
    -- high level vision but still low level cognitive processing
    takeYourTime is set to 'true' for the first frame which is
    necessary for convergence - see implementation
  */
  void intermediateLevel(bool takeYourTime = false);

  //! compute the new "action" and update "memory"
  /*! -- higher level reasoning */
  void highLevel( void );

  //! take a decision
  /*! based of the recommendation issued by highLevel()
    and passed sensations and actions */
  void decision( void );

  //! just do it
  void action( void );

  //! returns the current clustered layout
  void DEBUGgetClustered(Image< PixRGB<byte> >& im);

  //! returns the current action
  void DEBUGgetCurrentAction( BeobotAction & a )
  { a=currentAction; };

  //! set the current action
  void DEBUGsetCurrentAction( BeobotAction & a )
  { currentAction = a; };

private:
  // "brain"
  BeobotVisualCortex visualCortex;
  BeobotMemory memory;

  // "body"
  BeobotSensors sensors;
  nub::soft_ref<BeobotEffectors> effectors;

  // current action
  BeobotAction currentAction;

  // "corpus callosum"
  nub::soft_ref<Beowulf> beo;

  // parameters:
  int lmin, lmax, dmin, dmax, sml, nori, nortyp;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
