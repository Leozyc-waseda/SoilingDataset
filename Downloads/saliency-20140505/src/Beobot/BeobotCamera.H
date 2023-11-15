/*!@file Beobot/BeobotCamera.H A Beobot camera driver */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotCamera.H $
// $Id: BeobotCamera.H 7953 2007-02-19 22:51:57Z rjpeters $
//

#ifndef BEOBOTCAMERA_H_DEFINED
#define BEOBOTCAMERA_H_DEFINED

#include "Image/Image.H"
#include "Component/ModelComponent.H"
#include "Image/Pixels.H"
#include "Util/Types.H"

class IEEE1394grabber;

//! BeobotCamera event listener
/*! The BeoCamListener is called each time a frame is grabbed from the
  BeobotCamera. This is a virtual base class and users should define
  their own derived classes that will do something when a frame is
  received. */
class BeobotCameraListener {
public:
  //! Destructor
  virtual ~BeobotCameraListener();

  //! A frame was received
  /*! This is called by our grabbing thread and hence will execute in
    that thread. When overloading this function, you should be careful
    to enforce that processing done here will be very quick, or you
    will slow down and possibly choke the grabbing thread. */
  virtual void newFrame(const Image< PixRGB<byte> >& image,
                        const int frame) = 0;
};

//! This class abstracts an asynchronous Beobot camera
/*! This class continually grabs frames (in a thread) and makes them
  available to other classes, under the control of a mutex. By
  default, there is no listener, and users can just asynchronously
  query the BeobotCamera to get the most recent frame. Alternatively,
  users may register a BeobotCameraListener which will be called each
  time a new frame is grabbed. */
class BeobotCamera : public ModelComponent
{
public:
  //! Constructor
  BeobotCamera(OptionManager& mgr,
               const std::string& descrName = "Beobot Camera",
               const std::string& tagName = "BeobotCamera");

  //! Install a callback (listener)
  /*! This callback will be called eahc time a frame is grabbed. */
  void setListener(rutz::shared_ptr<BeobotCameraListener>& listener);

  //! Destructor
  ~BeobotCamera();

  //! Get a current view of the world
  /*! This function returns the last grabbed frame and also a frame number */
  void grab(Image< PixRGB<byte> >& image, int& frame);

  //! Running thread
  /*! Do not call directly, this is called by our grabbing thread */
  void run();

protected:
  //! get started
  void start2();

  //! get stopped
  void stop1();

private:
  nub::soft_ref<IEEE1394grabber> itsFG; // Our frame grabber
  Image< PixRGB<byte> > itsImage;   // Our internal image buffer
  int itsN;                         // frame number
  bool itsKeepgoing;                // true as long as we run
  rutz::shared_ptr<BeobotCameraListener> itsListener;
  pthread_t itsRunner;
  pthread_mutex_t itsLock;
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif
