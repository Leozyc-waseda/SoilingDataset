/*!@file Beobot/BeobotCamera.C A Beobot camera driver */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeobotCamera.C $
// $Id: BeobotCamera.C 15465 2013-04-18 01:45:18Z itti $
//

#include "Beobot/BeobotCamera.H"
#include "Devices/IEEE1394grabber.H"
#include "Image/Pixels.H"

#include <pthread.h>
#include <unistd.h>

// ######################################################################
BeobotCameraListener::~BeobotCameraListener()
{  }

// ######################################################################
void *camera_run(void *c)
{
  BeobotCamera *cc = (BeobotCamera *)c;
  cc->run();
  return NULL;
}

// ######################################################################
BeobotCamera::BeobotCamera(OptionManager& mgr,
                           const std::string& descrName,
                           const std::string& tagName) :
  ModelComponent(mgr, descrName, tagName),
  itsFG(new IEEE1394grabber(mgr)),
  itsImage(), itsN(0), itsKeepgoing(true), itsListener(NULL)
{
  // add our grabber as subcomponent:
  addSubComponent(itsFG);

  // setup some grabbing defaults:
  itsFG->setModelParamVal("FrameGrabberMode", VIDFMT_YUV444);
  itsFG->setModelParamVal("FrameGrabberDims", Dims(160, 120));
  itsFG->setModelParamVal("FrameGrabberNbuf", 2);
  pthread_mutex_init(&itsLock, NULL);
}

// ######################################################################
BeobotCamera::~BeobotCamera()
{ pthread_mutex_destroy(&itsLock); }

// ######################################################################
void BeobotCamera::setListener(rutz::shared_ptr<BeobotCameraListener>& listener)
{ itsListener = listener; }

// ######################################################################
void BeobotCamera::start2()
{
  itsKeepgoing = true;
  itsImage.resize(itsFG->getWidth(), itsFG->getHeight());
  pthread_create(&itsRunner, NULL, &camera_run, (void *)this);
}

// ######################################################################
void BeobotCamera::stop1()
{
  itsKeepgoing = false;
  usleep(300000); // make sure thread has exited
}

// ######################################################################
void BeobotCamera::run()
{
  while(itsKeepgoing)
    {
      itsFG->grabPrealloc(itsImage, &itsLock, &itsN);
      if (itsListener.get()) itsListener->newFrame(itsImage, itsN);
    }
  pthread_exit(0);
}

// ######################################################################
void BeobotCamera::grab(Image< PixRGB<byte> >& image, int& frame)
{
  image.resize(itsImage.getDims());

  pthread_mutex_lock(&itsLock);
  memcpy(image.getArrayPtr(), itsImage.getArrayPtr(), itsImage.getSize() * 3);
  frame = itsN;
  pthread_mutex_unlock(&itsLock);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
