/*!@file Beobot/BeoBotSim.H beobot Simulator */

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
// Primary maintainer for this file: Farhan Baluch <fbaluch@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/BeoBotSim.H $
// $Id: BeoBotSim.H 10794 2009-02-08 06:21:09Z itti $
//

#ifndef BeoBotSim_H_DEFINED
#define BeoBotSim_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "GUI/ViewPort.H"
#include "GUI/XWinManaged.H"
#include "Util/Types.H"
#include <deque>
#include <pthread.h>
#include <stdarg.h>
#include <ode/ode.h>
#include <ode/collision.h>

class BeoBotSim : public ModelComponent
{
public:
  //Constructor
  BeoBotSim(OptionManager& mgr,
          const std::string& descrName = "BeoBotSim",
          const std::string& tagName = "BeoBotSim",
          bool showWorld=false);
  ~BeoBotSim();

  //###########################################################
  void start2();
  void handleWinEvents(XEvent& event);
  Image<PixRGB<byte> > getFrame(int camera);
  void simLoop();
  void startSim(void);
  void setSpeed(float speed);
  void setSteering(float ang);

  dWorldID        getWorld() { return world; }
  dJointGroupID getContactgroup() { return contactgroup; }

 // void nearCallback (void *data, dGeomID o1, dGeomID o2);

private:

  void makeCar();
  void drawCar();

  //arena
  void drawArena();
  void drawGate(const double *pos);
  void drawBuoy(const double *pos);
  void drawBin(const double ori, const double *pos);
  void drawPipeline(const double ori, const double *pos);
  void drawPinger(const double *pos);

  dWorldID        world;
  dSpaceID        space;
  dGeomID          ground;
  dJointGroupID contactgroup;

  dBodyID itsCarBody;
  dGeomID itsCarGeom;

  double itsCarLength;
  double itsCarWidth;
  double itsCarWeight;

  ViewPort *vp;

  double itsSpeed;
  double itsSteeringAng;

  bool itsWorldView;
  bool itsShowWorld;
  XWinManaged *itsWorldDisp;

  pthread_mutex_t itsDispLock;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
