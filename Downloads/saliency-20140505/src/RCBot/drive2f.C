/*!@file RCBot/drive2f.C drive to a given feature                       */
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
// Primary maintainer for this file: Lior Elazary <lelazary@yahoo.com>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/drive2f.C $
// $Id: drive2f.C 13993 2010-09-20 04:54:23Z itti $
//

#include "Image/OpenCVUtil.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Component/ModelManager.H"
#include "Image/CutPaste.H"     // for inplacePaste()
#include "Util/Timer.H"
#include "Controllers/PID.H"
#include "Image/PyrBuilder.H"
#include "RCBot/TrackFeature.H"

#include <signal.h>
#include <unistd.h>
#include <netdb.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <stdlib.h>

#include "Corba/Objects/BotControlServerSK.hh"
#include "Corba/Objects/SceneRecServerSK.hh"
#include "Corba/ImageOrbUtil.H"
#include "Corba/CorbaUtil.H"

#define LAND_WINSIZE 50

char info[255];
Point2D<int> trackLoc(-1,-1);

static bool goforever = true;  //!< Will turn false on interrupt signal

//! Signal handler (e.g., for control-C)
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; exit(1); }
// ######################################################################
int main(int argc, char **argv)
{
  MYLOGVERB = LOG_INFO;

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  CORBA::ORB_ptr orb = CORBA::ORB_init(argc,argv,"omniORB4");

  // instantiate a model manager (for camera input):
  ModelManager manager("SaliencyMT Tester");

  nub::ref<SaliencyMT> smt(new SaliencyMT(manager, orb, 0));
  manager.addSubComponent(smt);

  //Bot controler
  CORBA::Object_ptr objBotControlRef[1]; int nBotControlObj;
  if (!getMultiObjectRef(orb, "saliency.BotControllers",
                         objBotControlRef, nBotControlObj))
    {
      LFATAL("Can not find any object to bind with");
    }

  BotControlServer_var robotController =
    BotControlServer::_narrow(objBotControlRef[0]);

  // Parse command-line:
  if (manager.parseCommandLine((const int)argc, (const char**)argv, "", 0, 1) == false)
    return(1);

  manager.start();

  PID<float> steer_pid(0.03, 0, 0, -1, 1);

  //init the robot controller
  robotController->init();

  short w, h;
  robotController->getImageSensorDims(w,h,0);
  LINFO("Dim %i %i", w, h);

  sprintf(info, "Testing...");

  //start the tracking thread
  TrackFeature trackFeature(smt);

  int leg = 0; //our current leg in the path

  //start with a stoped car
  robotController->setSpeed(0);
  robotController->setSteering(0);

  //Main loop. Get images and send them to the apropreate places
  while(goforever)
    {
      //get the image from the robot
      ImageOrb *imgOrb = robotController->getImageSensor(0);
      Image< PixRGB<byte> > ima;
      orb2Image(*imgOrb, ima);

      //send image for tracking
      trackFeature.setImg(ima);

      //Get the location we are tracking
      trackLoc = trackFeature.getTrackLoc();

      Point2DOrb locOrb;
      int key = robotController->getUserInput(locOrb);

      if (locOrb.i > 1 && locOrb.j > 1)
        { //we got input
          trackLoc.i = locOrb.i; trackLoc.j = locOrb.j;
          if (key != -1 || trackLoc.isValid())
            {
              LINFO("Key %i loc (%i,%i)", key, trackLoc.i, trackLoc.j);
            }

          //start tracking the given landmark position
          trackFeature.setTrackLoc(trackLoc, ima);
        }

      //Drive the robot toward the tracking point
      if (trackLoc.isValid())
        {
          double steer_cor = steer_pid.update(w/2, trackLoc.i);

          if (steer_cor > 1) steer_cor = 1;
          else if (steer_cor < -1) steer_cor = -1;

          LINFO("Steer %i %i %f\n", w/2, trackLoc.i, steer_cor);
          robotController->setSteering(steer_cor);
          //robotController->setSpeed(-0.325);
        }
      else
        {
          robotController->setSpeed(0);
        }

      sprintf(info, "%.1ffps l=%i", trackFeature.getFps(), leg);
      robotController->setInfo(info, toOrb(trackLoc), toOrb(trackLoc));
    }

  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
