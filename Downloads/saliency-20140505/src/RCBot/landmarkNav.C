/*!@file RCBot/landmarkNav.C navigate the robot based on learned landmark
  positions                                                             */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/landmarkNav.C $
// $Id: landmarkNav.C 13993 2010-09-20 04:54:23Z itti $
//

#include "Image/OpenCVUtil.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Component/ModelManager.H"
#include "Image/CutPaste.H"     // for inplacePaste()
#include "GUI/XWinManaged.H"
#include "Util/Timer.H"
#include "Controllers/PID.H"
#include "Image/PyrBuilder.H"
#include "RCBot/TrackFeature.H"
#include "RCBot/FindLandmark.H"

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

#define LAND_WINSIZE 55

char info[255];
Point2D<int> trackLoc(-1,-1);
Point2D<int> landmarkLoc(-1,-1);
Point2D<int> oldLandmarkLoc(-1,-1);

static bool goforever = true;  //!< Will turn false on interrupt signal

//! Signal handler (e.g., for control-C
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; exit(1); }

// ######################################################################
int main(int argc, char **argv)
{
  MYLOGVERB = LOG_INFO;

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate);  signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  CORBA::ORB_ptr orb = CORBA::ORB_init(argc,argv,"omniORB4");

  // instantiate a model manager (for camera input):
  ModelManager manager("SaliencyMT Tester");

  nub::ref<SaliencyMT> smt(new SaliencyMT(manager, orb, 0));
  manager.addSubComponent(smt);

  // Bot controller
  CORBA::Object_ptr objBotControlRef[10]; int nBotControlObj;
  if (!getMultiObjectRef(orb, "saliency.BotControllers",
                         objBotControlRef, nBotControlObj)){
    LFATAL("Cannot find any object to bind with");
  }
  BotControlServer_var robotController =
    BotControlServer::_narrow(objBotControlRef[0]);

  // SceneRec
  CORBA::Object_ptr objSceneRecRef[10]; int nSceneRecObj;
  if (!getMultiObjectRef(orb, "saliency.SceneRec", objSceneRecRef, nSceneRecObj))
    {
      LFATAL("Cannot find any SceneRec object to bind with");
    }
  SceneRecServer_var sceneRec = SceneRecServer::_narrow(objSceneRecRef[0]);

  // Parse command-line:
  if (manager.parseCommandLine((const int)argc, (const char**)argv, "", 0, 1) == false)
    return(1);

  manager.start();

  PID<float> steer_pid(0.03, 0, 0, -1, 1);

  // initialize the robot controller
  robotController->init();

  short w, h;
  robotController->getImageSensorDims(w,h,0);
  LINFO("Dim %i %i", w, h);

  sprintf(info, "Testing...");

  // start the tracking thread
  TrackFeature trackFeature(smt);

  // start the Landmark finding thread
  FindLandmark findLandmark(sceneRec);
  Image<PixRGB<byte> > landmarkImg;
  bool lookForLandmark = true;
  int leg = 0; // our current leg in the path

  // start with a stopped car
  robotController->setSpeed(0);
  robotController->setSteering(0);

  // Main loop. Get images and send them to the apropriate places
  while(goforever)
    {
      // get the image from the robot
      ImageOrb *imgOrb = robotController->getImageSensor(0);
      Image< PixRGB<byte> > ima;
      orb2Image(*imgOrb, ima);

      LINFO("Send image");
      // send image for tracking
      trackFeature.setImg(ima);

      LINFO("Send image 2");
      // send image to SceneRec to find the landmark we need to track
      findLandmark.setImg(ima);
      if (lookForLandmark){
        landmarkImg = ima;
        lookForLandmark = false;
      }

      LINFO("Compute locations");
      int tempLeg = findLandmark.getLandmark(landmarkLoc);
      if (tempLeg > leg) leg = tempLeg;
      LINFO("Got landmark");

      // if we have a valid landmark location,
      // and it is not the same landmark we had before
      // then update our template tracking
      if ((landmarkLoc.isValid() && oldLandmarkLoc != landmarkLoc) ||
          !trackLoc.isValid())
      {
        LINFO("Setting tracking location");
        trackFeature.setTrackLoc(landmarkLoc, landmarkImg);
        oldLandmarkLoc = landmarkLoc;
        lookForLandmark = true;
      }

      LINFO("Getting tracking");
      // Get the location we are tracking
      trackLoc = trackFeature.getTrackLoc();
      LINFO("Got tracking");

      // get the user input if we do not know which landmark to track
      // and we lost the feature we were tracking
      // increment the leg # when the space bar is pressed
      LINFO("Landmark (%i,%i) track (%i,%i)", landmarkLoc.i, landmarkLoc.j,
            trackLoc.i, trackLoc.j);
      Point2DOrb locOrb;
      int key = robotController->getUserInput(locOrb);
      if (key == 65) // space bar
        leg++;

      if (locOrb.i > 1 && locOrb.j > 1){ // we got input
        landmarkLoc.i = locOrb.i; landmarkLoc.j = locOrb.j;
        if (key != -1 || landmarkLoc.isValid()){
          LINFO("Key %i loc (%i,%i)", key, landmarkLoc.i, landmarkLoc.j);
        }

        // start tracking the given landmark position
        trackFeature.setTrackLoc(landmarkLoc, ima);
        oldLandmarkLoc = landmarkLoc;

        // learn the landmark using SIFT
        DimsOrb winOrb = {LAND_WINSIZE, LAND_WINSIZE};
        sceneRec->trainFeature(*image2Orb(ima), toOrb(landmarkLoc), winOrb, leg);

        //landmarkLoc.i = -1; landmarkLoc.j = -1;
      }

      // drive the robot toward the tracking point
      if (trackLoc.isValid()){

        double steer_cor = steer_pid.update(w/2, trackLoc.i);
        if (steer_cor > 1) steer_cor = 1;
        else if (steer_cor < -1) steer_cor = -1;
        //LINFO("Steer %i %i %f\n", w/2, fixation.i, steer_cor);
        robotController->setSteering(steer_cor);
        //robotController->setSpeed(-0.325);
        robotController->setSpeed(-0.4);
      } else {
        robotController->setSpeed(0);
      }

      sprintf(info, "%.1ffps l=%i", trackFeature.getFps(), leg);
      robotController->setInfo(info, toOrb(trackLoc), toOrb(landmarkLoc));
    }

  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
