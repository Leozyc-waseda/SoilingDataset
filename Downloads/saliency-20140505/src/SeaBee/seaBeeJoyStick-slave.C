/*!@file SeaBee/seaBeeJoyStick-slave */
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/seaBeeJoyStick-slave.C $
// $Id: seaBeeJoyStick-slave.C 10794 2009-02-08 06:21:09Z itti $
//
//////////////////////////////////////////////////////////////////////////


#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Devices/JoyStick.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Util/MathFunctions.H"
#include "Raster/Raster.H"
#include "SeaBee/SubController.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "Media/MediaOpts.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/ShapeOps.H"

#include <unistd.h>
#include <signal.h>
#include <math.h>

#define COM_B_NODE 0
#define INIT_DONE 10001
#define JS_AXIS_UPDATE   20000
#define SUB_GUI_UPDATE   20001
#define ABORT 90000

#define THRUSTER_UP_LEFT 0
#define THRUSTER_UP_RIGHT 4
#define THRUSTER_UP_BACK 2
#define THRUSTER_FWD_RIGHT 3
#define THRUSTER_FWD_LEFT 1

float currentXAxis = 0.0;
float currentYAxis = 0.0;

float currentR = 0.0;
float currentTheta = 0.0;


volatile bool keepGoing = false;
std::list<TCPmessage> inMessages;
std::list<TCPmessage> outMessages;

// gets the messages from COM_B and relay it to the agent manager
void checkInMessages
( nub::soft_ref<Beowulf> beo );

void packageGUIMessage
( nub::soft_ref<SubController> subController,
  Image<PixRGB<byte> > frontImg);

// ######################################################################
//! Signal handler (e.g., for control-C)
void terminate(int s) { LERROR("*** INTERRUPT ***"); keepGoing = false; }

// ######################################################################
// ######################################################################
int main( int argc, const char* argv[] )
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("JoyStick Manager");


  // Instantiate our various ModelComponents:
  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Slave", "BeowulfSlave", false));
  manager.addSubComponent(beo);

  nub::soft_ref<SubController> subController(new SubController(manager));
  manager.addSubComponent(subController);


  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  TCPmessage rmsg;     // buffer to receive messages
  TCPmessage smsg;     // buffer to send messages
  int32 rframe = 0, raction = 0, rnode = -1;

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // let's get all our ModelComponent instances started:
  manager.start();


  while(!beo->receive(rnode, rmsg, rframe, raction, 5));
  int initval  = uint(rmsg.getElementInt32());
  LINFO("got INIT: %d", initval);

  // send a message of ready to go
  smsg.reset(rframe, INIT_DONE);
  beo->send(rnode, smsg);

  keepGoing = true;
  subController->setMotorsOn(true);

  while(keepGoing)
    {
      // check messages recieved from COM_A
      checkInMessages(beo);

      subController->setThruster(THRUSTER_FWD_RIGHT, (int)(clampValue(-currentYAxis + currentXAxis,-100.0,100.0)));
      subController->setThruster(THRUSTER_FWD_LEFT, (int)(clampValue(-currentYAxis - currentXAxis,-100.0,100.0)));

      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE)
        break;

      //grab the images
      GenericFrame input = ifs->readFrame();
      if (!input.initialized())
        break;
      Image<PixRGB<byte> > frontImg = rescale(input.asRgb(), 320, 280);

      packageGUIMessage(subController, frontImg);

      //send GUI information back to master
      if(outMessages.size() > 0)
        {
          beo->send(-1, outMessages.front());
          outMessages.pop_front();
        }

    }

  subController->setMotorsOn(false);
  // we are done
  manager.stop();
  return 0;
}

// ######################################################################
void checkInMessages
( nub::soft_ref<Beowulf> beo)
{
  int32 rframe = 0, raction = 0, rnode = -1;
  TCPmessage rmsg;     // buffer to receive messages

  // get all the messages sent in
  while(beo->receive(rnode, rmsg, rframe, raction, 5))
    {
      LINFO("COM_B: recieving a message: %d", raction);

      // check the purpose of the message
      switch(raction)
        {
        case ABORT:
          {
            keepGoing = false; LINFO("Stop SeaBee COM-B");
            break;
          }

        case JS_AXIS_UPDATE:
          {
            int32 num =  rmsg.getElementInt32();
            float p = rmsg.getElementFloat();
            switch(num)
              {
              case 0:
                LINFO("X-Axis = %f", p);
                currentXAxis = p;
                break;
              case 1:
                LINFO("Y-Axis = %f", p);
                currentYAxis = p;
                break;
              case 2:
                LINFO("Depth = %f", p);
                break;
              default:
                LERROR("Unknown axis event recieved");
              }

            break;
          }
        default:
          LFATAL("Unknown purpose");
        }
    }
}


void packageGUIMessage
( nub::soft_ref<SubController> subController,
  Image<PixRGB<byte> > frontImg)
{
  TCPmessage smsg;     // buffer to send messages

  int32 sframe  = 0;
  int32 saction = SUB_GUI_UPDATE;

  smsg.reset(sframe, saction);

  smsg.addInt32(int32(*(subController->getIntPressurePtr())));
  smsg.addInt32(int32(*(subController->getHeadingPtr())));
  smsg.addInt32(int32(*(subController->getPitchPtr())));
  smsg.addInt32(int32(*(subController->getRollPtr())));
  smsg.addInt32(int32(*(subController->getDepthPtr())));
  smsg.addInt32(int32(*(subController->getThruster_Up_Left_Ptr())));
  smsg.addInt32(int32(*(subController->getThruster_Up_Right_Ptr())));
  smsg.addInt32(int32(*(subController->getThruster_Up_Back_Ptr())));
  smsg.addInt32(int32(*(subController->getThruster_Fwd_Left_Ptr())));
  smsg.addInt32(int32(*(subController->getThruster_Fwd_Right_Ptr())));

  smsg.addImage(frontImg);

  outMessages.push_back(smsg);

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
