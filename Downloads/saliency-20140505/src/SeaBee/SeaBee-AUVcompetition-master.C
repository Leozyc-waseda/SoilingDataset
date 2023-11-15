/*!@file SeaBee/SeaBee-AUVcompetition-master.C main 2007 competition code
  Run seabee-AUVcompetition-master at CPU_A
  Run seabee-AUVcompetition        at CPU_B                             */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/SeaBee-AUVcompetition-master.C $
// $Id: SeaBee-AUVcompetition-master.C 10794 2009-02-08 06:21:09Z itti $
//
//////////////////////////////////////////////////////////////////////////

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Raster/Raster.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "GUI/XWinManaged.H"

#include "Media/MediaOpts.H"
#include "Devices/DeviceOpts.H"
#include "Raster/GenericFrame.H"

#include "Image/CutPaste.H"

#include "AgentManagerA.H"
#include "SubGUI.H"
#include "Globals.H"
#include "SubController.H"

#include <signal.h>

#define SIM_MODE false

volatile bool goforever = false;

// gets the messages from COM_B and relay it to the agent manager
void checkInMessages
( nub::soft_ref<Beowulf> beo,
  nub::ref<AgentManagerA> agentManager);

// gets the out messages from the agent manager and send it to COM_B
void checkOutMessages
( nub::ref<AgentManagerA> agentManager,
  nub::soft_ref<Beowulf> beo);

// package an agent manager command to a TCP message to send
void packageAgentManagerCommand
(nub::ref<AgentManagerA> agentManager,
 rutz::shared_ptr<AgentManagerCommand> agentManagerCommand,
 TCPmessage  &smsg);

// unpackage an ocean object to a TCP message
rutz::shared_ptr<SensorResult> unpackageToSensorResult
( TCPmessage  rmsg );

// send new image to COM_B if in simMode
void sendImageUpdate
( Image<PixRGB<byte> > img, nub::soft_ref<Beowulf> beo );

// ######################################################################
//! Signal handler (e.g., for control-C)
void terminate(int s)
{
  LERROR("*** INTERRUPT ***");
  goforever = false;
  exit(1);
}

// ######################################################################
int main( int argc, const char* argv[] )
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("SeaBee 2008 Competition Master");

  // Instantiate our various ModelComponents:
  nub::soft_ref<Beowulf>
   beo(new Beowulf(manager, "Beowulf Master", "BeowulfMaster", true));
  manager.addSubComponent(beo);

#if SIM_MODE == false
  nub::soft_ref<InputFrameSeries> ifs;

  ifs.reset(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);
#endif

  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::soft_ref<SubGUI> subGUI(new SubGUI(manager));
  manager.addSubComponent(subGUI);

  nub::soft_ref<SubController> subController(new SubController(manager, "SubController", "SubController", SIM_MODE));
  manager.addSubComponent(subController);

  // create an Agent Manager
  nub::ref<AgentManagerA> agentManager(new AgentManagerA(subController,manager));
  manager.addSubComponent(agentManager);

  manager.exportOptions(MC_RECURSE);

  manager.setOptionValString(&OPT_InputFrameSource, "V4L2");
  manager.setOptionValString(&OPT_FrameGrabberMode, "YUYV");
  manager.setOptionValString(&OPT_FrameGrabberDims, "1024x576");
  manager.setOptionValString(&OPT_FrameGrabberByteSwap, "no");
  manager.setOptionValString(&OPT_FrameGrabberFPS, "30");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  int w, h;

#if SIM_MODE == false
    w = ifs->getWidth();
    h = ifs->getHeight();
#else
    w = 320;
    h = 240;
#endif

  std::string dims = convertToString(Dims(w, h));

  manager.setOptionValString(&OPT_InputFrameDims, dims);

  manager.setModelParamVal("InputFrameDims", Dims(w, h),
                           MC_RECURSE | MC_IGNORE_MISSING);

  TCPmessage rmsg;     // buffer to receive messages
  TCPmessage smsg;     // buffer to send messages

  int32 rframe = 0, raction = 0, rnode = 0;

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // let's do it!
  manager.start();

  // send init params to COM B to initialize contact
  smsg.reset(0, INIT_COMM);
  smsg.addInt32(int32(2008));
  beo->send(COM_B_NODE, smsg);

  // SYNCHRONIZATION: wait until COM B is ready
  LINFO("Waiting for COM_B...");
  rnode = COM_B_NODE;

  while(!beo->receive(rnode, rmsg, rframe, raction, 5));

  rmsg.reset(rframe, raction);
  LINFO("COM_B(%d) is ready", rnode);

  //eventually make this a command-line param
  bool competitionMode = false;

  // Setup GUI if not in competition mode
  if(!competitionMode)
    {
      subGUI->startThread(ofs);
      subGUI->setupGUI(subController.get(), true);
      subGUI->addMeter(subController->getIntPressurePtr(),
                       "Int Pressure", 500, PixRGB<byte>(255, 0, 0));
      subGUI->addMeter(subController->getHeadingPtr(),
                       "Heading", 360, PixRGB<byte>(192, 255, 0));
      subGUI->addMeter(subController->getPitchPtr(),
                       "Pitch", 256, PixRGB<byte>(192, 255, 0));
      subGUI->addMeter(subController->getRollPtr(),
                       "Roll", 256, PixRGB<byte>(192, 255, 0));
      subGUI->addMeter(subController->getDepthPtr(),
                       "Depth", 300, PixRGB<byte>(192, 255, 0));
      subGUI->addMeter(subController->getThruster_Up_Left_Ptr(),
                       "Motor_Up_Left", -100, PixRGB<byte>(0, 255, 0));
      subGUI->addMeter(subController->getThruster_Up_Right_Ptr(),
                       "Motor_Up_Right", -100, PixRGB<byte>(0, 255, 0));
      subGUI->addMeter(subController->getThruster_Up_Back_Ptr(),
                       "Motor_Up_Back", -100, PixRGB<byte>(0, 255, 0));
      subGUI->addMeter(subController->getThruster_Fwd_Left_Ptr(),
                       "Motor_Fwd_Left", -100, PixRGB<byte>(0, 255, 0));
      subGUI->addMeter(subController->getThruster_Fwd_Right_Ptr(),
                       "Motor_Fwd_Right", -100, PixRGB<byte>(0, 255, 0));

      subGUI->addImage(subController->getSubImagePtr());
      subGUI->addImage(subController->getPIDImagePtr());
    }

#if SIM_MODE == false
      //start streaming input frames
      ifs->startStream();
#endif


  agentManager->startRun();
  goforever = true;  uint fnum = 0;

  Image< PixRGB<byte> > img(w,h, ZEROS);

  while(goforever)
    {
#if SIM_MODE == false
          ifs->updateNext(); img = ifs->readRGB();
          if(!img.initialized()) {Raster::waitForKey(); break; }
#else
          //get sim image from bottom camera
          img = subController->getImage(1);
          sendImageUpdate(subController->getImage(2),beo);
#endif

          agentManager->setCurrentImage(img, fnum);
          fnum++;

          // check COM_B for in message
          checkInMessages(beo, agentManager);

          // check out_messages
          // to send to COM_B
          checkOutMessages(agentManager, beo);
    }

  // send abort to COM_B
  smsg.reset(0, ABORT);
  beo->send(COM_B_NODE, smsg);

  // we are done
  manager.stop();
  return 0;
}

// ######################################################################
void checkInMessages
( nub::soft_ref<Beowulf> beo,
  nub::ref<AgentManagerA> agentManager)
{
  int32 rframe = 0, raction = 0, rnode = 0;
  TCPmessage rmsg;     // buffer to receive messages

  // get all the messages sent in
  if(beo->receive(rnode, rmsg, rframe, raction, 5))
    {
      // check the purpose of the message
      switch(raction)
        {
        case SENSOR_RESULT:
          {
            // unpackage message to a sensor result
            rutz::shared_ptr<SensorResult> sensorResult =
              unpackageToSensorResult(rmsg);

            // update it in the agent manager
            agentManager->updateSensorResult(sensorResult);

            break;
          }

        default:
          LERROR("Unknown purpose");
        }
    }
}

// send new image to COM_B if in simMode
void sendImageUpdate(Image<PixRGB<byte> > img,
                     nub::soft_ref<Beowulf> beo)
{
  int32 rnode = 0;
  TCPmessage smsg;

  int32 sframe  = 0;
  int32 saction = IMAGE_UPDATE;

  smsg.reset(sframe, saction);

  smsg.addImage(img);

  // send the message to COM_B
  beo->send(rnode, smsg);
}

// ######################################################################
void checkOutMessages
( nub::ref<AgentManagerA> agentManager,
  nub::soft_ref<Beowulf> beo)
{
  int32 rnode = 0;
  TCPmessage smsg;     // buffer to send messages

  // while there is a message that needs sending
  if(agentManager->getNumCommands() > 0)
    {
      LINFO("COM_A: sending out a command");
      // pop the top of the queue
      // and packet the message properly
      packageAgentManagerCommand(agentManager,
                                 agentManager->popCommand(),
                                 smsg);

      // send the message to COM_B
      beo->send(rnode, smsg);
    }
}

// ######################################################################
void packageAgentManagerCommand
(nub::ref<AgentManagerA> agentManager,
 rutz::shared_ptr<AgentManagerCommand> agentManagerCommand,
 TCPmessage  &smsg)
{
  int32 sframe  = 0;
  int32 saction = agentManagerCommand->itsCommandType;

  smsg.reset(sframe, saction);

  switch(agentManagerCommand->itsCommandType)
    {

    case UPDATE_MISSION:
      agentManager->updateAgentsMission(agentManagerCommand->itsMission);
      smsg.addInt32(int32(agentManagerCommand->itsMission.missionName));
      smsg.addInt32(int32(agentManagerCommand->itsMission.timeForMission));
      smsg.addInt32(int32(agentManagerCommand->itsMission.missionState));
      break;
    default:
      LINFO("Unknown command type");
    }
}

// ######################################################################
rutz::shared_ptr<SensorResult> unpackageToSensorResult
(TCPmessage rmsg)
{

  SensorResult::SensorResultType type =
    (SensorResult::SensorResultType)(rmsg.getElementInt32());

  SensorResult::SensorResultStatus status =
    (SensorResult::SensorResultStatus)(rmsg.getElementInt32());

  rutz::shared_ptr<SensorResult> sensorResult(new SensorResult(type));

  sensorResult->setStatus(status);

  switch(type)
    {
    case SensorResult::BUOY:
      {
        int x = int(rmsg.getElementInt32());
        int y = int(rmsg.getElementInt32());
        int z = int(rmsg.getElementInt32());
        sensorResult->setPosition(Point3D(x,y,z));
        break;
      }
    case SensorResult::PIPE:
      {
        int x = int(rmsg.getElementInt32());
        int y = int(rmsg.getElementInt32());
        int z = int(rmsg.getElementInt32());
        sensorResult->setPosition(Point3D(x,y,z));
        //        LINFO("PIPE: %d,%d",x,z);
        double angle = rmsg.getElementDouble();
        sensorResult->setOrientation(Angle(angle));
        uint fnum = int(rmsg.getElementInt32());
        sensorResult->setFrameNum(fnum);
        break;
      }
    case SensorResult::BIN:
      {
        int x = int(rmsg.getElementInt32());
        int y = int(rmsg.getElementInt32());
        int z = int(rmsg.getElementInt32());
        sensorResult->setPosition(Point3D(x,y,z));
        break;
      }
    case SensorResult::PINGER:
      {

        double angle = rmsg.getElementDouble();
        sensorResult->setOrientation(Angle(angle));
        break;
      }
    case SensorResult::SALIENCY:
      {
        int x = int(rmsg.getElementInt32());
        int y = int(rmsg.getElementInt32());
        int z = int(rmsg.getElementInt32());
        sensorResult->setPosition(Point3D(x,y,z));
        break;
      }
    default: LINFO("Unknown sensor result type: %d.", type);
    }

  return sensorResult;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
