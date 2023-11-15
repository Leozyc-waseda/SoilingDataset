/*!@file SeaBee/SeaBeeMainB.C main 2007 competition code
  Run SeaBeeMainA at CPU_A
  Run SeaBeeMainB at CPU_B                             */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/SeaBeeMainB.C $
// $Id: SeaBeeMainB.C 10794 2009-02-08 06:21:09Z itti $
//
//////////////////////////////////////////////////////////////////////////

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Raster/Raster.H"
#include "Media/MediaOpts.H"
#include "GUI/XWinManaged.H"

#include "Video/VideoFormat.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/FrameGrabberFactory.H"

#include "Devices/DeviceOpts.H"
#include "Image/CutPaste.H"

#include "Globals.H"
#include "AgentManagerB.H"
#include "Mission.H"
#include "SubController.H"

#include <signal.h>

#define SIM_MODE false

volatile bool goforever = false;
uint fNum = 0;

// gets the messages from COM_A and relay it to the agent manager
void checkInMessages
( nub::soft_ref<Beowulf> beo,
  nub::ref<AgentManagerB> agentManager,
  rutz::shared_ptr<XWinManaged> cwin);

// gets the out messages from the agent manager and send it to COM_A
void checkOutMessages
( nub::ref<AgentManagerB> agentManager,
  nub::soft_ref<Beowulf> beo);

// unpackage an ocean object to a TCP message to process
void packageSensorResult
( SensorResult sensorResult,
  TCPmessage  &smsg);


// ######################################################################
//! Signal handler (e.g., for control-C)
void terminate(int s)
{
  LERROR("*** INTERRUPT ***");
  goforever = false;
  exit(1);
}

// ######################################################################
// ######################################################################
int main( int argc, const char* argv[] )
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("SeaBee 2008 competition slave");

  // Instantiate our various ModelComponents:
  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Slave", "BeowulfSlave", false));
  manager.addSubComponent(beo);

#if SIM_MODE == false

  //initiliaze forward and back bottom cameras
  //nub::soft_ref<FrameIstream> gbF(makeV4L2grabber(manager));
  nub::soft_ref<FrameIstream> gbB(makeV4L2grabber(manager));

  gbB->setModelParamVal("FrameGrabberDevice",std::string("/dev/video0"));
  gbB->setModelParamVal("FrameGrabberChannel",0);
  //  gbB->setModelParamVal("InputFrameSource", "V4L2");
  //  gbB->setModelParamVal("FrameGrabberMode", VIDFMT_YUYV);
  //      gbB->setModelParamVal("FrameGrabberByteSwap", false);
  //  gbB->setModelParamVal("FrameGrabberFPS", 30);
  manager.addSubComponent(gbB);
#endif


  // create an agent manager
  nub::ref<AgentManagerB> agentManager(new AgentManagerB(manager));
  manager.addSubComponent(agentManager);

  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);


//    std::string dims = convertToString(Dims(w, h));
//    LINFO("image size: [%dx%d]", w, h);
//    manager.setOptionValString(&OPT_InputFrameDims, dims);

//    manager.setModelParamVal("InputFrameDims", Dims(w, h),
//                             MC_RECURSE | MC_IGNORE_MISSING);

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
  LINFO("Recieved INIT(%d) from COM_A", initval);

  // Get camera image dimensions
  uint w, h;

#if SIM_MODE == false
  //    w = ifs->getWidth();
  //    h = ifs->getHeight();
  w = gbB->getWidth();
  h = gbB->getHeight();
#else
  w = 320;
  h = 240;
#endif


  rutz::shared_ptr<XWinManaged> cwin
    (new XWinManaged(Dims(w,h), 0, 0, "SeaBeeMainB Window"));

  // rutz::shared_ptr<XWinManaged> cwinB
  // (new XWinManaged(Dims(2*w,2*h), 2*w, 0, "Downward Vision WindowB"));
  //   Image<PixRGB<byte> >dispImageB(2*wB, 2*hB, ZEROS);

  agentManager->setWindow(cwin);
  //   agentManager->setWindow(cwinB,dispImageB);

  // send a message of ready to go
  smsg.reset(rframe, INIT_DONE);
  beo->send(rnode, smsg);

#if SIM_MODE == false
  //start streaming input frames
  //ifs->startStream();

  // STR: start streaming
  gbB->startStream();
#endif

  goforever = true;

  Image<PixRGB<byte> > img(w,h,ZEROS);

  //  Image<PixRGB<byte> > imgF(w,h,ZEROS);
  //  Image<PixRGB<byte> > imgB(w,h,ZEROS);

  while(goforever)
    {
      // get and store image
#if SIM_MODE == false

      //      ifs->updateNext(); img = ifs->readRGB();
      //      if(!img.initialized()) {Raster::waitForKey(); break; }
      img = gbB->readRGB();
      agentManager->setCurrentImageF(img, fNum);
      agentManager->setCurrentImageB(img, fNum);
//       imgF = gbF->readRGB();

#else
      // img = subController->getImage(2);
          //imgB = subController->getImage(2);
#endif

      fNum++;

      // check messages recieved from COM_A
      checkInMessages(beo, agentManager, cwin);

      // check if there is messages to send to COM_A
      checkOutMessages(agentManager, beo);
    }

  // we are done
  manager.stop();
  return 0;
}

// ######################################################################
void checkInMessages
( nub::soft_ref<Beowulf> beo,
  nub::ref<AgentManagerB> agentManager,
  rutz::shared_ptr<XWinManaged> cwin
)
{
  int32 rframe = 0, raction = 0, rnode = -1;
  TCPmessage rmsg;     // buffer to receive messages

  // get all the messages sent in
  if(beo->receive(rnode, rmsg, rframe, raction, 5))
    {
      //      LINFO("COM_B is recieving a command: %d", raction);

      Image<PixRGB<byte> > img;
      // check the purpose of the message
      switch(raction)
        {
        case ABORT:
          {
            goforever = false; LINFO("Stop SeaBee COM-B");
            break;
          }
        case UPDATE_MISSION:
          {
            //reconstruct mission object
            Mission theMission;
            theMission.missionName = Mission::MissionName(rmsg.getElementInt32());
            theMission.timeForMission = int(rmsg.getElementInt32());
            theMission.missionState = Mission::MissionState(rmsg.getElementInt32());

            //update downward vision agent's mission
            agentManager->updateAgentsMission(theMission);
            break;
          }
        case IMAGE_UPDATE:
          fNum++;
          img = rmsg.getElementColByteIma();
          agentManager->setCurrentImageF(img, fNum);
          //          cwin->drawImage(img, 0, 0);
          break;
        default:
          LINFO("Unknown purpose");
        }
    }
}

// ######################################################################
void checkOutMessages
( nub::ref<AgentManagerB> agentManager,
  nub::soft_ref<Beowulf> beo)
{
  int32 rnode = -1;
  TCPmessage smsg;     // buffer to send messages

  // while there is a message that needs sending
  if(agentManager->getNumResults() > 0)
    {
      // pop the back of the result vector
      SensorResult result = agentManager->popResult();

      // package data from ocean object into TCP message
      packageSensorResult(result, smsg);

      // send the message
      beo->send(rnode, smsg);
    }
}

// ######################################################################
void packageSensorResult
( SensorResult sensorResult,
  TCPmessage  &smsg)
{
  int32 rframe = 0;

  smsg.reset(rframe, SENSOR_RESULT);

  SensorResult::SensorResultType type =
    sensorResult.getType();

  smsg.addInt32(type);

  smsg.addInt32(sensorResult.getStatus());

  // package the message
  switch(type)
    {
    case SensorResult::BUOY:
      {
        Point3D pos = sensorResult.getPosition();
        smsg.addInt32(int32(pos.x));
        smsg.addInt32(int32(pos.y));
        smsg.addInt32(int32(pos.z));
        break;
      }
    case SensorResult::PIPE:
      {
        Point3D pos = sensorResult.getPosition();
        smsg.addInt32(int32(pos.x));
        smsg.addInt32(int32(pos.y));
        smsg.addInt32(int32(pos.z));
        Angle ori = sensorResult.getOrientation();
        smsg.addDouble(ori.getVal());
        uint fnum = sensorResult.getFrameNum();
        smsg.addInt32(int32(fnum));
        break;
      }
    case SensorResult::BIN:
      {
        Point3D pos = sensorResult.getPosition();
        smsg.addInt32(int32(pos.x));
        smsg.addInt32(int32(pos.y));
        smsg.addInt32(int32(pos.z));
        break;
      }
    case SensorResult::PINGER:
      {
        Angle ori = sensorResult.getOrientation();
        smsg.addDouble(ori.getVal());
        break;
      }
    case SensorResult::SALIENCY:
      {
        Point3D pos = sensorResult.getPosition();
        smsg.addInt32(int32(pos.x));
        smsg.addInt32(int32(pos.y));
        smsg.addInt32(int32(pos.z));
        break;
      }
      default: LINFO("Unknown sensor result type: %d",type);
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
