/*!@file BeoSub/BeeBrain/seabee-AUVcompetition.C main 2007 competition code
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/SeaBee-AUVcompetition.C $
// $Id: SeaBee-AUVcompetition.C 8623 2007-07-25 17:57:51Z rjpeters $
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

#include "BeoSub/BeeBrain/AgentManagerB.H"

#include "Image/CutPaste.H"

#include "BeoSub/BeeBrain/Globals.H"
#include <signal.h>

bool goforever = false;

// gets the messages from COM_B and relay it to the agent manager
void checkInMessages
( nub::soft_ref<Beowulf> beo,
  nub::ref<AgentManagerB> agentManager);

// gets the out messages from the agent manager and send it to COM_B
void checkOutMessages
( nub::ref<AgentManagerB> agentManager,
  nub::soft_ref<Beowulf> beo);

// unpackage an ocean object to a TCP message to process
void packageOceanObject
( rutz::shared_ptr<AgentManagerCommand> agentManagerCommand,
  rutz::shared_ptr<OceanObject> oceanObject,
  TCPmessage  &smsg);


// ######################################################################
//! Signal handler (e.g., for control-C)
void terminate(int s) { LERROR("*** INTERRUPT ***"); goforever = false; }

// ######################################################################
// ######################################################################
int main( int argc, const char* argv[] )
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("seabee 2007 competition");

  // Instantiate our various ModelComponents:
  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Slave", "BeowulfSlave", false));
  manager.addSubComponent(beo);

  //nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  //manager.addSubComponent(ifs);

  nub::soft_ref<FrameIstream> gbF(makeV4L2grabber(manager));
  nub::soft_ref<FrameIstream> gbB(makeV4L2grabber(manager));

  // create an agent manager
  nub::ref<AgentManagerB> agentManager(new AgentManagerB(manager));
  manager.addSubComponent(agentManager);

  manager.exportOptions(MC_RECURSE);

  //  manager.setOptionValString(&OPT_InputFrameSource, "V4L2");
  //manager.setOptionValString(&OPT_FrameGrabberMode, "YUYV");
  //  manager.setOptionValString(&OPT_FrameGrabberDims, "1024x576");
  //manager.setOptionValString(&OPT_FrameGrabberByteSwap, "no");
  //manager.setOptionValString(&OPT_FrameGrabberFPS, "30");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // STR
  gbF->setModelParamVal("FrameGrabberDevice",std::string("/dev/video0"));
  gbF->setModelParamVal("FrameGrabberChannel",0);
  //gbF->setModelParamVal("InputFrameSource", "V4L2");
  gbF->setModelParamVal("FrameGrabberMode", VIDFMT_YUYV);
  gbF->setModelParamVal("FrameGrabberByteSwap", false);
  //gbF->setModelParamVal("FrameGrabberFPS", 30);
  manager.addSubComponent(gbF);

  gbB->setModelParamVal("FrameGrabberDevice",std::string("/dev/video1"));
  gbB->setModelParamVal("FrameGrabberChannel",0);
  //gbB->setModelParamVal("InputFrameSource", "V4L2");
  gbB->setModelParamVal("FrameGrabberMode", VIDFMT_YUYV);
  gbB->setModelParamVal("FrameGrabberByteSwap", false);
  //gbB->setModelParamVal("FrameGrabberFPS", 30);
  manager.addSubComponent(gbB);

//    int w = ifs->getWidth(),  h = ifs->getHeight();
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
  LINFO("got INIT: %d", initval);

  // STR
  uint wF = gbF->getWidth(); uint hF = gbF->getHeight();
  uint wB = gbB->getWidth(); uint hB = gbB->getHeight();

  rutz::shared_ptr<XWinManaged> cwin
    (new XWinManaged(Dims(2*wF,2*hF), 0, 0, "Downward Vision WindowF"));
  Image<PixRGB<byte> >dispImage(2*wF, 2*hF, ZEROS);

  rutz::shared_ptr<XWinManaged> cwinB
    (new XWinManaged(Dims(2*wB,2*hB), 2*wF, 0, "Downward Vision WindowB"));
  Image<PixRGB<byte> >dispImageB(2*wB, 2*hB, ZEROS);

  agentManager->setWindow(cwin,dispImage);
  agentManager->setWindow(cwinB,dispImageB);

  // send a message of ready to go
  smsg.reset(rframe, INIT_DONE);
  beo->send(rnode, smsg);

  // STR: start streaming
  gbF->startStream();
  gbB->startStream();

  goforever = true; uint fNum = 0;
  while(goforever)
    {
      // get and store image
      //ifs->updateNext(); Image<PixRGB<byte> > ima = ifs->readRGB();
      Image<PixRGB<byte> > imaF = gbF->readRGB();
      Image<PixRGB<byte> > imaB = gbB->readRGB();
      agentManager->setCurrentImage(imaF, fNum);
      agentManager->setCurrentImageB(imaB, fNum);
      fNum++;

      // check messages recieved from COM_A
      checkInMessages(beo, agentManager);

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
  nub::ref<AgentManagerB> agentManager)
{
  int32 rframe = 0, raction = 0, rnode = -1;
  TCPmessage rmsg;     // buffer to receive messages

  // get all the messages sent in
  while(beo->receive(rnode, rmsg, rframe, raction, 5))
    {
      LINFO("COM_B: recieving a command: %d", raction);

      // check the purpose of the message
      switch(raction)
        {
        case ABORT:
          {
            goforever = false; LINFO("Stop SeaBee COM-B");
            break;
          }

        case SEARCH_OCEAN_OBJECT_CMD:
          {
            // get search command info
            DataTypes dType = DataTypes(rmsg.getElementInt32());
            uint ooId = int(rmsg.getElementInt32());
            OceanObject::OceanObjectType ooType =
              OceanObject::OceanObjectType(rmsg.getElementInt32());

            // set it to the agent manager
//             if(ooType != OceanObject::PINGER)
//               agentManager->getDownwardVisionAgent()
//                 ->msgFindAndTrackObject(ooId, ooType, dType);
//             else
              agentManager->getDownwardVisionAgent()
                ->msgFindAndTrackObject(ooId, ooType, dType);

            break;
          }
        case STOP_SEARCH_OCEAN_OBJECT_CMD:
          {
            // get search command info
            DataTypes dType = DataTypes(rmsg.getElementInt32());
            uint ooId = int(rmsg.getElementInt32());

            agentManager->getDownwardVisionAgent()
              ->msgStopLookingForObject(ooId, dType);
            break;
          }
        default:
          LFATAL("Unknown purpose");
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
  while(agentManager->getNumResults() > 0)
    {
      LINFO("COM_B: sending a result");

      // pop the top of the result queue
      std::pair<rutz::shared_ptr<AgentManagerCommand>,
        rutz::shared_ptr<OceanObject> >
        result = agentManager->popResult();

      // package data from ocean object into TCP message
      packageOceanObject(result.first, result.second, smsg);

      // send the message
      beo->send(rnode, smsg);
    }
}

// ######################################################################
void packageOceanObject
( rutz::shared_ptr<AgentManagerCommand> agentManagerCommand,
  rutz::shared_ptr<OceanObject> oceanObject,
  TCPmessage  &smsg)
{
  int32 rframe = 0;

  smsg.reset(rframe, agentManagerCommand->itsCommandType);

  // send the id
  smsg.addInt32(int32(oceanObject->getId()));

  // package the message
  switch(agentManagerCommand->itsCommandType)
    {
    case SEARCH_OCEAN_OBJECT_CMD:
      {
        smsg.addInt32(int32(agentManagerCommand->itsDataType));

        switch(agentManagerCommand->itsDataType)
          {
            case POSITION:
              {
                Point3D pos = oceanObject->getPosition();
                smsg.addInt32(int32(pos.x));
                smsg.addInt32(int32(pos.y));
                smsg.addInt32(int32(pos.z));
                break;
              }
          case ORIENTATION:
            {
              Angle ori = oceanObject->getOrientation();
              smsg.addDouble(ori.getVal());
              break;
            }
          case FREQUENCY:
            {
              smsg.addFloat(oceanObject->getFrequency());
              break;
            }
          case DISTANCE:
            {
              smsg.addFloat(oceanObject->getDistance());
              break;
            }
          case MASS:
            {
              smsg.addFloat(oceanObject->getMass());
              break;
            }
          default: LFATAL("unknown data type");
          }
        break;
      }

    case OCEAN_OBJECT_STATUS:
      {
        break;
      }
    default: LERROR("Unknown raction");
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
