/*!@file BeoSub/BeeBrain/seabee-AUVcompetition-master.C main 2007 competition code
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/SeaBee-AUVcompetition-master.C $
// $Id: SeaBee-AUVcompetition-master.C 8623 2007-07-25 17:57:51Z rjpeters $
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

#include "BeoSub/BeeBrain/AgentManagerA.H"

#include "BeoSub/BeeBrain/Globals.H"
#include <signal.h>

//#include <pthread.h>
//#include <unistd.h>



// #include "CaptainAgent.h"
// #include "MovementAgent.h"
// #include "ForwardVisionAgent.h"
// #include "DownwardVisionAgent.h"
// #include "SonarAgent.h"


bool goforever = false;

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
(rutz::shared_ptr<AgentManagerCommand> agentManagerCommand,
 TCPmessage  &smsg);

// unpackage an ocean object to a TCP message to process
void unpackageToOceanObject
( TCPmessage  rmsg,
  rutz::shared_ptr<OceanObject> oceanObject,
  DataTypes dType);

// ######################################################################
//! Signal handler (e.g., for control-C)
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; exit(1); }

// ######################################################################
int main( int argc, const char* argv[] )
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("seabee 2007 competition");

  // Instantiate our various ModelComponents:
  nub::soft_ref<Beowulf>
   beo(new Beowulf(manager, "Beowulf Master", "BeowulfMaster", true));
  manager.addSubComponent(beo);

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  // create an agent manager
  nub::ref<AgentManagerA> agentManager(new AgentManagerA(manager));
  manager.addSubComponent(agentManager);

  manager.exportOptions(MC_RECURSE);

  manager.setOptionValString(&OPT_InputFrameSource, "V4L2");
  manager.setOptionValString(&OPT_FrameGrabberMode, "YUYV");
  manager.setOptionValString(&OPT_FrameGrabberDims, "1024x576");
  manager.setOptionValString(&OPT_FrameGrabberByteSwap, "no");
  manager.setOptionValString(&OPT_FrameGrabberFPS, "30");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  //  int w = ifs->getWidth(),  h = ifs->getHeight();
  int w = 320, h = 240;
  std::string dims = convertToString(Dims(w, h));
  LINFO("image size: [%dx%d]", w, h);
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

  // send params to dorsal and ventral node to initialize contact:
  smsg.reset(0, INIT_COMM);
  smsg.addInt32(int32(235));

  // send the same initial values message to CPU_B
  beo->send(COM_B_NODE, smsg);

  // SYNCHRONIZATION: wait until the other board is ready
  LINFO("waiting until COM_B is ready to go");
  rnode = COM_B_NODE;
  while(!beo->receive(rnode, rmsg, rframe, raction, 5));
  rmsg.reset(rframe, raction);
  LINFO("%d is ready", rnode);

  //start streaming
  ifs->startStream();

  rutz::shared_ptr<XWinManaged> cwin
    (new XWinManaged(Dims(2*w,2*h), 20, 20, "Forward Vision Window"));
  Image<PixRGB<byte> >dispImage(2*w, 2*h, ZEROS);

  agentManager->setWindow(cwin,dispImage);

  agentManager->startRun();
  goforever = true;  uint fnum = 0;
  while(goforever)
    {
      // check user input

      // get and store image
   //    ifs->updateNext();

      Image<PixRGB<byte> > ima;// = ifs->readRGB();

      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE)
        break;

      //grab the images
      GenericFrame input = ifs->readFrame();
      if (!input.initialized())
        break;
      ima = rescale(input.asRgb(), 320, 240);

      agentManager->setCurrentImage(ima, fnum);
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
  while(beo->receive(rnode, rmsg, rframe, raction, 5))
    {
      //LINFO("COM_A: recieving a result");

      rutz::shared_ptr<OceanObject> oceanObject(new OceanObject());

      // get the ocean object id
      uint oceanObjectId = int(rmsg.getElementInt32());

      oceanObject->setId(oceanObjectId);

      // check the purpose of the message
      switch(raction)
        {
        case SEARCH_OCEAN_OBJECT_CMD:
          {
            // get ocean object data type
            DataTypes oceanObjectDataType = DataTypes(rmsg.getElementInt32());

            LINFO("Recieved result. DataType: %d", oceanObjectDataType);

            // unpackage message to an ocean object
            unpackageToOceanObject(rmsg, oceanObject, oceanObjectDataType);

            // update it in the agent manager
            agentManager->updateOceanObject(oceanObject, oceanObjectDataType);

            break;
          }

        case OCEAN_OBJECT_STATUS:
          {
            agentManager->getPreFrontalCortexAgent()->msgOceanObjectUpdate();
            break;
          }

        default:
          LERROR("Unknown purpose");
        }
    }
}

// ######################################################################
void checkOutMessages
( nub::ref<AgentManagerA> agentManager,
  nub::soft_ref<Beowulf> beo)
{
  int32 rnode = 0;
  TCPmessage smsg;     // buffer to send messages

  // while there is a message that needs sending
  while(agentManager->getNumCommands() > 0)
    {
      LINFO("COM_A: sending out a command");
      // pop the top of the queue
      // and packet the message properly
      packageAgentManagerCommand(agentManager->popCommand(), smsg);

      // send the message to COM_B
      beo->send(rnode, smsg);
    }
}

// ######################################################################
void packageAgentManagerCommand
(rutz::shared_ptr<AgentManagerCommand> agentManagerCommand, TCPmessage  &smsg)
{
  int32 sframe  = 0;
  int32 saction = agentManagerCommand->itsCommandType;

  smsg.reset(sframe, saction);
  smsg.addInt32(int32(agentManagerCommand->itsDataType));
  smsg.addInt32(int32(agentManagerCommand->itsOceanObjectId));
  smsg.addInt32(int32(agentManagerCommand->itsOceanObjectType));
}

// ######################################################################
void unpackageToOceanObject
(TCPmessage  rmsg,
 rutz::shared_ptr<OceanObject> oceanObject,
 DataTypes dType)
{
  DataTypes dataType = dType;

  switch(dataType)
    {
    case POSITION:
      {
        int x = int(rmsg.getElementInt32());
        int y = int(rmsg.getElementInt32());
        int z = int(rmsg.getElementInt32());
        LINFO("Recieved a point (%d,%d,%d)",x,y,z);
        oceanObject->setPosition(Point3D(x,y,z));
        break;
      }
    case ORIENTATION:
      {
        double angle = rmsg.getElementDouble();
        oceanObject->setOrientation(Angle(angle));
        break;
      }
    case FREQUENCY:
      {
        float frequency = rmsg.getElementFloat();
        oceanObject->setFrequency(frequency);
        break;
      }
    case DISTANCE:
      {
        float distance = rmsg.getElementFloat();
        oceanObject->setDistance(distance);
        break;
      }
    case MASS:
      {
        float mass = rmsg.getElementFloat();
        oceanObject->setMass(mass);
        break;
      }
    default: LFATAL("unknown data type");
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
