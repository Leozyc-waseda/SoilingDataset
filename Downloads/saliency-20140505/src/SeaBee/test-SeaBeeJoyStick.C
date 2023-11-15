/*!@file src/SeaBee/test-SeaBeeJoyStick.C test Joystick interface to SeaBee */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   //
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/test-SeaBeeJoyStick.C $
// $Id:
//

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Devices/JoyStick.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Raster/Raster.H"

#include <unistd.h>
#include <signal.h>

#ifdef HAVE_LINUX_JOYSTICK_H

#define COM_B_NODE 0
#define INIT_COMM 10000
#define JS_AXIS_UPDATE   20000
#define ABORT 90000

#define X_MIN -32767
#define X_MAX 32767
#define Y_MIN -32767
#define Y_MAX 32767
#define D_MIN -32767
#define D_MAX 32767


nub::soft_ref<Beowulf> beo;
volatile bool keepGoing = false;

//! A simple joystick listener
class TestJoyStickListener : public JoyStickListener
{
public:
  virtual ~TestJoyStickListener() { }

  virtual void axis(const uint num, const int16 val)
  {
    if(keepGoing)
      {

        int32 rnode = 0;
        TCPmessage smsg;     // buffer to send messages

        int32 sframe  = 0;
        int32 saction = JS_AXIS_UPDATE;

        smsg.reset(sframe, saction);
        smsg.addInt32(int32(num));

        //p is the percentage of thrust being applied [-100.0...100.0] for the axis
        float p = 0.0;
        float v = 0.0;
        v = (float)val;

        switch(num)
          {
          case 0:
            p =  (float)(v/X_MAX*100.0);
            LINFO("X-Axis = %f", p);
            break;
          case 1:
            p =  v/Y_MAX*100.0;
            LINFO("Y-Axis = %f", p);
            break;
          case 2:
            p =  v/D_MAX*100.0;
            LINFO("Depth = %f", p);
            break;
          default:
            LERROR("Unknown axis event recieved");
          }

        smsg.addFloat(p);

        // send the message to COM_B
        beo->send(rnode, smsg);

      }
  }

  virtual void button(const uint num, const bool state)
  {
    LINFO("Button[%d] = %s", num, state?"true":"false");
  }
};

#endif

// ######################################################################
//! Signal handler (e.g., for control-C)
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); keepGoing = false; exit(1); }


// ######################################################################
//! Test JoyStick code
/*! Test Joystick code. */
int main(const int argc, const char **argv)
{
#ifndef HAVE_LINUX_JOYSTICK_H

  LFATAL("<linux/joystick.h> must be installed to use this program");

#else

  // get a manager going:
  ModelManager manager("JoyStick Manager");

  // instantiate our model components:
  nub::soft_ref<JoyStick> js(new JoyStick(manager) );
  manager.addSubComponent(js);

  beo.reset(new Beowulf(manager, "Beowulf Master", "BeowulfMaster", true));
  manager.addSubComponent(beo);

  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "[device]", 0, 1) == false)
    return(1);

  // let's configure our device:
  if (manager.numExtraArgs() > 0)
    js->setModelParamVal("JoyStickDevName",
                         manager.getExtraArg(0), MC_RECURSE);

  // register a listener:
  rutz::shared_ptr<TestJoyStickListener> lis(new TestJoyStickListener);
  rutz::shared_ptr<JoyStickListener> lis2; lis2.dynCastFrom(lis); // cast down
  js->setListener(lis2);

  TCPmessage rmsg;     // buffer to receive messages
  TCPmessage smsg;     // buffer to send messages
  int32 rframe = 0, raction = 0, rnode = 0;


  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // get started:
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
  Raster::waitForKey();

  keepGoing = true;
  // Everything is event-driven so in our main loop here we just sleep:
  while(keepGoing) sleep(100);

  // send abort to COM_B
  smsg.reset(0, ABORT);
  beo->send(COM_B_NODE, smsg);

  // stop everything and exit:
  manager.stop();
  return 0;

#endif // HAVE_LINUX_JOYSTICK_H

}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
