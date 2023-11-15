/*!@file AppDevices/test-JoyStick.C test Linux JoyStick interfacing */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-JoyStick.C $
// $Id: test-JoyStick.C 7880 2007-02-09 02:34:07Z itti $
//

#include "Component/ModelManager.H"
#include "Devices/JoyStick.H"
#include "Util/Types.H"
#include "Util/log.H"
#include <unistd.h>

#ifdef HAVE_LINUX_JOYSTICK_H

//! A simple joystick listener
class TestJoyStickListener : public JoyStickListener
{
public:
  virtual ~TestJoyStickListener() { }

  virtual void axis(const uint num, const int16 val)
  { LINFO("Axis[%d] = %d", num, val); }

  virtual void button(const uint num, const bool state)
  { LINFO("Button[%d] = %s", num, state?"true":"false"); }
};

#endif

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

  // get started:
  manager.start();

  // Everything is event-driven so in our main loop here we just sleep:
  while(1) sleep(100);

  /*
  // Alternatively to the event-driven approach used above where a
  // JoyStickListener is activated each time an event is received from
  // the joystick, below is an example of how you would do a polling
  // driver. It is not recommended as you may miss some events if you
  // don't poll fast enough:

  // get our joystick's vitals:
  uint nax = js->getNumAxes();
  uint nbu = js->getNumButtons();

  char msg[2000], tmp[20];

  // main loop: display the values at regular time intervals
  for(;;)
    {
      // get the axes:
      sprintf(msg, "Axes:   ");
      for (uint i = 0; i < nax; i ++)
        {
          int16 val = js->getAxisValue(i);
          sprintf(tmp, " %-05d", val); strcat(msg, tmp);
        }
      LINFO(msg);

      // get the buttons:
      sprintf(msg, "Buttons:");
      for (uint i = 0; i < nbu; i ++)
        {
          bool val = js->getButtonState(i);
          if (val) strcat(msg, " 1"); else strcat(msg, " 0");
        }
      LINFO(msg);

      // sleep a bit:
      usleep(100000);
    }
  */

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
