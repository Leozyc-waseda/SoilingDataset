/*!@file AppDevices/test-BeoJoyStick.C test Linux JoyStick interfacing */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-BeoJoyStick.C $
// $Id: test-BeoJoyStick.C 8267 2007-04-18 18:24:24Z rjpeters $

#include "Beowulf/Beowulf.H"
#include "Beowulf/BeowulfOpts.H"
#include "Component/ModelManager.H"
#include "Devices/JoyStick.H"
#include "Image/Image.H"

//! A simple joystick listener
class TestJoyStickListener : public JoyStickListener
{
public:
  TestJoyStickListener(const nub::soft_ref<Beowulf> b) :
    data(6, 1, ZEROS), beo(b)
  { }

  virtual ~TestJoyStickListener() { }

  virtual void axis(const uint num, const int16 val)
  {
    LINFO("Axis[%d] = %d", num, val);

    bool changed = false;

    switch(num)
      {
      case 0:
        // left-right
        data.setVal(1, val);
        changed = true;
        break;

      case 1:
        // front-back
        data.setVal(0, val);
        changed = true;
        break;

      case 2:
        // rotation of the handle
        break;

      case 3:
        // little accessory slider
        data.setVal(2, val);
        changed = true;
        break;
      }

    if (changed) sendData();
  }

  virtual void button(const uint num, const bool state)
  {
    LINFO("Button[%d] = %s", num, state?"pressed":"released");

    bool changed = false;

    switch(num)
      {
      case 5:
        // third largest button on the base
        data.setVal(3, state ? 1.0F : 0.0F);
        changed = true;
        break;
      case 6:
        // second largest button on base
        data.setVal(4, state ? 1.0F : 0.0F);
        changed = true;
        break;

      case 7:
        // largest button on base
        data.setVal(5, state ? 1.0F : 0.0F);
        changed = true;
        break;
      }

    if (changed) sendData();
  }

virtual void sendData()
{
  TCPmessage smsg(0, 0);
  smsg.addImage(data);
  beo->send(0, smsg);
}

Image<float> data;
nub::soft_ref<Beowulf> beo;
};

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
  nub::soft_ref<JoyStick> js(new JoyStick(manager));
  manager.addSubComponent(js);

  nub::soft_ref<Beowulf> beow(new Beowulf(manager, "Beowulf Master",
                                          "BeowulfMaster", true));
  manager.addSubComponent(beow);

  manager.setOptionValString(&OPT_BeowulfSlaveNames, "192.168.0.215");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "[device]", 0, 1) == false)
    return(1);

  // let's configure our device:
  if (manager.numExtraArgs() > 0)
    js->setModelParamVal("JoyStickDevName",
                         manager.getExtraArg(0), MC_RECURSE);

  // get started:
  manager.start();

  // register a listener:
  rutz::shared_ptr<TestJoyStickListener> lis(new TestJoyStickListener(beow));
  rutz::shared_ptr<JoyStickListener> lis2; lis2.dynCastFrom(lis); // cast down
  js->setListener(lis2);

  // Everything is event-driven so in our main loop here we just sleep:
  while(1) sleep(100);

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
