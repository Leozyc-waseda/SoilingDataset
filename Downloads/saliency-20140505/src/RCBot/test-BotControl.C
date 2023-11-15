/*!@file RCBot/test-BotControl.C test the Robot Controller
 */

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
// Primary maintainer for this file: Zack Gossman <gossman@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/test-BotControl.C $
// $Id: test-BotControl.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "GUI/XWinManaged.H"
#include "Image/CutPaste.H"     // for inplacePaste()
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "RCBot/BotControl.H"
#include "Raster/Raster.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

static bool goforever = true;  //!< Will turn false on interrupt signal

//! Signal handler (e.g., for control-C)
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; exit(1); }

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager (for camera input):
  ModelManager manager("BotControl Tester");

  nub::ref<BotControl>
    botControl(new BotControl(manager, "Robot control",
                              "BotControl", BotControl::SIMULATION));
  manager.addSubComponent(botControl);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 1) == false)
    return(1);

  manager.start();

  //init the robot controller
  botControl->init();

  short w, h;
  botControl->getImageSensorDims(w,h,0);

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  botControl->setSpeed(0);
  botControl->setSteering(0);

  // main loop
  while(goforever)
    {
      Point2D<int> loc;
      int key = botControl->getUserInput(loc);

      Image<PixRGB<byte> > ima = botControl->getImageSensor();
      if(!ima.initialized()) break;
      switch (key) {
      case 98: //up
        botControl->setSpeed(0.1);
        break;
      case 104: //down
        botControl->setSpeed(-0.1);
        break;
      case 102: //right
        botControl->setSteering(0.4);
        break;
      case 100: //left
        botControl->setSteering(-0.4);
        break;
      case 65: //space
        LINFO("stop");
        botControl->setSpeed(0);
        botControl->setSteering(0);
        break;
      default:
        if (key != -1)
          LINFO("Unknown Key %i", key);
        botControl->setSpeed(0);
        botControl->setSteering(0);
        break;
      }
    }

  // get ready to terminate:
  manager.stop();

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
