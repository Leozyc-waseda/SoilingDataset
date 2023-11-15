/*!@file AppDevices/test-pelco.C test the pelco controller  */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-pelco.C $
// $Id: test-pelco.C 13712 2010-07-28 21:00:40Z itti $
//

#include "Component/ModelManager.H"
#include "Devices/Pelco.H"
#include "Util/log.H"

#include <stdio.h>

const char* USAGE = "-- <cmd> <optional data>\nwhere cmd is one of the following\n"
"Cmd=resetPanTilt        Reset pan/tilt\n"
"Cmd=zoom  data=x    Zoom by <x> from 0 (wide), 1023 (tele)} \n"
"Cmd=focus  data=x    focus by <x> from 4096 (infinity), 40959 (close)} \n"
"Cmd=panTilt  data=x y   move pan to x and tilt to y\n"
"Cmd=raw               send raw data (data1 data2 data3 data4)\n";

int main(int argc, const char **argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Test Model for pelco Class");

  // Instantiate our various ModelComponents:
  nub::soft_ref<Pelco> pelco(new Pelco(manager));
  manager.addSubComponent(pelco);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               USAGE, 1, 5) == false) return(1);

  // Let's get some of the option values, to configure our window:
  std::string cmd = manager.getExtraArg(0).c_str();
  //int serpos = atoi(manager.getExtraArg(1).c_str());


  // let's get all our ModelComponent instances started:
  manager.start();

  if (cmd == "resetPanTilt") pelco->resetPanTilt();
  else if (cmd == "zoom")
    pelco->zoom(atoi(manager.getExtraArg(1).c_str()));
  else if (cmd == "focus")
    pelco->setFocus(atoi(manager.getExtraArg(1).c_str()));
  else if (cmd == "panTilt")
    pelco->movePanTilt(manager.getExtraArgAs<int>(1), atoi(manager.getExtraArg(2).c_str()));
  else if (cmd == "raw")
    pelco->sendRawCmd(manager.getExtraArgAs<int>(1),
                      manager.getExtraArgAs<int>(2),
                      manager.getExtraArgAs<int>(3),
                      manager.getExtraArgAs<int>(4));
  else
    LINFO("%s", USAGE);

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
