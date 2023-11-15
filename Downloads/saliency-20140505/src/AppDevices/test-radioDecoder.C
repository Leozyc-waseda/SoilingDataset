/*!@file AppDevices/test-radioDecoder.C Test RC radio decoder */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-radioDecoder.C $
// $Id: test-radioDecoder.C 14697 2011-04-08 21:34:48Z farhan $
//

#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/RadioDecoder.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <signal.h>
#include <cstdio>

static bool goforever = true;  //!< Will turn false on interrupt signal

//! Signal handler (e.g., for control-C)
void terminate(int s) { LERROR("*** INTERRUPT ***"); goforever = false; }

int main(int argc, const char **argv)
{
  // setup signal handling:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);

  // Instantiate a ModelManager:
  ModelManager manager("Test Model for RadioDecoder Class");

  // Instantiate our various ModelComponents:
  nub::soft_ref<RadioDecoder> rd(new RadioDecoder(manager));
  manager.addSubComponent(rd);

  // decide on which command-line options our model should export:
  // NOTE: see the implementation of exportOptions() in
  // RadioDecoder.C; it blocks some of the options of AudioGrabber so
  // that users cannot mess with those.
  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // get some of the configured params back:
  const bool stereo =
    manager.getModelParamVal<uint>("AudioGrabberChans", MC_RECURSE);
  int nchan = stereo ? 2 : 1;

  // let's get all our ModelComponent instances started:
  manager.start();

  // let's run a calibration
  rd->zeroCalibrate();  rd->rangeCalibrate();

  // main loop:
  char txt[1024], xx[1024];
  while(goforever)
    {
      sprintf(txt, "RADIO: ");
      for (int i = 0; i < nchan; i ++)
        { sprintf(xx, "%d: %+2.2f ", i, rd->getVal(i)); strcat(txt, xx); }
      LINFO("%s", txt);
      usleep(150000);
    }

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
