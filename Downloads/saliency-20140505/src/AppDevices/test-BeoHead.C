/*!@file AppDevices/test-BeoHead.C Test Robot Head  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-BeoHead.C $
// $Id: test-BeoHead.C 8267 2007-04-18 18:24:24Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Devices/BeoHead.H"
#include "Util/log.H"

#include <stdio.h>
#include <math.h>

int main(int argc, const char **argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Test Robot Head");

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeoHead> beoHead(new BeoHead(manager));
  manager.addSubComponent(beoHead);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  //subject the eye to a sin wave
  for (float i=(M_PI/2); i<(16.5*M_PI); i += 0.01)
  {
    float tilt_pos = sin(i);
    float pan_pos = sin(i+(M_PI/2));

    beoHead->setLeftEyeTilt(tilt_pos);
    beoHead->setLeftEyePan(pan_pos);

    beoHead->setRightEyeTilt(tilt_pos);
    beoHead->setRightEyePan(pan_pos);

    beoHead->setHeadTilt(tilt_pos);
    beoHead->setHeadPan(pan_pos);

    usleep(1000);
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
