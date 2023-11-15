/*!@file BeoSub/test-KillSwitch.C */

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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-KillSwitch.C $
// $Id: test-KillSwitch.C 6526 2006-04-25 17:26:17Z rjpeters $
//

#include "Devices/ParPort.H"
#include "Util/log.H"

#include <unistd.h>
#include <iostream>
#include "Component/ModelManager.H"

int main() {
#ifndef HAVE_LINUX_PARPORT_H

  LFATAL("<linux/parport.h> must be installed to use this program");

#else

  ModelManager mgr("test killswitch");
  unsigned char status;
  nub::soft_ref<ParPort> lp0(new ParPort(mgr));
  mgr.addSubComponent(lp0);

  mgr.start();

  while(1) {
    status = lp0->ReadStatus();
    std::cout << (int) status;
    std::cout << !((status & PARPORT_STATUS_BUSY) || !(status & PARPORT_STATUS_ACK));

    std::cout << std::endl;
    sleep(1);
  }

  mgr.stop();

  return 0;

#endif

}
