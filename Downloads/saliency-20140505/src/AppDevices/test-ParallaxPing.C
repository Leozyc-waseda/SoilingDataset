/*!@file AppDevices/test-ParallaxPing.C test the parallax Ping))) sonar range finder   */

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
// Primary maintainer for this file: Farhan Baluch <fbaluch@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-ParallaxPing.C $
// $Id: test-ParallaxPing.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Component/ModelManager.H"
#include "Devices/RangeFinder.H"
#include "Util/log.H"

#include <stdio.h>
#include "Image/DrawOps.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "Devices/Serial.H"
#include "Util/StringUtil.H"

int main(int argc, const char **argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Test Rage Finder");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::ref<Serial> itsSerial(new Serial(manager));
  manager.addSubComponent(itsSerial);
  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,"", 1, 1) == false) return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  LINFO("configuring serial %s",argv[1]);
  itsSerial->configure(argv[1],115200,"8N1",false,false,0);
  itsSerial->enablePort(argv[1]);
  LINFO("done configuring");

  while(1)
    {


          unsigned char start ={255};
          unsigned char end = {255};
          std::vector<unsigned char> frame = itsSerial->readFrame(start,end,4,-1);


      if(frame.size() == 4)
      {
          unsigned int dist = ((0x0FF & frame[0])  << 24) |
              ((0x0FF & frame[1])  << 16) |
              ((0x0FF & frame[2])  << 8)  |
              ((0x0FF & frame[3])  << 0);

          LINFO("dist : %i", dist);

      }

      else
      {
          LFATAL("bad packets");
      }

      usleep(10000);

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
