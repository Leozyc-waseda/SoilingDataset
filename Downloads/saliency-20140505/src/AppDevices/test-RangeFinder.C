/*!@file AppDevices/test-RangeFinder.C test the range finder  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-RangeFinder.C $
// $Id: test-RangeFinder.C 10794 2009-02-08 06:21:09Z itti $
//

#include "Component/ModelManager.H"
#include "Devices/RangeFinder.H"
#include "Util/log.H"

#include <stdio.h>
#include "Image/DrawOps.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"

int main(int argc, const char **argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Test Rage Finder");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  // Instantiate our various ModelComponents:
  nub::soft_ref<RangeFinder> rangeFinder(new RangeFinder(manager));
  manager.addSubComponent(rangeFinder);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,"", 0, 0) == false) return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  while(1)
  {
    Image<PixRGB<byte> > img(512,512,ZEROS);

    std::vector<int> rangeData = rangeFinder->getRangeData();

    for(uint i=0; i<rangeData.size(); i++)
    {
      float angle = i * (M_PI/74);
      float rad = ((float)rangeData[i]/4094.0) * 256.0;//1.9/(tan(((float)(rangeData[i]))/1000))*(256.0/100.0);
      LINFO("Range %i %f", i, rad);
      if (rad > 256) rad = 256;


      Point2D<int> pt;
      pt.i = 256-(int)(rad * cos(angle));
      pt.j = 256-(int)(rad * sin(angle));
      drawCircle(img, pt, 3, PixRGB<byte>(255,0,0));
      drawLine(img, Point2D<int>(256,256), pt, PixRGB<byte>(0,255,0));
    }


    ofs->writeRGB(img, "Output", FrameInfo("output", SRC_POS));
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
