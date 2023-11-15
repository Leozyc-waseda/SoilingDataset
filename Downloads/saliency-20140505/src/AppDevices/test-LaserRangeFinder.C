/*!@file AppDevices/test-LaserRangeFinder.C Test the Hokuyo laser
range finder */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-LaserRangeFinder.C $
// $Id: test-LaserRangeFinder.C 12962 2010-03-06 02:13:53Z irock $
//
// NOTE: need Liburg-0.1.0-9mdv2008.1.x86_64.rpm
//       which may be obtained at: /lab/mviswana/rpm/RPMS/

#include "Component/ModelManager.H"
#include "Raster/Raster.H"
#include "Image/Image.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "Transport/FrameInfo.H"
#include "GUI/ImageDisplayStream.H"
#include "GUI/XWinManaged.H"
#include "Robots/LoBot/io/LoLaserRangeFinder.H"
#include <algorithm>
#include <iterator>
//#include <urg/urg.h>


int main(int argc, const char **argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Test laser range finder");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  manager.start();
  LINFO("Testing the Hokuyo laser range finder");

  //declare a new laserRangeFinder object
  lobot::LaserRangeFinder* lrf = new lobot::LaserRangeFinder();

  Image<int> dists;
  Image<int>::iterator aptr, stop;
  float rad; int dist;
  int  min,max;
  int angle;

  while(true)
  {
      lrf->update();
      dists = lrf->get_distances();
      aptr = dists.beginw();
      stop = dists.endw();
      LDEBUG("dims= %d", dists.getDims().w());

      // some scaling
      getMinMax(dists, min, max);
      if (max == min) max = min + 1;

      angle = -141; int count = 0;
      Image<PixRGB<byte> > dispImg(512,512,ZEROS);
      while(aptr!=stop)
        {
          dist = *aptr++;
          rad = dist; rad = ((rad - min)/(max-min))*256;
          if (rad < 0) rad = 1.0;

          Point2D<int> pt;
          pt.i = 256 - int(rad*sin((double)angle*M_PI/180.0));
          pt.j = 256 - int(rad*cos((double)angle*M_PI/180.0));

          drawCircle(dispImg, pt, 2, PixRGB<byte>(255,0,0));
          drawLine(dispImg, Point2D<int>(256,256),pt,PixRGB<byte>(0,255,0));

          LDEBUG("[%4d] <%4d>: %13d mm", count, angle, dist);
          angle++; count++;
        }
      ofs->writeRGB(dispImg,"Output",FrameInfo("output",SRC_POS));
      usleep(200);

      //      std::copy(dists.begin(), dists.end(),
      //          std::ostream_iterator<int>(std::cout, " ")) ;
      //std::cout << '\n\n\n---------' ;
  }

  manager.stop();
  return 0;

}



// ######################################################################


/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
