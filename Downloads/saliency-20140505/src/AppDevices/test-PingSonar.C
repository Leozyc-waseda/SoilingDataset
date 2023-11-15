 /*!@file AppDevices/test-PingSonar.C test the ping sonar device  */

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
// Primary maintainer for this file: farhan baluch <fbaluch@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-PingSonar.C $
// $Id: test-PingSonar.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Component/ModelManager.H"
#include "Devices/PingSonar.H"
#include "Util/log.H"
#include "Util/Types.H"
#include "Util/sformat.H"
#include "Image/DrawOps.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"

#include <stdio.h>
#include <cstring>
#include <cstdlib>

// ######################################################################
//! Visualize distance received from sensor
Image<PixRGB<byte> > vizDist(std::vector<int> dists,int divisions)
{

    Image<PixRGB<byte> > img(800,800,ZEROS);
    int startAng = 0;
    int increment = 180/dists.size();
    int beginAng=startAng, endAng=increment;

    for(uint s=0;s<dists.size();s++)
    {

        for(int i=1; i<=divisions;i++)
            for (int ang=beginAng;ang<=endAng;ang++)
            {
                int rad = i*5;
                Point2D<int> pt;
                pt.i = 200+100*s - (int) (rad*cos(ang*M_PI/180.0));
                pt.j = 400 - (int) (rad*sin(ang*M_PI/180.0));

                if(dists.at(s) <= i*250)
                    drawPoint(img,pt.i,pt.j,PixRGB<byte>(255,0,0));
                else
                    drawPoint(img,pt.i,pt.j,PixRGB<byte>(0,0,255));

                writeText(img,Point2D<int>(10,10),sformat(" %d ",dists.at(s)).c_str(),
                          PixRGB<byte>(255),PixRGB<byte>(0));

            }
        beginAng = endAng;
        endAng = endAng + increment;
    }

     return img;

}


// ######################################################################

int main(int argc, const char **argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Test Ping Sonar");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  // Instantiate our various ModelComponents:
  nub::soft_ref<PingSonar> pingSonar(new PingSonar(manager,"PingSonar",
                                                   "PingSonar","/dev/ttyUSB0",3));
  manager.addSubComponent(pingSonar);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,"", 0, 0) == false) return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  Image<PixRGB<byte> > img(800,800,ZEROS);
  int divisions=12;

  while(1)
  {

      std::vector<int> dists = pingSonar->getDists();

      ofs->writeRGB(vizDist(dists,divisions), "Output", FrameInfo("output", SRC_POS));


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
