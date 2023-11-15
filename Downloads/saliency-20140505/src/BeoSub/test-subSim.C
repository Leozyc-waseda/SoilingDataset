/*!@file AppDevices/test-armSim.C Test the sub simulator */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-subSim.C $
// $Id: test-subSim.C 11564 2009-08-09 01:10:32Z rand $
//



#include "Component/ModelManager.H"
#include "Raster/GenericFrame.H"
#include "Image/Layout.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Image/MatrixOps.H"
#include "GUI/ImageDisplayStream.H"
#include "GUI/XWinManaged.H"
#include "BeoSub/SubSim.H"
#include <stdio.h>
#include <stdlib.h>

void handle_keys(nub::soft_ref<OutputFrameSeries> ofs, nub::soft_ref<SubSim> subSim)
{
    //handle keyboard input
    const nub::soft_ref<ImageDisplayStream> ids =
      ofs->findFrameDestType<ImageDisplayStream>();

    const rutz::shared_ptr<XWinManaged> uiwin =
      ids.is_valid()
      ? ids->getWindow("subSim")
      : rutz::shared_ptr<XWinManaged>();

    int key = uiwin->getLastKeyPress();
    if (key != -1)
    {
      float panTruster = 0;
      float tiltTruster = 0;
      float forwardTruster = 0;
      float upTruster = 0;
      switch(key)
      {
        case 38: upTruster = -3.0; break; //a
        case 52: upTruster = 3.0; break; //z
        case 33: panTruster = 1.0; break; //p
        case 32: panTruster = -1.0; break; //o
        case 40: forwardTruster = 1.0; break; //d
        case 54: forwardTruster = -1.0; break; //c
        case 39: tiltTruster = 1.0; break; //s
        case 53: tiltTruster = -1.0; break; //x
        case 65: //stop
                 panTruster = 0;
                 tiltTruster = 0;
                 forwardTruster = 0;
                 upTruster = 0;
                 break; // space

      }
      subSim->setTrusters(panTruster, tiltTruster, forwardTruster, upTruster);

      LINFO("Key is %i\n", key);
    }
}


int main(int argc, char *argv[])
{
  // Instantiate a ModelManager:
  ModelManager manager("Sub Simulator");

  //nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  //manager.addSubComponent(ifs);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  // Instantiate our various ModelComponents:
  nub::soft_ref<SubSim> subSim(new SubSim(manager));
  manager.addSubComponent(subSim);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);


  // let's get all our ModelComponent instances started:
        manager.start();



        while(1){
          Layout<PixRGB<byte> > outDisp;

          subSim->simLoop();
          Image<PixRGB<byte> > forwardCam = flipVertic(subSim->getFrame(1));
          Image<PixRGB<byte> > downwardCam = flipVertic(subSim->getFrame(2));

          outDisp = vcat(outDisp, hcat(forwardCam, downwardCam));

          ofs->writeRgbLayout(outDisp, "subSim", FrameInfo("subSim", SRC_POS));

          handle_keys(ofs, subSim);

        }
        return 0;

}
