/*!@file src/Features/test-DPM.C */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// Primary maintainer for this file: Lior Elazary
// $HeadURL$
// $Id$
//

#include "Component/ModelManager.H"
#include "Image/DrawOps.H"
#include "Image/Kernels.H"
#include "Image/CutPaste.H"
#include "Image/ColorOps.H"
#include "Image/FilterOps.H"
#include "Raster/Raster.H"
#include "Media/FrameSeries.H"
#include "Util/Timer.H"
#include "Util/CpuTimer.H"
#include "Util/StringUtil.H"
#include "FeatureMatching/DPM.H"
#include "rutz/rand.h"
#include "rutz/trace.h"
#include "GUI/DebugWin.H"

#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>


int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager manager("Test DPM");

  if (manager.parseCommandLine(
        (const int)argc, (const char**)argv, "<model> <Image> <outputImg>", 3, 3) == false)
    return 0;

  manager.start();

  DPM dpm;
  dpm.readModel(manager.getExtraArg(0).c_str());
  Image<PixRGB<byte> > modelImg = dpm.getModelImage();
  //SHOWIMG(modelImg);

  Image<PixRGB<byte> > img = Raster::ReadRGB(manager.getExtraArg(1));
  //SHOWIMG(img);
  LINFO("Compute pyramid");
  dpm.computeFeaturePyramid(img);
  LINFO("Done");

  dpm.convolveModel();

  std::vector<DPM::Detection> detections = dpm.getBoundingBoxes(-0.50F);

  //non maximal suppression
  detections = dpm.filterDetections(detections, 0.0);
  for(uint i=0; i<detections.size(); i++)
  {
    if (img.rectangleOk(detections[i].bb))
    {
      printf("Detection score %f comp %i bb %i %i %i %i\n", 
          detections[i].score,
          detections[i].component,
          detections[i].bb.left(),
          detections[i].bb.top(),
          detections[i].bb.rightI(),
          detections[i].bb.bottomI()
          );

      drawRect(img, detections[i].bb, PixRGB<byte>(255,0,0));
    }
    //SHOWIMG(img);
  }
  Raster::WriteRGB(img, manager.getExtraArg(2));

  manager.stop();

}



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */



