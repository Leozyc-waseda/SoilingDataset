/*!@file BeoSub/test-BeoSubSaliency.C tests the multi-threaded beosub salincy code */

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
// Primary maintainer for this file: Zack Gossman <gossman@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoSubSaliency.C $
// $Id: test-BeoSubSaliency.C 6454 2006-04-11 00:47:40Z rjpeters $
//

#ifndef TESTBEOSUBSALIENCY_H_DEFINED
#define TESTBEOSUBSALIENCY_H_DEFINED

#include "BeoSub/BeoSubSaliency.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Component/ModelManager.H"
#include "Devices/FrameGrabberFactory.H"
#include "Devices/DeviceOpts.H"

int main(const int argc, const char **argv)
{
  // instantiate a model manager (for camera input):
  ModelManager manager("BeoSubSaliency Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameIstream>
    gb(makeIEEE1394grabber(manager, "smtcam", "smt"));
  manager.addSubComponent(gb);

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeoSubSaliency> smt(new BeoSubSaliency(manager));
  manager.addSubComponent(smt);

  // set the camera number (in IEEE1394 lingo, this is the
  // "subchannel" number):
  gb->setModelParamVal("FrameGrabberSubChan", 0);

  // let's get started:
  manager.start();

  while(1)
    {
      Image< PixRGB<byte> > img = gb->readRGB();
      smt->run(img, true);
    }

  return 0;
}

#endif
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
