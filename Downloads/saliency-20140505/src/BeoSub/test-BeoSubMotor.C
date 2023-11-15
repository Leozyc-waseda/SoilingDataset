/*!@file BeoSub/test-BeoSubMotor.C Test BeoSub submarine motors */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoSubMotor.C $
// $Id: test-BeoSubMotor.C 12074 2009-11-24 07:51:51Z itti $
//

#include "BeoSub/BeoSubMotor.H"
#include "Component/ModelManager.H"
#include <cstdio>

//! Simple test program for the BeoSub submarine
/*! Simple test program for the BeoSub submarine */
int main(const int argc, const char **argv)
{
  // instantiate a model manager:
  ModelManager manager("BeoSub Motor Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeoSubMotor> mot(new BeoSubMotor(manager));
  manager.addSubComponent(mot);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<chan> <val>",
                               2, 2) == false) return(1);

  // do post-command-line configs:
  int chan = manager.getExtraArgAs<int>(0);
  int val = manager.getExtraArgAs<int>(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  // main loop:
  LINFO("[RETURN] to set channel %d to value %d...", chan, val);
  getchar();
  mot->setValue(val, chan, true);
  LINFO("All done. -- [RETURN] to exit");
  getchar();

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
