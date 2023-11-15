/*! @file SceneUnderstanding/test-tensor.C Test the various vision comp
 * with simple stimulus*/

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/test-SFS.C $
// $Id: test-SFS.C 13413 2010-05-15 21:00:11Z itti $
//

#include "Image/Image.H"
#include "Component/ModelManager.H"
#include "plugins/SceneUnderstanding/TensorVoting.H"
#include "plugins/SceneUnderstanding/V2.H"
#include "plugins/SceneUnderstanding/SFS.H"
#include "Raster/Raster.H"
#include "GUI/DebugWin.H"

#include <signal.h>
#include <sys/types.h>

int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager manager("Test SFS");

  nub::ref<SFS> sfs(new SFS(manager));

  if (manager.parseCommandLine(
        (const int)argc, (const char**)argv, "", 0, 0) == false)
    return 1;

  Image<float> img = Raster::ReadFloat("tests/inputs/lenna_face.pfz");
  manager.start();
  sfs->evolve(img);
  manager.stop();

  return 0;
}

