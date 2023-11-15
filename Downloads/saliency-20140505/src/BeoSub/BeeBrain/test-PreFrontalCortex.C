/*!@file Beosub/BeeBrain/test-PreFrontalCortex.C                             */
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/test-PreFrontalCortex.C $
// $Id: test-PreFrontalCortex.C 8623 2007-07-25 17:57:51Z rjpeters $

#include "Component/ModelManager.H"
#include "Raster/Raster.H"

#include "Globals.H"

#include "PreFrontalCortex.H"
#include "OceanObject.H"


int main( int argc, const char* argv[] )
{

 // instantiate a model manager:
  ModelManager manager("seabee 2007 competition");

 // create an agent manager
  nub::ref<AgentManagerA> agentManager(new AgentManagerA(manager));
  manager.addSubComponent(agentManager);

  manager.exportOptions(MC_RECURSE);

  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  manager.start();

  agentManager->startRun();

//   rutz::shared_ptr<PreFrontalCortexAgent> pfc(PreFrontalCortexAgent
// ("preFrontalCortexAgent"), agentManager);

//   pfc->start();

//   while(pfc->pickAndExecuteAnAction());

  Raster::waitForKey();

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
