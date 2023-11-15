/*!@file Script/NeuroScript.C */

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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Script/NeuroScript.C $
// $Id: NeuroScript.C 11876 2009-10-22 15:53:06Z icore $
//

#ifndef SCRIPT_NEUROSCRIPT_C_DEFINED
#define SCRIPT_NEUROSCRIPT_C_DEFINED

#include "Script/NeuroScript.H"

#include "Media/FrameSeries.H"
#include "Neuro/AttentionGuidanceMap.H"
#include "Neuro/Brain.H"
#include "Neuro/GistEstimator.H"
#include "Neuro/InferoTemporal.H"
#include "Neuro/Retina.H"
#include "Neuro/EyeHeadController.H"
#include "Neuro/NeuroSimEvents.H"
#include "Neuro/SaliencyMap.H"
#include "Neuro/ShapeEstimator.H"
#include "Neuro/SimulationViewer.H"
#include "Neuro/StdBrain.H"
#include "Neuro/TaskRelevanceMap.H"
#include "Neuro/VisualCortex.H"
#include "Neuro/WinnerTakeAll.H"
#include "Script/ImageScript.H" // for tcl<->Image conversions
#include "Script/MediaScript.H" // for tcl<->SimTime conversions
#include "Script/ModelScript.H" // for registerComponentCreator()
#include "tcl/objpkg.h"
#include "tcl/list.h"
#include "tcl/pkg.h"

namespace
{
  tcl::list brainEvolve(nub::soft_ref<Brain> brain,
                        nub::soft_ref<SimEventQueue> q)
  {
    const SimStatus status = q->evolve(); // note: this should come last...
    bool covertshift = false;
    if (SeC<SimEventWTAwinner> e = q->check<SimEventWTAwinner>(0))
      covertshift = true;

    tcl::list result;
    result.append(q->now());
    result.append(covertshift);
    result.append(SIM_BREAK == status);

    return result;
  }
}

extern "C"
int Brain_Init(Tcl_Interp* interp)
{
  GVX_PKG_CREATE(pkg, interp, "Brain", "4.$Revision: 1$");
  pkg->inherit_pkg("Modelcomponent");
  tcl::def_basic_type_cmds<Brain>(pkg, SRC_POS);

  registerComponentCreator<Brain>();

  pkg->def("evolve", "objref seq", &brainEvolve, SRC_POS);

  //  pkg->def("saveResults", "objref ofs", &Brain::saveResults, SRC_POS);

  GVX_PKG_RETURN(pkg);
}


extern "C"
int Stdbrain_Init(Tcl_Interp* interp)
{
  GVX_PKG_CREATE(pkg, interp, "StdBrain", "4.$Revision: 1$");
  pkg->inherit_pkg("Brain");
  tcl::def_basic_type_cmds<StdBrain>(pkg, SRC_POS);

  registerComponentCreator<StdBrain>();

  GVX_PKG_RETURN(pkg);
}

extern "C"
int Winnertakeall_Init(Tcl_Interp* interp)
{
  GVX_PKG_CREATE(pkg, interp, "WinnerTakeAll", "4.$Revision: 1$");
  pkg->inherit_pkg("Modelcomponent");
  tcl::def_basic_type_cmds<WinnerTakeAll>(pkg, SRC_POS);

  GVX_PKG_RETURN(pkg);
}

extern "C"
int Winnertakeallstd_Init(Tcl_Interp* interp)
{
  GVX_PKG_CREATE(pkg, interp, "WinnerTakeAllStd", "4.$Revision: 1$");
  pkg->inherit_pkg("Winnertakeall");
  tcl::def_basic_type_cmds<WinnerTakeAllStd>(pkg, SRC_POS);

  GVX_PKG_RETURN(pkg);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // SCRIPT_NEUROSCRIPT_C_DEFINED
