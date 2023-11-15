/*!@file Script/ModelScript.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Script/ModelScript.C $
// $Id: ModelScript.C 11876 2009-10-22 15:53:06Z icore $
//

#ifndef SCRIPT_MODELSCRIPT_C_DEFINED
#define SCRIPT_MODELSCRIPT_C_DEFINED

#include "Script/ModelScript.H"

#include "tcl/stdconversions.h"

#include "Component/ModelManager.H"
#include "Neuro/StdBrain.H"
#include "Util/log.H"
#include "tcl/list.h"
#include "tcl/objpkg.h"
#include "tcl/pkg.h"

#include <sstream>
#include <string>
#include <tcl.h>

namespace
{
  nub::soft_ref<ModelManager> theManager;

  nub::soft_ref<StdBrain> makeStdBrain()
  {
    return nub::soft_ref<StdBrain>(new StdBrain(*theManager));
  }

  void setMPString(nub::soft_ref<ModelComponent> comp,
                   const std::string& name,
                   const std::string& value)
  {
    comp->setModelParamString(name, value);
  }

  std::string getMPString(nub::soft_ref<ModelComponent> comp,
                          const std::string& name)
  {
    return comp->getModelParamString(name);
  }

  std::string printout(nub::soft_ref<ModelComponent> comp)
  {
    std::ostringstream strm;
    comp->printout(strm);
    return strm.str();
  }

  bool mmParse(nub::soft_ref<ModelManager> mm,
               tcl::list argv, const char* usage,
               int minarg, int maxarg)
  {
    const char* args[argv.length()];

    for (uint i = 0; i < argv.length(); ++i)
      args[i] = argv.get<const char*>(i);

    return mm->parseCommandLine(argv.length(), args, usage,
                                minarg, maxarg);
  }

  tcl::list mmExtraArgs(nub::soft_ref<ModelManager> mm)
  {
    tcl::list result;
    for (uint i = 0; i < mm->numExtraArgs(); ++i)
      result.append(mm->getExtraArg(i));
    return result;
  }

  int evolveMaster(nub::ref<SimEventQueue> seq)
  {
    const SimStatus status = seq->evolve();
    return status == SIM_BREAK;
  }
}

OptionManager& getManager()
{
  return *theManager;
}

extern "C"
int Modelcomponent_Init(Tcl_Interp* interp)
{
  GVX_PKG_CREATE(pkg, interp, "Modelcomponent", "4.$Revision: 1$");
  pkg->inherit_pkg("Obj");
  tcl::def_basic_type_cmds<ModelComponent>(pkg, SRC_POS);

  pkg->def("start", "objref",
           &ModelComponent::start, SRC_POS);
  pkg->def("stop", "objref",
           &ModelComponent::stop, SRC_POS);
  pkg->def("started", "objref",
           &ModelComponent::started, SRC_POS);
  pkg->def("descriptiveName", "objref",
           &ModelComponent::descriptiveName, SRC_POS);
  pkg->def("tagName", "objref",
           &ModelComponent::tagName, SRC_POS);
  pkg->def("tagName", "objref newname",
           &ModelComponent::setTagName, SRC_POS);
  pkg->def("addSubComponent", "objref subcomp propgrealm",
           &ModelComponent::addSubComponent, SRC_POS);
  pkg->def("removeSubComponentByIndex", "objref index",
           (void (ModelComponent::*)(uint))
           &ModelComponent::setTagName, SRC_POS);
  pkg->def("removeSubComponentByName", "objref tagname ",
           (void (ModelComponent::*)(const std::string&))
           &ModelComponent::setTagName, SRC_POS);
  pkg->def("removeAllSubComponents", "objref",
           &ModelComponent::removeAllSubComponents, SRC_POS);
  pkg->def("numSubComp", "objref",
           &ModelComponent::numSubComp, SRC_POS);
  pkg->def("subCompByIndex", "objref index",
           (nub::ref<ModelComponent> (ModelComponent::*)(uint) const)
           &ModelComponent::subComponent, SRC_POS);
  pkg->def("subCompByName", "objref tagname flags",
           (nub::ref<ModelComponent> (ModelComponent::*)(const std::string&, int) const)
           &ModelComponent::subComponent, SRC_POS);
  pkg->def("hasSubComponent", "objref tagname flags",
           (bool (ModelComponent::*)(const std::string&, int) const)
           &ModelComponent::hasSubComponent, SRC_POS);
  pkg->def("printout", "objref",
           &printout, SRC_POS);
  pkg->def("reset", "objref do_recurse",
           &ModelComponent::reset, SRC_POS);

  pkg->def("param", "objref paramname",
           &getMPString, SRC_POS);

  pkg->def("param", "objref paramname paramvalue",
           &setMPString, SRC_POS);

  GVX_PKG_RETURN(pkg);
}

extern "C"
int Modelmanager_Init(Tcl_Interp* interp)
{
  GVX_PKG_CREATE(pkg, interp, "ModelManager", "4.$Revision: 1$");
  pkg->inherit_pkg("Modelcomponent");
  tcl::def_basic_type_cmds<ModelManager>(pkg, SRC_POS);

  theManager.reset(new ModelManager("iLab Neuromorphic Vision Toolkit"));

  registerComponentCreator<StdBrain>();

  pkg->def("parseCommandLine", "objref argv usage minarg maxarg",
           &mmParse, SRC_POS);

  pkg->def("extraArgs", "objref", &mmExtraArgs, SRC_POS);

  GVX_PKG_RETURN(pkg);
}

extern "C"
int Simmodule_Init(Tcl_Interp* interp)
{
  GVX_PKG_CREATE(pkg, interp, "SimModule", "4.$Revision: 1$");
  pkg->inherit_pkg("Modelcomponent");
  tcl::def_basic_type_cmds<SimModule>(pkg, SRC_POS);

  GVX_PKG_RETURN(pkg);
}

extern "C"
int Simeventqueue_Init(Tcl_Interp* interp)
{
  GVX_PKG_CREATE(pkg, interp, "SimEventQueue", "4.$Revision: 1$");
  pkg->inherit_pkg("Modelcomponent");
  tcl::def_basic_type_cmds<SimEventQueue>(pkg, SRC_POS);

  registerComponentCreator<SimEventQueue>();

  pkg->def("evolveMaster", "objref", &evolveMaster, SRC_POS);

  //pkg->def("now", "objref", &SimEventQueue::now, SRC_POS);

  GVX_PKG_RETURN(pkg);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // SCRIPT_MODELSCRIPT_C_DEFINED
