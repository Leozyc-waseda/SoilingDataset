/*!@file Script/MediaScript.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Script/MediaScript.C $
// $Id: MediaScript.C 11876 2009-10-22 15:53:06Z icore $
//

#ifndef SCRIPT_MEDIASCRIPT_C_DEFINED
#define SCRIPT_MEDIASCRIPT_C_DEFINED

#include "Script/MediaScript.H"

#include "Component/OptionManager.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Media/FrameSeries.H"
#include "Media/SimFrameSeries.H"
#include "Script/ImageScript.H" // for tcl conversions
#include "Script/ModelScript.H" // for registerComponentCreator()
#include "rutz/error.h"
#include "rutz/sfmt.h"
#include "tcl/list.h"
#include "tcl/objpkg.h"
#include "tcl/pkg.h"
#include "tcl/stdconversions.h"

using nub::soft_ref;

namespace
{
  tcl::list ifsDims(soft_ref<InputFrameSeries> ifs)
  {
    Dims d = ifs->peekDims();
    tcl::list result;
    result.append(d.w());
    result.append(d.h());
    return result;
  }

  Image< PixRGB<byte> > ifsReadRGB(soft_ref<InputFrameSeries> ifs)
  { return ifs->readRGB(); }

  Image<byte> ifsReadGray(soft_ref<InputFrameSeries> ifs)
  { return ifs->readGray(); }

  void ofsWriteRGB(soft_ref<OutputFrameSeries> ofs,
                   const Image< PixRGB<byte> >& image,
                   const char* stem)
  { ofs->writeRGB(image, stem); }

  void ofsWriteGray(soft_ref<OutputFrameSeries> ofs,
                    const Image<byte>& image,
                    const char* stem)
  { ofs->writeGray(image, stem); }

  void ofsWriteFloat(soft_ref<OutputFrameSeries> ofs,
                     const Image<float>& image,
                     const char* stem)
  { ofs->writeFloat(image, FLOAT_NORM_0_255, stem); }
}

tcl::obj tcl::aux_convert_from(const FrameState s)
{
  switch (s)
    {
    case FRAME_SAME:     return tcl::convert_from("SAME");
    case FRAME_NEXT:     return tcl::convert_from("NEXT");
    case FRAME_FINAL:    return tcl::convert_from("FINAL");
    case FRAME_COMPLETE: return tcl::convert_from("COMPLETE");
    }

  // else...
  throw rutz::error(rutz::sfmt("unknown FrameState '%d'", int(s)),
                    SRC_POS);

  /* can't happen */ return tcl::obj();
}

FrameState tcl::aux_convert_to(Tcl_Obj* obj, FrameState*)
{
  const rutz::fstring f = tcl::aux_convert_to(obj, (rutz::fstring*)0);

  if      (f == "SAME")     return FRAME_SAME;
  else if (f == "NEXT")     return FRAME_NEXT;
  else if (f == "FINAL")    return FRAME_FINAL;
  else if (f == "COMPLETE") return FRAME_COMPLETE;

  // else...
  throw rutz::error(rutz::sfmt("unknown FrameSeries:State '%s'",
                               f.c_str()), SRC_POS);

  /* can't happen */ return FrameState(0);
}

tcl::obj tcl::aux_convert_from(SimTime t)
{
  return tcl::convert_from(t.nsecs());
}

SimTime tcl::aux_convert_to(Tcl_Obj* obj, SimTime*)
{
  return SimTime::NSECS(tcl::aux_convert_to(obj, (int64*)0));
}

extern "C"
int Frameseries_Init(Tcl_Interp* interp)
{
  GVX_PKG_CREATE(pkg, interp, "FrameSeries", "4.$Revision: 1$");
  pkg->inherit_pkg("Modelcomponent");
  tcl::def_basic_type_cmds<ModelComponent>(pkg, SRC_POS);

  GVX_PKG_RETURN(pkg);
}

extern "C"
int Inputframeseries_Init(Tcl_Interp* interp)
{
  GVX_PKG_CREATE(pkg, interp, "InputFrameSeries", "4.$Revision: 1$");
  pkg->inherit_pkg("Frameseries");
  tcl::def_basic_type_cmds<InputFrameSeries>(pkg, SRC_POS);

  registerComponentCreator<InputFrameSeries>();

  pkg->def("update", "stime", &InputFrameSeries::update, SRC_POS);
  pkg->def("updateNext", "objref", &InputFrameSeries::updateNext, SRC_POS);
  pkg->def("shouldWait", "objref", &InputFrameSeries::shouldWait, SRC_POS);

  pkg->def("dims", "objref", &ifsDims, SRC_POS);
  pkg->def("readRGB", "objref", &ifsReadRGB, SRC_POS);
  pkg->def("readGray", "objref", &ifsReadGray, SRC_POS);

  GVX_PKG_RETURN(pkg);
}

extern "C"
int Outputframeseries_Init(Tcl_Interp* interp)
{
  GVX_PKG_CREATE(pkg, interp, "OutputFrameSeries", "4.$Revision: 1$");
  pkg->inherit_pkg("Frameseries");
  tcl::def_basic_type_cmds<OutputFrameSeries>(pkg, SRC_POS);

  registerComponentCreator<OutputFrameSeries>();

  pkg->def("update", "stime new_event", &OutputFrameSeries::update, SRC_POS);
  pkg->def("updateNext", "objref", &OutputFrameSeries::updateNext, SRC_POS);
  pkg->def("shouldWait", "objref", &OutputFrameSeries::shouldWait, SRC_POS);

  pkg->def("writeRGB", "ofs img stem", &ofsWriteRGB, SRC_POS);
  pkg->def("writeGray", "ofs img stem", &ofsWriteGray, SRC_POS);
  pkg->def("writeFloat", "ofs img stem", &ofsWriteFloat, SRC_POS);

  GVX_PKG_RETURN(pkg);
}

extern "C"
int Siminputframeseries_Init(Tcl_Interp* interp)
{
  GVX_PKG_CREATE(pkg, interp, "SimInputFrameSeries", "4.$Revision: 1$");
  pkg->inherit_pkg("Simmodule");
  tcl::def_basic_type_cmds<SimInputFrameSeries>(pkg, SRC_POS);

  registerComponentCreator<SimInputFrameSeries>();

  GVX_PKG_RETURN(pkg);
}

extern "C"
int Simoutputframeseries_Init(Tcl_Interp* interp)
{
  GVX_PKG_CREATE(pkg, interp, "SimOutputFrameSeries", "4.$Revision: 1$");
  pkg->inherit_pkg("Simmodule");
  tcl::def_basic_type_cmds<SimOutputFrameSeries>(pkg, SRC_POS);

  registerComponentCreator<SimOutputFrameSeries>();

  GVX_PKG_RETURN(pkg);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // SCRIPT_MEDIASCRIPT_C_DEFINED
