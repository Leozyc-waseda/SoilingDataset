/*!@file SceneUnderstanding/EGMM.C features based on edges of mixture of Gaussian  */

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
// $HeadURL: $
// $Id: $
//

#ifndef EGMM_C_DEFINED
#define EGMM_C_DEFINED

#include "Component/ModelManager.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
//#include "Image/OpenCVUtil.H"
#include "Image/Layout.H"
#include "plugins/SceneUnderstanding/EGMM.H"
#include "Simulation/SimEventQueue.H"
#include "Simulation/SimEvents.H"
#include "Media/MediaSimEvents.H"
#include "Channels/InputFrame.H"
#include "Image/Kernels.H"
#include "Image/FilterOps.H"
#include "Image/Convolutions.H"
#include "GUI/DebugWin.H"
#include <math.h>
#include <fcntl.h>
#include <limits>
#include <string>

const ModelOptionCateg MOC_EGMM = {
  MOC_SORTPRI_3,   "EGMM-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_EGMMShowDebug =
  { MODOPT_ARG(bool), "EGMMShowDebug", &MOC_EGMM, OPTEXP_CORE,
    "Show debug img",
    "egmm-debug", '\0', "<true|false>", "false" };


// ######################################################################
EGMM::EGMM(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventInputFrame),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  itsShowDebug(&OPT_EGMMShowDebug, this)

{
}

// ######################################################################
EGMM::~EGMM()
{
	LINFO("Destory Module");

}

// ######################################################################
void EGMM::onSimEventInputFrame(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventInputFrame>& e)
{
  // here is the inputs image:
  const Image<PixRGB<byte> > inimg = e->frame().asRgb();

  itsCurrentImg = inimg;


  evolve();

}

// ######################################################################
void EGMM::onSimEventSaveOutput(SimEventQueue& q, rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage();
      ofs->writeRgbLayout(disp, "EGMM", FrameInfo("EGMM", SRC_POS));
    }
}

// ######################################################################
void EGMM::evolve()
{

}

Layout<PixRGB<byte> > EGMM::getDebugImage()
{
  Layout<PixRGB<byte> > outDisp;
  outDisp = itsCurrentImg;
  return outDisp;

}

//Create and destory the brain
extern "C" nub::ref<SimModule> createObj( OptionManager& manager,
 const std::string& name)
{
  return nub::ref<SimModule>(new EGMM(manager));
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif

