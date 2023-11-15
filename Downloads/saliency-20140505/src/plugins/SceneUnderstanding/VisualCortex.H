/*!@file Neuro/VisualCortex.H A  */

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
// Primary maintainer for this file: Lior Elazary
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/VisualCortex.H $
// $Id: VisualCortex.H 13765 2010-08-06 18:56:17Z lior $
//


#ifndef VISUALCORTEX_H_DEFINED
#define VISUALCORTEX_H_DEFINED

//#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Simulation/SimEventQueue.H"
#include "Simulation/SimModule.H"
#include "Image/VisualTracker.H"

#include "plugins/SceneUnderstanding/LGN.H"
#include "plugins/SceneUnderstanding/V1.H"
#include "plugins/SceneUnderstanding/V2.H"
#include "plugins/SceneUnderstanding/CornersFeatures.H"
//#include "plugins/SceneUnderstanding/TwoHalfDSketch.H"
#include "plugins/SceneUnderstanding/Geons3D.H"

#include "plugins/SceneUnderstanding/PCamera.H"
//#include "plugins/SceneUnderstanding/Contour.H"
//#include "plugins/SceneUnderstanding/TensorVoting.H"
#include "plugins/SceneUnderstanding/SMap.H"
//#include "plugins/SceneUnderstanding/Regions.H"
//#include "plugins/SceneUnderstanding/V4d.H"
//#include "plugins/SceneUnderstanding/V4.H"
//#include "plugins/SceneUnderstanding/Objects.H"
////#include "plugins/SceneUnderstanding/Geons2D.H"
//#include "plugins/SceneUnderstanding/IT.H"
//#include "Neuro/VisualCortexConfigurator.H"

class ModelManager;

class VisualCortex  : public SimModule
{
public:
  //! Construct using a VisualCortex object with standard channels
  /*! @param mgr our ModelManager (see ModelManager.H)
      @param descrName descriptive name for human useage
      @param tagName name for ParamMap usage */
  VisualCortex(OptionManager& mgr,
      const std::string& descrName = "VC",
      const std::string& tagName = "VC");

  //! virtual destructor for safe inheritance
  virtual ~VisualCortex();

private:
  nub::ref<PCamera> itsPCamera;
  nub::ref<LGN> itsLGN;
  nub::ref<SMap> itsSMap;
  nub::ref<V1> itsV1;
  nub::ref<V2> itsV2;
  nub::ref<CornersFeatures> itsCornersFeatures;
//  nub::ref<Regions> itsRegions;
 // nub::ref<TwoHalfDSketch> itsTwoHalfDSketch;
  nub::ref<Geons3D> itsGeons3D;
//  nub::ref<Objects> itsObjects;
 // nub::ref<V2> itsV2;
 // nub::ref<V4d> itsV4d;
 // nub::ref<V4> itsV4;
////  nub::ref<Geons2D> itsGeons2D;
 // nub::ref<IT> itsIT;

//  nub::ref<VisualCortexConfigurator> itsVcc;



 // nub::ref<VisualTracker> itsVisualTracker;
};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
