/*!@file Neuro/VisualCortex.C */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/VisualCortex.C $
// $Id: VisualCortex.C 13413 2010-05-15 21:00:11Z itti $
//

#include "plugins/SceneUnderstanding/VisualCortex.H"

// ######################################################################
VisualCortex::VisualCortex(OptionManager& mgr,
    const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  itsPCamera(new PCamera(mgr)),
  itsLGN(new LGN(mgr)),
  itsSMap(new SMap(mgr)),
  itsV1(new V1(mgr)),
  itsV2(new V2(mgr)),
  itsCornersFeatures(new CornersFeatures(mgr)),
  //itsRegions(new Regions(mgr)),
  //itsTwoHalfDSketch(new TwoHalfDSketch(mgr))
  itsGeons3D(new Geons3D(mgr))
 // itsObjects(new Objects(mgr))
  //itsV2(new V2(mgr)),
  //itsV4d(new V4d(mgr)),
  //itsV4(new V4(mgr)),
//  itsGeons2D(new Geons2D(mgr)),
  //itsIT(new IT(mgr)),
  //itsVcc(new VisualCortexConfigurator(mgr)),
  //itsVisualTracker(new VisualTracker(mgr))
{
  // NOTE: if callback priorities are all zero, the order here defines
  // the order in which the callbacks will be called when several
  // modules are catching a given event:
  //addSubComponent(itsPCamera);
  addSubComponent(itsLGN);
  addSubComponent(itsSMap);
  addSubComponent(itsV1);
  addSubComponent(itsV2);
  addSubComponent(itsCornersFeatures);
 // addSubComponent(itsRegions);
  //addSubComponent(itsTwoHalfDSketch);
  addSubComponent(itsGeons3D);
  //addSubComponent(itsContour);
  //addSubComponent(itsObjects);

}

// ######################################################################
VisualCortex::~VisualCortex()
{  }

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
