/*!@file SceneUnderstanding/SMap.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/SMap.H $
// $Id: SMap.H 13765 2010-08-06 18:56:17Z lior $
//

#ifndef SMap_H_DEFINED
#define SMap_H_DEFINED

//#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "Image/Point3D.H"
#include "plugins/SceneUnderstanding/LGN.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
#include "Media/MediaSimEvents.H"
#include "Neuro/EnvVisualCortex.H"
#include "Component/ModelOptionDef.H"
#include "Component/OptionManager.H" // for REQUEST_OPTIONALIAS_NEURO()
#include "Component/ModelParam.H"

#include <vector>
#include <string>

class SMap : public SimModule
{
public:

  struct SMapState
  {
    std::vector<Point2D<int> > region;
    double mu; //the mean of the
    double sigma;
    Point2D<int> center;
    SMapState() : mu(-1),sigma(-1) {}
  };


  SMap(OptionManager& mgr, const std::string& descrName = "SMap",
      const std::string& tagName = "SMap");

  //! Destructor
  ~SMap();

  void evolve();

  Layout<PixRGB<byte> > getDebugImage();

protected:
  //! Callback for when a new LGN output is ready
  SIMCALLBACK_DECLARE(SMap, SimEventInputFrame);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(SMap, SimEventSaveOutput);

  //! Callback for every time we have a user event
  SIMCALLBACK_DECLARE(SMap, SimEventUserInput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;

private:
  nub::soft_ref<EnvVisualCortex> itsEvc;
  Image<PixRGB<byte> > itsSMapCellsInput; //The input to the cells
  std::vector<SMapState> itsSMapState;
  Image<byte> itsSMap;
  Image<byte> tmpEdges;

  float itsTempColor;

};

/* ############################### SMap sim events ######################## */
class SimEventSMapOutput : public SimEvent
{
public:
  SimEventSMapOutput(SimModule* src, std::vector<SMap::SMapState>& cellsOutput,
      Image<byte>& smap) :
    SimEvent(src), itsCells(cellsOutput), itsSMap(smap)
  {}

  virtual ~SimEventSMapOutput(){}
  std::vector<SMap::SMapState> getCells() { return itsCells; }
  Image<byte> getSMap() { return itsSMap; }

private:
  const std::vector<SMap::SMapState>& itsCells;
  const Image<byte>& itsSMap;
};


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
