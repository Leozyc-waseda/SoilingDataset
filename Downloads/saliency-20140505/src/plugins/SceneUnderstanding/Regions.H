/*!@file SceneUnderstanding/Regions.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/Regions.H $
// $Id: Regions.H 13413 2010-05-15 21:00:11Z itti $
//

#ifndef Regions_H_DEFINED
#define Regions_H_DEFINED

#include "Util/Types.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "Image/Point3D.H"
#include "plugins/SceneUnderstanding/LGN.H"
#include "plugins/SceneUnderstanding/SMap.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
#include "Media/MediaSimEvents.H"
#include "Component/ModelOptionDef.H"
#include "Component/OptionManager.H" // for REQUEST_OPTIONALIAS_NEURO()
#include "Component/ModelParam.H"

#include <vector>
#include <string>

class SimEventRegionsPrior;

class Regions : public SimModule
{
public:

  struct RegionState
  {
    std::vector<Point2D<int> > region;
    std::vector<Point2D<int> > notRegion;
    double prob;
    double mu; //the mean of the
    double sigma;
    RegionState() {}
  };


  Regions(OptionManager& mgr, const std::string& descrName = "Regions",
      const std::string& tagName = "Regions");

  //! Destructor
  ~Regions();

  void evolve();

  Layout<PixRGB<byte> > getDebugImage();

  void calcRegionLikelihood(RegionState& rs);

  double calcRegionLikelihood(Image<float>& mu);

protected:
  //! Callback for when a new LGN output is ready
  SIMCALLBACK_DECLARE(Regions, SimEventLGNOutput);

  //! Callback for when a new LGN output is ready
  SIMCALLBACK_DECLARE(Regions, SimEventSMapOutput);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(Regions, SimEventSaveOutput);

  //! Callback for every time we have a user event
  SIMCALLBACK_DECLARE(Regions, SimEventUserInput);

  //! Callback for every time we have a prior input
  SIMCALLBACK_DECLARE(Regions, SimEventRegionsPrior);


  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;

private:

  ImageSet<float> itsRegionsCellsInput; //The input to the cells
  std::vector<SMap::SMapState> itsSMapInput; //the proposed regions from the saliency map
  Point2D<int> itsProbe;
  RegionState itsBackgroundRegion;
  std::vector<RegionState> itsRegionsState;
  std::vector<RegionState> itsRegionsPrior;
  std::vector<Point2D<int> > itsUserPolygon;
  float itsTempColor;

};

/* ############################### Regions sim events ######################## */
class SimEventRegionsOutput : public SimEvent
{
public:
  SimEventRegionsOutput(SimModule* src, std::vector<Regions::RegionState>& cellsOutput) :
    SimEvent(src), itsCells(cellsOutput)
  {}

  virtual ~SimEventRegionsOutput(){}
  std::vector<Regions::RegionState> getCells() { return itsCells; }

private:
  const std::vector<Regions::RegionState>& itsCells;
};

/* ############################### Objects sim events ######################## */
class SimEventRegionsPrior : public SimEvent
{
public:
  SimEventRegionsPrior(SimModule* src,
      std::vector<Regions::RegionState>& regions) :
    SimEvent(src), itsRegions(regions)
  {}

  virtual ~SimEventRegionsPrior(){}
  std::vector<Regions::RegionState> getRegions() { return itsRegions; }

private:
  const std::vector<Regions::RegionState>& itsRegions;
};



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
