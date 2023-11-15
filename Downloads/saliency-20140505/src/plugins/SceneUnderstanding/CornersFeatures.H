/*!@file SceneUnderstanding/CornersFeatures.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/CornersFeatures.H $
// $Id: CornersFeatures.H 13765 2010-08-06 18:56:17Z lior $
//

#ifndef CornersFeatures_H_DEFINED
#define CornersFeatures_H_DEFINED

//#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "plugins/SceneUnderstanding/V2.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
//#include "Image/OpenCVUtil.H"
#include "Learn/SOFM.H"

#include <vector>
#include <string>


class CornersFeatures : public SimModule
{
public:

  struct CornerState
  {
    Point2D<float> center; //Rename to loc
    std::vector<float> angles;
    double prob;

    CornerState(std::vector<float>& an) :
      center(-1,-1), angles(an), prob(1)
    {}

    CornerState() 
    {}
  };

  CornersFeatures(OptionManager& mgr, const std::string& descrName = "CornersFeatures",
      const std::string& tagName = "CornersFeatures");

  //! Destructor
  ~CornersFeatures();


  void evolve(SimEventQueue& q);

  std::vector<CornersFeatures::CornerState> getCorners(std::vector<V2::LineSegment>& lines);
  
  

  Layout<PixRGB<byte> > getDebugImage(SimEventQueue& q);

protected:
  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(CornersFeatures, SimEventV2Output);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(CornersFeatures, SimEventSaveOutput);

  //! Callback for every time we have a user event
  SIMCALLBACK_DECLARE(CornersFeatures, SimEventUserInput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;


private:
  Dims itsPatchSize; //The patch used to describe the corners
  std::vector<V2::LineSegment> itsLines;
  Image<float> itsCornersProposals;
  SOFM* itsSOFM; //SOFM used to quanite corners into codewords
  Image<PixRGB<byte> > itsSOFMMap;

  std::vector<CornerState> itsCorners;
  std::vector<CornerState> itsCornersDB;

};

/* ############################### Geons3D sim events ######################## */
class SimEventCornersOutput : public SimEvent
{
public:
  SimEventCornersOutput(SimModule* src,
      std::vector<CornersFeatures::CornerState>& corners) :
    SimEvent(src), itsCorners(corners)
  {}

  virtual ~SimEventCornersOutput(){}
  std::vector<CornersFeatures::CornerState> getCorners() { return itsCorners; }

private:
  const std::vector<CornersFeatures::CornerState>& itsCorners;
};


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
