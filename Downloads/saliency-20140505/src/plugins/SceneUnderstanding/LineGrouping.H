/*!@file SceneUnderstanding/LineGrouping.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/LineGrouping.H $
// $Id: LineGrouping.H 14179 2010-10-28 19:49:24Z lior $
//

#ifndef LineGrouping_H_DEFINED
#define LineGrouping_H_DEFINED

#include "Image/OpenCVUtil.H" //Need to be first to avoid type def conf
#include "Util/Types.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "plugins/SceneUnderstanding/V2.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
#include "GUI/ViewPort3D.H"
#include "Learn/HMM.H"

#include <vector>
#include <string>
#include <map>
#include <queue>

class SimEventLineGroupingPrior;


class LineGrouping : public SimModule
{
public:

  struct LinesGroup
  {
    std::vector<V2::LineSegment> lines;
  };

  struct LineInfo
  {
    uint idx;
    double cost;

    LineInfo(uint i) :
      idx(i), cost(0)
    {
    }

    LineInfo(uint i, double c) :
      idx(i), cost(c)
    {
    }

  };

  struct LineInfoCmp
  {
    bool operator()(const LineInfo& l1, const LineInfo& l2)
    {
      return l1.cost > l2.cost;
    }
  };

  LineGrouping(OptionManager& mgr, const std::string& descrName = "LineGrouping",
      const std::string& tagName = "LineGrouping");

  //! Destructor
  ~LineGrouping();

  void evolve(SimEventQueue& q);

  Layout<PixRGB<byte> > getDebugImage(SimEventQueue& q);

  std::vector<V2::LineSegment>  getAndRemoveLinesNearLoc(Image<std::vector<uint> >& linesIndices,
      const Point2D<int> loc, const int radius);

  //! Get lines that are local (within radius) and have not been used
  std::vector<LineInfo>  getLocalLines(const Point2D<int> loc, const int radius, 
      std::vector<uint>& lineColour,
      const Image<std::vector<uint> >& linesIndices);
  
 
  void setTopDownCost(std::vector<LineInfo>& newLines,const std::vector<LineInfo>& contour);
  

protected:
  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(LineGrouping, SimEventV2Output);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(LineGrouping, SimEventSaveOutput);

  ////! Callback for every time we have a user event
  SIMCALLBACK_DECLARE(LineGrouping, SimEventUserInput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;


private:
  std::vector<V2::LineSegment> itsLines;
  std::vector<LinesGroup> itsLineGroups;
  HMM<uint> itsHMM;

  Dims itsInputDims;
  
};

/* ############################### V2 sim events ######################## */
/*
class SimEventLineGroupingOutput : public SimEvent
{
public:
  SimEventLineGroupingOutput(SimModule* src,
      std::vector<LineGrouping::SurfaceState>& surfaces) :
    SimEvent(src), itsLineGroups(surfaces)
  {}

  virtual ~SimEventLineGroupingOutput(){}
  std::vector<LineGrouping::SurfaceState> getSurfaces() { return itsSurfaces; }

private:
  const std::vector<LineGrouping::SurfaceState>& itsSurfaces;
};
*/



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
