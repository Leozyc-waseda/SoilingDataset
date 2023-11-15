/*!@file plugins/SceneUnderstanding/LineMatching.H  */

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

#ifndef LineMatching_H_DEFINED
#define LineMatching_H_DEFINED

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "plugins/SceneUnderstanding/LineFitting.H"
#include "plugins/SceneUnderstanding/V2.H"
#include "plugins/SceneUnderstanding/LMLineMatcher/LMLineMatcher.h"

#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"

#include <limits>
#include <vector>
#include <string>
#include <stack>
#include <list>
#include <map>

class LineMatching : public SimModule
{
public:


  struct Shape2D
  {
    std::string name;
    double score;
    Rectangle bb;
    std::vector<Point2D<int> > polygon;

    Shape2D(std::string n, double s, Rectangle b, std::vector<Point2D<int> > p) :
      name(n),
      score(s),
      bb(b),
      polygon(p)
    {}
  };

  LineMatching(OptionManager& mgr, const std::string& descrName = "LineMatching",
      const std::string& tagName = "LineMatching");

  //! Destructor
  ~LineMatching();

  void evolve(SimEventQueue& q);
  Layout<PixRGB<byte> > getDebugImage();

protected:
  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(LineMatching, SimEventV2Output);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(LineMatching, SimEventSaveOutput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;


private:
  std::vector<V2::LineSegment> itsLines;
  Image<PixRGB<byte> > itsInImage;
  std::vector<Shape2D> itsShapes;

};

class SimEventLineMatchingOutput : public SimEvent
{
public:
  SimEventLineMatchingOutput(SimModule* src,
      std::vector<LineMatching::Shape2D>& shapes) :
    SimEvent(src), itsShapes(shapes)
  {}

  virtual ~SimEventLineMatchingOutput(){}

  std::vector<LineMatching::Shape2D> getShapes() { return itsShapes; }

private:
  const std::vector<LineMatching::Shape2D>& itsShapes;
};


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
