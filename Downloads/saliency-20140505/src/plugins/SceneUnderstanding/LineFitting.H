/*!@file plugins/SceneUnderstanding/LineFitting.H  */

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

#ifndef LineFitting_H_DEFINED
#define LineFitting_H_DEFINED

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "plugins/SceneUnderstanding/V1.H"
#include "plugins/SceneUnderstanding/V2.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"

#include <limits>
#include <vector>
#include <string>
#include <stack>
#include <list>
#include <map>

class LineFitting : public SimModule
{
public:

  LineFitting(OptionManager& mgr, const std::string& descrName = "LineFitting",
      const std::string& tagName = "LineFitting");

  //! Destructor
  ~LineFitting();

  void evolve(SimEventQueue& q);
  Layout<PixRGB<byte> > getDebugImage();


  std::vector<V2::LineSegment> FitLine(const Image<float>& edges);
  

protected:
  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(LineFitting, SimEventV1Output);

  //! Callback for when a new input frame is available
  SIMCALLBACK_DECLARE(LineFitting, SimEventInputFrame);
  

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(LineFitting, SimEventSaveOutput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;


private:
  ImageSet<float>  itsInput;
  Image<PixRGB<byte> > itsInImage;
  std::vector<V2::LineSegment> itsLines;

};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
