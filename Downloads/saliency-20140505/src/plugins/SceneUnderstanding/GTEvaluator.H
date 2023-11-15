/*!@file plugins/SceneUnderstanding/GTEvaluator.H  */

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

#ifndef GTEvaluator_H_DEFINED
#define GTEvaluator_H_DEFINED

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "plugins/SceneUnderstanding/LineMatching.H"

#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"

#include <limits>
#include <vector>
#include <string>
#include <stack>
#include <list>
#include <map>

class GTEvaluator : public SimModule
{
public:


  GTEvaluator(OptionManager& mgr, const std::string& descrName = "GTEvaluator",
      const std::string& tagName = "GTEvaluator");

  //! Destructor
  ~GTEvaluator();

  void evolve(SimEventQueue& q);
  Layout<PixRGB<byte> > getDebugImage();

protected:
  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(GTEvaluator, SimEventLineMatchingOutput);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(GTEvaluator, SimEventSaveOutput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;


private:
  Image<PixRGB<byte> > itsInImage;
  std::vector<LineMatching::Shape2D> itsShapes;
  //Get the ground truth
  rutz::shared_ptr<TestImages::SceneData> itsSceneData;
};



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
