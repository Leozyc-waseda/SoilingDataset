/*!@file SceneUnderstanding/Geons2D.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/Geons2D.H $
// $Id: Geons2D.H 13765 2010-08-06 18:56:17Z lior $
//

#ifndef Geons2D_H_DEFINED
#define Geons2D_H_DEFINED

//#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "Image/Point3D.H"
#include "plugins/SceneUnderstanding/Camera.H"
#include "plugins/SceneUnderstanding/Ganglion.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
#include "Media/MediaSimEvents.H"
#include "Component/ModelOptionDef.H"
#include "Component/OptionManager.H" // for REQUEST_OPTIONALIAS_NEURO()
#include "Component/ModelParam.H"

#include <vector>
#include <string>

class Geons2D : public SimModule
{
public:

  struct Geons2DState
  {
    Point2D<int> pos;
    float radius;

    Geons2DState() {}
    Geons2DState(Point2D<int> p, float r) :
      pos(p),
      radius(r)
    {}
  };


  Geons2D(OptionManager& mgr, const std::string& descrName = "Geons2D",
      const std::string& tagName = "Geons2D");

  //! Destructor
  ~Geons2D();

  void evolve();

  Layout<PixRGB<byte> > getDebugImage();

  void setBias(const Image<float> &biasImg);

protected:
  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(Geons2D, SimEventGanglionOutput);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(Geons2D, SimEventSaveOutput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;

private:

  Image<float> itsGeons2DCellsInput; //The input to the cells
  std::vector<Geons2DState> itsGeons2DState;

#ifdef HAVE_OPENCV
  CvMemStorage* itsMemStorage;
#endif

};

/* ############################### Geons2D sim events ######################## */
class SimEventGeons2DOutput : public SimEvent
{
public:
  SimEventGeons2DOutput(SimModule* src, std::vector<Geons2D::Geons2DState>& cellsOutput) :
    SimEvent(src), itsCells(cellsOutput)
  {}

  virtual ~SimEventGeons2DOutput(){}
  std::vector<Geons2D::Geons2DState> getCells() { return itsCells; }

private:
  const std::vector<Geons2D::Geons2DState>& itsCells;
};


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
