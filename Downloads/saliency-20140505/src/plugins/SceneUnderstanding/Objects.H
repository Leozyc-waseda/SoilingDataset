/*!@file SceneUnderstanding/Objects.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/Objects.H $
// $Id: Objects.H 13765 2010-08-06 18:56:17Z lior $
//

#ifndef Objects_H_DEFINED
#define Objects_H_DEFINED

#include "Util/Types.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "plugins/SceneUnderstanding/V1.H"
#include "plugins/SceneUnderstanding/Geons3D.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
//#include "Image/OpenCVUtil.H"
#include "GUI/ViewPort.H"

#include <vector>
#include <string>

class Objects : public SimModule
{
public:

  enum ObjectType {TABLE};

  struct ObjectState
  {
    ObjectType type;
    Point3D<double> pos;
    Point3D<double> rotation;
    int lastSample;

    ObjectState () {}

    ObjectState(const ObjectType t,
        const Point3D<float>& p,
        const Point3D<float>& r) :
      type(t), pos(p), rotation(r), lastSample(0)
    {}
  };

  struct Table : public ObjectState
  {
    float tableWidth;
    float tableLength;
    float legHeight;
    float legWidth;
    Table() { type = TABLE; }
  };


  Objects(OptionManager& mgr, const std::string& descrName = "Objects",
      const std::string& tagName = "Objects");

  //! Destructor
  ~Objects();

  void evolve(SimEventQueue& q);

  void calcObjectLikelihood(ObjectState& object);

  Layout<PixRGB<byte> > getDebugImage(SimEventQueue& q);

  //! Get the geons prior from an object
  std::vector<Geons3D::GeonState> getPrior(ObjectState* object);

  //! Get a new object state
  ObjectState* proposeState(ObjectState* object);


protected:
  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(Objects, SimEventGeons3DOutput);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(Objects, SimEventSaveOutput);

  //! Callback for every time we have a user event
  SIMCALLBACK_DECLARE(Objects, SimEventUserInput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;


private:
  std::vector<Geons3D::GeonState> itsGeons;
  std::vector<ObjectState*> itsObjectsState;
  Image<PixRGB<byte> > itsCurrentFrame;
  double itsProb;

};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
