/*!@file SceneUnderstanding/Geons3D.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/Geons3D.H $
// $Id: Geons3D.H 13875 2010-09-03 00:54:58Z lior $
//

#ifndef Geons3D_H_DEFINED
#define Geons3D_H_DEFINED

#include "plugins/SceneUnderstanding/TwoHalfDSketch.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "Transport/World3DInput.H"
#include "plugins/SceneUnderstanding/LGN.H"
#include "plugins/SceneUnderstanding/SMap.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
#include "FeatureMatching/GeometricHashing.H"
#include "FeatureMatching/GHough.H"
#include "GUI/ViewPort3D.H"
#include "GUI/SuperQuadric.H"

#include "Util/CpuTimer.H"
#include "Util/JobServer.H"
#include "Util/JobWithSemaphore.H"
#include "Util/WorkThreadServer.H"

#include <fcntl.h>
#include <vector>
#include <string>
#include <map>

class SimEventGeons3DPrior;

class Geons3D : public SimModule
{
public:

  struct GeonState
  {
    Point3D<float> pos;
    Point3D<float> rot;
    PixRGB<byte> color;
    SuperQuadric superQuadric;
    double prob;
  };

  Geons3D(OptionManager& mgr, const std::string& descrName = "Geons3D",
      const std::string& tagName = "Geons3D");

  //! Destructor
  ~Geons3D();

  void evolve(SimEventQueue& q);

  std::vector<GeonState> proposeGeons(Rectangle& attenLoc);

  void  calcGeonLikelihood(GeonState& geon);
  double  calcGeonEdgeLikelihood(GeonState& geon, Image<float>& edges, Image<float>& surface);
  double  calcGeonSurfaceLikelihood(GeonState& geon, Image<float>& edges, Image<float>& surface);


  //! Get the probability from a model and data
  double getEdgeProb(Image<float>& mag, Image<float>& modelOri, Image<float>& modelMag);

  //! Get the probability of the surface
  double getSurfaceProb(Image<float>& data, Image<float>& model);
  

  //! Test the liklihood function
  void testLikelihood();


  Image<PixRGB<byte> > getGeonImage(GeonState& geon);
  
  void drawGeon(const GeonState& geon);
  void renderScene(const GeonState& geon, std::vector<ViewPort3D::Line>& lines, Image<PixRGB<byte> >& frame);

  Layout<PixRGB<byte> > getDebugImage(SimEventQueue& q);

protected:

  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(Geons3D, SimEventTwoHalfDSketchOutput);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(Geons3D, SimEventSaveOutput);

  //! Callback for every time we have a user event
  SIMCALLBACK_DECLARE(Geons3D, SimEventUserInput);

  //! Callback for every time we have a prior input
  SIMCALLBACK_DECLARE(Geons3D, SimEventGeons3DPrior);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;


private:
  ViewPort3D *itsViewPort;
  ImageSet<float>  itsLGNInput;
  Image<float> itsEdges;
  Image<float> itsLinesMag;
  Image<float> itsLinesOri;
  Image<float> itsEdgesDT;
  ImageSet<float> itsOriEdgesDT;
  Image<float> itsTableDepth;
  Image<byte> itsSMap;
  rutz::shared_ptr<World3DInput::ObjectsData> itsObjectsData;
  std::vector<TwoHalfDSketch::SurfaceState> itsSurfaces;
  std::vector<GeonState> itsGeonsState;
  std::vector<GeonState> itsProposals;
  Point3D<double> itsGlobalRotation;
  Point3D<double> itsGlobalPos;
  Image<PixRGB<byte> > itsCurrentFrame;
  GeometricHashing itsGHash;
  GHough itsGHough;
  GHough itsGHough2;
  bool itsLearn;

  double itsTotalProb;

  float itsTrainingThreshold;

  rutz::shared_ptr<WorkThreadServer> itsThreadServer;
  

};

/* ############################### Geons3D sim events ######################## */
class SimEventGeons3DOutput : public SimEvent
{
public:
  SimEventGeons3DOutput(SimModule* src,
      std::vector<Geons3D::GeonState>& geons, double prob) :
    SimEvent(src), itsGeons(geons), itsProb(prob)
  {}

  virtual ~SimEventGeons3DOutput(){}
  std::vector<Geons3D::GeonState> getGeons() { return itsGeons; }
  double getProb() { return itsProb; }

private:
  const std::vector<Geons3D::GeonState>& itsGeons;
  double itsProb;
};

/* ############################### Objects sim events ######################## */
class SimEventGeons3DPrior : public SimEvent
{
public:
  SimEventGeons3DPrior(SimModule* src,
      std::vector<Geons3D::GeonState>& geons,
      Point3D<double>& pos, Point3D<double>& rotation) :
    SimEvent(src), itsGeons(geons),
    itsPos(pos), itsRotation(rotation)
  {}

  virtual ~SimEventGeons3DPrior(){}
  std::vector<Geons3D::GeonState> getGeons() { return itsGeons; }
  Point3D<double> getRotation() { return itsRotation; }
  Point3D<double> getPos() { return itsPos; }

private:
  const std::vector<Geons3D::GeonState>& itsGeons;
  Point3D<double> itsPos;
  Point3D<double> itsRotation;
};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
