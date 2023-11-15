/*!@file SceneUnderstanding/V4.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/V4.H $
// $Id: V4.H 13765 2010-08-06 18:56:17Z lior $
//

#ifndef V4_H_DEFINED
#define V4_H_DEFINED

//#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "Image/Point3D.H"
#include "plugins/SceneUnderstanding/Camera.H"
#include "plugins/SceneUnderstanding/V4d.H"
#include "plugins/SceneUnderstanding/V2.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
#include "Util/WorkThreadServer.H"

#include <vector>
#include <string>
#include <google/dense_hash_map>
using google::dense_hash_map;

class V4 : public SimModule
{
public:

  enum GEON_TYPE  {TRIANGLE,SQUARE, CIRCLE};

  struct GeonState
  {
    Point3D<float> pos;
    float rot;
    double prob;
    double weight;
    GEON_TYPE geonType;

    Point3D<float> posSigma;
    float rotSigma;

    double distance(const GeonState &geonState)
    {
      double dist = 1.0e100;
      if (geonState.geonType == geonType)
      {
        double dPoint = pos.squdist(geonState.pos);
        double dRot = geonState.rot - rot;
        dRot = acos(cos(geonState.rot - rot));

        dist = sqrt(dPoint + (dRot*dRot));
      } else {
        dist =  1.0e100; //can not compare the distance across features for now
      }

      return dist;
    }
  };

  struct RTableEntry
  {
    Point2D<int> loc;
    float rot;
    V4d::NAF_TYPE featureType;
  };

  struct GHTAcc
  {
    Point2D<int> pos;
    int ang;
    int scale;
    GEON_TYPE geonType;
    float sum;

    bool operator<(const GHTAcc& acc) const
    {
      return sum < acc.sum;
    }

  };

  struct GeonOutline
  {
    std::vector<Point3D<float> > outline;
    std::vector<V4d::NAFState> NAFTemplate;
    std::vector<RTableEntry> rTable;
    GEON_TYPE geonType;
  };

  V4(OptionManager& mgr, const std::string& descrName = "V4",
      const std::string& tagName = "V4");

  //! Destructor
  ~V4();

  void buildRTables();
  void init(Dims dims);

  void setInput(const std::vector<V1::EdgeState> &edgesState);
  void setInput(const std::vector<V4d::NAFState> &nafStates);

  std::vector<V4::GeonState> getOutput();
  std::vector<V4d::NAFState> getBias();

  void showGeon(GeonOutline& object);

  void evolve();

  //! Align template so that center of mass is at 0,0
  void alignToCenterOfMass(GeonOutline& featureTemplate);

  Layout<PixRGB<byte> > getDebugImage();

  void getOutlineLikelihood(GeonState& geon);
  void getGeonLikelihood(GeonState& geon);

  double getLineProbability(const Point2D<int>& p1, const Point2D<int>& p2);

  void resampleParticles(std::vector<GeonState>& geonParticles);
  void resampleParticles2(std::vector<GeonState>& geonParticles);
  void proposeParticles(std::vector<GeonState>& geonParticles, const double Neff);

  std::vector<Point2D<int> > getImageGeonOutline(GeonState& geon);

  float evaluateGeonParticles(std::vector<GeonState>& geonParticles);


  void GHT(std::vector<GHTAcc>& accRet, GeonOutline& geonOutline);
  void normalizeAcc(std::vector<GHTAcc>& acc);
  void voteForFeature(Image<float>& acc, int angIdx, std::vector<RTableEntry>& rTable);


  Image<PixRGB<byte> > showParticles(const std::vector<GeonState>& geonParticles);

protected:
  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(V4, SimEventV2Output);

  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(V4, SimEventV4dOutput);


  //! Callback for every time we have a user event
  SIMCALLBACK_DECLARE(V4, SimEventUserInput);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(V4, SimEventSaveOutput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;

private:

  std::vector<V1::EdgeState> itsEdgesState;
  dense_hash_map<int, V1::EdgeState> itsHashedEdgesState;

  std::vector<V4d::NAFState> itsNAFParticles;

  float itsMaxVal;
  int itsGHTAngStep;
  double itsBestProb;
  float itsObjectsDist; //Objects distance from the camera from the camera


  Camera itsCamera;
  std::vector<GeonState> itsGeonsParticles;

  std::vector<GeonOutline> itsGeons;

  //Images for debug
  Image<float> itsHoughSpaceImg;

  rutz::shared_ptr<WorkThreadServer> itsThreadServer;

  Image<PixRGB<byte> > itsDebugImg;

};

/* ############################### V4d sim events ######################## */
class SimEventV4Output : public SimEvent
{
public:
  SimEventV4Output(SimModule* src, std::vector<V4::GeonState>& cellsOutput) :
    SimEvent(src), itsCells(cellsOutput)
  {}

  virtual ~SimEventV4Output(){}
  std::vector<V4::GeonState> getCells() { return itsCells; }

private:
  const std::vector<V4::GeonState>& itsCells;
};

//class SimEventV4BiasOutput : public SimEvent
//{
//public:
//  SimEventV4BiasOutput(SimModule* src, std::vector<V4d::NAFState>& cellsOutput) :
//    SimEvent(src), itsCells(cellsOutput)
//  {}
//
//  virtual ~SimEventV4BiasOutput(){}
//  std::vector<V4d::NAFState> getCells() { return itsCells; }
//
//private:
//  const std::vector<V4d::NAFState>& itsCells;
//};



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
