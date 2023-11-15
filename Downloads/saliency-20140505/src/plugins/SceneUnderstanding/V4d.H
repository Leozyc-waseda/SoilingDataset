/*!@file SceneUnderstanding/V4d.H non-accidental features */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/V4d.H $
// $Id: V4d.H 13765 2010-08-06 18:56:17Z lior $
//

#ifndef V4d_H_DEFINED
#define V4d_H_DEFINED

//#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "Image/Point3D.H"
#include "plugins/SceneUnderstanding/Camera.H"
//#include "plugins/SceneUnderstanding/V4.H"
#include "plugins/SceneUnderstanding/V4d.H"
#include "plugins/SceneUnderstanding/V2.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
#include "SIFT/Histogram.H"
#include "Util/WorkThreadServer.H"

#include <vector>
#include <string>
#include <queue>
#include <google/dense_hash_map>
using google::dense_hash_map;

class SimEventV4BiasOutput;

class V4d : public SimModule
{
public:

  enum NAF_TYPE  {RIGHT_VERTIX,VERTIX, ARC};

  struct NAFState
  {
    Point3D<float> pos;
    float rot;
    NAF_TYPE featureType;
    Point3D<float> posSigma;
    float rotSigma;

    double prob;
    double weight;

    NAFState () {}

    NAFState (Point3D<float> p, float r, NAF_TYPE ft, Point3D<float> ps, float rs) :
      pos(p), rot(r), featureType(ft),
      posSigma(ps), rotSigma(rs),
      prob(0), weight(0)
    {}

    double distance(const NAFState &nafState)
    {
      double dist = 1.0e100;
      if (nafState.featureType == featureType)
      {
        double dPoint = pos.squdist(nafState.pos);
        double dRot = nafState.rot - rot;
        dRot = acos(cos(nafState.rot - rot));

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
  };

  struct FeatureTemplate
  {
    std::vector<V1::EdgeState> edges;
    std::vector<Point3D<float> > outline;
    std::vector<Point3D<float> > var;
    std::vector<RTableEntry> rTable;
    NAF_TYPE featureType;
  };

  struct GHTAcc
  {
    Point2D<int> pos;
    int ang;
    int scale;
    NAF_TYPE featureType;
    float sum;

    bool operator<(const GHTAcc& acc) const
    {
      return sum < acc.sum;
    }
  };

  V4d(OptionManager& mgr, const std::string& descrName = "V4",
      const std::string& tagName = "V4");

  //! Destructor
  ~V4d();

  //! Align template so that center of mass is at 0,0
  void alignToCenterOfMass(FeatureTemplate& featureTemplate);

  void buildRTables();

  //void setBias(std::vector<V4d::NAFState> bias);

  void evolve();
  void evolveSift();

  float calculateOrientationVector(Point2D<int>& loc, Histogram& OV);
  float createDescriptor(Point2D<int>& loc, Histogram& OV, float mainOri);

  void harrisDetector(std::vector<V4d::GHTAcc>& accRet, const FeatureTemplate& featureTemplate);
  void proposeParticlesHarris(std::vector<NAFState>& particles, const double Neff);


  Layout<PixRGB<byte> > getDebugImage();

  void getParticleLikelihood(NAFState& particle);
  double getLineProbability(const Point2D<int>& p1, const Point2D<int>& p2);

  void resampleParticles(std::vector<NAFState>& particles);
  void resampleParticles2(std::vector<NAFState>& particles);
  void proposeParticles(std::vector<NAFState>& particles, const double Neff);
  float evaluateParticles(std::vector<NAFState>& particles);


  std::vector<Point2D<int> > getImageOutline(NAFState& nafState);

  void findAngMerge(std::vector<GHTAcc>& acc, const NAFState& nafState, std::vector<GHTAcc> tmpAcc);
  void mergeBias(std::vector<GHTAcc>& acc);



  void GHT(std::vector<GHTAcc>& accRet, const FeatureTemplate& featureTemplate);
  float voteForFeature(Image<float>& acc, int angIdx, std::vector<V1::EdgeState>& rTable);

  void normalizeAcc(std::vector<GHTAcc>& acc);

  Image<PixRGB<byte> > showParticles(const std::vector<NAFState>& particles);

protected:
  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(V4d, SimEventV2Output);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(V4d, SimEventSaveOutput);

  //! Callback for every time we have a user event
  SIMCALLBACK_DECLARE(V4d, SimEventUserInput);

//  //! Callback for getting bias
  SIMCALLBACK_DECLARE(V4d, SimEventV4BiasOutput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;

private:

  std::vector<V1::EdgeState> itsEdgesState;
  std::priority_queue<V2::CornerState> itsCornersState;
  dense_hash_map<int, V1::EdgeState> itsHashedEdgesState;

  float itsMaxVal;
  int itsGHTAngStep;
  double itsBestProb;
  float itsObjectsDist; //Objects distance from the camera from the camera

  Camera itsCamera;
  std::vector<FeatureTemplate> itsNAFeatures;

  std::vector<NAFState> itsFeaturesParticles;
  std::vector<NAFState> itsBias;

  rutz::shared_ptr<WorkThreadServer> itsThreadServer;

  //Images for debug
  //Image<float> itsHoughSpaceImg;
  Image<PixRGB<byte> > itsDebugImg;

};

/* ############################### V4d sim events ######################## */
class SimEventV4dOutput : public SimEvent
{
public:
  SimEventV4dOutput(SimModule* src, std::vector<V4d::NAFState>& cellsOutput) :
    SimEvent(src), itsCells(cellsOutput)
  {}

  virtual ~SimEventV4dOutput(){}
  std::vector<V4d::NAFState> getCells() { return itsCells; }

private:
  const std::vector<V4d::NAFState>& itsCells;
};

class SimEventV4BiasOutput : public SimEvent
{
public:
  SimEventV4BiasOutput(SimModule* src, std::vector<V4d::NAFState>& cellsOutput) :
    SimEvent(src), itsCells(cellsOutput)
  {}

  virtual ~SimEventV4BiasOutput(){}
  std::vector<V4d::NAFState> getCells() { return itsCells; }

private:
  const std::vector<V4d::NAFState>& itsCells;
};


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
