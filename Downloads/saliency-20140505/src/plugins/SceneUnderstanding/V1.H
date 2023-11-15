/*!@file SceneUnderstanding/V1.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/V1.H $
// $Id: V1.H 14065 2010-09-28 14:52:03Z lior $
//

#ifndef V1_H_DEFINED
#define V1_H_DEFINED

//#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "Image/Point3D.H"
#include "Image/TensorOps.H"
#include "plugins/SceneUnderstanding/LGN.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
#include "Media/MediaSimEvents.H"
#include "Component/ModelOptionDef.H"
#include "Component/OptionManager.H" // for REQUEST_OPTIONALIAS_NEURO()
#include "Component/ModelParam.H"
#include "Util/CpuTimer.H"

#include <vector>
#include <string>

class SimEventV1Bias;
class V1 : public SimModule
{
public:

  struct EdgesState
  {
    TensorField lumTensorField;
    TensorField rgTensorField;
    TensorField byTensorField;
    Image<float> prob;
    Image<float> distTr; //the distance transform
    //double distance(EdgeState &e)
    //{
    //  double dist = 1.0e100;
    //  double dPoint = pos.squdist(e.pos);
    //  double dRot = e.ori - ori;
    //  dRot = atan(sin(dRot)/cos(dRot))*180/M_PI; //wrap to 180 deg to 0

    //  dist = sqrt(dPoint + (dRot*dRot));
    //  return dist;
    //}
  };


  struct SpatialBias
  {
    Point2D<int> loc;
    Dims dims;
    float threshold;

    SpatialBias() :
      loc(-1,-1),
      dims(0,0),
      threshold(0)
    {}

    SpatialBias(int i, int j, int w, int h, float t) :
      loc(i,j),
      dims(w,h),
      threshold(t)
    {}
    bool contains(int x, int y) 
    {
      if (!loc.isValid())
        return false;

      if (x > loc.i - (dims.w()/2) &&
          x < loc.i + (dims.w()/2) &&
          y > loc.j - (dims.h()/2) &&
          y < loc.j + (dims.h()/2))
        return true;
      else
        return false;

    }


  };


  V1(OptionManager& mgr, const std::string& descrName = "V1",
      const std::string& tagName = "V1");

  //! Destructor
  ~V1();

  void evolve(SimEventQueue& q);
  void evolveGabor();
  void evolveSobel();
  void evolveCanny();
  void evolveTensor();

  Layout<PixRGB<byte> > getDebugImage();

  void setBias(const Image<float> &biasImg);

  void applyThreshold(TensorField& tensorField, std::vector<SpatialBias>& spatialBias);
  
protected:
  //! Callback for when a new LGN output is ready
  SIMCALLBACK_DECLARE(V1, SimEventLGNOutput);

  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(V1, SimEventV1Bias);
  

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(V1, SimEventSaveOutput);

  //! Callback for every time we have a user event
  SIMCALLBACK_DECLARE(V1, SimEventUserInput);
    
  

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;

private:

  ImageSet<float> itsLGNData; //The input to the cells
  Image<float> itsInput;
  EdgesState itsEdgesState;
  float itsThreshold;
  float itsBiasThreshold;
  float itsAngBias;
  std::vector<SpatialBias> itsSpatialBias;
  Point2D<int> itsAttenLoc;
  Dims itsWinSize;
  CpuTimer itsTimer;

};

/* ############################### V1 sim events ######################## */
class SimEventV1Output : public SimEvent
{
public:
  SimEventV1Output(SimModule* src,
      V1::EdgesState& edgesState) :
    SimEvent(src), itsEdgesState(edgesState)
  {}

  virtual ~SimEventV1Output(){}
  V1::EdgesState getEdgesState() { return itsEdgesState; }

private:
  const V1::EdgesState itsEdgesState;
};

class SimEventV1Bias : public SimEvent
{
public:
  SimEventV1Bias(SimModule* src,
      std::vector<V1::SpatialBias>& bias) :
    SimEvent(src), itsSpatialBias(bias)
  {}

  virtual ~SimEventV1Bias(){}
  std::vector<V1::SpatialBias> getSpatialBias() { return itsSpatialBias; }

private:
  const std::vector<V1::SpatialBias>& itsSpatialBias;
};




// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
