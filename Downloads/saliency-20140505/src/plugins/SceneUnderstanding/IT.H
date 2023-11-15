/*!@file SceneUnderstanding/IT.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/IT.H $
// $Id: IT.H 13413 2010-05-15 21:00:11Z itti $
//

#ifndef IT_H_DEFINED
#define IT_H_DEFINED

#include "Util/Types.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "Image/Point3D.H"
#include "plugins/SceneUnderstanding/Camera.H"
#include "plugins/SceneUnderstanding/V4.H"
#include "plugins/SceneUnderstanding/V4d.H"
#include "plugins/SceneUnderstanding/V1.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
#include <vector>
#include <string>
#include <google/dense_hash_map>
using google::dense_hash_map;

class IT : public SimModule
{
public:

  enum OBJECT_TYPE {HOUSE, WOMAN, HAT, MAN};

  struct ObjectState
  {
    Point3D<float> pos;
    float rot;
    double prob;
    double weight;
    OBJECT_TYPE objectType;
  };

  struct RTableEntry
  {
    Point2D<int> loc;
    float rot;
    V4::GEON_TYPE geonType;
  };

  struct Object
  {
    std::vector<V4::GeonState> geons;
    OBJECT_TYPE objectType;
    std::vector<RTableEntry> rTable;
  };


  struct GHTAcc
  {
    Point2D<int> pos;
    int ang;
    int scale;
    OBJECT_TYPE objectType;
    float sum;

    bool operator<(const GHTAcc& acc) const
    {
      return sum < acc.sum;
    }

  };


  IT(OptionManager& mgr, const std::string& descrName = "IT",
      const std::string& tagName = "IT");

  //! Destructor
  ~IT();

  void buildRTables();
  void init(Dims dims);

  void setInput(const std::vector<V4::GeonState> &geonsState);

  void evolve();

  Layout<PixRGB<byte> > getDebugImage();

  void getObjectLikelihood(ObjectState& objectState);

  V4::GeonState findNearestGeon(const V4::GeonState& geonState);
  double getGeonDistance(const V4::GeonState& g1, const V4::GeonState& g2);
  void fixAngle(float& ang);


  void resampleParticles(std::vector<ObjectState>& objectParticles);
  void proposeParticles(std::vector<ObjectState>& objectParticles, const double Neff);

  float evaluateObjectParticles(std::vector<ObjectState>& objectParticles);

  //void GHT(Image<float>& acc);
  void GHT(std::vector<GHTAcc>& accRet, OBJECT_TYPE objectType);
  void voteForFeature(Image<float>& acc, int angIdx, std::vector<RTableEntry>& rTable);
  void normalizeAcc(std::vector<GHTAcc>& acc);

  Image<float> showParticles(const std::vector<ObjectState>& objectParticles);

  void showObject(Image<PixRGB<byte> > &img, Object& object, Point3D<float>& pos, float rot);

protected:
  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(IT, SimEventV4Output);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(IT, SimEventSaveOutput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;


private:

  dense_hash_map<int, V4::GeonState> itsHashedGeonsState;
  std::vector<V4::GeonState> itsGeonsState;

  Camera itsCamera;

  std::vector<ObjectState> itsObjectsParticles;

  float itsObjectsDist;

  std::vector<Point2D<int> > rTable;

  Image<float> itsHoughSpace;

  std::vector<Object> itsObjects;

};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
