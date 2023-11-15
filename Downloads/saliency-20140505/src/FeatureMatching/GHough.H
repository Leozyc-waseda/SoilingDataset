/*!@file SceneUnderstanding/GHough.H  Generalized Hough */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/FeatureMatching/GHough.H $
// $Id: GHough.H 13815 2010-08-22 17:58:48Z lior $
//

#ifndef GHough_H_DEFINED
#define GHough_H_DEFINED

#include "Util/Types.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ImageSet.H"
#include "Image/Point2D.H"
#include "Image/Point3D.H"
#include "Util/CpuTimer.H"
#include "Util/JobServer.H"
#include "Util/JobWithSemaphore.H"
#include "Util/WorkThreadServer.H"
#include "Learn/SOFM.H"

#include <vector>
#include <map>

class GHough
{
public:

  struct Feature
  {
    Point2D<float> loc;
    std::vector<float> values;
  };

  struct RTable
  {
    std::map<int, std::vector<Point2D<float> > > entries; 
    std::map<long, std::vector<Feature> > featureEntries;
  };

  struct Model
  {
    int id;
    int type;
    Point2D<int> center;
    std::map<int,std::vector<double> > invRTable; 
    std::vector<RTable> rTables;
    Point2D<float> imgPos;
    Point3D<float> pos;
    Point3D<float> rot;
    int numFeatures;
  };



  struct Acc
  {
    int id;
    Point2D<int> pos;
    float votes;
    float prob;

    Acc(int i, Point2D<int> p, float w) :
      id(i), pos(p), votes(w), prob(0)
    {}
    Acc(int i, int x, int y, float w) :
      id(i), pos(x,y), votes(w), prob(0)
    {}

    Acc() {}
  };


  struct AccCmp
  {
    bool operator()(const Acc& c1, const Acc& c2)
    {
      return c1.prob > c2.prob;
    }
  };


  GHough();

  //! Destructor
  virtual ~GHough();

  //! Create an Inverient RTable
  void createInvRTable(const Image<byte>& img, const Image<float>& ang);
  
  //! Create an RTable
  RTable createRTable(const Image<byte>& img, const Image<float>& ang, Point2D<float>& imgPos, int& numFeatures,
      Point2D<float>& imgLoc);

  RTable createRTable(const std::vector<Feature>& features, Point2D<float>& imgPos, Point2D<float>& imgLoc);

  RTable createRTable(const Image<byte>& img, const Image<float>& ang,
      Point2D<int>& center, int& numFeatures);
  

  //! Get votes from an RTable
  Image<float> getInvVotes(const Image<byte>& img, const Image<float>& ang);

  //The hash function
  long getIndex(const std::vector<float>& values);
  

  std::vector<Acc> getVotes(const Image<byte>& img, const Image<float>& ang);

  std::vector<Acc> getVotes(const std::vector<Feature>& features);

  std::vector<Acc> getVotes(int id, const RTable& rTable, const Image<byte>& img, const Image<float>& ang);

  std::vector<Acc> getVotes(int id, const RTable& rTable, const std::vector<Feature>& features, float& maxVotes);
  std::vector<Acc> getVotes2(int id, const RTable& rTable, const std::vector<Feature>& features, float& maxVotes);

  std::vector<Acc> getVotes(const Image<float>& img);

  std::vector<Acc> getVotes(const Image<float>& img, const Image<float>& ori);


  Point2D<float> addModel(int& id, const Image<byte>& img, const Image<float>& ang,
      Point3D<float> pos, Point3D<float> rot);

  Point2D<float> addModel(int& id, int type, const std::vector<Feature>& features,
      Point3D<float> pos, Point3D<float> rot);


  void addModel(int id, const Image<float>& img);

  Point2D<int> addModel(int id, const std::vector<Point2D<int> >& polygon);
  
  void setPosOffset(int id, Point3D<float> pos);

  void writeTable(const char* filename);

  void readTable(const char* filename);

  Point2D<float> getModelImgPos(const int id);

  Point3D<float> getModelRot(const int id);

  Point3D<float> getModelPosOffset(const int id);

  int getModelType(const int id);
  
  Image<PixRGB<byte> > getRTableImg(const int id);

  uint getNumFeatures(const int id);
  
  Image<float> getAccImg(std::vector<GHough::Acc>& acc);
 
  Point2D<int> findInvFeature(const int x, const int y,
      const Image<float>& img, const Image<float>& ang);
  
  void trainSOFM();
 
private:
  std::vector<Model> itsModels; //A list of models
  int itsNumEntries;
  rutz::shared_ptr<WorkThreadServer> itsThreadServer;
 
  SOFM* itsSOFM; 
  
};


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
