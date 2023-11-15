/*!@file SceneUnderstanding/Contours.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/Contours.H $
// $Id: Contours.H 13904 2010-09-09 21:37:01Z lior $
//

#ifndef Contours_H_DEFINED
#define Contours_H_DEFINED

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "plugins/SceneUnderstanding/V2.H"
#include "plugins/SceneUnderstanding/TensorVoting.H"
#include "Image/TensorOps.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"

#include <vector>
#include <string>
class SimEventContoursBias;

class Contours : public SimModule
{
public:
  const static float UNKNOWN = -1.0F;
  const static float NOTDEFINED = -2.0F;

  struct Contour
  {
    std::vector<Point2D<int> > points;
    std::vector<float> ori;

    size_t size() const
    {
      return points.size();
    }
  };

  struct ContourCmp
  {
    bool operator()(const Contour& c1, const Contour& c2)
    {
      return c1.size() > c2.size();
    }
  };
  

  Contours(OptionManager& mgr, const std::string& descrName = "Contours",
      const std::string& tagName = "Contours");

  //! Destructor
  ~Contours();

  void evolve(SimEventQueue& q);

  Layout<PixRGB<byte> > getDebugImage();

protected:
  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(Contours, SimEventV2Output);

  //! Callback for when a new ganglion output is ready
  SIMCALLBACK_DECLARE(Contours, SimEventContoursBias);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(Contours, SimEventSaveOutput);

  //! Callback for every time we have a user event
  SIMCALLBACK_DECLARE(Contours, SimEventUserInput);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;

  std::vector<Point2D<int> > getPointList(TensorVoting& tv, Image<float>& contoursImg,
      Image<float>& edgesMag, Image<float>& edgesOri);

  std::vector<Point2D<int> > getPointList(Image<float>& inMag, Image<float>& inOri,
      Image<float>& contoursImg);

  
  Contour followContour(int idx, Point2D<int> startLoc, Image<float>& contoursImg,
      const Image<float>& edgesMag, const Image<float>& edgesOri);

  Point2D<int> findEdgeToFollow(const Point2D<int>& edgeLoc,
      const float origEdgeMag,
      const Image<float>& edgesMag, const Image<float>& edgesOri,
      const Image<float>& contoursImg,
      const Point2D<int>& pEdge);
  
  

private:
  std::vector<TensorVoting> itsTensorFields;
  std::vector<Contour> itsContours;
  std::vector<Contour> itsBiasContours;

  int itsCurrentContour;

};

/* ############################### V2 sim events ######################## */
class SimEventContoursOutput : public SimEvent
{
public:
  SimEventContoursOutput(SimModule* src,
      std::vector<Contours::Contour>& contour,
      Image<float>& img) :
    SimEvent(src), itsContours(contour), itsImg(img)
  {}

  virtual ~SimEventContoursOutput(){}
  std::vector<Contours::Contour> getContours() { return itsContours; }
  Image<float> getImg() { return itsImg; }

private:
  const std::vector<Contours::Contour>& itsContours;
  const Image<float>& itsImg;
};

class SimEventContoursBias : public SimEvent
{
public:
  SimEventContoursBias(SimModule* src,
      std::vector<Contours::Contour>& contour) :
    SimEvent(src), itsContours(contour)
  {}

  virtual ~SimEventContoursBias(){}
  std::vector<Contours::Contour> getContours() { return itsContours; }

private:
  const std::vector<Contours::Contour>& itsContours;
};


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
