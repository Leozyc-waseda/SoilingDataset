/*!@file SceneUnderstanding/PCamera.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/PCamera.H $
// $Id: PCamera.H 13765 2010-08-06 18:56:17Z lior $
//

#ifndef PCamera_H_DEFINED
#define PCamera_H_DEFINED

//#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/Layout.H"
#include "Image/Point3D.H"
#include "Image/VisualTracker.H"
#include "plugins/SceneUnderstanding/Camera.H"
#include "plugins/SceneUnderstanding/Ganglion.H"
#include "Simulation/SimEvents.H"
#include "Simulation/SimModule.H"
#include "Media/MediaSimEvents.H"
#include "Component/ModelOptionDef.H"
#include "Component/OptionManager.H" // for REQUEST_OPTIONALIAS_NEURO()
#include "Component/ModelParam.H"
#include "Devices/Visca.H"
#include "GUI/ViewPort3D.H"

#include <vector>
#include <string>

class PCamera : public SimModule
{
public:

  struct Square
  {
    Point2D<int> p[4];
  };

  PCamera(OptionManager& mgr, const std::string& descrName = "PCamera",
      const std::string& tagName = "PCamera");

  //! Destructor
  ~PCamera();

  void evolve();

  Layout<PixRGB<byte> > getDebugImage();

  void calibrate(std::vector<CvPoint2D32f>& corners);

  void displayExtrinsic(std::vector<CvPoint2D32f>& corners);

  void projectPoints(std::vector<Point3D<float> >& points);

  void projectGrid();

  void findSquares(const Image<PixRGB<byte> >& in, CvMemStorage* storage,
      const int minarea, const int maxarea, const double mincos);

  void getTransMat();


protected:

  //! Callback for when a new input frame is available
  SIMCALLBACK_DECLARE(PCamera, SimEventInputFrame);

  //! Callback for every time we should save our outputs
  SIMCALLBACK_DECLARE(PCamera, SimEventSaveOutput);

  //! Callback for every time we have a user event
  SIMCALLBACK_DECLARE(PCamera, SimEventUserInput);

  void onSimEventPTZToLoc(SimEventQueue& q,
      rutz::shared_ptr<SimEventUserInput>& e);

  //! Should we show our debug info
  OModelParam<bool> itsShowDebug;
  nub::soft_ref<Visca> itsCameraCtrl;   //! The controller of the camera

private:
  CvMemStorage* itsStorage;
  Image<PixRGB<byte> > itsCurrentImg;
  Image<PixRGB<byte> > itsRenderImg;
  ImageSet<float> itsPCameraCellsInput; //The input to the cells
  Image<byte> itsDebugImg;
  std::vector<Square> itsSquares;
        CvMat* itsIntrinsicMatrix;
        CvMat* itsDistortionCoeffs;
  CvMat* itsCameraRotation;
  CvMat* itsCameraTranslation;
  std::vector<CvPoint2D32f> itsCorners;
  std::string itsCurrentObjName;
  ViewPort3D* itsVP;


  Point2D<int> itsCurrentTargetLoc;
  bool itsZoomOut;


  bool itsIntrinsicInit;
  bool itsSaveCorners;
  bool itsDrawGrid;
  bool itsChangeRot;

};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
