/*!@file BeoSub/BeoSub.H An autonomous submarine */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSub.H $
// $Id: BeoSub.H 10845 2009-02-13 08:49:12Z itti $
//

#ifndef BEOSUB_H_DEFINED
#define BEOSUB_H_DEFINED

#include "Util/Angle.H"
#include "BeoSub/Attitude.H"
#include "BeoSub/BeoSubDB.H"
#include "BeoSub/BeoSubCanny.H"
#include "BeoSub/BeoSubTaskDecoder.H"
#include "BeoSub/ColorTracker.H"
#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ImageSet.H"
#include "SIFT/VisualObject.H"
#include "Util/Timer.H"
#include <pthread.h>

class RawVisualCortex;
class VisualObjectDB;
class ShapeModel;

// ANDRE - NOTE: For a Global Position Variable (GPV), I made a struct
struct Location {
  float x, y;
};

//! A simple enum to access one of our three cameras
enum BeoSubCamera { BEOSUBCAMDOWN = 0, BEOSUBCAMFRONT = 1,
                    BEOSUBCAMUP = 2 };

//! Simple function to print the name (Front, Down, Up) of a camera
const char *beoSubCameraName(const BeoSubCamera cam);

//! Definition and access functions for the BeoSub base class
/*! BeoSub is an autonomous visually-guided submarine. This class
  implements the high-level functionality, relying on low-level
  drivers to handle motor activation, compass reading, etc. This is
  the base class. Some functions are purely virtual and need to be
  overloaded in various derived classes. */
class BeoSub : public ModelComponent
{
public:
  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  BeoSub(OptionManager& mgr,
         const std::string& descrName = "BeoSub",
         const std::string& tagName = "BeoSub");

  //! Destructor
  virtual ~BeoSub();

  //@}

  // ######################################################################
  /*! @name Low-level behaviors */
  //@{

  //NOTE: checkpoint may need FIXed
  /*
  //! Acquire and save a bunch of measurements
  virtual void checkpoint(const char *fmt, ...)
    // NOTE: this __attribute__ tells gcc that it should issue
    // printf-style warnings when compiling calls to
    // BeoSub::checkpoint(), treating the 1st argument (fmt) as the
    // format string, and the 2nd and subsequent arguments as the
    // printf-style parameters (SUBNOTE: because this is a member
    // function, there is a hidden 'this' parameter that counts as arg
    // 1, so the listed arguments are counted starting from 2)
    __attribute__((format(__printf__, 2, 3))) = 0;

  //@}
  */
  // ######################################################################
  /*! @name Mid-level access functions */
  //@{

  //! Have we reached the latest set target?
  /*! A tol of 1.0F corresponds to +/- 10deg in heading and +/- 20cm
    in depth, these tolerances scaling linearly with tol. Pitch and
    roll are ignored, we just assume they will be stabilized by the
    PID controller and the sub will always remain level. */
  bool targetReached(const float tol = 1.0F) const;

  //! Wait until a move is complete. Timeout is in seconds.
  void waitMove(const double timeout);

  //! Turn the sub to the given heading
  /*! @param blocking if true will block until completion or some
    timeout occurs */
  void turnAbs(const Angle finalHeading, const bool blocking = false);

  //! Turns the sub to the given heading relative to our current heading
  void turnRel(const Angle relHeading, const bool blocking = false);

  virtual void turnOpen(const Angle openHeading, const bool stop = true) = 0;

  //! Pitch the sub to a given angle
  void pitchAbs(const Angle finalPitch, const bool blocking = false);

  //! Makes the sub dive to a specific depth in meters
  void diveAbs(const float finalDepth, const bool blocking = false);

  //! Makes the sub dive to a specific depth relative to current
  /*! Negative depth is up, positive is down. Distance is in meters. */
  void diveRel(const float relDepth, const bool blocking = false);

  //! Advance the sub by some distance in meters
  /*! Negative distance is to the rear, positive to the
    front. Distance is in meters (approximate). Note: this blocks for
    some time... */
  virtual void advanceRel(const float relDist, const bool stop = true) = 0;

  //! Strafes the sub by some distance in meters
  /*! Negative distance is to the left, positive to the
    right. Distance is in meters (approximate). Note: this blocks for
    some time... */
  virtual void strafeRel(const float relDist);

  //! Drop a marker
  virtual void dropMarker(const bool blocking = false) = 0;

  //@}

  // ######################################################################
  /*! @name Visual processing functions */
  //@{

  //! Grab a an image from one of the cameras
  virtual Image< PixRGB<byte> >
  grabImage(const enum BeoSubCamera cam) const = 0;

  //! Get saliency map from a given camera
  /*! For ease of use, the output saliency map returned here is
    rescaled (with bilinear interpolation) to the same dims as the
    grabbed input image. */
  virtual Image<float> getSaliencyMap(const enum BeoSubCamera cam) const;

  //! match a known object to a grabbed image
  bool matchSIFT(const enum BeoSubCamera cam,
                 const rutz::shared_ptr<VisualObject>& obj) const;

  //! Recognize an image from our camera against our image databases
  /*! Returns true if reliable recognition was achieved. */
  virtual bool recognizeSIFT(const enum BeoSubCamera cam,
                             MappingData& data, Angle& myHeading) const;

  virtual bool affineSIFT(const enum BeoSubCamera cam,
                          rutz::shared_ptr<VisualObject> goal);

  virtual bool findShape(rutz::shared_ptr<ShapeModel>& shapeArg, const char* colorArg, const enum BeoSubCamera camArg) const;

  virtual bool centerColor(const char* colorArg, const enum BeoSubCamera camArg, float& thresholdMass);

  virtual bool approachArea(std::string name, const enum BeoSubCamera cam, float stepdist);

  virtual bool approachArea(MappingData goalArea, MappingData currentArea, float stepDist);

  virtual bool Decode();

  virtual bool TaskGate();

  virtual bool TaskA();

  virtual bool TaskB();

  virtual bool TaskC();




  //behaviors for the tasks

  bool TaskScheduler(int TaskOrder);

  //===============TaskA===================
  bool LookForRedLight();
  bool CenterRedLight();
  bool ApproachRedLight();
  bool PushRedLight();

  //===============TaskB===================
  //break into LookForPipeLine and ApproachPipeLine
  virtual bool ApproachPipeLine();

  bool FollowPipeLine();

  int RecognizeBin();

  bool TestBin(nub::soft_ref<FrameIstream> gb, int testColor);

  bool CenterBin();

  bool DropMarker();

  bool PassBin();
  //@}

  // ######################################################################
  /*! @name Low-level access functions */
  //@{

  //! Get time since started, in seconds
  double getTime() const;

  //! Gets the Current Attitude of the Sub
  Attitude getCurrentAttitude() const;

  //! Gets the Target Attitude of the Sub
  Attitude getTargetAttitude() const;

  //! Get current heading (yaw axis), in degrees, ]-180.0 .. 180.0]
  Angle getHeading() const;

  //! Get current pitch, in degrees, ]-180.0 .. 180.0]
  Angle getPitch() const;

  //! Get current roll, in degrees, ]-180.0 .. 180.0]
  Angle getRoll() const;

  //! Shorthand to get all 3 compass readings
  void getCompass(Angle& heading, Angle& pitch, Angle& roll) const;

  //! Get current depth (units in meters)
  float getDepth() const;

  //@}

  //private:
protected:
  OModelParam<std::string> itsFrontVODBfname; //!< file name of front object db
  rutz::shared_ptr<VisualObjectDB> itsFrontVODB;     //!< front object db
  OModelParam<std::string> itsDownVODBfname;  //!< file name of down object db
  rutz::shared_ptr<VisualObjectDB> itsDownVODB;      //!< down object db
  OModelParam<std::string> itsUpVODBfname;    //!< file name of up object db
  rutz::shared_ptr<VisualObjectDB> itsUpVODB;        //!< up object db

  Timer itsMasterClock;        //!< our master clock, unit is useconds
  int itsCkPt;                 //!< our checkpoint number
  Attitude itsCurrentAttitude; //!< the current Attitude of the sub
  // Andre: NOTE: we need a global variable for attitude so we can
  //+/- our direction
  // whenever we rotate in TurnOpen(), with some drift error that cancels
  // itself slightly if we do a relative good number of +/- changes in direction
  // But I'm not sure if we need a new Attitude or just use itsCurrentAttitude
  Location itsGlobalPosition;
  Angle itsGlobalHeading;

  Attitude itsTargetAttitude;  //!< target (desired) Attitude of the sub
  nub::soft_ref<RawVisualCortex> itsVisualCortex; //!< To compute saliency maps

  std::string itsFrontDBfname;                //!< file name of front data db
  rutz::shared_ptr<BeoSubDB> itsFrontDB;             //!< front data db
  std::string itsDownDBfname;                 //!< file name of down data db
  rutz::shared_ptr<BeoSubDB> itsDownDB;              //!< down data db
  std::string itsUpDBfname;                   //!< file name of up data db
  rutz::shared_ptr<BeoSubDB> itsUpDB;                //!< up data db

  nub::soft_ref<BeoSubCanny> itsShapeDetector;
  nub::soft_ref<BeoSubTaskDecoder> itsTaskDecoder;
  nub::soft_ref<ColorTracker> itsColorTracker;
  nub::soft_ref<ColorTracker> itsColorTracker2;

  bool decoderIsRed;
public:
  bool taskAdone;
  bool taskBdone;
  bool taskCdone;
  std::list<char> itsTasks;
  std::list<char>::iterator itsTasksIter;

  rutz::shared_ptr<VisualObject> itsVOtaskAdown, itsVOtaskAfront;
  rutz::shared_ptr<VisualObject> itsVOtaskBdown, itsVOtaskBfront;
  rutz::shared_ptr<VisualObject> itsVOtaskCdown, itsVOtaskCfront, itsVOtaskCup;
  rutz::shared_ptr<VisualObject> itsVOtaskGdown, itsVOtaskGfront, itsVOtaskGup;
  rutz::shared_ptr<VisualObject> itsVOtaskDfront;

  Location taskAposition, taskBposition, taskCposition, taskGposition;

protected:
  pthread_mutex_t itsLock;     //!< lock to control access to our internal data

  virtual void start1(); //!< get started
};


#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
