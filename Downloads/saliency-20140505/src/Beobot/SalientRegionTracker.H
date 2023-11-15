/*!@file Beobot/BeobotBrainMT.H template matching tracker
  on conspicuity maps                                                   */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/SalientRegionTracker.H $
// $Id: SalientRegionTracker.H 12962 2010-03-06 02:13:53Z irock $
//


// ######################################################################

#ifndef BEOBOT_SALIENTREGIONTRACKER_H_DEFINED
#define BEOBOT_SALIENTREGIONTRACKER_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Neuro/VisualCortex.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/ColorOps.H"
#include "Image/PyramidOps.H"
#include "Image/Pixels.H"
#include "GUI/XWinManaged.H"
#include "Raster/Raster.H"
#include "Util/Timer.H"

#include "Beobot/BeobotBrainMT.H"

#include "SIFT/VisualObject.H"
#include "SIFT/VisualObjectMatch.H"



// ######################################################################
//!
class SalientRegionTracker : public ModelComponent
{
public:

  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  SalientRegionTracker(OptionManager& mgr,
                       const std::string& descrName = "Salient Region Tracker",
                       const std::string& tagName   = "SalientRegionTracker");

  //! Destructor
  virtual ~SalientRegionTracker();

  //@}

  void clear();

  void move(nub::soft_ref<SalientRegionTracker> tracker2, uint i);

  void input
  (Image<PixRGB<byte> > image, ImageSet<float> cmap,
   bool resetTracker,
   std::vector<Point2D<int> > points, std::vector<Rectangle> rects,
   std::vector<rutz::shared_ptr<VisualObject> > visualObjects);


  inline uint getNumTrackedPoints() const;

  inline Point2D<int> getCurrTrackedPoints(uint i) const;
  inline std::vector<Point2D<int> > getCurrTrackedPoints() const;

  inline Rectangle getCurrTrackedROI(uint i) const;
  inline std::vector<Rectangle> getCurrTrackedROI() const;

  inline Point2D<int> getPrevTrackedPointsScaled(uint i) const;
  inline std::vector<Point2D<int> > getPrevTrackedPointsScaled() const;

  inline ImageSet<float> getTrackerBias(uint i) const;
  inline std::vector<ImageSet<float> > getTrackerBias() const;

  inline Point2D<int> getTrackerBiasOffset(uint i) const;
  inline std::vector<Point2D<int> > getTrackerBiasOffset() const;

  inline Image<PixRGB<byte> > getOriginalInputImage() const;
  inline void setOriginalInputImage(Image<PixRGB<byte> > img);

  inline Image<PixRGB<byte> > getCurrInputImage() const;
  inline void setCurrInputImage(Image<PixRGB<byte> > img);

protected:
  //void start1(); //!< get started
  //void stop2();  //!< get stopped

private:

  void track();
  void trackCmaps();
  void trackVisualObjects();

  ImageSet<float> setNewBias
  (Point2D<int> inTrackLoc, Point2D<int> &biasOffset);

  Point2D<int> trackPoint
  (ImageSet<float> &bias, Point2D<int> biasOffset,
   Point2D<int> trackLoc);

  void updateTemplate(Point2D<int> upLeft, ImageSet<float> &bias);

  Image<float> getBiasedSMap(ImageSet<float> bias);

  // tracker and current results
  ImageSet<float> itsCurrCmap;

  std::vector<ImageSet<float> > itsTrackerBias;
  std::vector<Point2D<int> >    itsTrackerBiasOffset;

  std::vector<Point2D<int> > itsCurrTrackedPoints;
  std::vector<Point2D<int> > itsPrevTrackedPointsScaled;

  std::vector<rutz::shared_ptr<VisualObject> >
  itsCurrTrackedVisualObjects;
  std::vector<Rectangle> itsCurrTrackedROI;

  Image<PixRGB<byte> > itsOriginalInputImage;
  Image<PixRGB<byte> > itsCurrInputImage;

  Timer itsTimer;

  bool itsResetTracker;

  rutz::shared_ptr<XWinManaged> itsWin;
};

// ######################################################################
// Implementation for SalientRegionTracker inline functions
// ######################################################################


// ######################################################################
inline uint SalientRegionTracker::getNumTrackedPoints() const
{ return itsCurrTrackedPoints.size(); }

inline Point2D<int>
SalientRegionTracker::getCurrTrackedPoints(uint i) const
{
  ASSERT(i < itsCurrTrackedPoints.size());
  return itsCurrTrackedPoints[i];
}

inline std::vector<Point2D<int> >
SalientRegionTracker::getCurrTrackedPoints() const
{ return itsCurrTrackedPoints; }


inline Rectangle
SalientRegionTracker::getCurrTrackedROI(uint i) const
{
  ASSERT(i < itsCurrTrackedROI.size());
  return itsCurrTrackedROI[i];
}

inline std::vector<Rectangle>
SalientRegionTracker::getCurrTrackedROI() const
{ return itsCurrTrackedROI; }


inline Point2D<int>
SalientRegionTracker::getPrevTrackedPointsScaled(uint i) const
{
  ASSERT(i < itsPrevTrackedPointsScaled.size());
  return itsPrevTrackedPointsScaled[i];
}

inline std::vector<Point2D<int> >
SalientRegionTracker::getPrevTrackedPointsScaled() const
{
  return itsPrevTrackedPointsScaled;
}

inline ImageSet<float>
SalientRegionTracker::getTrackerBias(uint i) const
{
  ASSERT(i < itsTrackerBias.size());
  return itsTrackerBias[i];
}

inline std::vector<ImageSet<float> >
SalientRegionTracker::getTrackerBias() const
{
  return itsTrackerBias;
}

inline Point2D<int>
SalientRegionTracker::getTrackerBiasOffset(uint i) const
{
  ASSERT(i < itsTrackerBiasOffset.size());
  return itsTrackerBiasOffset[i];
}

inline std::vector<Point2D<int> >
SalientRegionTracker::getTrackerBiasOffset() const
{
  return itsTrackerBiasOffset;
}

inline Image<PixRGB<byte> >
SalientRegionTracker::getOriginalInputImage() const
{ return itsOriginalInputImage; }
inline void
SalientRegionTracker::setOriginalInputImage
(Image<PixRGB<byte> > img)
{ itsOriginalInputImage = img;  }

inline Image<PixRGB<byte> >
SalientRegionTracker::getCurrInputImage() const
{ return itsCurrInputImage; }
inline void SalientRegionTracker::setCurrInputImage
(Image<PixRGB<byte> > img)
{ itsCurrInputImage = img;  }

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
