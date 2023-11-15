/*!@file RCBot/TrackFeature.H track a location using the saliency map and
 * template matching                                                    */
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
// Primary maintainer for this file: Lior Elazary <lelazary@yahoo.com>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/TrackFeature.H $
// $Id: TrackFeature.H 9412 2008-03-10 23:10:15Z farhan $
//

#include "CMapDemo/SaliencyCMapMT.H"
#include "Image/Image.H"
#include "Image/CutPaste.H"
#include "Util/Timer.H"

#ifndef TRACK_FEATURE_H
#define TRACK_FEATURE_H

#define NAVG 20
#define WINSIZE 25

//Tracking thread
static omni_mutex trackMutex;
static omni_mutex trackLocMutex;
static omni_mutex trackImgMutex;
static omni_condition trackCond(&trackMutex);

class TrackFeature : public omni_thread
{
public:
  TrackFeature(nub::ref<SaliencyMT> insmt);
  ~TrackFeature();
  void run(void *ptr);
  void calcFps();
  float getFps();
  void setImg(Image<PixRGB<byte> > &inImg);
  void setTrackLoc(Point2D<int> &inTrackLoc,
                   Image<PixRGB<byte> > &img);
  Point2D<int> getTrackLoc();
  void updateTemplate();

private:
  bool doneTracking;
  Image<PixRGB<byte> > img;
  nub::ref<SaliencyMT> smt;
  Point2D<int> lastWinner;
  Point2D<int> trackLoc;
  float templThresh;
  Timer timer;
  ImageSet<float> bias;
  ImageSet<float> newBias;
  int width, height;

  // fps calculation
  uint64 avgtime;
  int avgn;
  float fps;
};
#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
