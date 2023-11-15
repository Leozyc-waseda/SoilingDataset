/*!@file Beobot/SalientRegionTracker.C template matching tracker
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/SalientRegionTracker.C $
// $Id: SalientRegionTracker.C 15426 2012-11-02 21:44:22Z siagian $
//

// ######################################################################

#include "Image/OpenCVUtil.H"
#include "Beobot/SalientRegionTracker.H"

#include "Component/ModelManager.H"
#include "Raster/Raster.H"
#include "GUI/XWinManaged.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Util/Timer.H"

#include <signal.h>

#include "Image/ShapeOps.H"
#include "Image/CutPaste.H"     // for inplacePaste()
#include "Image/MathOps.H"      // for findMax
#include "Image/DrawOps.H"

#define WINSIZE           7
#define templThresh       2000.0F

// ######################################################################
SalientRegionTracker::SalientRegionTracker(OptionManager& mgr,
                             const std::string& descrName,
                             const std::string& tagName)
  :
  ModelComponent(mgr, descrName, tagName),
  itsTimer(1000000)
{
  itsCurrTrackedPoints.clear();

  //uint w = 160; uint h = 120;
  //itsWin.reset
  //  (new XWinManaged(Dims(7*w, 2*h), 0, 0, "Cmap Window" ));
}

// ######################################################################
SalientRegionTracker::~SalientRegionTracker()
{ }

// ######################################################################
void SalientRegionTracker::input
(Image<PixRGB<byte> > image, ImageSet<float> cmap, bool resetTracker,
 std::vector<Point2D<int> > points, std::vector<Rectangle> rects,
std::vector<rutz::shared_ptr<VisualObject> > visualObjects)
{
  itsTimer.reset();
  itsCurrCmap = cmap;

  // check if the tracker is currently inactive
  // if so, automatic reset tracker
  itsResetTracker = resetTracker;

  if(itsResetTracker == false && itsCurrTrackedPoints.size() == 0)
    return;

  // process the tracking task
  if(itsResetTracker)
    {
      itsOriginalInputImage       = image;
      itsCurrInputImage           = image;
      itsCurrTrackedPoints        = points;
      itsCurrTrackedVisualObjects = visualObjects;

      itsCurrTrackedROI = rects;
    }

  track();

  // print timer
  LDEBUG("Time: %6.3f ms", itsTimer.get()/1000.0);
}

// ######################################################################
void SalientRegionTracker::clear()
{
  itsPrevTrackedPointsScaled.clear();
  itsTrackerBias.clear();
  itsTrackerBiasOffset.clear();
  itsCurrTrackedPoints.clear();
  itsCurrTrackedROI.clear();
}

// ######################################################################
void SalientRegionTracker::move
(nub::soft_ref<SalientRegionTracker> tracker2, uint i)
{
  itsPrevTrackedPointsScaled.push_back
    (tracker2->getPrevTrackedPointsScaled(i));
  itsTrackerBias.push_back
    (tracker2->getTrackerBias(i));
  itsTrackerBiasOffset.push_back
    (tracker2->getTrackerBiasOffset(i));
  itsCurrTrackedPoints.push_back
    (tracker2->getCurrTrackedPoints(i));
  itsCurrTrackedROI.push_back
    (tracker2->getCurrTrackedROI(i));

  // FIXX: maybe want to move this as well
  //itsCurrTrackedVisualObjects = visualObjects;
}

// ######################################################################
void SalientRegionTracker::track()
{
  std::vector<Point2D<int> > diffs;
  for(uint i = 0; i < itsCurrTrackedPoints.size(); i++)
    {
      Point2D<int> pt =
        itsCurrTrackedROI[i].topLeft() - itsCurrTrackedPoints[i];
      diffs.push_back(pt);

      LDEBUG("diff[%3d]: (%4d,%4d) - (%4d,%4d) = (%4d,%4d)", i,
             itsCurrTrackedROI[i].topLeft().i,
             itsCurrTrackedROI[i].topLeft().j,
             itsCurrTrackedPoints[i].i,
             itsCurrTrackedPoints[i].j,
             diffs[i].i, diffs[i].j);
    }

  trackCmaps();

  Dims imgDims = itsCurrInputImage.getDims();
  Rectangle imgRect(Point2D<int>(0,0), imgDims);
  for(uint i = 0; i <  itsCurrTrackedPoints.size(); i++)
    {
      Point2D<int> tl = diffs[i] + itsCurrTrackedPoints[i];
      Dims d = itsCurrTrackedROI[i].dims();

      LDEBUG("imgRect[%4d,%4d,%4d,%4d]",
             imgRect.topLeft().i, imgRect.topLeft().j,
             imgRect.bottomRight().i, imgRect.bottomRight().j);

      Rectangle newRect(tl, d);
      LDEBUG("newRect[%4d,%4d,%4d,%4d]",
             newRect.topLeft().i, newRect.topLeft().j,
             newRect.bottomRight().i, newRect.bottomRight().j);

      itsCurrTrackedROI[i] = imgRect.getOverlap(newRect);

      LDEBUG("Resulting ROI[%4d,%4d,%4d,%4d]",
             itsCurrTrackedROI[i].topLeft().i,
             itsCurrTrackedROI[i].topLeft().j,
             itsCurrTrackedROI[i].bottomRight().i,
             itsCurrTrackedROI[i].bottomRight().j);
    }

  //trackVisualObjects();
}

// ######################################################################
void SalientRegionTracker::trackVisualObjects()
{
  if(itsResetTracker) return;

  // create a visual object for the scene
  std::string sName("scene");
  std::string sfName = sName + std::string(".png");
  rutz::shared_ptr<VisualObject>
    scene(new VisualObject(sName, sfName, itsCurrInputImage));

  // match the input visual objects
  for(uint i = 0; i < itsCurrTrackedVisualObjects.size(); i++)
    {
      // check for match
      Timer tim(1000000);
      VisualObjectMatchAlgo voma(VOMA_SIMPLE);
      rutz::shared_ptr<VisualObjectMatch> matchRes
        (new VisualObjectMatch(scene, itsCurrTrackedVisualObjects[i], voma));
      uint64 t = tim.get();

      // let's prune the matches:
      uint orgSize = matchRes->size();
       tim.reset();
      uint np = matchRes->prune();
      uint t2 = tim.get();

      LDEBUG("Found %u matches (%s & %s) in %.3fms:"
             " pruned %u in %.3fms",
             orgSize, scene->getName().c_str(),
             itsCurrTrackedVisualObjects[i]->getName().c_str(),
             float(t) * 0.001F,
             np, float(t2) * 0.001F);

      // matching score
      float kpAvgDist    = matchRes->getKeypointAvgDist();
      float afAvgDist    = matchRes->getAffineAvgDist();
      float score        = matchRes->getScore();
      bool isSIFTaffine  = matchRes->checkSIFTaffine
        (M_PI/4,5.0F,0.25F);
      SIFTaffine siftAffine = matchRes->getSIFTaffine();
      LDEBUG("kpAvgDist = %.4f|affAvgDist = %.4f|"
             " score: %.4f|aff? %d",
             kpAvgDist, afAvgDist, score, isSIFTaffine);

      if (!isSIFTaffine)
        LINFO("### Affine is too weird -- BOGUS MATCH");
      else
        {
          // show our final affine transform:
          LINFO("[testX]  [ %- .3f %- .3f ] [refX]   [%- .3f]",
                siftAffine.m1, siftAffine.m2, siftAffine.tx);
          LINFO("[testY]= [ %- .3f %- .3f ] [refY] + [%- .3f]",
                siftAffine.m3, siftAffine.m4, siftAffine.ty);
        }

      bool isSIFTfit = (isSIFTaffine && (score > 2.5) &&
                        (matchRes->size() > 3));
      LINFO("OD isSIFTfit %d", isSIFTfit);
    }
}

// ######################################################################
void SalientRegionTracker::trackCmaps()
{
  int smscale = (int)(pow(2,sml));

  // reset tracker?
  if(itsResetTracker)
    {
      itsPrevTrackedPointsScaled.clear();
      itsTrackerBias.clear();
      itsTrackerBiasOffset.clear();
    }

  for(uint i = 0; i < itsCurrTrackedPoints.size(); i++)
    {
      // if we are resetting the tracker
      if(itsResetTracker)
        {
          if(!itsCurrTrackedPoints[i].isValid())
            LFATAL("invalid input tracked point[%d]", i);

          itsPrevTrackedPointsScaled.push_back
            (Point2D<int>(itsCurrTrackedPoints[i].i/smscale,
                          itsCurrTrackedPoints[i].j/smscale));

          Point2D<int> tempOffset;
          itsTrackerBias.push_back
            (setNewBias(itsPrevTrackedPointsScaled[i], tempOffset));
          itsTrackerBiasOffset.push_back(tempOffset);
        }
      // else we are tracking (pt still not lost)
      else if(itsPrevTrackedPointsScaled[i].isValid())
        {
          LDEBUG("tracking current point[%d]", i);
          itsPrevTrackedPointsScaled[i] = trackPoint
            (itsTrackerBias[i],
             itsTrackerBiasOffset[i],
             itsPrevTrackedPointsScaled[i]);
        }
      // else it's previously lost
      else { LINFO("lost current point[%d]", i); }

      if(itsPrevTrackedPointsScaled[i].isValid())
        itsCurrTrackedPoints[i] =
          Point2D<int>(itsPrevTrackedPointsScaled[i].i*smscale,
                       itsPrevTrackedPointsScaled[i].j*smscale);
      else  itsCurrTrackedPoints[i] = Point2D<int>(-1,-1);
      LDEBUG("current track[%d] result: [%d,%d] -> [%d,%d]", i,
             itsCurrTrackedPoints[i].i,
             itsCurrTrackedPoints[i].j,
             itsPrevTrackedPointsScaled[i].i,
             itsPrevTrackedPointsScaled[i].j);
    }
}

// ######################################################################
ImageSet<float> SalientRegionTracker::setNewBias
(Point2D<int> inTrackLoc, Point2D<int> &biasOffset)
{
  int w = itsCurrCmap[0].getWidth();
  int h = itsCurrCmap[0].getHeight();

  ImageSet<float> bias(NUM_CHANNELS);

  // set bias offset
  if(inTrackLoc.i < (WINSIZE/2))
    biasOffset.i = inTrackLoc.i;
  else if(inTrackLoc.i > ((w - 1) - (WINSIZE/2)))
    biasOffset.i = WINSIZE - (w - inTrackLoc.i);
  else
    biasOffset.i = WINSIZE/2;

  if(inTrackLoc.j < (WINSIZE/2))
    biasOffset.j = inTrackLoc.j;
  else if(inTrackLoc.j > ((h - 1) - (WINSIZE/2)))
    biasOffset.j = WINSIZE - (h - inTrackLoc.j);
  else
    biasOffset.j = WINSIZE/2;

  LDEBUG("Set new bias[%d,%d]: offset: (%d, %d)",
         inTrackLoc.i, inTrackLoc.j, biasOffset.i, biasOffset.j);

  // get the features at the loc point
  for(int i = 0; i < NUM_CHANNELS; i++)
    {
      Point2D<int> upLeftsc(inTrackLoc.i - biasOffset.i,
                       inTrackLoc.j - biasOffset.j);
      Image<float> target = crop(itsCurrCmap[i], upLeftsc,
                                 Dims(WINSIZE,WINSIZE));
      bias[i] = target;
    }
  return bias;
}

// ######################################################################
Point2D<int> SalientRegionTracker::trackPoint
(ImageSet<float> &bias, Point2D<int> biasOffset,
  Point2D<int> trackLoc)
{
  int w = itsCurrCmap[0].getWidth();
  int h = itsCurrCmap[0].getHeight();

  // match templates
  Image<float> smap = getBiasedSMap(bias);

  // add a value to saliency based on distance from last point
  int i = 0; float maxDist = sqrt(w*w + h*h); //LINFO("maxDist: %f", maxDist);
  int wsmap = smap.getWidth();
  Point2D<int> prevLoc(trackLoc.i - biasOffset.i, trackLoc.j - biasOffset.j);
  for (Image<float>::iterator itr = smap.beginw(), stop = smap.endw();
       itr != stop; ++itr, i++)
    {
      int x = i % wsmap;
      int y = i / wsmap;
      float dist = (prevLoc.distance(Point2D<int>(x,y))+.1)/maxDist;
      *itr = *itr * dist;
    }

  // get the min val location
  // since the smap is corroded by WINSIZE all around,
  // winner coordinate point is actually the topleft of the bias window
  float minval; Point2D<int> upLeft; findMin(smap, upLeft, minval);

  // update the template
  updateTemplate(upLeft, bias);

  // get new tracking point
  Point2D<int> newTrackLoc = upLeft + biasOffset;
  return newTrackLoc;
}

// ######################################################################
void SalientRegionTracker::updateTemplate
( Point2D<int> upLeft, ImageSet<float> &bias)
{
  double dist = 0;
  ImageSet<float> newBias(NUM_CHANNELS);

  for(int i = 0; i < NUM_CHANNELS; i++)
    {
      Image<float> target =
        crop(itsCurrCmap[i], upLeft, Dims(WINSIZE,WINSIZE));

      // take more of the old template but still incorporate the new template
      newBias[i] = bias[i]*0.9 + target*(1 - 0.9);
      dist += distance(bias[i], newBias[i]);
    }

  // if the difference is too big, then do not update the template
  LDEBUG("Distance %f (thresh: %f)", dist, templThresh);
  if (dist < templThresh)
    {
      bias = newBias;
    }
  else LDEBUG("not adding bias");

  // did we lose the tracking completely?
  //float winDist = lastWinner.distance(trackLoc);
}

// ######################################################################
Image<float> SalientRegionTracker::getBiasedSMap(ImageSet<float> bias)
{
#ifndef HAVE_OPENCV
  LFATAL("OpenCV must be installed to use this function");
  return Image<float>();
#else

  int w = itsCurrCmap[0].getWidth();
  int h = itsCurrCmap[0].getHeight();
  //   int scale = (int)(pow(2,sml));

  Image<float> biasedCMap(w - WINSIZE + 1, h - WINSIZE + 1, ZEROS);
  Image<float> res(w - WINSIZE + 1, h - WINSIZE + 1, ZEROS);

  // add the bias of all the channels
  for(uint i = 0; i < NUM_CHANNELS; i++)
    {
      cvMatchTemplate(img2ipl(itsCurrCmap[i]), img2ipl(bias[i]),
                      img2ipl(biasedCMap), CV_TM_SQDIFF); //CV_TM_CCOEFF);

      // Add to saliency map: //save the cmap
      res += biasedCMap;
    }

  return res;

#endif
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
