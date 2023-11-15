/*!@file RCBot/TrackFeature.C track a location using the saliency map and
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/RCBot/TrackFeature.C $
// $Id: TrackFeature.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "RCBot/TrackFeature.H"

//#include "GUI/XWinManaged.H"
//XWinManaged xwin(Dims(320,240), -1, -1, "Track Feature");

// ######################################################################
TrackFeature::TrackFeature(nub::ref<SaliencyMT> insmt) :
  smt(insmt), trackLoc(-1,-1), bias(14), newBias(14)
{
  doneTracking = false;
  avgtime = 0;
  avgn = 0;
  fps = 0.0F;

  templThresh = 50.0F;

  width = -1; height = -1;
  timer.reset();
  start();
}

// ######################################################################
TrackFeature::~TrackFeature(){}

// ######################################################################
void TrackFeature::run(void *ptr)
{
  LINFO("Starting track thread");
  while(!doneTracking){

    // do we have a point to track?
    trackLocMutex.lock();
    bool isLocValid = (trackLoc.i > (WINSIZE/2) &&
                       trackLoc.i < width - (WINSIZE/2) &&
                       trackLoc.j > (WINSIZE/2) &&
                       trackLoc.j < height - (WINSIZE/2)
                       );
    trackLocMutex.unlock();

    if (isLocValid){

      // get the lock and wait for images
      trackMutex.lock();
      trackCond.wait();
      // process the image and track
      trackMutex.unlock();

      if (smt->outputReady()){
        Image<float> SMap = smt->getOutput();
        if (SMap.initialized()){
          //xwin.drawImage(SMap);
          // add a value to saliency based on distance from last point
          int w = SMap.getWidth();
          int i = 0;
          for (Image<float>::iterator itr = SMap.beginw(), stop = SMap.endw();
               itr != stop; ++itr, i++) {
            int x = i % w;
            int y = i / w;
            float dist = lastWinner.distance(Point2D<int>(x,y));
            *itr = *itr - (0.5*dist);
          }
          float maxval; Point2D<int> currWinner; findMin(SMap, currWinner, maxval);

          // since the saliency map is smaller than winsize,
          // remap the winner point
          currWinner.i = currWinner.i + (WINSIZE-1)/2;
          currWinner.j = currWinner.j + (WINSIZE-1)/2;

          trackImgMutex.lock();
          smt->newInput(img);
          trackImgMutex.unlock();

          trackLocMutex.lock();
          trackLoc = currWinner;
          trackLocMutex.unlock();

          // update the template
          updateTemplate();

          // calculate fps
          calcFps();

          lastWinner = currWinner;
        }
      }

    } else {
      trackLocMutex.lock();
      trackLoc.i = -1;
      trackLoc.j = -1;
      trackLocMutex.unlock();
    }
  }
}

// ######################################################################
void TrackFeature::updateTemplate()
{
  double dist = 0;
  trackLocMutex.lock();
  Point2D<int> currLoc = trackLoc;
  trackLocMutex.unlock();
  static int badTrack = 0;

  bool isLocValid = (currLoc.i > (WINSIZE/2) &&
                     currLoc.i < width - (WINSIZE/2) &&
                     currLoc.j > (WINSIZE/2) &&
                     currLoc.j < height - (WINSIZE/2));

  if (isLocValid){
    for(int i = 0; i < 14; i++){
      if (smt->cmaps[i].initialized()){
        if (bias[i].initialized()){
          Point2D<int> center_fixation(currLoc.i-((WINSIZE-1)/2),
                                  currLoc.j-((WINSIZE-1)/2));
          Image<float> target = crop(smt->cmaps[i],
                                     center_fixation,
                                     Dims(WINSIZE,WINSIZE));

          // take more of the old template but still incorporate the new template
          newBias[i] = bias[i]*0.9 + target*(1-0.9);
          dist += distance(bias[i], newBias[i]);
          bias[i] = newBias[i];
        }
      }
    }
  } else {
    dist = 999999999;
  }

  // if the difference is too big, then do not update the template
  if (dist < templThresh){
    smt->setSMBias(newBias);
    badTrack = 0;
  } else {
    badTrack++;
  }
  //LINFO("Distance %f", dist);

  // did we lose the tracking completely?
  //float winDist = lastWinner.distance(trackLoc);
  //if (dist > templThresh){

  // try 3 times
  if (badTrack > 3){ // we lost the tracking
    trackLocMutex.lock();
    trackLoc.i = -1;
    trackLoc.j = -1;
    trackLocMutex.unlock();
  }
}

// ######################################################################
Point2D<int> TrackFeature::getTrackLoc()
{
  trackLocMutex.lock();
  Point2D<int> loc = trackLoc;
  trackLocMutex.unlock();

  return loc;
}

// ######################################################################
void TrackFeature::calcFps()
{
  avgtime += timer.getReset(); avgn ++;
  if (avgn == NAVG)
    {
      fps = 1000.0F / float(avgtime) * float(avgn);
      avgtime = 0; avgn = 0;
    }
}

// ######################################################################
float TrackFeature::getFps(){ return fps; }

// ######################################################################
void TrackFeature::setImg(Image<PixRGB<byte> > &inImg)
{
  trackImgMutex.lock();
  img = inImg;
  width = img.getWidth();
  height = img.getHeight();
  trackImgMutex.unlock();
  trackCond.signal();
}

// ######################################################################
void TrackFeature::setTrackLoc(Point2D<int> &inTrackLoc, Image<PixRGB<byte> > &img)
{
  trackLocMutex.lock();
  LINFO("Setting tracking location");
  bool isLocValid = (inTrackLoc.i > (WINSIZE/2) &&
                     inTrackLoc.i < width - (WINSIZE/2) &&
                     inTrackLoc.j > (WINSIZE/2) &&
                     inTrackLoc.j < height - (WINSIZE/2));
  trackLocMutex.unlock();

  if (isLocValid)
  {
    // discard the currently processed saliency map
    // build a new unbiased saliency map, and find the highest local value within it

    LINFO("Wait for smt");
    while(!smt->outputReady()){ // wait for any processing to finish
      usleep(100);
    }
    LINFO("Done waiting");

    smt->setBiasSM(false);        // let the system know we don't want an unbiased smap
    //smt->setSaliencyMapLevel(0); // set the saliency map level to 0
    smt->getSMap(img);

    LINFO("Set biases");
    // get the features at the loc point
    for(int i=0; i<14; i++){
      if (smt->cmaps[i].initialized()){
        Point2D<int> center_loc(inTrackLoc.i - ((WINSIZE - 1)/2),
                           inTrackLoc.j - ((WINSIZE - 1)/2));
        Image<float> target = crop(smt->cmaps[i],center_loc, Dims(WINSIZE,WINSIZE));
        bias[i] = target;
      }
    }

    LINFO("Done setting bias");
    smt->setSMBias(bias);
    smt->setBiasSM(true);

    trackLocMutex.lock();
    trackLoc = inTrackLoc; // update the track location
    trackLocMutex.unlock();

    smt->newInput(img);
  }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
