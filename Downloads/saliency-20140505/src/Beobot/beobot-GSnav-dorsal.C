/*!@file Beobot/beobot-GSnav-dorsal.C Robot navigation using saliency and gist.
  Run beobot-GSnav-master at CPU_A to run Gist-Saliency model
  Run beobot-GSnav        at CPU_B to run SIFT                          */

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
// ///////////////////////////////////////////////////////////////////////
//
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/beobot-GSnav-dorsal.C $
// $Id: beobot-GSnav-dorsal.C 15426 2012-11-02 21:44:22Z siagian $
//
//////////////////////////////////////////////////////////////////////////
// beobot-Gist-Sal-Nav.C <input_train.txt>
//
//
// This is an on-going project for biologically-plausible
// mobile-robotics navigation.
// It accepts any inputs: video  clip <input.mpg>, camera feed, frames.
//
// The system uses Gist to recognize places and saliency
// to get better localization within the place
// The program also is made to be streamline for fast processing using
// parallel computation. That is the V1 features of different channels are
// computed in parallel
//
// Currently it is able to recognize places through the use of gist features.
// The place classifier uses a neural networks,
// passed in a form of <input_train.txt> -
// the same file is used in the training phase by train-FFN.C.
//
// Related files of interest: GistEstimator.C (and .H) and
// GistEstimatorConfigurator.C (and .H) used by Brain.C to compute
// the gist features.
// test-Gist.C uses GistEstimator to extract gist features from an image.
//
// In parallel we use saliency to get a better spatial resolution
// as well as better place accuracy. The saliency model is used to obtain
// salient locations. We then use ShapeEstimator algorithm to segment out
// the sub-region to get a landmark. Using SIFT we can identify the object,
// create a database, etc.
//
// for localization, path planning we perform landmark-hopping
// to get to the final destination
//
//
//

#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Raster/Raster.H"
#include "GUI/XWinManaged.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Util/Timer.H"

#include "Beobot/beobot-GSnav-def.H"
#include "Beobot/BeobotBrainMT.H"
#include <signal.h>

#include "Image/ShapeOps.H"
#include "Image/CutPaste.H"     // for inplacePaste()
#include "Image/MathOps.H"      // for findMax
#include "Image/DrawOps.H"

#define WINSIZE           7
#define templThresh       2000.0F

static bool goforever = true;  //!< Will turn false on interrupt signal

// ######################################################################
// ######################################################################

//! get the request sent by Visual Cortex
void getTrackCommand
( TCPmessage &rmsg, int32 rframe, int32 rnode, int32 raction,
  ImageSet<float> &cmap,
  bool &resetCurrLandmark, std::vector<Point2D<int> > &clmpt,
  bool &resetNextLandmark, std::vector<Point2D<int> > &nlmpt);

//! process the request of the Visual Cortex
void processTrackCommand
( ImageSet<float> cmap,
  std::vector<ImageSet<float> > &clmbias, std::vector<Point2D<int> > &clmbiasOffset,
  std::vector<ImageSet<float> > &nlmbias, std::vector<Point2D<int> > &nlmbiasOffset,
  bool resetCurrLandmark,
  std::vector<Point2D<int> > &clmpt, std::vector<Point2D<int> > &prevClmptsc,
  bool resetNextLandmark,
  std::vector<Point2D<int> > &nlmpt, std::vector<Point2D<int> > &prevNlmptsc,
  rutz::shared_ptr<XWinManaged> dispWin);

//! setup the tracking result packet to be sent to Visual Cortex
void setupTrackingResultPacket
(TCPmessage  &smsg, int rframe,
 std::vector<Point2D<int> > clmpt, std::vector<Point2D<int> > nlmpt);

//! resets the bias template to the region around the point passed
ImageSet<float> setNewBias
( Point2D<int> inTrackLoc, Point2D<int> &biasOffset, ImageSet<float> cmap,
  rutz::shared_ptr<XWinManaged> dispWin);

//! track the object at the point
Point2D<int> trackPoint
( ImageSet<float> cmap, ImageSet<float> &bias, Point2D<int> biasOffset,
  Point2D<int> trackLoc, rutz::shared_ptr<XWinManaged> dispWin);

//! update the template used for tracking
void updateTemplate
( Point2D<int> upLeft, ImageSet<float> cmap, ImageSet<float> &bias,
  rutz::shared_ptr<XWinManaged> dispWin);

//! get the biased saliency map for tracking
Image<float> getBiasedSMap(ImageSet<float> cmap, ImageSet<float> bias,
                           rutz::shared_ptr<XWinManaged> dispWin);

// ######################################################################
//! Signal handler (e.g., for control-C)
void terminate(int s) { LERROR("*** INTERRUPT ***"); goforever = false; }

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("beobot Navigation using Gist and Saliency - Dorsal");

  // Instantiate our various ModelComponents:
  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Slave", "BeowulfSlave", false));
  manager.addSubComponent(beo);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false)
    return(1);

  // setup signal handling:
  signal(SIGHUP,  terminate); signal(SIGINT,  terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);

  TCPmessage rmsg;            // message being received and to process
  TCPmessage smsg;            // message being sent
  int32 rframe, raction, rnode = -1;

  // let's get all our ModelComponent instances started:
  manager.start();

  // receive the image dimension first
  while(!beo->receive(rnode, rmsg, rframe, raction, 5));
  LINFO("Dorsal size: %d", rmsg.getSize());
  const int fstart = int(rmsg.getElementInt32());
  const int w = int(rmsg.getElementInt32());
  const int h = int(rmsg.getElementInt32());
  const int opMode = int(rmsg.getElementInt32());
  int nSegment = int(rmsg.getElementInt32());
  int currSegNum = int(rmsg.getElementInt32());
  std::string envFName = rmsg.getElementString();
  std::string testRunFPrefix = rmsg.getElementString();
  std::string saveFilePath  = rmsg.getElementString();

  int ldpos = envFName.find_last_of('.');
  std::string testRunEnvFName = envFName.substr(0, ldpos) +
    std::string("-") + testRunFPrefix + std::string(".env");

  LINFO("fstart: %d", fstart);
  LINFO("envFName: %s", envFName.c_str());
  LINFO("Where we save data: %s", saveFilePath.c_str());
  LINFO("test run prefix: %s", testRunFPrefix.c_str());
  LINFO("testRunEnvFName: %s", testRunEnvFName.c_str());
  LINFO("Image dimension %d by %d", w,h);
  switch(opMode)
    {
    case TRAIN_MODE:
      LINFO("number of segments: %d", nSegment);
      LINFO("current segment: %d", currSegNum);
      LINFO("TRAIN_MODE: build landmark DB");
      break;
    case TEST_MODE:
      LINFO("TEST_MODE: let's roll");
      break;
    default: LERROR("Unknown operation mode");
    }
  rmsg.reset(rframe, raction);

  // send a message of ready to go
  smsg.reset(rframe, INIT_DONE);
  beo->send(rnode, smsg);

  // image buffer and window for display:
  Image<PixRGB<byte> > disp(w * 4, h, ZEROS); disp += PixRGB<byte>(128);
  rutz::shared_ptr<XWinManaged> dispWin;
  //dispWin.reset(new XWinManaged(Dims(w*4,h), 0, 0, "Dorsal display"));

  // tracker and current results
  std::vector<ImageSet<float> > clmbias(NUM_CHANNELS);
  std::vector<Point2D<int> > clmbiasOffset(NUM_CHANNELS);
  std::vector<ImageSet<float> > nlmbias(NUM_CHANNELS);
  std::vector<Point2D<int> > nlmbiasOffset(NUM_CHANNELS);
  std::vector<Point2D<int> > prevClmptsc;
  std::vector<Point2D<int> > prevNlmptsc;

  // MAIN LOOP
  LINFO("MAIN LOOP");
  Timer tim(1000000); uint64 t[NAVG]; float frate = 0.0f; uint fcount = 0;
  while(goforever)
    {
      // if we get a new message from V1
      // check the type of action to take with the data
      if(beo->receive(rnode, rmsg, rframe, raction, 1))
        {
          // abort: break out of the loop and finish the operation
          if(raction == ABORT)
            { LINFO("done: BREAK"); goforever = false; break; }

          // tracking task
          if(raction == TRACK_LM)
            {
              LINFO("message received: TRACK_LM");

              tim.reset();
              // get all the Visual Cortex information
              // FIX: need the sal map??
              ImageSet<float> cmap;
              bool resetCurrLandmark, resetNextLandmark;
              std::vector<Point2D<int> > clmpt;   std::vector<Point2D<int> > nlmpt;
              //std::vector<Point2D<int> > clmptsc; std::vector<Point2D<int> > nlmptsc;
              getTrackCommand
                (rmsg, rframe, rnode, raction, cmap,
                 resetCurrLandmark, clmpt, resetNextLandmark, nlmpt);

              // process the tracking task
              processTrackCommand
                (cmap, clmbias, clmbiasOffset, nlmbias, nlmbiasOffset,
                 resetCurrLandmark, clmpt, prevClmptsc,
                 resetNextLandmark, nlmpt, prevNlmptsc, dispWin);

              // return the tracking results
              setupTrackingResultPacket(smsg, rframe, clmpt, nlmpt);
              beo->send(rnode, smsg);

              LINFO("done TRACK_LM\n");

              // compute and show framerate over the last NAVG frames:
              t[fcount % NAVG] = tim.get(); fcount++; tim.reset();
              if (fcount % NAVG == 0 && fcount > 0)
                {
                  uint64 avg = 0ULL;
                  for(int i = 0; i < NAVG; i ++) avg += t[i];
                  frate = 1000000.0F / float(avg) * float(NAVG);
                  LINFO("[%6d] Frame rate: %6.3f fps -> %8.3f ms/frame",
                        fcount, frate, 1000.0/frate);
                }
            }
        }
    }

  // Received abort signal
  smsg.reset(rframe, raction);
  beo->send(rnode, smsg);
  LINFO("received ABORT signal");

  // ending operations
  switch(opMode)
    {
    case TRAIN_MODE:
      break;

    case TEST_MODE:
      // report some statistical data
      break;
    default:
      LERROR("Unknown operation mode");
    }

  // we got broken:
  manager.stop();
  return 0;
}

// ######################################################################
void getTrackCommand
( TCPmessage &rmsg, int32 rframe, int32 rnode, int32 raction,
  ImageSet<float> &cmap,
  bool &resetCurrLandmark, std::vector<Point2D<int> > &clmpt,
  bool &resetNextLandmark, std::vector<Point2D<int> > &nlmpt )
{
  cmap = rmsg.getElementFloatImaSet();

  int tempbool = int(rmsg.getElementInt32());
  if (tempbool == 0) resetCurrLandmark = false;
  else  resetCurrLandmark = true;

  tempbool = int(rmsg.getElementInt32());
  if (tempbool == 0) resetNextLandmark = false;
  else  resetNextLandmark = true;

  LINFO("[%4d] reset currLM? %d nextLM? %d",
        rframe, resetCurrLandmark, resetNextLandmark);

  uint nClmpt = int(rmsg.getElementInt32());
  clmpt.clear();
  for(uint i = 0; i < nClmpt; i++)
    {
      uint cI = int(rmsg.getElementInt32());
      uint cJ = int(rmsg.getElementInt32());
      clmpt.push_back(Point2D<int>(cI,cJ));
      LINFO("currLM[%d]: (%3d, %3d)", i, clmpt[i].i, clmpt[i].j);
    }

  uint nNlmpt = int(rmsg.getElementInt32());
  nlmpt.clear();
  for(uint i = 0; i < nNlmpt; i++)
    {
      uint nI = int(rmsg.getElementInt32());
      uint nJ = int(rmsg.getElementInt32());
      nlmpt.push_back(Point2D<int>(nI,nJ));
      LINFO("nextLM[%d]: (%3d, %3d)", i, nlmpt[i].i, nlmpt[i].j);
    }

  rmsg.reset(rframe, raction);
}

// ######################################################################
void processTrackCommand
( ImageSet<float> cmap,
  std::vector<ImageSet<float> > &clmbias, std::vector<Point2D<int> > &clmbiasOffset,
  std::vector<ImageSet<float> > &nlmbias, std::vector<Point2D<int> > &nlmbiasOffset,
  bool resetCurrLandmark,
  std::vector<Point2D<int> > &clmpt, std::vector<Point2D<int> > &prevClmptsc,
  bool resetNextLandmark,
  std::vector<Point2D<int> > &nlmpt, std::vector<Point2D<int> > &prevNlmptsc,
  rutz::shared_ptr<XWinManaged> dispWin)
{
  int smscale = (int)(pow(2,sml));

  // current landmark points:
  if(resetCurrLandmark)
    { prevClmptsc.clear(); clmbias.clear(); clmbiasOffset.clear(); }
  for(uint i = 0; i < clmpt.size(); i++)
    {

      // if we are resetting current landmark
      if(resetCurrLandmark)
        {
          if(!clmpt[i].isValid())  LFATAL("invalid currLmk[%d]", i);
          prevClmptsc.push_back(Point2D<int>(clmpt[i].i/smscale,
                                             clmpt[i].j/smscale));
          Point2D<int> tempOffset;
          clmbias.push_back(setNewBias(prevClmptsc[i], tempOffset, cmap,
                                       dispWin));
          clmbiasOffset.push_back(tempOffset);
        }
      // else we are tracking (pt still not lost)
      else if(prevClmptsc[i].isValid())
        {
          LINFO("tracking current Landmark[%d]", i);
          prevClmptsc[i] = trackPoint
            (cmap, clmbias[i], clmbiasOffset[i], prevClmptsc[i], dispWin);
        }
      // else it's previously lost
      else { LINFO("lost current Landmark[%d]", i); }

      if(prevClmptsc[i].isValid())
        clmpt[i] = Point2D<int>(prevClmptsc[i].i*smscale, prevClmptsc[i].j*smscale);
      else  clmpt[i] = Point2D<int>(-1,-1);
      LINFO("current landmark[%d] result: [%d,%d] -> [%d,%d]",
            i, clmpt[i].i, clmpt[i].j, prevClmptsc[i].i, prevClmptsc[i].j);
    }

  // next landmark point:
  if(resetNextLandmark)
    { prevNlmptsc.clear(); nlmbias.clear(); nlmbiasOffset.clear(); }
  for(uint i = 0; i < nlmpt.size(); i++)
    {
      // if we are resetting and it's a valid point
      if(resetNextLandmark)
        {
          if(!nlmpt[i].isValid())  LFATAL("invalid nextLmk[%d]", i);
          prevNlmptsc.push_back(Point2D<int>(nlmpt[i].i/smscale,
                                        nlmpt[i].j/smscale));
          Point2D<int> tempOffset;
          nlmbias.push_back(setNewBias(prevNlmptsc[i], tempOffset, cmap,
                                       dispWin));
          nlmbiasOffset.push_back(tempOffset);
        }
      // else we are tracking (pt still not lost)
      else if(prevNlmptsc[i].isValid())
        {
          LINFO("tracking next Landmark[%d]", i);
          prevNlmptsc[i] = trackPoint
            (cmap, nlmbias[i], nlmbiasOffset[i], prevNlmptsc[i], dispWin);
        }
      // else it's previously lost
      else { LINFO("lost next Landmark[%d]", i); }

      // resulting tracking/resetting
      if(prevNlmptsc[i].isValid())
        nlmpt[i] = Point2D<int>(prevNlmptsc[i].i*smscale, prevNlmptsc[i].j*smscale);
      else  nlmpt[i] = Point2D<int>(-1,-1);
      LINFO("next landmark[%d] result: [%d,%d] -> [%d,%d]",
            i, nlmpt[i].i, nlmpt[i].j, prevNlmptsc[i].i, prevNlmptsc[i].j);
    }
}

// ######################################################################
void setupTrackingResultPacket
(TCPmessage  &smsg, int rframe,
 std::vector<Point2D<int> > clmpt, std::vector<Point2D<int> > nlmpt)
{
  smsg.reset(rframe, TRACK_LM_RES);
  smsg.addInt32(int32(clmpt.size()));
  for(uint i = 0; i < clmpt.size(); i++)
    {
      smsg.addInt32(int32(clmpt[i].i));
      smsg.addInt32(int32(clmpt[i].j));
      LINFO("curr[%d]: (%d, %d)", i, clmpt[i].i, clmpt[i].j);
    }

  smsg.addInt32(int32(nlmpt.size()));
  for(uint i = 0; i < nlmpt.size(); i++)
    {
      smsg.addInt32(int32(nlmpt[i].i));
      smsg.addInt32(int32(nlmpt[i].j));
      LINFO("next[%d]: (%d %d)", i, nlmpt[i].i, nlmpt[i].j);
    }
}

// ######################################################################
ImageSet<float> setNewBias
(Point2D<int> inTrackLoc, Point2D<int> &biasOffset, ImageSet<float> cmap,
 rutz::shared_ptr<XWinManaged> dispWin)
{
  int w = cmap[0].getWidth();
  int h = cmap[0].getHeight();

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

  LINFO("Set new bias[%d,%d]: offset: (%d, %d)",
        inTrackLoc.i, inTrackLoc.j, biasOffset.i, biasOffset.j);

  // get the features at the loc point
  for(int i = 0; i < NUM_CHANNELS; i++)
    {
      Point2D<int> upLeftsc(inTrackLoc.i - biasOffset.i,
                       inTrackLoc.j - biasOffset.j);
      Image<float> target = crop(cmap[i], upLeftsc, Dims(WINSIZE,WINSIZE));
      bias[i] = target;
    }
  return bias;
}

// ######################################################################
Point2D<int> trackPoint
( ImageSet<float> cmap, ImageSet<float> &bias, Point2D<int> biasOffset,
  Point2D<int> trackLoc, rutz::shared_ptr<XWinManaged> dispWin)
{
  int w = cmap[0].getWidth();
  int h = cmap[0].getHeight();

  // match templates
  Image<float> smap = getBiasedSMap(cmap, bias, dispWin);

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
  updateTemplate(upLeft, cmap, bias, dispWin);

  // get new tracking point
  Point2D<int> newTrackLoc = upLeft + biasOffset;
  return newTrackLoc;
}

// ######################################################################
void updateTemplate
( Point2D<int> upLeft, ImageSet<float> cmap, ImageSet<float> &bias,
  rutz::shared_ptr<XWinManaged> dispWin)
{
  double dist = 0;
  ImageSet<float> newBias(NUM_CHANNELS);

  for(int i = 0; i < NUM_CHANNELS; i++)
    {
      Image<float> target = crop(cmap[i], upLeft, Dims(WINSIZE,WINSIZE));

      // take more of the old template but still incorporate the new template
      newBias[i] = bias[i]*0.9 + target*(1 - 0.9);
      dist += distance(bias[i], newBias[i]);
    }

  // if the difference is too big, then do not update the template
  LINFO("Distance %f (thresh: %f)", dist, templThresh);
  if (dist < templThresh)
    {
      bias = newBias;
    }
  else LINFO("not adding bias");

  // did we lose the tracking completely?
  //float winDist = lastWinner.distance(trackLoc);
}

// ######################################################################
Image<float> getBiasedSMap(ImageSet<float> cmap, ImageSet<float> bias,
                           rutz::shared_ptr<XWinManaged> dispWin)
{
#ifndef HAVE_OPENCV
  LFATAL("OpenCV must be installed to use this function");
  return Image<float>();
#else

  int w = cmap[0].getWidth();
  int h = cmap[0].getHeight();
  //   int scale = (int)(pow(2,sml));

  Image<float> biasedCMap(w - WINSIZE + 1, h - WINSIZE + 1, ZEROS);
  Image<float> res(w - WINSIZE + 1, h - WINSIZE + 1, ZEROS);

  // add the bias of all the channels
  for(uint i = 0; i < NUM_CHANNELS; i++)
    {
      cvMatchTemplate(img2ipl(cmap[i]), img2ipl(bias[i]),
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
