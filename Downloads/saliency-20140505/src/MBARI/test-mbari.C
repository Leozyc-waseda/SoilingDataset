/*!@file MBARI/test-mbari.C test program to detect marine animals
 */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   //
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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/test-mbari.C $
// $Id: test-mbari.C 10982 2009-03-05 05:11:22Z itti $
//

#include "Channels/ChannelOpts.H"
#include "Component/GlobalOpts.H"
#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Image/ColorOps.H"
#include "Image/FilterOps.H"    // for lowPass5y()
#include "Image/ImageCache.H"
#include "Image/Kernels.H"      // for twofiftyfives()
#include "Image/MorphOps.H"     // for openImg(), closeImg()
#include "Image/PyramidOps.H"
#include "Image/Transforms.H"
#include "MBARI/FOEestimator.H"
#include "MBARI/MbariFrameSeries.H"
#include "MBARI/MbariResultViewer.H"
#include "MBARI/VisualEvent.H"
#include "MBARI/mbariFunctions.H"
#include "Media/FrameRange.H"
#include "Media/MediaOpts.H"
#include "Neuro/NeuroOpts.H"
#include "Neuro/SimulationViewer.H"
#include "Neuro/SpatialMetrics.H"
#include "Neuro/StdBrain.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "Util/StringConversions.H"
#include "Util/log.H"
#include "Util/sformat.H"

#include <algorithm>
#include <cstdio>
#include <deque>
#include <iostream>
#include <sstream>

// Used by: InputMbariFrameSeries
static const ModelOptionDef OPT_InputMbariFrameRange =
  { MODOPT_ARG(FrameRange), "InputMbariFrameRange", &MOC_MBARIRV, OPTEXP_MRV,
    "Input frame range and delay in ms",
    "mbari-input-frames", 'M', "<first>-<last>", "0-0@0.0" };

namespace
{
  struct MbariFrameRange
  {
    MbariFrameRange() : first(0), last(0) {}

    int first, last;
  };

  bool operator==(const MbariFrameRange& r1,
                  const MbariFrameRange& r2)
  {
    return (r1.first == r2.first && r1.last == r2.last);
  }

  std::string convertToString(const MbariFrameRange& val)
  {
    return sformat("%d-%d", val.first, val.last);
  }

  void convertFromString(const std::string& str, MbariFrameRange& val)
  {
    std::stringstream s; int first = -2, last = -2; char c;
    s<<str; s>>first>>c>>last;
    if (first == -2 || last == -2 || c != '-')
      conversion_error::raise<MbariFrameRange>(str);

    val.first = first; val.last = last;
  }
}

int main(const int argc, const char** argv)
{
  // ######## Initialization of variables, reading of parameters etc.
  // a few constants
  const float maxEvolveTime = 0.5F;
  const uint maxNumSalSpots = 20;
  const uint minFrameNum = 5;
  const int minSizeRatio = 10000;
  const int maxDistRatio = 18;
  const int foaSizeRatio = 19;
  const int circleRadiusRatio = 40;
  const byte threshold = 5;
  const Image<byte> se = twofiftyfives(3);
  const int numFrameDist = 5;

  // initialize a few things
  ModelManager manager("MBARI test program");

  nub::soft_ref<SimEventQueueConfigurator>
    seqc(new SimEventQueueConfigurator(manager));
  manager.addSubComponent(seqc);

  nub::soft_ref<InputMbariFrameSeries> imfs(new InputMbariFrameSeries(manager));
  manager.addSubComponent(imfs);

  nub::soft_ref<OutputMbariFrameSeries> omfs(new OutputMbariFrameSeries(manager));
  manager.addSubComponent(omfs);

  nub::soft_ref<MbariResultViewer> rv(new MbariResultViewer(manager,omfs));
  manager.addSubComponent(rv);

  nub::soft_ref<StdBrain> brain(new StdBrain(manager));
  manager.addSubComponent(brain);

  nub::ref<SpatialMetrics> metrics(new SpatialMetrics(manager));
  manager.addSubComponent(metrics);

  // set up a frame range
  OModelParam<MbariFrameRange> frameRange
    (&OPT_InputMbariFrameRange, &manager);

  // set a bunch of paramters
  manager.setOptionValString(&OPT_OriInteraction,"SubtractMean");
  manager.setOptionValString(&OPT_OrientComputeType,"Steerable");
  manager.setOptionValString(&OPT_RawVisualCortexChans,"O:5IC");
  manager.setOptionValString(&OPT_UseRandom,"false");
  manager.setOptionValString(&OPT_ShapeEstimatorMode,"ConspicuityMap");
  manager.setOptionValString(&OPT_ShapeEstimatorSmoothMethod,"None");
  manager.setOptionValString(&OPT_IORtype,"ShapeEstCM");
  manager.setOptionValString(&OPT_SVdisplayFOA,"true");
  manager.setOptionValString(&OPT_SVdisplayPatch,"false");
  manager.setOptionValString(&OPT_SVdisplayFOALinks,"false");
  manager.setOptionValString(&OPT_SVdisplayAdditive,"true");
  manager.setOptionValString(&OPT_SVdisplayTime,"false");
  manager.setOptionValString(&OPT_SVdisplayBoring,"false");

  // parse the command line for the file names and set them
  if (!manager.parseCommandLine(argc, argv, "<input> <output>",1,2))
    return(1);

  // get the file names straight
  imfs->setFileStem(manager.getExtraArg(0));
  std::string outFileStem;
  if (manager.numExtraArgs() == 1)
    {
      outFileStem = "Res_";
      outFileStem.append(manager.getExtraArg(0));
      omfs->setFileStem(outFileStem);
    }
  else
    {
      outFileStem = manager.getExtraArg(1);
      omfs->setFileStem(outFileStem);
    }


  // get image dimensions and set a few paremeters that depend on it
  const Dims dims = imfs->peekDims(frameRange.getVal().first);
  const int minSize = dims.sz() / minSizeRatio;
  LINFO("minSize = %i",minSize);
  const int circleRadius = dims.w() / circleRadiusRatio;
  const int maxDist = dims.w() / maxDistRatio;
  LINFO("maxDist = %i",maxDist);
  const int foaSize = dims.w() / foaSizeRatio;
  metrics->setFOAradius(foaSize);
  nub::soft_ref<SimEventQueue> seq = seqc->getQ();


  // start all the ModelComponents
  manager.start();

  LINFO("after manager.start();");

  // initialize the visual event set
  VisualEventSet eventSet(maxDist, minFrameNum, minSize, manager.getExtraArg(0));
  int countFrameDist = 1;

  // are we loading the event structure from a file?
  const bool loadedEvents = rv->isLoadEventsNameSet();

  LINFO("before load events");
  if (loadedEvents) rv->loadVisualEventSet(eventSet);
  LINFO("after load events");

  PropertyVectorSet pvs;
  FOEestimator foeEst(20,0);

  // are we loading the set of property vectors from a file?
  const bool loadedProperties = rv->isLoadPropertiesNameSet();
  if (loadedProperties) rv->loadProperties(pvs);

  // initialize some more
  ImageCacheAvg< PixRGB<byte> > avgCache(rv->getAvgCacheSize());
  ImageCache< PixRGB<byte> > outCache(0);
  std::deque<int> outFrameNum;
  Image< PixRGB<byte> > img;

  // do we actually need to process the frames?
  if (rv->needFrames())
    {
      // pre-load and low-pass a few frames to get a valid average
      int currentFrame = frameRange.getVal().first;
      while(avgCache.size() < rv->getAvgCacheSize())
        {
          if (currentFrame > frameRange.getVal().last)
            {
              LERROR("Less input frames than necessary for sliding average - "
                     "using all the frames for caching.");
              break;
            }
          LINFO("Caching frame %06d.",currentFrame);
          img = lowPass5y(imfs->readRGB(currentFrame));
          avgCache.push_back(img);
          outCache.push_back(img);
          outFrameNum.push_back(currentFrame);
          ++currentFrame;
        }
    } // end if needFrames

  // ######## loop over frames ####################
  for (int curFrame = frameRange.getVal().first; curFrame <= frameRange.getVal().last; ++curFrame)
    {
      if (rv->needFrames())
        {
          // get image from cache or load and low-pass
          uint cacheFrameNum = curFrame - frameRange.getVal().first;
          if (cacheFrameNum < avgCache.size())
            {
              // we have cached this guy already
              LINFO("Processing frame %06d from cache.",curFrame);
              img = avgCache[cacheFrameNum];
            }
          else
            {
              // we need to load and low pass it and put it in the cache
              LINFO("Loading frame %06d.",curFrame);
              if (curFrame > frameRange.getVal().last)
                {
                  LERROR("Premature end of frame sequence - bailing out.");
                  break;
                }
              img = lowPass5y(imfs->readRGB(curFrame));
              avgCache.push_back(img);
              outCache.push_back(img);
              outFrameNum.push_back(curFrame);
              ++curFrame;
            }

          // subtract the running average from the image
          rv->output(img,curFrame,"LowPassed");
          img = avgCache.clampedDiffMean(img);
          rv->output(img,curFrame,"diffAvg");

        } // end if needFrames


      // all this we do not have to do if we load the event structure from a file
      if (!loadedEvents)
        {

          // create bw and binary versions of the img
          Image<byte> bwImg = maxRGB(img);
          rv->output(bwImg,curFrame,"BW");

          Image<byte> bitImg = makeBinary(bwImg, threshold);
          rv->output(bitImg,curFrame,"bin");

          Vector2D curFOE = foeEst.updateFOE(bitImg);

          /*
          std::cout << "Frame " << curFrame << ": FOE = ";
          if (curFOE.isValid())
            std::cout << curFOE.x() << " , " << curFOE.y() << "\n";
          else
            std::cout << "invalid\n";
          */
          if (curFOE.isValid()) std::cout << curFOE.x() << ' ' << curFOE.y() <<'\n';
          else std::cout << '\n';

          bitImg = closeImg(openImg(bitImg,se),se);
          rv->output(bitImg,curFrame,"Eroded");

          // update the events using the binary version
          eventSet.updateEvents(bitImg, curFOE, curFrame);

          // is counter at 0?
          --countFrameDist;
          if (countFrameDist == 0)
            {
              countFrameDist = numFrameDist;

              // get BitObjects at winning locations
              std::list<BitObject> sobjs = getSalRegions(brain, seq,
                                                         img, bitImg,
                                                         maxEvolveTime,
                                                         maxNumSalSpots,
                                                         minSize);
              //Rectangle region = img.getBounds();
              //std::list<BitObject> sobjs = getLargestObjects(bitImg,
              //                                             region,
              //                                             maxNumSalSpots,
              //                                             minSize);

              // display all the extracted objects
              rv->output(showAllObjects(sobjs),curFrame,"salient objects");

              // initiate events with these objects
              eventSet.initiateEvents(sobjs, curFrame);
            }

          // last frame? -> close everyone
          if (curFrame == frameRange.getVal().last) eventSet.closeAll();

          // weed out migit events (a.k.a too few frames)
          eventSet.cleanUp(curFrame);
        } // end if (!loadedEvents)

      // any closed events need flushing? -> flush out images
      int readyFrame;
      if ((curFrame == frameRange.getVal().last) || loadedEvents)
        readyFrame = curFrame;
      else
        //readyFrame = eventSet.getAllClosedFrameNum(curFrame);
        readyFrame = std::max(curFrame - int(minFrameNum), -1);

      // no frame ready -> go on
      if (readyFrame == -1) continue;

      // need to obtain the property vector set?
      if (!loadedProperties) pvs = eventSet.getPropertyVectorSet();

      // do this only when we actuall loaded frames
      if (rv->needFrames())
        {
          // see which frames are ready - output them and pop them off the cache
          while(outFrameNum.front() <= readyFrame)
            {
              rv->outputResultFrame(outCache.front(),outFileStem,
                                    outFrameNum.front(),
                                    eventSet,pvs,circleRadius);

              // need to save any event clips?
              uint csavenum = rv->numSaveEventClips();
              for (uint idx = 0; idx < csavenum; ++idx)
                {
                  uint evnum = rv->getSaveEventClipNum(idx);
                  if (!eventSet.doesEventExist(evnum)) continue;

                  VisualEvent event = eventSet.getEventByNumber(evnum);
                  if (event.isFrameOk(outFrameNum.front()))
                    rv->saveSingleEventFrame(outCache.front(),
                                             outFrameNum.front(),event);
                }

              outCache.pop_front();
              outFrameNum.pop_front();
              if (outFrameNum.empty()) break;
            }
        }

    } // end loop over all frames

  // write out eventSet?
  if (rv->isSaveEventsNameSet()) rv->saveVisualEventSet(eventSet);

  // write out property vector set?
  if (rv->isSavePropertiesNameSet()) rv->saveProperties(pvs);

  // write out positions?
  if (rv->isSavePositionsNameSet()) rv->savePositions(eventSet);

} // end main


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
