/*!@file MBARI/mbariFunctions.C a few functions used by MBARI programs
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/mbariFunctions.C $
// $Id: mbariFunctions.C 10712 2009-02-01 06:22:17Z itti $
//


#include "MBARI/mbariFunctions.H"
#include "Image/ColorOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Transforms.H"
#include "Media/MediaSimEvents.H"
#include "MBARI/BitObject.H"
#include "Neuro/Brain.H"
#include "Neuro/NeuroSimEvents.H"
#include "Simulation/SimEventQueue.H"
#include "Util/Timer.H"
#include "rutz/shared_ptr.h"


// ######################################################################
std::list<BitObject> extractBitObjects(const Image<byte>& bImg,
                                       Rectangle region,
                                       int minSize)
{
  Timer timer;
  Image<byte> bitImg = replaceVals(bImg, byte(0), byte(0), byte(1));
  int tmask = 0, tobj = 0;
  std::list<BitObject> bos;
  region = region.getOverlap(bitImg.getBounds());
  Image<byte> labelImg(bitImg.getDims(),ZEROS);

  for (int ry = region.top(); ry < region.bottomO(); ++ry)
    for (int rx = region.left(); rx < region.rightO(); ++rx)
      {
        // this location doesn't have anything -> never mind
        if (bitImg.getVal(rx,ry) == 0) continue;

        // got this guy already -> never mind
        if (labelImg.getVal(rx,ry) > 0) continue;

        //Image<byte> dest(bitImg.getDims(), ZEROS);

        //timer.reset();
        //if (floodClean(bitImg,labelImg, Point2D<int>(rx,ry), byte(1), byte(1)) < minSize)
        //continue
        //int area = floodClean(bitImg,dest, Point2D<int>(rx,ry), byte(1), byte(1));
        //tflood += timer.get();
        //labelImg = takeMax(labelImg, dest);
        //if (area < minSize) continue;

        timer.reset();
        BitObject obj;
        Image<byte> dest = obj.reset(bitImg, Point2D<int>(rx,ry));
        tobj += timer.get();

        timer.reset();
        labelImg = takeMax(labelImg, dest);
        tmask += timer.get();

        if (obj.getArea() > minSize) bos.push_back(obj);
        //bos.push_back(BitObject(bitImg,Point2D<int>(rx,ry)));
        //bos.push_back(BitObject(dest));

        //LINFO("found object of size: %d",bos.back().getArea());
      }
  LINFO("tobj = %i; tmask = %i",tobj,tmask);
  return bos;
}

// ######################################################################
std::list<BitObject> getLargestObjects(const Image<byte>& bImg,
                                       Rectangle region,
                                       int numObjects,
                                       int minSize)
{
  std::list<BitObject> objs = extractBitObjects(bImg, region, minSize);
  std::list<BitObject> result;

  for (int i = 0; i < numObjects; ++i)
    {
      int maxSize = 0;
      std::list<BitObject>::iterator mx, cur;

      if (objs.empty()) break;
      for (cur = objs.begin(); cur != objs.end(); ++cur)
        {
          if (cur->getArea() > maxSize)
            {
              maxSize = cur->getArea();
              mx = cur;
            }
        }

      result.push_back(*mx);
      objs.erase(mx);
    }
  return result;
}

// ######################################################################
std::list<BitObject> getSalRegions(nub::soft_ref<Brain> brain,
                                   nub::soft_ref<SimEventQueue> q,
                                   const Image< PixRGB<byte> >& img,
                                   const Image<byte>& bitImg,
                                   float maxEvolveTime,
                                   int maxNumSalSpots,
                                   int minSize)
{
  // this should be 2^(smlev - 1)
  const int rectRad = 8;

  std::list<BitObject> bos;
  brain->reset(MC_RECURSE);

  rutz::shared_ptr<SimEventInputFrame>
    e(new SimEventInputFrame(brain.get(), GenericFrame(img), 0));
  q->post(e); // post the image to the brain

  int numSpots = 0;
  while ((q->now().secs() < maxEvolveTime) && (numSpots < maxNumSalSpots))
    {
      q->evolve();

      if (SeC<SimEventWTAwinner> e = q->check<SimEventWTAwinner>(0))
        {
          const Point2D<int> winner = e->winner().p;

          ++numSpots;

          // extract all the bitObjects at the salient location
          Rectangle region =
            Rectangle::tlbrI(winner.j - rectRad, winner.i - rectRad,
                            winner.j + rectRad, winner.i + rectRad);
          region = region.getOverlap(bitImg.getBounds());
          std::list<BitObject> sobjs = extractBitObjects(bitImg,region,minSize);

          // loop until we find a new object that doesn't overlap with anything
          // that we have found so far, or until we run out of objects
          bool keepGoing = true;
          while(keepGoing)
            {
              // no object left -> go to the next salient point
              if (sobjs.empty()) break;

              std::list<BitObject>::iterator biter, siter, largest;

              // find the largest object
              largest = sobjs.begin();
              int maxSize = 0;
              for (siter = sobjs.begin(); siter != sobjs.end(); ++siter)
                if (siter->getArea() > maxSize)
                  {
                    maxSize = siter->getArea();
                    largest = siter;
                  }

              // does the largest objects intersect with any of the already stored guys?
              keepGoing = false;
              for (biter = bos.begin(); biter != bos.end(); ++biter)
                if (biter->doesIntersect(*largest))
                  {
                    // no need to store intersecting objects -> get rid of largest
                    // and look for the next largest
                    sobjs.erase(largest);
                    keepGoing = true;
                    break;
                  }

              // so, did we end up finding a BitObject that we can store?
              if (!keepGoing) bos.push_back(*largest);

            } // end while keepGoing

        } // end if we found a winner

    } // end while we keep looking for salient points

  return bos;
}

// ######################################################################
Image<byte> showAllObjects(std::list<BitObject>& objs)
{
  Image<byte> result;
  std::list<BitObject>::iterator currObj;
  for (currObj = objs.begin(); currObj != objs.end(); ++currObj)
    {
      Image<byte> mask = currObj->getObjectMask(byte(255));
      if (result.initialized())
        result = takeMax(result,mask);
      else
        result = mask;
    }
  return result;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
