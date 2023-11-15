/*!@file MBARI/VisualEvent.C classes useful for event tracking */

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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/VisualEvent.C $
// $Id: VisualEvent.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "MBARI/VisualEvent.H"

#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/Rectangle.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"
#include "Image/colorDefs.H"
#include "MBARI/Geometry2D.H"
#include "MBARI/mbariFunctions.H"
#include "Util/Assert.H"
#include "Util/StringConversions.H"

#include <algorithm>
#include <istream>
#include <ostream>

// ######################################################################
// ###### Token
// ######################################################################
Token::Token()
  : bitObject(),
    location(),
    prediction(),
    line(),
    angle(0.0F),
    frame_nr(0)
{}

// ######################################################################
Token::Token (float x, float y, uint frame, BitObject bo)
  : bitObject(bo),
    location(x,y),
    prediction(),
    line(),
    angle(0.0F),
    frame_nr(frame)
{}

// ######################################################################
Token::Token (BitObject bo, uint frame)
  : bitObject(bo),
    location(bo.getCentroidXY()),
    prediction(),
    line(),
    angle(0.0F),
    frame_nr(frame)
{}

// ######################################################################
Token::Token (std::istream& is)
{
  readFromStream(is);
}

// ######################################################################
void Token::writeToStream(std::ostream& os) const
{
  os << frame_nr << ' ';
  location.writeToStream(os);
  prediction.writeToStream(os);
  line.writeToStream(os);
  os << angle << '\n';
  bitObject.writeToStream(os);
  os << "\n";
}

// ######################################################################
void Token::readFromStream(std::istream& is)
{
  is >> frame_nr;
  location.readFromStream(is);
  prediction.readFromStream(is);
  line.readFromStream(is);
  is >> angle;
  bitObject = BitObject(is);
}

// ######################################################################
void Token::writePosition(std::ostream& os) const
{
  location.writeToStream(os);
}


// ######################################################################
// ####### PropertyVectorSet
// ######################################################################
PropertyVectorSet::PropertyVectorSet()
{}

// ######################################################################
PropertyVectorSet::PropertyVectorSet(std::istream& is)
{
  readFromStream(is);
}

// ######################################################################
void PropertyVectorSet::writeToStream(std::ostream& os) const
{
  uint s2;
  if (itsVectors.empty()) s2 = 0;
  else s2 = itsVectors.front().size();

  os << itsVectors.size() << " " << s2 << "\n";

  for (uint i = 0; i < itsVectors.size(); ++i)
    {
      for (uint j = 0; j < s2; ++j)
        os << itsVectors[i][j] << " ";

      os << "\n";
    }
}

// ######################################################################
void PropertyVectorSet::readFromStream(std::istream& is)
{
  uint s1, s2;
  is >> s1; is >> s2;
  itsVectors = std::vector< std::vector<float> > (s1, std::vector<float>(s2));

  for (uint i = 0; i < s1; ++i)
    for (uint j = 0; j < s2; ++j)
      is >> itsVectors[i][j];
}

// ######################################################################
std::vector<float> PropertyVectorSet::getPropertyVectorForEvent(const int num)
{
  for (uint i = 0; i < itsVectors.size(); ++i)
    if ((int)(itsVectors[i][0]) == num) return itsVectors[i];

  LFATAL("property vector for event number %d not found!", num);
  return std::vector<float>();
}

// ######################################################################
// ####### VisualEvent
// ######################################################################
VisualEvent::VisualEvent(Token tk, int maxDist)
  : startframe(tk.frame_nr),
    endframe(tk.frame_nr),
    max_size(tk.bitObject.getArea()),
    maxsize_framenr(tk.frame_nr),
    closed(false),
    itsMaxDist(maxDist),
    xTracker(tk.location.x(),0.1F,10.0F),
    yTracker(tk.location.y(),0.1F,10.0F)
{
  LINFO("tk.location = (%g, %g); area: %i",tk.location.x(),tk.location.y(),
        tk.bitObject.getArea());
  tokens.push_back(tk);
  ++counter;
  myNum = counter;
}
// initialize static variable
uint VisualEvent::counter = 0;

// ######################################################################
VisualEvent::VisualEvent(std::istream& is)
{
  readFromStream(is);
}

// ######################################################################
void VisualEvent::writeToStream(std::ostream& os) const
{
  os << myNum << " " << startframe << " " << endframe << "\n";
  os << max_size << " " << maxsize_framenr << "\n";

  if (closed) os << "1\n";
  else os << "0\n";

  os << itsMaxDist << "\n";

  xTracker.writeToStream(os);
  yTracker.writeToStream(os);

  os << tokens.size() << "\n";

  for (uint i = 0; i < tokens.size(); ++i)
    tokens[i].writeToStream(os);

  os << "\n";
}

// ######################################################################
void VisualEvent::readFromStream(std::istream& is)
{
  is >> myNum;
  is >> startframe;
  is >> endframe;
  is >> max_size;
  is >> maxsize_framenr;

  int n; is >> n;
  closed = (n == 1);

  is >> itsMaxDist;

  xTracker.readFromStream(is);
  yTracker.readFromStream(is);

  int t;
  is >> t;
  tokens.clear();
  for (int i = 0; i < t; ++i)
    tokens.push_back(Token(is));
}

// ######################################################################
void VisualEvent::writePositions(std::ostream& os) const
{
  for (uint i = 0; i < tokens.size(); ++i)
    tokens[i].writePosition(os);

  os << "\n";
}

// ######################################################################
Point2D<int> VisualEvent::predictedLocation() const
{
  int x = int(xTracker.getEstimate() + 0.5F);
  int y = int(yTracker.getEstimate() + 0.5F);
  return Point2D<int>(x,y);
}

// ######################################################################
bool VisualEvent::isTokenOk(const Token& tk) const
{
  return ((tk.frame_nr - endframe) == 1) && !closed;
}

// ######################################################################
float VisualEvent::getCost(const Token& tk) const
{
  if (!isTokenOk(tk)) return -1.0F;

  float cost = (xTracker.getCost(tk.location.x()) +
                yTracker.getCost(tk.location.y()));

  LINFO("Event no. %i; obj location: %g, %g; predicted location: %g, %g; cost: %g",
        myNum, tk.location.x(), tk.location.y(), xTracker.getEstimate(),
        yTracker.getEstimate(), cost);
  return cost;
}

// ######################################################################
void VisualEvent::assign(const Token& tk, const Vector2D& foe)
{
  ASSERT(isTokenOk(tk));

  tokens.push_back(tk);

  tokens.back().prediction = Vector2D(xTracker.getEstimate(),
                                      yTracker.getEstimate());
  tokens.back().location = Vector2D(xTracker.update(tk.location.x()),
                                    yTracker.update(tk.location.y()));

  // update the straight line
  //Vector2D dir(xTracker.getSpeed(), yTracker.getSpeed());
  Vector2D dir = tokens.front().location - tokens.back().location;
  tokens.back().line.reset(tokens.back().location, dir);

  if (foe.isValid())
    tokens.back().angle = dir.angle(tokens.back().location - foe);
  else
    tokens.back().angle = 0.0F;

  if (tk.bitObject.getArea() > max_size)
    {
      max_size = tk.bitObject.getArea();
      maxsize_framenr = tk.frame_nr;
    }
  endframe = tk.frame_nr;
}

// ######################################################################
bool VisualEvent::doesIntersect(const BitObject& obj, int frameNum) const
{
  if (!isFrameOk(frameNum)) return false;
  else return getToken(frameNum).bitObject.doesIntersect(obj);
}


// ######################################################################
std::vector<float>  VisualEvent::getPropertyVector()
{
  std::vector<float> vec;
  Token tk = getMaxSizeToken();
  BitObject bo = tk.bitObject;

  // 0 - event number
  vec.push_back(getEventNum());

  // 1 - interesting value
  vec.push_back(-1);

  // not valid?
  if (!bo.isValid())
    {
      // 2  - set area to -1
      vec.push_back(-1);

      // 3-12 set everyone to 0
      for (uint i = 3; i <= 12; ++i)
        vec.push_back(0);

      // done
      return vec;
    }

  // we're valid

  // 2 - area
  vec.push_back(bo.getArea());

  // 3, 4, 5 - uxx, uyy, uxy
  float uxx, uyy, uxy;
  bo.getSecondMoments(uxx, uyy, uxy);
  vec.push_back(uxx);
  vec.push_back(uyy);
  vec.push_back(uxy);

  // 6 - major axis
  vec.push_back(bo.getMajorAxis());

  // 7 - minor axis
  vec.push_back(bo.getMinorAxis());

  // 8 - elongation
  vec.push_back(bo.getElongation());

  // 9 - orientation angle
  vec.push_back(bo.getOriAngle());

  // 10, 11, 12 - max, min, avg intensity
  float maxIntens,minIntens,avgIntens;
  bo.getMaxMinAvgIntensity(maxIntens, minIntens, avgIntens);
  vec.push_back(maxIntens);
  vec.push_back(minIntens);
  vec.push_back(avgIntens);

  // 13 - angle with respect to expansion
  vec.push_back(tk.angle);

  // done -> return the vector
  return vec;
 }

// ######################################################################
Dims VisualEvent::getMaxObjectDims() const
{
  int w = -1, h = -1;
  std::vector<Token>::const_iterator t;
  for (t = tokens.begin(); t != tokens.end(); ++t)
    {
      Dims d = t->bitObject.getObjectDims();
      w = std::max(w, d.w());
      h = std::max(h, d.h());
    }
  return Dims(w,h);
}

// ######################################################################
// ###### VisualEventSet
// ######################################################################
VisualEventSet::VisualEventSet(const int maxDist,
                               const uint minFrameNum,
                               const int minSize,
                               const std::string& fileName)
  : itsMaxDist(maxDist),
    itsMinFrameNum(minFrameNum),
    itsMinSize(minSize),
    startframe(-1),
    endframe(-1),
    itsFileName(fileName)
{
  float maxAreaDiff = maxDist * maxDist / 4.0F;
  itsMaxCost = maxDist * maxDist + maxAreaDiff * maxAreaDiff;
}

// ######################################################################
VisualEventSet::VisualEventSet(std::istream& is)
{
  readFromStream(is);
}

// ######################################################################
void VisualEventSet::writeToStream(std::ostream& os) const
{
  os << itsFileName << "\n";
  os << itsMaxDist << " "
     << itsMaxCost << " "
     << itsMinFrameNum << " "
     << itsMinSize << "\n";
  os << startframe << ' ' << endframe << '\n';

  os << itsEvents.size() << "\n";
  std::list<VisualEvent>::const_iterator currEvent;
  for (currEvent = itsEvents.begin(); currEvent != itsEvents.end(); ++currEvent)
    currEvent->writeToStream(os);

  os << itsFOE.size() << '\n';
  for (uint i = 0; i < itsFOE.size(); ++i)
    itsFOE[i].writeToStream(os);

  os << "\n";
}

// ######################################################################
void VisualEventSet::readFromStream(std::istream& is)
{
  is >> itsFileName; LINFO("filename: %s",itsFileName.c_str());
  is >> itsMaxDist;
  is >> itsMaxCost;
  is >> itsMinFrameNum;
  is >> itsMinSize;
  is >> startframe;
  is >> endframe;

  int n; is >> n;
  itsEvents.clear();
  for (int i = 0; i < n; ++i)
    itsEvents.push_back(VisualEvent(is));

  is >> n;
  itsFOE.clear();
  for (int i = 0; i < n; ++i)
    itsFOE.push_back(Vector2D(is));
}

// ######################################################################
void VisualEventSet::writePositions(std::ostream& os) const
{
  std::list<VisualEvent>::const_iterator currEvent;
  for (currEvent = itsEvents.begin(); currEvent != itsEvents.end(); ++currEvent)
    currEvent->writePositions(os);
}

// ######################################################################
void VisualEventSet::updateEvents(const Image<byte>& binMap,
                                  const Vector2D& curFOE,
                                  int frameNum)
{
  if (startframe == -1) {startframe = frameNum; endframe = frameNum;}
  ASSERT((frameNum == endframe) || (frameNum == endframe+1));
  if (frameNum > endframe) endframe = frameNum;

  itsFOE.push_back(curFOE);

  std::list<VisualEvent>::iterator currEvent;
  for (currEvent = itsEvents.begin(); currEvent != itsEvents.end(); ++currEvent)
    {
      if (currEvent->isClosed()) continue;

      // get the predicted location
      Point2D<int> pred = currEvent->predictedLocation();

      // is the prediction too far outside the image?
      //int gone = itsMaxDist / 2;
      int gone = 0;
      if ((pred.i < -gone) || (pred.i >= (binMap.getWidth() + gone)) ||
          (pred.j < -gone) || (pred.j >= (binMap.getHeight() + gone)))
        {
          currEvent->close();
          LINFO("Event %i out of bounds - closed",currEvent->getEventNum());
          continue;
        }

      // get the region used for searching for a match
      Rectangle region =
        Rectangle::tlbrI(pred.j - itsMaxDist, pred.i - itsMaxDist,
                        pred.j + itsMaxDist, pred.i + itsMaxDist);
      region = region.getOverlap(binMap.getBounds());

      // extract all the BitObjects from the region
      std::list<BitObject> objs = extractBitObjects(binMap, region, itsMinSize);
      //LINFO("pred. location: %s; region: %s; Number of extracted objects: %i",
      //    toStr(pred).c_str(),toStr(region).c_str(),objs.size());

      // now look which one fits best
      float lCost = -1.0F;
      std::list<BitObject>::iterator cObj, lObj = objs.end();
      for (cObj = objs.begin(); cObj != objs.end(); ++cObj)
        {
          if (doesIntersect(*cObj, frameNum)) continue;

          float cost = currEvent->getCost(Token(*cObj,frameNum));

          //LINFO("Event no. %i; cost: %g; lowest cost: %g",currEvent->getEventNum(),
          //    cost, lCost);
         if (cost < 0.0F) continue;
         if ((lCost == -1.0F) || (cost < lCost))
           {
             lCost = cost;
             lObj = cObj;
             //LINFO("best cost: %g",lCost);
           }
        }

      // cost too high no fitting object found? -> close event
      if ((lCost > itsMaxCost) || (lCost == -1.0))
        {
          currEvent->close();
          LINFO("Event %i - no token found, closing event",
                currEvent->getEventNum());
        }
      else
        {
          // associate the best fitting guy
          Token tk(*lObj, frameNum);
          currEvent->assign(tk, curFOE);

          LINFO("Event %i - token found at %g, %g",currEvent->getEventNum(),
                currEvent->getToken(frameNum).location.x(),
                currEvent->getToken(frameNum).location.y());
        }
    }
}

// ######################################################################
void VisualEventSet::initiateEvents(std::list<BitObject>& bos, int frameNum)
{
  if (startframe == -1) {startframe = frameNum; endframe = frameNum;}
  ASSERT((frameNum == endframe) || (frameNum == endframe+1));
  if (frameNum > endframe) endframe = frameNum;

  std::list<BitObject>::iterator currObj;

  // loop over the BitObjects
  currObj = bos.begin();
  while(currObj != bos.end())
    {
      // is there an intersection with an event?
      if (doesIntersect(*currObj, frameNum))
        {
          //LINFO("Object at %g, %g intersects with event %i, erasing the object",
          //    currObj->getCentroidX(), currObj->getCentroidY(),
          //    currEvent->getEventNum());
          currObj = bos.erase(currObj);
        }
      else
        {
          //LINFO("Object at %g, %g does not intersect with event %i",
          //currObj->getCentroidX(), currObj->getCentroidY(),
          //currEvent->getEventNum());
          ++currObj;
        }
    }

  // now go through all the remaining BitObjects and create new events for them
  for (currObj = bos.begin(); currObj != bos.end(); ++currObj)
    {
      itsEvents.push_back(VisualEvent(Token(*currObj, frameNum), itsMaxDist));
      LINFO("assigning object of area: %i to new event %i",currObj->getArea(),
            itsEvents.back().getEventNum());
    }
}

// ######################################################################
bool VisualEventSet::doesIntersect(const BitObject& obj, int frameNum) const
{
  std::list<VisualEvent>::const_iterator cEv;
  for (cEv = itsEvents.begin(); cEv != itsEvents.end(); ++cEv)
    if (cEv->doesIntersect(obj,frameNum)) return true;

  return false;
}

/*
// ######################################################################
Vector2D VisualEventSet::updateFOE()
{
  const uint minTrackFrames = 3;
  std::vector<StraightLine2D> lines;
  std::vector<std::list<VisualEvent>::iterator> events =
    getEventsForFrame(endframe);

  // go through all the events and extract the straight lines
  for (uint i = 0; i < events.size(); ++i)
    {
      // must have at least minTrackFrames points until here
      if ((endframe - events[i]->getStartFrame() + 1) >= minTrackFrames)
        {
          StraightLine2D l = events[i]->getToken(endframe).line;
          if (l.isValid()) lines.push_back(l);
        }
    }

  // get all the intersection points of the straight lines
  std::vector<float> xCoords, yCoords;
  for (uint i = 0; i < lines.size(); ++i)
    for (uint j = 0; j < lines.size(); ++j)
      {
        if (i == j) continue;
        float n,m;
        Vector2D inter = lines[i].intersect(lines[j],n,m);
        if (inter.isValid())
          {
            xCoords.push_back(inter.x());
            yCoords.push_back(inter.y());
          }
      }

  // update the FOE
  Vector2D currFOE;

  if (xCoords.empty())
    {
      //itsFOE.push_back(Vector2D());
    }
  else
    {

      // this is all for taking the median
      // sort xCoords and yCoords
      //std::sort(xCoords.begin(),xCoords.end());
      //std::sort(yCoords.begin(),yCoords.end());

      // take the median to be the FOE
      //currFOE = Vector2D(xCoords[int(xCoords.size()/2)],
      //           yCoords[int(yCoords.size()/2)]);

      // compute the mean of x and y
      float sumX = 0.0F, sumY = 0.0F;
      for (uint i = 0; i < xCoords.size(); ++i)
        { sumX += xCoords[i]; sumY += yCoords[i]; }
      currFOE = Vector2D(sumX/float(xCoords.size()),
                         sumY/float(yCoords.size()));

      //if (sumFOE.isValid()) sumFOE += currFOE;
      //else sumFOE = currFOE;
      //++numFOE;
    }

  if (xFOE.isInitialized())
    {
      // update KalmanFilters with currFOE
      if (currFOE.isValid())
        itsFOE.push_back(Vector2D(xFOE.update(currFOE.x()),
                                  yFOE.update(currFOE.y())));
      else
        itsFOE.push_back(Vector2D(xFOE.update(), yFOE.update()));
    }
  else
    {
      if (currFOE.isValid())
        {
          // initialize KalmanFilters
          xFOE.init(currFOE.x(), 0.0001F, 10000.0F);
          yFOE.init(currFOE.y(), 0.0001F, 10000.0F);
        }

      // store currFOE
      itsFOE.push_back(currFOE);
    }

  //if (sumFOE.isValid()) itsFOE.push_back(sumFOE/numFOE);
  //else itsFOE.push_back(Vector2D());

  return itsFOE.back();
}
*/

// ######################################################################
Vector2D VisualEventSet::getFOE(int frameNum) const
{

  int idx = frameNum - startframe;
  //LINFO("frameNum = %i, startframe = %i, idx = %i, itsFOE.size() = %i",
  //    frameNum, startframe, idx, itsFOE.size());
  ASSERT((idx >= 0)&& (idx < (int)itsFOE.size()));

  return itsFOE[idx];
}

// ######################################################################
uint VisualEventSet::numEvents() const
{
  return itsEvents.size();
}

// ######################################################################
void VisualEventSet::reset()
{
  itsEvents.clear();
}

// ######################################################################
void VisualEventSet::cleanUp(uint currFrame, uint maxFrameSkip)
{
  std::list<VisualEvent>::iterator currEvent = itsEvents.begin();
  while (currEvent != itsEvents.end())
    {
      if (currEvent->isClosed() &&
          (currEvent->getNumberOfFrames() < itsMinFrameNum))
        {
          LINFO("Erasing event %i, because it has only %i frames.",
                currEvent->getEventNum(), currEvent->getNumberOfFrames());
          currEvent = itsEvents.erase(currEvent);
        }
      else ++currEvent;

    } // end while loop over events
}

// ######################################################################
void VisualEventSet::closeAll()
{
  std::list<VisualEvent>::iterator cEvent;
  for (cEvent = itsEvents.begin(); cEvent != itsEvents.end(); ++cEvent)
    cEvent->close();
}

// ######################################################################
std::vector<Token> VisualEventSet::getTokens(uint frameNum)
{
  std::vector<Token> tokens;
  std::list<VisualEvent>::iterator currEvent;
  for (currEvent = itsEvents.begin(); currEvent != itsEvents.end(); ++currEvent)
    {
      // does this guy participate in frameNum?
      if (!currEvent->isFrameOk(frameNum)) continue;

      tokens.push_back(currEvent->getToken(frameNum));
    } // end loop over events

  return tokens;
}

// ######################################################################
void VisualEventSet::drawTokens(Image< PixRGB<byte> >& img,
                                uint frameNum,
                                PropertyVectorSet& pvs,
                                int circleRadius,
                                BitObjectDrawMode mode,
                                float opacity,
                                PixRGB<byte> colorInteresting,
                                PixRGB<byte> colorCandidate,
                                PixRGB<byte> colorPred,
                                PixRGB<byte> colorFOE,
                                bool showEventLabels)
{
  // dimensions of the number text and location to put it at
  const int numW = 10;
  const int numH = 21;

  std::list<VisualEvent>::iterator currEvent;
  for (currEvent = itsEvents.begin(); currEvent != itsEvents.end(); ++currEvent)
    {
      // does this guy participate in frameNum?
      if (!currEvent->isFrameOk(frameNum)) continue;

      PixRGB<byte> circleColor;
      Token tk = currEvent->getToken(frameNum);
      Point2D<int> center = tk.location.getPoint2D();

      if (isEventInteresting(pvs.getPropertyVectorForEvent
                             (currEvent->getEventNum())))
        circleColor = colorInteresting;
      else
        circleColor = colorCandidate;

      // if requested, prepare the event labels
      Image< PixRGB<byte> > textImg;
      if (showEventLabels)
        {
          // write the text and create the overlay image
          std::string numText = toStr(currEvent->getEventNum());
          textImg.resize(numW * numText.length(), numH, NO_INIT);
          textImg.clear(COL_WHITE);
          writeText(textImg, Point2D<int>(0,0), numText.c_str());
        }

      // draw the event object itself if requested
      if (circleColor != COL_TRANSPARENT)
        {
          // the box so that the text knows where to go
          Rectangle bbox;

          // draw rectangle or circle and determine the pos of the number label
          if (tk.bitObject.isValid())
            {
              tk.bitObject.draw(mode, img, circleColor, opacity);
              bbox = tk.bitObject.getBoundingBox(BitObject::IMAGE);
              //drawRect(img, bbox,circleColor);
            }
          else
            {
              LINFO("BitObject is invalid: area: %i;",tk.bitObject.getArea());
              LFATAL("bounding box: %s",toStr(tk.bitObject.getBoundingBox()).c_str());
              drawCircle(img, center, circleRadius, circleColor);
              bbox = Rectangle::tlbrI(center.j - circleRadius, center.i - circleRadius,
                               center.j + circleRadius, center.i + circleRadius);
              bbox = bbox.getOverlap(img.getBounds());
            }

          // if requested, write the event labels into the image
          if (showEventLabels)
            {
              Point2D<int> numLoc = getLabelPosition(img.getDims(),bbox,textImg.getDims());
              Image<PixRGB <byte> > textImg2 = replaceVals(textImg,COL_BLACK,circleColor);
              textImg2 = replaceVals(textImg2,COL_WHITE,COL_TRANSPARENT);
              pasteImage(img,textImg2,COL_TRANSPARENT, numLoc, opacity);

            } // end if (showEventLabels)

        } // end if we're not transparent

      // now do the same for the predicted value
      if ((colorPred != COL_TRANSPARENT) && tk.prediction.isValid())
        {
          Point2D<int> ctr = tk.prediction.getPoint2D();
          drawCircle(img, ctr,10,colorPred);
          Rectangle ebox =
            Rectangle::tlbrI(ctr.j - 10, ctr.i - 10, ctr.j + 10, ctr.i + 10);
          ebox = ebox.getOverlap(img.getBounds());
          if (showEventLabels)
            {
              Point2D<int> numLoc = getLabelPosition(img.getDims(), ebox, textImg.getDims());
              Image< PixRGB<byte> > textImg2 = replaceVals(textImg,COL_BLACK,colorPred);
              textImg2 = replaceVals(textImg2,COL_WHITE,COL_TRANSPARENT);
              pasteImage(img,textImg2,COL_TRANSPARENT, numLoc, opacity);
            }
        }

    } // end loop over events

  // draw the focus of expansion
  if ((colorFOE != COL_TRANSPARENT) && getFOE(frameNum).isValid())
    {
      Point2D<int> ctr = getFOE(frameNum).getPoint2D();
      drawDisk(img, ctr,2,colorFOE);
    }

}

// ######################################################################
Point2D<int> VisualEventSet::getLabelPosition(Dims imgDims,
                                         Rectangle bbox,
                                         Dims textDims) const
{
  // distance of the text label from the bbox
  const int dist = 2;

  Point2D<int> loc(bbox.left(),(bbox.top() - dist - textDims.h()));

  // not enough space to the right? -> shift as apropriate
  if ((loc.i + textDims.w()) > imgDims.w())
    loc.i = imgDims.w() - textDims.w() - 1;

  // not enough space on the top? -> move to the bottom
  if (loc.j < 0)
    loc.j = bbox.bottomI() + dist;

  return loc;
}

// ######################################################################
PropertyVectorSet VisualEventSet::getPropertyVectorSet()
{
  PropertyVectorSet pvs;

  std::list<VisualEvent>::iterator currEvent;
  for (currEvent = itsEvents.begin(); currEvent != itsEvents.end();
       ++currEvent)
    pvs.itsVectors.push_back(currEvent->getPropertyVector());

  return pvs;
}


// ######################################################################
int VisualEventSet::getAllClosedFrameNum(uint currFrame)
{
  std::list<VisualEvent>::iterator currEvent;
  for (int frame = (int)currFrame; frame >= -1; --frame)
    {
      bool done = true;

      for (currEvent = itsEvents.begin(); currEvent != itsEvents.end();
           ++currEvent)
        {
          done &= ((frame < (int)currEvent->getStartFrame())
                   || currEvent->isClosed());
          if (!done) break;
        }

      if (done) return frame;
    }
  return -1;
}

// ######################################################################
bool VisualEventSet::isEventInteresting(std::vector<float> propVec) const
{
  const float interestThresh = 5;
  const float uxyThresh = 0.4F;

  // did we get set from outside? -> threshold with interstingness
  if (propVec[1] >= 0.0F)
    return (propVec[1] >= interestThresh);
  // otherwise threshold uxy and set interestingness value
  else
    return (fabs(propVec[5]) >= uxyThresh);
}

// ######################################################################
bool VisualEventSet::doesEventExist(uint eventNum) const
{
  std::list<VisualEvent>::const_iterator evt;
  for (evt = itsEvents.begin(); evt != itsEvents.end(); ++evt)
    if (evt->getEventNum() == eventNum) return true;

  return false;
}

// ######################################################################
VisualEvent VisualEventSet::getEventByNumber(uint eventNum) const
{
  std::list<VisualEvent>::const_iterator evt;
  for (evt = itsEvents.begin(); evt != itsEvents.end(); ++evt)
    if (evt->getEventNum() == eventNum) return *evt;

  LFATAL("Event with number %i does not exist.",eventNum);

  return *evt;
}

// ######################################################################
std::vector<std::list<VisualEvent>::iterator>
VisualEventSet::getEventsForFrame(uint framenum)
{
  std::vector<std::list<VisualEvent>::iterator> result;
  std::list<VisualEvent>::iterator event;
  for (event = itsEvents.begin(); event != itsEvents.end(); ++event)
    if (event->isFrameOk(framenum)) result.push_back(event);

  return result;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
