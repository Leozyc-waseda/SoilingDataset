/*!@file MBARI/VisualEvent.H classes useful for event tracking */

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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/VisualEvent.H $
// $Id: VisualEvent.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef VISUALEVENT_H_DEFINED
#define VISUALEVENT_H_DEFINED

#include "Image/KalmanFilter.H"
#include "Image/Point2D.H"
#include "Image/Rectangle.H"
#include "MBARI/BitObject.H"
#include "Util/Assert.H"
#include "Util/Types.H"

#include <list>
#include <string>
#include <vector>

template <class T> class Image;
template <class T> class PixRGB;
class Vector2D;
class StraighLine2D;

// ######################################################################
//! public class that contains information for a visual token for tracking
class Token
{
public:

  //!default constructor
  Token();

  //!constructor that fills the members with initial values
  Token (float x, float y, uint frame, BitObject bo);

  //!constructor with the location being the centroid of the BitObject
  Token (BitObject bo, uint frame);

  //!read the Token from the input stream is
  Token (std::istream& is);

  //!the BitObject that holds info about the shape of this Token
  BitObject bitObject;

  //!the x and y coordinates of the token in the frame
  Vector2D location;

  //!the expected location for this token
  Vector2D prediction;

  //!the straight line on which this token is moving
  StraightLine2D line;

  //!the angle between line and the connection to the FOE
  float angle;

  //!the frame number at which the token occurred
  uint frame_nr;

  //! write the entire Token to the output stream os
  void writeToStream(std::ostream& os) const;

  //! read the Token from the input stream is
  void readFromStream(std::istream& is);

  //! write the Token's position to the output streeam os
  void writePosition(std::ostream& os) const;

};

// ######################################################################
//! class that contains a set of property vectors for events
/*! each property vector contains the following fields:<br>
  <table><tr><td><b>index</b><td><b>content</b></th>
  <tr><td>0<td>event number
  <tr><td>1<td>interestingness index
  <tr><td>2<td>area
  <tr><td>3<td>Uxx
  <tr><td>4<td>Uyy
  <tr><td>5<td>Uxy
  <tr><td>6<td>major Axis
  <tr><td>7<td>minor Axis
  <tr><td>8<td>elongation
  <tr><td>9<td>orientation angle
  <tr><td>10<td>maximum intensity
  <tr><td>11<td>minimum intensity
  <tr><td>12<td>average intensity</table>*/
class PropertyVectorSet
{
public:

  //! default contructor
  PropertyVectorSet();

  //! read from stream constructor
  PropertyVectorSet(std::istream& is);

  //! write the entire PropertyVectorSet to the output stream os
  void writeToStream(std::ostream& os) const;

  //! read the PropertyVectorSet from the input stream is
  void readFromStream(std::istream& is);

  //! return the PropertyVector for an event by its number
  std::vector<float> getPropertyVectorForEvent(const int num);

  //! the vector of property vectors
  std::vector< std::vector<float> > itsVectors;
};


// ######################################################################
//!stores a visual event as a sequence of tokens with additional information
class VisualEvent
{
public:

  //! constructor
  /*!@param tk the first token for this event*/
  VisualEvent(Token tk, int maxDist);

  //! read the VisualEvent from the input stream is
  VisualEvent(std::istream& is);

  //! write the entire VisualEvent to the output stream os
  void writeToStream(std::ostream& os) const;

  //! read the VisualEvent from the input stream is
  void readFromStream(std::istream& is);

  //! write all the positions for this event to the output stream os
  void writePositions(std::ostream& os) const;

  //! get the prediction for the location of the next token
  Point2D<int> predictedLocation() const;

  //! is this token allowed as the next one?
  bool isTokenOk(const Token& tk) const;

  //! returns the cost of associating tk with this event
  /*!@return returns -1.0F if the token is not valid for this event*/
  float getCost(const Token& tk) const;

  //! assign tk to this event, use foe as the focus of expansion
  void assign(const Token& tk, const Vector2D& foe);

  //! if the BitObject intersects with the one for this event at frameNum
  bool doesIntersect(const BitObject& obj, int frameNum) const;

  //! mark this event as "closed"
  inline void close();

  //! whether this event is closed
  inline bool isClosed();

  //! return the event identification number of this event
  inline uint getEventNum() const;

  //! return the frame number of the first token
  inline uint getStartFrame() const;

  //! return the frame number of the last token
  inline uint getEndFrame() const;

  //! return the number of frames that this event spans
  inline uint getNumberOfFrames() const;

  //! return the maximum object size of any of the tokens
  inline int getMaxSize() const;

  //! return the token that has the maximum object size
  inline Token getMaxSizeToken() const;

  //!return a token based on a frame number
  inline Token getToken(const uint frame_num) const;

  //!return whether frame_num is between start frame and end frame
  inline bool isFrameOk(const uint frame_num) const;

  //! return a vector that contains a bunch of numbers that describe the event
  /*! see source code for details */
  std::vector<float>  getPropertyVector();

  //! returns the maximum dimensions of the tracked object in any of the frames
  Dims getMaxObjectDims() const;

private:
  static uint counter;
  uint myNum;
  std::vector<Token> tokens;
  uint startframe, endframe;
  int max_size;
  uint maxsize_framenr;
  bool closed;
  int itsMaxDist;
  KalmanFilter xTracker, yTracker;
};

// ######################################################################
// ######## VisualEventSet
// ######################################################################
//! contains a set of visual events to which a set of tokens can be matched
class VisualEventSet
{
public:

  //! constructor
  /*!@param maxDist the maximum distance that a token can have from an
    event and still be considered for becoming part of the event
    @param minFrameNum the minimum number of frames for which an event needs
    be present
    @param minSize the minimum size (in pixels) for an "interesting" event
    @param fileName the fileName for saving intermediate result*/
  VisualEventSet(const int maxDist,
                 const uint minFrameNum,
                 const int minSize,
                 const std::string& filename);

  //! read the VisualEventSet from the input stream is
  VisualEventSet(std::istream& is);

  //! write the entire VisualEventSet to the output stream os
  void writeToStream(std::ostream& os) const;

  //! read the VisualEventSet from the input stream is
  void readFromStream(std::istream& is);

  //! write th positions of all events to the output stream os
  void writePositions(std::ostream& os) const;

  //! extract BitObjects from binMap and assign them to the events
  /*!@param curFOE the current focus of expansion for detecting unusual
    motion*/
  void updateEvents(const Image<byte>& binMap, const Vector2D& curFOE,
                    int frameNum);

  //! initiate new events for all BitObjects in bos if they aren't tracked yet
  void initiateEvents(std::list<BitObject>& bos, int frameNum);

  //! if obj intersects with any of the event at frameNum
  bool doesIntersect(const BitObject& obj, int frameNum) const;

  //! update the focus of expansion
  //Vector2D updateFOE();

  //! returns the current focus of expansion for frameNum
  Vector2D getFOE(int frameNum) const;

  //! return the number of stored events
  uint numEvents() const;

  //! delete all stored events
  void reset();

  //! clean up the event list - erase all unsuccessful candidates
  /*!@param currFrame - the current frame number in processing
    @param maxFrameSkip how many frames may be skipped using interpolation
    (default: 1)*/
  void cleanUp(uint currFrame, uint maxFrameSkip = 1);

  //! close all events (for clean-up at the end)
  void closeAll();

  //! returns a set of all tokens stored at a particular frame number
  std::vector<Token> getTokens(uint frameNum);

  //! draw the tokens for a particular frame number into an image
  /*!@param img the image to draw on
    @param frameNum the frame number for which to draw the tokens
    @param circleRadius the radius of the circle used to mark the tokens
    @param colorInteresting the color for "interesting" tokens
    @param colorCandidate the color for unsuccessful candidate tokens
    @param colorPred the colot for event location predictions
    @param showEventLabels whether to write the event labels into the image*/
  void drawTokens(Image< PixRGB<byte> >& img,
                  uint frameNum,
                  PropertyVectorSet& pvs,
                  int circleRadius,
                  BitObjectDrawMode mode,
                  float opacity,
                  PixRGB<byte> colorInteresting,
                  PixRGB<byte> colorCandidate,
                  PixRGB<byte> colorPred,
                  PixRGB<byte> colorFOE,
                  bool showEventLabels);

  //! returns a PropertyVectorSet for this VisualEventSet
  PropertyVectorSet getPropertyVectorSet();

  //! return the latest frame number before currFrame for which events have been closed already
  int getAllClosedFrameNum(uint currFrame);

  //! whether event is "interesting"
  bool isEventInteresting(std::vector<float> propVec) const;

  //! test whether the event with eventNum exists in this set
  bool doesEventExist(uint eventNum) const;

  //! Returns the event with number eventNum
  VisualEvent getEventByNumber(uint eventNum) const;

protected:
  //! Returns a vector with iterators pointing to all events at framenum
  std::vector<std::list<VisualEvent>::iterator> getEventsForFrame(uint framenum);

private:
  // compute the right position for a text label
  Point2D<int> getLabelPosition(Dims imgDims,Rectangle bbox,
                           Dims textDims) const;
  std::list<VisualEvent> itsEvents;
  int itsMaxDist;
  float itsMaxCost;
  uint itsMinFrameNum;
  int itsMinSize;
  int startframe, endframe;
  std::string itsFileName;
  std::vector<Vector2D> itsFOE;
  //Vector2D sumFOE;
  //int numFOE;
  KalmanFilter xFOE, yFOE;
};



// ######################################################################
// ########### INLINED METHODS
// ######################################################################
inline void VisualEvent::close()
{ closed = true; }

// ######################################################################
inline bool VisualEvent::isClosed()
{ return closed; }

// ######################################################################
inline uint VisualEvent::getEventNum() const
{ return myNum; }

// ######################################################################
inline uint VisualEvent::getStartFrame() const
{ return startframe; }

// ######################################################################
inline uint VisualEvent::getEndFrame() const
{ return endframe; }

// ######################################################################
inline uint VisualEvent::getNumberOfFrames() const
{ return (endframe - startframe + 1); }

// ######################################################################
inline int VisualEvent::getMaxSize() const
{ return max_size; }

// ######################################################################
inline Token VisualEvent::getMaxSizeToken() const
{ return getToken(maxsize_framenr); }

// ######################################################################
inline Token VisualEvent::getToken(uint frame_num) const
{
  ASSERT (isFrameOk(frame_num));
  return tokens[frame_num - startframe];
}

// ######################################################################
inline bool VisualEvent::isFrameOk(const uint frame_num) const
{
  return ((frame_num >= startframe) && (frame_num <= endframe));
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
