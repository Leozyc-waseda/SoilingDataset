/*!@file Beobot/TopologicalMap.H topological map for localization */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/TopologicalMap.H $
// $Id: TopologicalMap.H 15472 2013-04-26 18:58:29Z kai $
//

// ######################################################################
/*! Topological Map
  actually it's metric augmented topological map
  there are coordinates for the nodes in the map
  DIRECTED edges/segments connect the nodes

  Also generate path: vector of segment numbers
*/

#ifndef BEOBOT_TOPOLOGICALMAP_H_DEFINED
#define BEOBOT_TOPOLOGICALMAP_H_DEFINED

#include "Beobot/Map.H"
#include "Beobot/Graph.H"
#include <vector>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/optional.hpp> 
#include <boost/foreach.hpp>
#include <cassert>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#define   TOPOMAP_MOVE_FORWARD     -1
#define   TOPOMAP_TURN_AROUND      -2

//! topological map using a list of landmarks
class TopologicalMap : public Map
{
public:

  // ######################################################################
  /*! @name Constructors, assignment, and Destructors */
  //@{

  //! Constructor: generate a blank map
  TopologicalMap();

  //! Constructor: retrieve the map from a file
  TopologicalMap(std::string fileName);

  //! Destructor
  ~TopologicalMap();

  //! read a map from a file
  bool read(std::string fileName);

  //! read a map from a file
  bool readJSON(std::string fileName);

  //! write a map to a file
  bool write(std::string fileName);

  //@}

  // ######################################################################
  //! @name Access functions
  //@{

  //! get segment size
  inline uint getSegmentNum();

  //! get the map dimension
  inline Dims getMapDims();

  //! get the map scale in (ft/unit map)
  inline double getMapScale();

  //! get node - segment distance matrix
  inline Image<double> getNodeSegmentDistanceMatrix();

  //! get node or edge of the passed in index
  inline rutz::shared_ptr<Node> getNode(uint index);
  inline rutz::shared_ptr<Edge> getEdge(uint index);

  //! get node and edge size
  inline uint getNumNode();
  inline uint getNumEdge();

  //! get maximum node-to-node distance
  inline double getMaxDistance();

  //! get maximum node-to-segment distance
  double getNodeSegmentMaxDistance();

  //@}

  // ######################################################################
  /*! @name member functions */
  //@{

  //! get the distance from the point to the segment
  //! also return the intersecting node
  double getSegmentDistance(uint sseg, double ltrav, uint dseg,
                           uint &intIndex);

  //! get the distance between two points
  double getDistance(uint asnum, double altrav, uint bsnum, double bltrav);

  //! return a path from points 'a' to 'b'
  double getPath
  (uint asnum, double altrav, uint bsnum, double bltrav,
   std::vector<int> &moves);

  //! get the length of the segment
  double getSegmentLength(uint index);

  //! get Point2D<int> location of the (segnum, ltrav) descriptor
  Point2D<int> getLocation(uint cseg, double ltrav);

  //! get Point2D<double> location of the (segnum, ltrav) descriptor
  Point2D<float> getLocationFloat(uint cseg, double ltrav);

  //! get the edge in which the point (snum, ltrav) is at
  rutz::shared_ptr<Edge> getEdge(uint cseg, double ltrav,
                                 double &sdist,
                                 double &edist);

  //! get the edge in which the point (snum, ltrav) is at
  rutz::shared_ptr<Edge> getEdge(uint cseg, double ltrav,
                                 double &sdist,
                                 double &edist,
                                 uint& eindex);

  //! get the edge in which the point is at
  rutz::shared_ptr<Edge> getEdge(Point2D<int> loc,
                                 double &sdist,
                                 double &edist);

  //! get start location of an edge given edge index and offset length on edge
  //! return matching segment number and length of traveld
  void getEdgeStartLocation(uint edge_index, uint &cseg, double &ltrav, double edge_dist = 0.0F);

  //! get the angle (in radians) between two edges
  //! the two edges do not have to have a common node
  //! angles are intuitive from the robot's perspective
  //! 0 degrees/radians means going straight,
  //! 180/-180 degrees or M_PI/-M_PI means turn around
  //! positive angle to the right, negative angle to the left
  double getAngle(rutz::shared_ptr<Edge> e1, rutz::shared_ptr<Edge> e2);

  //! convert Point2D<int> location the segment number and length traveled
  void getLocation(Point2D<int>, uint &cseg, double &ltrav);

  //! convert Point2D<int> location to segment number
  uint getSegmentLocation(Point2D<int> loc);

  //! convert Point2D<int> location to segment length traveled
  double getSegmentLengthTraveled(Point2D<int> loc);

  //! get the list of nodes in between the inputted interval
  //! two different versions of the results
  std::vector<rutz::shared_ptr<Node> >
  getNodesInInterval(uint index, double fltrav, double lltrav);
  std::vector<std::pair<uint,float> >
  getNodeLocationsInInterval(uint index, double fltrav, double lltrav);

  //! returns an image representation of the map
  //! the w and h is the ideal size while keeping aspect ratio
  //! so most likely the image returned will be smaller in size
  Image<PixRGB<byte> > getMapImage(Image<PixRGB<byte> > backgroudImg,
    double scale = 1.0,Point2D<int> offset = Point2D<int>(0,0));
  Image<PixRGB<byte> > getMapImage(uint w, uint h);

  // Find nearest node label
  std::string getNodeLabel (Point2D<int> loc);



  //! ADD: modify the map

  //@}

private:

  //! compute all the shortcuts for shortest-distance related operations
  //! node-to-segment, and segment-to-segment
  void computeDistances();

  //! set all the segment lengths
  void setSegmentLength();

  rutz::shared_ptr<Graph> itsGraph;

  //! the map size and scale in ft/unit
  int itsMapWidth;
  int itsMapHeight;
  double itsMapScale;

  //! the list of segments in the map
  //! each segment is an ordered list of directed edges
  std::vector< std::vector<rutz::shared_ptr<Edge> > > itsSegmentList;
  std::vector< std::vector<uint> > itsSegmentEdgeIndexList;

  //! the segment numbers for each edge
  std::vector<uint> itsEdgeSegmentNum;

  //! store the segment lengths
  std::vector<double> itsSegmentLength;

  //! the segment to segment distance matrix
  //! note that this is to the closest point in the segment
  Image<double> itsNodeSegmentDistanceMatrix;

  //! segment labels for each edge
  //std::vector<int> segmentLabels;
  //std::vector<std::string> landmarkNames;
};

// ######################################################################
// Implementation for Topological Map inline functions
// ######################################################################
inline uint TopologicalMap::getSegmentNum()
{
  return itsSegmentList.size();
}

inline Dims TopologicalMap::getMapDims()
{
  return Dims(itsMapWidth, itsMapHeight);
}

inline double TopologicalMap::getMapScale()
{
  return itsMapScale;
}
inline Image<double> TopologicalMap::getNodeSegmentDistanceMatrix()
{
  return itsNodeSegmentDistanceMatrix;
}

inline double TopologicalMap::getMaxDistance()
{
  return itsGraph->getMaxUndirectedDistance();
}

inline uint TopologicalMap::getNumNode()
{
  return itsGraph->getNumNode();
}

inline uint TopologicalMap::getNumEdge()
{
  return itsGraph->getNumEdge();
}

inline rutz::shared_ptr<Node> TopologicalMap::getNode(uint index)
{
  ASSERT(index < getNumNode());
  return itsGraph->getNode(index);
}

inline rutz::shared_ptr<Edge> TopologicalMap::getEdge(uint index)
{
  ASSERT(index < getNumEdge());
  return itsGraph->getEdge(index);
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
