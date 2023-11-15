/*!@file Beobot/Graph.H basic graph class can be both directed
  and undirected */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/Graph.H $
// $Id $
//

#ifndef BEOBOT_GRAPH_H_DEFINED
#define BEOBOT_GRAPH_H_DEFINED

#include "Beobot/Node.H"
#include "Beobot/Edge.H"
#include "Image/Image.H"

//! basic graph class
//! has all the access functions and shortest path
//! no input/output file function because usually the file
//! has to be filled with extra information
class Graph
{
public:
  // ######################################################################
  //! @name Constructor, assigment and destructor
  //@{

  //! Constructor: generate a blank graph
  Graph();

  //! Constructor: generate a graph with edges
  Graph(std::vector<rutz::shared_ptr<Node> > nodes,
        std::vector<rutz::shared_ptr<Edge> > edges);

  //! Destructor
  ~Graph();

  //! add edges and nodes
  void addNode(rutz::shared_ptr<Node>);
  void addEdge(rutz::shared_ptr<Edge>);

  //@}

  // ######################################################################
  //! @name Access functions
  //@{

  //! get node and edge size
  inline uint getNumNode();
  inline uint getNumEdge();

  //! get node or edge of the passed in index
  inline rutz::shared_ptr<Node> getNode(uint index);
  inline rutz::shared_ptr<Edge> getEdge(uint index);

  //! get edge index connecting the two nodes
  inline int getUndirectedAdjecencyEdgeIndex(uint sindex, uint dindex);
  inline int getDirectedAdjecencyEdgeIndex  (uint sindex, uint dindex);

  //@}

  // ######################################################################
  //! @name Compute functions
  //@{

  //! compute the adjecency list
  void computeAdjecencyList();

  //! compute just the cost list
  //! assume the adjecency list already set
  //! usually done after changing some weights
  void computeAdjecencyCostList();

  //! compute all the shortcuts for shortest-distance related operations
  void computeDistances();

  //! get the shortest distance from node A to node B
  //! using Dijkstra shortest-path algorithm
  //! for the directed graph
  float getDirectedDistance
  (uint a, uint b, std::vector<uint> &edges);
  float getDirectedDistance(uint a, uint b);

  //! get the shortest distance from node A to node B
  //! using Dijkstra shortest-path algorithm
  //! for the undirected graph
  float getUndirectedDistance
  (uint a, uint b, std::vector<uint> &edges);
  float getUndirectedDistance(uint a, uint b);

  //! get the shortest distance from node A to node B
  //! using A* shortest-path algorithm
  //! for the directed graph
  float getDirectedDistance
  (uint a, uint b, std::vector<float> h);
  float getDirectedDistance
  (uint a, uint b, std::vector<float> h, std::vector<uint> &edges);

  //! get the maximum distance of any 2 points in the map
  //! return -1.0F if the graph is unconnected
  //! for the directed graph
  float getMaxDirectedDistance();

  //! get the maximum distance of any 2 points in the map
  //! return -1.0F if the graph is unconnected
  //! for undirected graph
  float getMaxUndirectedDistance();

  //! get the directed path from node A to Note B
  std::vector<uint> getDirectedPath(uint a, uint b);

  //! get the undirected path from node A to Note B
  std::vector<uint> getUndirectedPath(uint a, uint b);

  //! get all the edge that connects to this node
  std::vector<uint> getEdges(rutz::shared_ptr<Node> n);

  //! get the angle (in radians) between two edges
  //! they two edges do not have to have a common node
  //! angles are intuitive from the tip of
  //! the incoming (e1) edge perspective
  //! 0 degrees/radians means continue the same direction
  //! 180/-180 degrees or M_PI/-M_PI means turn around
  //! positive angle to the right, negative angle to the left
  float getAngle(rutz::shared_ptr<Edge> e1, rutz::shared_ptr<Edge> e2);

  //@}

private:

  //! all the nodes and edge information
  std::vector<rutz::shared_ptr<Node> > itsNodes;
  std::vector<rutz::shared_ptr<Edge> > itsEdges;

  std::vector<std::vector<uint> >  itsDirectedAdjecencyList;
  std::vector<std::vector<float> > itsDirectedAdjecencyCostList;
  std::vector<std::vector<uint> >  itsDirectedAdjecencyInEdgeList;
  std::vector<std::vector<uint> >  itsDirectedAdjecencyOutEdgeList;
  Image<int>                       itsDirectedAdjecencyEdgeIndexList;

  std::vector<std::vector<uint> >  itsUndirectedAdjecencyList;
  std::vector<std::vector<float> > itsUndirectedAdjecencyCostList;
  std::vector<std::vector<uint> >  itsUndirectedAdjecencyEdgeList;
  Image<int>                       itsUndirectedAdjecencyEdgeIndexList;

  //! all the distances (pre-computed if 'computeDistances' is called)
  Image<float> itsDirectedDistanceMatrix;
  Image<float> itsUndirectedDistanceMatrix;

  std::vector<std::vector<std::vector<uint> > > itsDirectedPathList;
  std::vector<std::vector<std::vector<uint> > > itsUndirectedPathList;

  bool itsDistanceMatrixComputed;
};

// ######################################################################
// Implementation for Node inline functions
// ######################################################################

inline uint Graph::getNumNode()
{
  return itsNodes.size();
}

inline uint Graph::getNumEdge()
{
  return itsEdges.size();
}

inline rutz::shared_ptr<Node> Graph::getNode(uint index)
{
  ASSERT(index < itsNodes.size());
  return itsNodes[index];
}

inline rutz::shared_ptr<Edge> Graph::getEdge(uint index)
{
  ASSERT(index < itsEdges.size());
  return itsEdges[index];
}

inline int Graph::getUndirectedAdjecencyEdgeIndex(uint sindex, uint dindex)
{
  ASSERT(sindex < itsEdges.size());
  ASSERT(dindex < itsEdges.size());

  return itsUndirectedAdjecencyEdgeIndexList.getVal(sindex, dindex); 
}

inline int Graph::getDirectedAdjecencyEdgeIndex(uint sindex, uint dindex)
{
  ASSERT(sindex < itsEdges.size());
  ASSERT(dindex < itsEdges.size());

  return itsDirectedAdjecencyEdgeIndexList.getVal(sindex, dindex); 
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
