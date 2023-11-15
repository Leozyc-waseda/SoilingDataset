/*!@file Beobot/Node.H basic node in a graph class */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/Node.H $
// $Id $
//

#ifndef BEOBOT_NODE_H_DEFINED
#define BEOBOT_NODE_H_DEFINED

#include "Image/Point2D.H"
//#include "Beobot/Edge.H"
#include "rutz/shared_ptr.h"
#include <string>
#include <vector>

class Node;
class Edge;

//! basic node in a graph class
//! the Point2D<int> location is only for display purposes, it is not necessary
//! to enter it (unless we want to draw the graph image)
class Node
{
public:
  // ######################################################################
  //! @name Constructor, assigment and destructor
  //@{

  //! Constructor: generate a node with coordinates and label
  Node(uint index,
       const Point2D<int> point = Point2D<int>(-1,-1),
       const std::string label = std::string(""));

  //! Destructor
  ~Node();

  //! inline get the index
  inline uint getIndex();

  //! inline get the index
  inline void setIndex(int index);

  //! get the Point2D<int> location for printing/display
  inline Point2D<int> getCoordinate();

  //! set the Point2D<int> location for printing/display
  inline void setCoordinate(Point2D<int> coor);

  //! get the label for printing/display
  inline std::string getLabel();

  //! set the label for printing/display
  inline void setLabel(std::string label);

//   //! set edge in
//   inline void setEdgeIn(std::vector<rutz::shared_ptr<Edge> > edgeIn);

//   //! set edge out
//   inline void setEdgeOut(std::vector<rutz::shared_ptr<Edge> > edgeOut);

  //@}

private:

  //! its index - the one thing that matter
  uint itsIndex;

  //! points and label for display
  Point2D<int> itsCoor;
  std::string itsLabel;

  //  std::vector<rutz::shared_ptr<Edge> > itsEdgeIn;
  //std::vector<rutz::shared_ptr<Edge> > itsEdgeOut;
};

// ######################################################################
// Implementation for Node inline functions
// ######################################################################
inline uint Node::getIndex()
{
  return itsIndex;
}

inline void Node::setIndex(int index)
{
  itsIndex = index;
}

inline Point2D<int> Node::getCoordinate()
{
  return itsCoor;
}

inline void Node::setCoordinate(Point2D<int> coor)
{
  itsCoor = coor;
}

inline std::string Node::getLabel()
{
  return itsLabel;
}

inline void Node::setLabel(std::string label)
{
  itsLabel = label;
}

// inline void Node::setEdgeIn(std::vector<rutz::shared_ptr<Edge> > edgeIn)
// {
//   itsEdgeIn.clear();
//   for(uint i = 0; i < edgeIn.size(); i++)
//     itsEdgeIn.push_back(edgeIn[i]);
// }

// inline void Node::setEdgeOut(std::vector<rutz::shared_ptr<Edge> > edgeOut)
// {
//   itsEdgeOut.clear();
//   for(uint i = 0; i < edgeOut.size(); i++)
//     itsEdgeOut.push_back(edgeOut[i]);
// }
#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
