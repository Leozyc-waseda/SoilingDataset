/*!@file Beobot/Edge.H basic edge in a graph class */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/Edge.H $
// $Id $
//

#ifndef BEOBOT_EDGE_H_DEFINED
#define BEOBOT_EDGE_H_DEFINED

#include "Beobot/Node.H"
#include "rutz/shared_ptr.h"
#include "Image/Pixels.H"
#include <string>
#include <vector>

class Node;
class Edge;

//! basic edge in a graph class
//! the label is for display only, not necessary
//! note its directionality is decided by the graph
//! despite the label source and destination node
class Edge
{
public:
  // ######################################################################
  //! @name Constructor, assigment and destructor
  //@{

  //! Constructor: generate a blank edge (no end-nodes)
  Edge(float cost = 1.0F,
       const std::string label = std::string(""));

  //! Constructor: generate a directed edge with two nodes
  Edge(rutz::shared_ptr<Node> sourceNode, rutz::shared_ptr<Node> destNode,
       float cost = 1.0F,
       const std::string label = std::string(""));

  //! Destructor
  ~Edge();

  //@}

  // ######################################################################
  /*! @name member functions */
  //@{

  //! cost of traversing the edge
  //! default: 1.0
  inline float getCost();

  //! set the cost of the edge
  inline void setCost(float cost);

  //! get the source node
  inline rutz::shared_ptr<Node> getSourceNode();

  //! get the destination node
  inline rutz::shared_ptr<Node> getDestNode();

  //! set the source node
  inline void setSourceNode(rutz::shared_ptr<Node> sourceNode);

  //! set the destination node
  inline void setDestNode(rutz::shared_ptr<Node> destNode);

  //! get the label of the edge
  inline std::string getLabel();


  //! get display edge color
  inline PixRGB<byte> getDisplayColor();

  //! set display edge color
  inline void setDisplayColor(PixRGB<byte> col);

  //@}

private:

  rutz::shared_ptr<Node> itsSourceNode;
  rutz::shared_ptr<Node> itsDestNode;
  float itsCost;
  std::string itsLabel;
  PixRGB<byte> itsDisplayColor;
};

// ######################################################################
// Implementation for Node inline functions
// ######################################################################
inline float Edge::getCost()
{
  return itsCost;
}

inline void Edge::setCost(float cost)
{
  itsCost = cost;
}

inline rutz::shared_ptr<Node> Edge::getSourceNode()
{
  return itsSourceNode;
}

inline rutz::shared_ptr<Node> Edge::getDestNode()
{
  return itsDestNode;
}

inline void Edge::setSourceNode(rutz::shared_ptr<Node> sourceNode)
{
  itsSourceNode = sourceNode;
}

inline void Edge::setDestNode(rutz::shared_ptr<Node> destNode)
{
  itsDestNode = destNode;
}

inline std::string Edge::getLabel()
{
  return itsLabel;
}

inline PixRGB<byte> Edge::getDisplayColor()
{
  return itsDisplayColor;
}

inline void  Edge::setDisplayColor(PixRGB<byte> col)
{
  itsDisplayColor = col;
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
