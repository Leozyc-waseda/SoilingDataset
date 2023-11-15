/*!@file Beobot/GridMap.H grid map for localization */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/GridMap.H $
// $Id: GridMap.H 15320 2012-06-22 22:51:37Z kai $
//

// ######################################################################
/*! Grid Map                                                            */

#ifndef BEOBOT_GRIDMAP_H_DEFINED
#define BEOBOT_GRIDMAP_H_DEFINED

#include "Beobot/Map.H"

//! topological map using a list of landmarks
class GridMap : public Map
{
public:
  //! Constructor: generate a blank map
  GridMap();

  //! Constructor: retrieve the map from a file
  GridMap(std::string fileName);

  //! Constructor: create a map from dimensions
  GridMap(uint w, uint h);

  //! Destructor
  ~GridMap();

  //! read a map from a file
  bool read(std::string fileName);

  //! write a map to a file
  bool write(std::string fileName);

  //! returns an image representation of the map
  Image<PixRGB<byte> > getMapImage(uint w, uint h);

  //! update the weights of the graph
  void updateGridMap(Image<float> map);

  //! get the shortest path from 'a' to 'b'
  //! by specifying the intermediate steps 
  float getShortestPath
  (Point2D<int> a, Point2D<int> b, std::vector<Point2D<int> > &steps);
  
  //! get edgeweights for a point
  std::vector<float> getDirectedEdgeWeights(Point2D<int> pt);

  //! set edge weights for a point
  void setDirectedEdgeWeights(Point2D<int> pt, std::vector<float> weights);

private:  

  //! setup appropriate storage for the specified sized grid map
  void setupGridMap(uint w, uint h);

  //! directed edge weights for each coordinate location
  //! to get shortest path
  Image<std::vector<float> > itsDirectedEdgeWeights;

  //! store the distances between neighboring coordinates
  std::vector<std::pair<Point2D<int>, float> > itsNeighbors;

};
#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
