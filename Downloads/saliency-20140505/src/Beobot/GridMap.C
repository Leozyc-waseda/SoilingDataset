/*!@file Beobot/GridMap.C Grid map                                      */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/GridMap.C $
// $Id:GridMap.C 6891 2006-05-25 06:46:56Z siagian $
//

// ######################################################################
/*! Grid map for localization                                          */

#include "Beobot/GridMap.H"

#include "Util/Timer.H"
#include "Raster/Raster.H"

// ######################################################################
//! Constructor: generate a blank map
GridMap::GridMap()
  : Map()
{ }

// ######################################################################
//! Constructor: retrieve the map from a file
GridMap::GridMap(std::string fileName)
  : Map()
{
  read(fileName);
}

// ######################################################################
//! Constructor: create a map from dimensions
GridMap::GridMap(uint w, uint h)
  : Map()
{
  setupGridMap(w,h);
}

// ######################################################################
GridMap::~GridMap() { }

// ######################################################################
void GridMap::setupGridMap(uint w, uint h)
{
  // set the neighborhood of a coordinate location
  itsNeighbors.clear();
  for(int ii = -1; ii <= 1; ii++)
    for(int jj = -1; jj <= 1; jj++)
      {
        if(ii == 0 && jj == 0) continue;
        float dist = pow(ii*ii+jj*jj, 0.5);
        itsNeighbors.push_back
          (std::pair<Point2D<int>, float>(Point2D<int>(ii,jj), dist));
      }
  
  uint num_neighbors = 8;
 
  // setup the edge weight storage
  itsDirectedEdgeWeights = Image<std::vector<float> >(w,h,NO_INIT);
  Image<std::vector<float> >::iterator 
    aptr = itsDirectedEdgeWeights.beginw(), 
    stop = itsDirectedEdgeWeights.endw();
  while(aptr != stop)
    *aptr++ = std::vector<float>(num_neighbors);
}

// ######################################################################
void GridMap::updateGridMap(Image<float> map)
{
  uint w = map.getWidth();
  uint h = map.getHeight();

  // check that map is of same size
  //Dims a = itsDirectedEdgeWeights.getDims();
  //Dims b = map.getDims();
  if(itsDirectedEdgeWeights.getDims() != map.getDims()) 
    setupGridMap(w,h);

  Image<std::vector<float> >::iterator
    aptr = itsDirectedEdgeWeights.beginw();

  float max_distance = 2*(w+h);

  // update the edge-weights
  for(uint j = 0; j < h; j++)
    for(uint i = 0; i < w; i++)
      {
        // NOTE: to speed up, don't allocate anything
        Point2D<int> p(i,j);
        
        // go through the neighbors
        for(uint dij = 0; dij < itsNeighbors.size(); dij++)
          {
            Point2D<int> dp = itsNeighbors[dij].first;
            float  dist_weight  = itsNeighbors[dij].second;
            Point2D<int> p2 = p+dp;
            
            if(!map.coordsOk(p2)) 
              {
                (*aptr)[dij] = max_distance; continue;
              }
                       
            float map_val2 = map.getVal(p+dp);
            if(map_val2 >= 1.0F)           
              {
                (*aptr)[dij] = max_distance; continue;                
              }

            // formula of weight so that it still accounts for distance
            // even if the destination location is 100% unobstructed
            (*aptr)[dij] = 100*map_val2*dist_weight + dist_weight;
          }
        aptr++;
      }
}

// ######################################################################
//! read a map from a file
bool GridMap::read(std::string fileName)
{
  // file creation info:
  //  user id, creation date, general map description

  // evidence grid map parameters
  //  # of columns, # of rows,
  //  resolution (m/cell - assume square aspect ratio)

  // list of the likelihood for each cell

  LFATAL("GridMap::read  not yet implemented");

  return true;
}

// ######################################################################
float GridMap::getShortestPath
(Point2D<int> a, Point2D<int> b, std::vector<Point2D<int> > &steps)
{
  steps.clear();

  Timer tim(1000000);

  // check that both points are within range
  ASSERT(itsDirectedEdgeWeights.coordsOk(a) && 
         itsDirectedEdgeWeights.coordsOk(b));
  if(a == b) { return 0.0; }

  int w = itsDirectedEdgeWeights.getWidth();
  int h = itsDirectedEdgeWeights.getHeight();

  // calculate heuristic distance to goal
  Image<float> h_score(w,h, NO_INIT);  
  Image<float>::iterator hptr = h_score.beginw();

  for(int j = 0; j < h; j++)
    for(int i = 0; i < w; i++)
      {
        Point2D<int> p(i,j);
        double dist = b.distance(p);
        *hptr++ = dist;
      }

  // A* algorithm
  Image<float> g_score(w,h, NO_INIT);
  Image<float> f_score(w,h, NO_INIT);

  Image<float>::iterator gptr = g_score.beginw(), stop = g_score.endw();
  Image<float>::iterator fptr = f_score.beginw();
  while(gptr != stop) { *gptr++ = -1.0F; *fptr++ = 1.0F; }

  g_score.setVal(a, 0.0F);
  f_score.setVal(a, g_score.getVal(a) + h_score.getVal(a));

  // two forms of open and closed set to speed up the access
  std::vector<Point2D<int> > closedset;
  std::vector<Point2D<int> > openset;
  Image<bool> closedset_i(w,h,ZEROS);
  Image<bool> openset_i  (w,h,ZEROS);
  Image<int>  openset_pos(w,h, NO_INIT);

  Image<int> came_from_id(w,h, NO_INIT);

  openset.push_back(a); openset_i.setVal(a, true);

  float max_distance = 2*(w+h);

  // while openset is not empty
  while(openset.size() > 0)
    {
      // find a node in the openset with the lowest fscore
      Point2D<int> min_point(-1,-1); float minval = -1.0; 
      int min_index = -1;
      for(uint i = 0; i < openset.size(); i++)
        {
          Point2D<int> p = openset[i];
          float val      = f_score[p];
          //LINFO("[%3d %3d]: %f", p.i, p.j, val);

          if((minval == -1.0F && val != -1.0F) || val < minval)
            { minval = val; min_point = p; min_index = i; }
        }

      Point2D<int> x = min_point;      

      // if we arrive at the goal node
      if(x == b) 
        {
          steps.clear();
          std::vector<Point2D<int> > tsteps;

          // construct the path
          Point2D<int> cnode = x; float tcost = 0.0F;
          while(cnode != a && itsDirectedEdgeWeights.coordsOk(cnode))
            { 
              int id = came_from_id[cnode];
              Point2D<int>  diff = itsNeighbors[id].first;
              Point2D<int>  pcnode = cnode - diff;             
              float cost = itsDirectedEdgeWeights.getVal(pcnode)[id];

              if(cost == -1.0F) LFATAL("something wrong 1");    

              tsteps.push_back(diff);               
              tcost+= cost;              
              cnode = pcnode;
              
              //LINFO("step[%3d](%3d %3d): %f", 
              //      int(tsteps.size()), diff.i, diff.j, cost);
            }
          
          if(!itsDirectedEdgeWeights.coordsOk(cnode)) 
            LFATAL("something wrong 2");

          uint tsize = tsteps.size()-1;
          for(uint i = 0; i <= tsize; i++) 
            steps.push_back(tsteps[tsize - i]);

          return tcost;
        }

      // remove x from openset and add to closed set
      openset.erase(openset.begin()+min_index); closedset.push_back(x);
      openset_i.setVal(x, false);               closedset_i.setVal(x, true);

      // for each of x's directed edges
      float g_score_x = g_score[x];
      for(uint i = 0; i < itsNeighbors.size(); i++)
        {
          Point2D<int> diff   = itsNeighbors[i].first;
          float        weight = itsDirectedEdgeWeights[x][i];
 
          Point2D<int> y = x+diff;
          if(!itsDirectedEdgeWeights.coordsOk(y) || closedset_i.getVal(y)) 
            continue;

          // automatically skip the edge if 
          // the weight is above the maximum distance
          // NOTE: done to speed up operation
          if(weight >= max_distance) 
            { closedset_i.setVal(y, true);  continue; }

          float tentative_g_score = g_score_x + weight;

          bool tentative_is_better = false;
          float c_g_score = g_score[y];

          if(!openset_i.getVal(y))
            {
              openset.push_back(y); openset_i.setVal(y, true);
              tentative_is_better = true;
            }
          else if(c_g_score != -1.0F && tentative_g_score < c_g_score)
            {
              tentative_is_better = true;
            }
          
          if(tentative_is_better)
            {
              came_from_id[y] = i; // y came_from x using edge i;
              g_score[y] = tentative_g_score;
              float y_f_score = tentative_g_score + h_score[y]; 
              f_score[y] = y_f_score;              
            }
        }
    }
  
  // path not found
  steps.clear();
  return -1.0F;  
}

// ######################################################################
std::vector<float> GridMap::getDirectedEdgeWeights(Point2D<int> pt)
{
  ASSERT(itsDirectedEdgeWeights.coordsOk(pt));
  return itsDirectedEdgeWeights[pt];
}

// ######################################################################
void GridMap::setDirectedEdgeWeights
(Point2D<int> pt, std::vector<float> weights)
{
  ASSERT(itsDirectedEdgeWeights.coordsOk(pt));
  itsDirectedEdgeWeights[pt] = weights;
}

// ######################################################################
//! write a map to a file
bool GridMap::write(std::string fileName)
{
  LFATAL("GridMap::write not yet implemented");

  return true;
}

// ######################################################################
//! returns an image representation of the map
Image<PixRGB<byte> > GridMap::getMapImage(uint w, uint h)
{
  Image<PixRGB<byte> > res;

  LFATAL("GridMap::getImageMap not yet implemented");

  return res;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
