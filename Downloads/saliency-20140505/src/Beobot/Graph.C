/*!@file Beobot/Graph.C basic graph class can be both directed
  and undirected                                                        */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/Graph.C $
// $Id $
//

#include "Beobot/Graph.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include <unistd.h>

#include "Raster/Raster.H"

// ######################################################################
Graph::Graph():
  itsNodes(),
  itsEdges(),
  itsDirectedAdjecencyList(),
  itsDirectedAdjecencyCostList(),
  itsDirectedAdjecencyInEdgeList(),
  itsDirectedAdjecencyOutEdgeList(),
  itsUndirectedAdjecencyList(),
  itsUndirectedAdjecencyCostList(),
  itsUndirectedAdjecencyEdgeList(),
  itsDirectedPathList(),
  itsUndirectedPathList()
{
  itsDistanceMatrixComputed = false;
}

// ######################################################################
Graph::Graph(std::vector<rutz::shared_ptr<Node> > nodes,
             std::vector<rutz::shared_ptr<Edge> > edges):
  itsNodes(),
  itsEdges(),
  itsDirectedAdjecencyList(),
  itsDirectedAdjecencyCostList(),
  itsDirectedAdjecencyInEdgeList(),
  itsDirectedAdjecencyOutEdgeList(),
  itsUndirectedAdjecencyList(),
  itsUndirectedAdjecencyCostList(),
  itsUndirectedAdjecencyEdgeList(),
  itsDirectedPathList(),
  itsUndirectedPathList()
{
  for(uint i = 0; i < nodes.size(); i++)
    itsNodes.push_back(nodes[i]);

  for(uint i = 0; i < edges.size(); i++)
    itsEdges.push_back(edges[i]);

  itsDistanceMatrixComputed = false;
}

// ######################################################################
Graph::~Graph()
{ }

// ######################################################################
void Graph::addNode(rutz::shared_ptr<Node> node)
{
  itsNodes.push_back(node);
}

// ######################################################################
void Graph::addEdge(rutz::shared_ptr<Edge> edge)
{
  itsEdges.push_back(edge);
}

// ######################################################################
void Graph::computeAdjecencyList()
{
  itsDirectedAdjecencyList.clear();
  itsDirectedAdjecencyCostList.clear();
  itsDirectedAdjecencyInEdgeList.clear();
  itsDirectedAdjecencyOutEdgeList.clear();

  itsUndirectedAdjecencyList.clear();
  itsUndirectedAdjecencyCostList.clear();
  itsUndirectedAdjecencyEdgeList.clear();

  itsDirectedAdjecencyList.resize(itsNodes.size());
  itsDirectedAdjecencyCostList.resize(itsNodes.size());
  itsDirectedAdjecencyInEdgeList.resize(itsNodes.size());
  itsDirectedAdjecencyOutEdgeList.resize(itsNodes.size());

  itsUndirectedAdjecencyList.resize(itsNodes.size());
  itsUndirectedAdjecencyCostList.resize(itsNodes.size());
  itsUndirectedAdjecencyEdgeList.resize(itsNodes.size());

  uint nnode = itsNodes.size();
  itsDirectedAdjecencyEdgeIndexList   = Image<int>(nnode, nnode, ZEROS) - 1;
  itsUndirectedAdjecencyEdgeIndexList = Image<int>(nnode, nnode, ZEROS) - 1;
  
  // for each edge get the nodes
  for(uint i = 0; i < itsEdges.size(); i++)
    {
      uint sind = itsEdges[i]->getSourceNode()->getIndex();
      uint dind = itsEdges[i]->getDestNode()->getIndex();
      float cost = itsEdges[i]->getCost();
      LDEBUG("[%d -> %d]: %f", sind, dind, cost);

      // adjecency list:

      //for directed case
      itsDirectedAdjecencyList[sind].push_back(dind);
      itsDirectedAdjecencyCostList[sind].push_back(cost);

      itsDirectedAdjecencyInEdgeList[dind].push_back(i);
      itsDirectedAdjecencyOutEdgeList[sind].push_back(i);

      itsDirectedAdjecencyEdgeIndexList.setVal(sind, dind, i);

      //for undirected case
      itsUndirectedAdjecencyList[sind].push_back(dind);
      itsUndirectedAdjecencyList[dind].push_back(sind);
      itsUndirectedAdjecencyCostList[sind].push_back(cost);
      itsUndirectedAdjecencyCostList[dind].push_back(cost);

      itsUndirectedAdjecencyEdgeList[dind].push_back(i);
      itsUndirectedAdjecencyEdgeList[sind].push_back(i);

      itsUndirectedAdjecencyEdgeIndexList.setVal(sind, dind, i);
      itsUndirectedAdjecencyEdgeIndexList.setVal(dind, sind, i);
    }
}

// ######################################################################
void Graph::computeAdjecencyCostList()
{
  uint nnode = itsNodes.size();

  std::vector<int> countDCL(nnode);
  std::vector<int> countUCL(nnode);
  for(uint i = 0; i < nnode; i++)
    {
      countDCL[i] = 0;
      countUCL[i] = 0;
    }

  // for each edge get the nodes
  for(uint i = 0; i < itsEdges.size(); i++)
    {
      uint sind = itsEdges[i]->getSourceNode()->getIndex();
      uint dind = itsEdges[i]->getDestNode()->getIndex();
      float cost = itsEdges[i]->getCost();
      LDEBUG("[%d -> %d]: %f", sind, dind, cost);

      uint countD  = countDCL[sind];
      uint countUs = countUCL[sind];
      uint countUd = countUCL[dind];

      // adjecency list:

      //for directed case
      itsDirectedAdjecencyCostList[sind][countD] = cost;
      countDCL[sind] = countD + 1;

      //for undirected case
      itsUndirectedAdjecencyCostList[sind][countUs] = cost;
      itsUndirectedAdjecencyCostList[dind][countUd] = cost;
      countUCL[sind] = countUs + 1;
      countUCL[dind] = countUd + 1; 
    }
}

// ######################################################################
void Graph::computeDistances()
{
  // to avoid flag in getDistance()
  itsDistanceMatrixComputed = false;
  uint nnode = itsNodes.size();

  // create distance matrix
  itsDirectedDistanceMatrix.resize(nnode, nnode);
  itsDirectedPathList.resize(nnode);
  itsUndirectedDistanceMatrix.resize(nnode, nnode);
  itsUndirectedPathList.resize(nnode);
  for(uint i = 0; i < nnode; i++)
    {
      itsDirectedPathList[i].resize(nnode);
      itsUndirectedPathList[i].resize(nnode);
    }

  for(uint i = 0; i < nnode; i++)
    {
      for(uint j = 0; j < nnode; j++)
        {
          // directed case
          std::vector<uint> dedges;
          float ddist = getDirectedDistance(i,j, dedges);
          itsDirectedDistanceMatrix.setVal(i,j, ddist);
          itsDirectedPathList[i][j] = dedges;

          // undirected case
          std::vector<uint> uedges;
          float udist = getUndirectedDistance(i,j, uedges);
          itsUndirectedDistanceMatrix.setVal(i,j, udist);
          itsUndirectedPathList[i][j] = uedges;
        }
    }
  itsDistanceMatrixComputed = true;
}

// ######################################################################
float Graph::getDirectedDistance(uint a, uint b)
{
  std::vector<uint> edges;
  return getDirectedDistance(a, b, edges);
}

// ######################################################################
float Graph::getDirectedDistance
(uint a, uint b, std::vector<uint> &edges)
{
  ASSERT(a < itsNodes.size() && b < itsNodes.size());
  if(a == b) return 0.0F;

  if(itsDistanceMatrixComputed)
    {
      edges.clear();
      for(uint i = 0; i < itsDirectedPathList[a][b].size(); i++)
        edges.push_back(itsDirectedPathList[a][b][i]);
      return itsDirectedDistanceMatrix[Point2D<int>(a,b)];
    }

  uint nnode = itsNodes.size();
  std::vector<float> edist(nnode);
  std::vector<uint>  epath(nnode);

  // NOTE: this is Dijkstra algorithm
  //       we can't have negative edges
  for(uint i = 0; i < nnode; i++) edist[i] = -1.0F;
  std::vector<bool> selected(nnode);
  for(uint i = 0; i < nnode; i++) selected[i] = false;

  // Directed edges
  uint nsadj = itsDirectedAdjecencyList[a].size();
  for(uint i = 0; i < nsadj; i++)
    {
      uint adjnode   = itsDirectedAdjecencyList[a][i];
      edist[adjnode] = itsDirectedAdjecencyCostList[a][i];
      epath[adjnode] = itsDirectedAdjecencyOutEdgeList[a][i];
    }
  edist[a] = 0.0F; selected[a] = true;

  //for(uint ii = 0; ii < edist.size(); ii++)
  //  LINFO("[%d]: %d %f",ii, int(selected[ii]), edist[ii]);

  // while the minimum vertex is not our destination
  // or all the queue minimums are negative (can't reach anything else)
  // for un-connected graph
  uint mindex = a; bool end = false;
  while(mindex != b && !end)
    {
      // find the minimum distance in the queue
      float mindist = -1.0F;
      for(uint i = 0; i < nnode; i++)
        if(!selected[i] && edist[i] != -1.0F &&
           (mindist == -1.0F || mindist > edist[i]))
          {
            mindist = edist[i]; mindex = i;
            //LINFO("in: %d: mindex: %d mindist: %f", i, mindex, mindist);
          }

      //LINFO("min index[%d] %f", mindex, edist[mindex]);

      // if we still have a reacheable node
      if(mindist != -1.0F)
        {
          selected[mindex] = true;

          uint nadj = itsDirectedAdjecencyList[mindex].size();
          for(uint i = 0; i < nadj; i++)
            {
              uint eindex   = itsDirectedAdjecencyOutEdgeList[mindex][i];
              uint adjindex = itsDirectedAdjecencyList[mindex][i];
              float cost    = itsDirectedAdjecencyCostList[mindex][i];

              //LINFO("eindex: %d adjindex: %d cost: %f", eindex, adjindex, cost);

              if(edist[adjindex] == -1.0F ||
                 edist[adjindex] > edist[mindex] + cost)
                {
                  //LINFO("relaxing: %d: %f -> %f", adjindex,
                  //      edist[adjindex], edist[mindex] + cost);
                  edist[adjindex] = edist[mindex] + cost;
                  epath[adjindex] = eindex;

                  //LINFO("edist[%d]: %f epath[%d]: %d",
                  //      adjindex, edist[adjindex],
                  //      adjindex, epath[adjindex]);
                }
            }
        }
      else { end = true; }

      //for(uint ii = 0; ii < edist.size(); ii++)
      //  LINFO("[%d]: %d %f",ii, int(selected[ii]), edist[ii]);
    }

  // use the distances to calculate the list of edges
  edges.clear();

  // if node b is not unreachable
  if(edist[b] != -1.0F)
    {
      std::vector<uint> tEdges;
      // backtrack from the destination
      uint index = b;
      while(index != a)
        {
          tEdges.push_back(epath[index]);
          //LINFO("index: %d: epath: %d",index, epath[index]);
          index = getEdge(epath[index])->getSourceNode()->getIndex();
        }
      for(int i = tEdges.size()-1; i >= 0; i--)
        edges.push_back(tEdges[i]);
    }

  return edist[b];
}

// ######################################################################
float Graph::getUndirectedDistance(uint a, uint b)
{
  std::vector<uint> edges;
  return getUndirectedDistance(a, b, edges);
}

// ######################################################################
float Graph::getUndirectedDistance
(uint a, uint b, std::vector<uint> &edges)
{
  ASSERT(a < itsNodes.size() && b < itsNodes.size());
  if(a == b) return 0.0F;
  //LINFO("a: %d, b: %d", a, b);

  if(itsDistanceMatrixComputed)
    {
      edges.clear();
      for(uint i = 0; i < itsUndirectedPathList[a][b].size(); i++)
        {
          edges.push_back(itsUndirectedPathList[a][b][i]);
          //LINFO("edge[%d]: %d", i, edges[i]);
        }
      return itsUndirectedDistanceMatrix[Point2D<int>(a,b)];
    }

  uint nnode = itsNodes.size();
  std::vector<float> edist(nnode);
  std::vector<uint>  epath(nnode);

  // NOTE: this is Dijkstra algorithm
  //       we can't have negative edges
  //       here we use -1.0F to denote infinity
  for(uint i = 0; i < nnode; i++) edist[i] = -1.0F;
  std::vector<bool> selected(nnode);
  for(uint i = 0; i < nnode; i++) selected[i] = false;

  // undirected edges
  uint nsadj = itsUndirectedAdjecencyList[a].size();
  //LINFO("nsadj: %d", nsadj);
  for(uint i = 0; i < nsadj; i++)
    {
      uint adjnode = itsUndirectedAdjecencyList[a][i];
      edist[adjnode] = itsUndirectedAdjecencyCostList[a][i];
      epath[adjnode] = itsUndirectedAdjecencyEdgeList[a][i];
    }
  edist[a] = 0.0F; selected[a] = true;

  // for(uint ii = 0; ii < edist.size(); ii++)
  //   LINFO("[%d]:sel:%d %f ep:%d",
  //         ii, int(selected[ii]), edist[ii], epath[ii]);

  // while the minimum vertex is not our destination
  // or all the queue minimums are negative (can't reach anything else)
  // for un-connected graph
  uint mindex = a; bool end = false;
  while(mindex != b && !end)
    {
      // find the minimum distance in the queue
      float mindist = -1.0F;
      for(uint i = 0; i < nnode; i++)
        if(!selected[i] && edist[i] != -1.0F &&
           (mindist == -1.0F || mindist > edist[i]))
          {
            mindist = edist[i]; mindex = i;
            //LINFO("in: %d: mindex: %d mindist: %f", i, mindex, mindist);
          }

      //LINFO("min index[%d] %f", mindex, edist[mindex]);

      // if we still have a reacheable node
      if(mindist != -1.0F)
        {
          selected[mindex] = true;

          uint nadj = itsUndirectedAdjecencyList[mindex].size();
          for(uint i = 0; i < nadj; i++)
            {
              uint eindex   = itsUndirectedAdjecencyEdgeList[mindex][i];
              uint adjindex = itsUndirectedAdjecencyList[mindex][i];
              float cost    = itsUndirectedAdjecencyCostList[mindex][i];

              if(edist[adjindex] == -1.0F ||
                 edist[adjindex] > edist[mindex] + cost)
                {
                  //LINFO("relaxing: %d: %f -> %f", adjindex,
                  //      edist[adjindex], edist[mindex] + cost);
                  edist[adjindex] = edist[mindex] + cost;
                  epath[adjindex] = eindex;
                }
            }
        }
      else { end = true; }

      //for(uint ii = 0; ii < edist.size(); ii++)
      //  LINFO("[%d]: %d %f",ii, int(selected[ii]), edist[ii]);
    }

  // use the distances to calculate the list of edges
  edges.clear();

  // if node b is not unreachable
  if(edist[b] != -1.0F)
    {
      std::vector<uint> tEdges;
      // backtrack from the destination
      uint index = b;
      while(index != a)
        {
          tEdges.push_back(epath[index]);

          // get next edge 
          // need to check index because it's undirected
          uint is = getEdge(epath[index])->getSourceNode()->getIndex();
          uint id = getEdge(epath[index])->getDestNode()->getIndex();
          if(index == is) index = id;
          else index = is;
          //LINFO("edge index: %d",index);
        }

      for(int i = tEdges.size()-1; i >= 0; i--)
        edges.push_back(tEdges[i]);
    }
  return edist[b];
}

// ######################################################################
float Graph::getDirectedDistance(uint a, uint b, std::vector<float> h)
{
  std::vector<uint> edges;
  return getDirectedDistance(a, b, h, edges);
}

// ######################################################################
float Graph::getDirectedDistance
(uint a, uint b, std::vector<float> h_score, std::vector<uint> &edges)
{
  ASSERT(a < itsNodes.size() && b < itsNodes.size());
  if(a == b) return 0.0F;
  //LINFO("a: %d, b: %d", a, b);

  if(itsDistanceMatrixComputed)
    {
      edges.clear();
      for(uint i = 0; i < itsDirectedPathList[a][b].size(); i++)
        {
          edges.push_back(itsDirectedPathList[a][b][i]);
          //LINFO("edge[%d]: %d", i, edges[i]);
        }
      return itsDirectedDistanceMatrix[Point2D<int>(a,b)];
    }

  // A* algorithm
  uint nnode = itsNodes.size();
  std::vector<float> g_score(nnode);
  std::vector<float> f_score(nnode);
  for(uint i = 0; i < nnode; i++) g_score[i] = -1.0F;
  for(uint i = 0; i < nnode; i++) f_score[i] = -1.0F;
  g_score[a] = 0.0F;
  f_score[a] = g_score[a] + h_score[a]; 

  std::vector<uint>  came_from(nnode);
  for(uint i = 0; i < nnode; i++) came_from[i] = nnode;  

  std::vector<bool> closedset(nnode);
  std::vector<bool> openset(nnode);
  for(uint i = 0; i < nnode; i++) closedset[i] = false;
  for(uint i = 0; i < nnode; i++) openset[i]   = false;
  int countOpenSet = 1;
  openset[a]   = true;
  closedset[a] = false;
  
  // while openset is not empty
  while(countOpenSet > 0)
    {
      // find the node with the lowest fscore
      uint minindex = 0; float minval = -1.0; 
      for(uint i = 0; i < nnode; i++)
        {
          if(!openset[i]) continue;
          
          float val = f_score[i];
          if((minval == -1.0F && val != -1.0F) || 
             val < minval)
            { minval = val; minindex = i; }
        }

      uint x = minindex;
      
      //LINFO("x: %d: %f", x, minval);

      // if we arrive at the goal node
      if(x == b) 
        {
          edges.clear();
          std::vector<uint> tedges;

          // construct the path
          uint cnode = x; float tcost = 0.0F;
          while(cnode != a && cnode != nnode)
            { 
              uint pcnode = came_from[cnode];

              float cost = -1.0F;
              uint nsadj = itsDirectedAdjecencyList[pcnode].size();
              uint mindex = 0;
              for(uint i = 0; i < nsadj; i++)
                {
                  uint ncnode = itsDirectedAdjecencyList[pcnode][i];
                  if(ncnode == cnode)
                    {
                      cost = itsDirectedAdjecencyCostList[pcnode][i];
                      mindex = i; i = nsadj;
                    }
                }
              //LINFO("DONE: %d: a: %d", pcnode, a);
              if(cost == -1.0F) LFATAL("something wrong");    

              tedges.push_back
                (itsDirectedAdjecencyOutEdgeList[pcnode][mindex]); 
              
              cnode = pcnode;
              tcost+= cost;
            }
          
          if(cnode == nnode) LFATAL("something wrong");
          uint tsize = tedges.size()-1;
          for(uint i = 0; i <= tsize; i++) 
            edges.push_back(tedges[tsize - i]);

          return tcost;
        }

      // remove x from openset and add to closed set
      openset[x] = false; countOpenSet--; closedset[x] = true;

      // for each of x's directed edges
      uint nsadj = itsDirectedAdjecencyList[x].size();
      //LINFO("nneigh[%3d]: %d", x, nsadj);
      
      for(uint i = 0; i < nsadj; i++)
        {
          uint y = itsDirectedAdjecencyList[x][i];
          //LINFO("  n[%3d]: %d -- %d", int(i), int(y), int(closedset[y]));
          if(closedset[y]) continue;

          float tentative_g_score = g_score[x] + 
            itsDirectedAdjecencyCostList[x][i];
          //LINFO("  cost: %f + %f = %f ", g_score[x], 
          //      itsDirectedAdjecencyCostList[x][i],
          //      tentative_g_score);
          
          bool tentative_is_better = false;
          if(!openset[y])
            {
              openset[y] = true; countOpenSet++;
              tentative_is_better = true;
            }
          else if(g_score[y] != -1.0F && tentative_g_score < g_score[y])
            {
              tentative_is_better = true;
            }
          
          if(tentative_is_better)
            {
              came_from[y] = x;
              g_score[y] = tentative_g_score;
              f_score[y] = g_score[y] + h_score[y];
            }
        }
    }

  // path not found
  edges.clear(); 
  return -1.0F;  
}

// ######################################################################
float Graph::getMaxDirectedDistance()
{
  float min, max;
  getMinMax(itsDirectedDistanceMatrix, min, max);
  if(min == -1.0F) return -1.0F;
  else return max;
}

// ######################################################################
float Graph::getMaxUndirectedDistance()
{
  float min, max;
  getMinMax(itsUndirectedDistanceMatrix, min, max);
  if(min == -1.0F) return -1.0F;
  else return max;
}

// ######################################################################
std::vector<uint> Graph::getDirectedPath(uint a, uint b)
{
  std::vector<uint> edges;

  // if distance matrix is already computed
  if(itsDistanceMatrixComputed) return itsDirectedPathList[a][b];
  else
    {
      getDirectedDistance(a, b, edges);
      return edges;
    }
}

// ######################################################################
std::vector<uint> Graph::getUndirectedPath(uint a, uint b)
{
  std::vector<uint> edges;

  // if distance matrix is already computed
  if(itsDistanceMatrixComputed) return itsUndirectedPathList[a][b];
  else
    {
      getUndirectedDistance(a, b, edges);
      return edges;
    }
}

// ######################################################################
std::vector<uint> Graph::getEdges(rutz::shared_ptr<Node> n)
{
  std::vector<uint> res;

  // for each edge get the nodes
  for(uint i = 0; i < itsEdges.size(); i++)
    {
      //res.push_back();
    }
  LFATAL("not yet implemented");

  return res;
}

// ######################################################################
float Graph::getAngle
(rutz::shared_ptr<Edge> e1, rutz::shared_ptr<Edge> e2)
{
  // get the end coordinates of each edge
  Point2D<int> s1 = e1->getSourceNode()->getCoordinate();
  Point2D<int> d1 = e1->getDestNode()->getCoordinate();
  Point2D<int> s2 = e2->getSourceNode()->getCoordinate();
  Point2D<int> d2 = e2->getDestNode()->getCoordinate();
  LDEBUG("(s1(%d,%d)-d1(%d,%d))-----(s2(%d,%d)-d2(%d,%d))",
         s1.i, s1.j,d1.i, d1.j, s2.i, s2.j,d2.i, d2.j);

  // calculate differences
  float xD1 = d1.i - s1.i;
  float xD2 = d2.i - s2.i;
  float yD1 = d1.j - s1.j;
  float yD2 = d2.j - s2.j;

  // calculate the lengths of the two lines
  float len1 = sqrt(xD1*xD1+yD1*yD1);
  float len2 = sqrt(xD2*xD2+yD2*yD2);

  float dot   = (xD1*xD2+yD1*yD2); // dot   product
  float cross = (xD1*yD2-xD2*yD1); // cross product

  // calculate angle between the two lines
  float ang = atan2(cross/(len1*len2), dot/(len1*len2));
  return ang;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
