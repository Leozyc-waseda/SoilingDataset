/*!@file Beobot/TopologicalMap.C topological map                        */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/TopologicalMap.C $
// $Id:TopologicalMap.C 6891 2006-05-25 06:46:56Z siagian $
//

// ######################################################################
/*! Topological map for localization                                    */

#include "Beobot/TopologicalMap.H"
#include "Raster/Raster.H"
#include "Image/DrawOps.H"      // for drawing
#include "Image/MathOps.H"      // for minmax
#include "Image/CutPaste.H" // for inplacePaste
#include "Util/Geometry2DFunctions.H" //for line dist
#include <cstdio>

// ######################################################################
TopologicalMap::TopologicalMap()
  : Map(),
    itsGraph(new Graph())
{
}

// ######################################################################
TopologicalMap::TopologicalMap(std::string fileName)
  : Map(),
    itsGraph(new Graph())
{
  readJSON(fileName);
}

// ######################################################################
TopologicalMap::~TopologicalMap()
{
}

// ######################################################################
bool TopologicalMap::readJSON(std::string fileName)
{
  try
  {
    std::ifstream input(fileName);
    std::stringstream ss;
    ss << input.rdbuf(); 

    boost::property_tree::ptree pt;
    boost::property_tree::read_json(ss, pt);

    int w = pt.get("USC.width",0);
    int h = pt.get("USC.height",0);
    double scale = pt.get("USC.scale",0.0);
    int segnum = pt.get("USC.segnum",0);

    itsMapWidth = w; itsMapHeight = h;
    itsMapScale = scale;
    itsSegmentList.resize(segnum);
    itsSegmentEdgeIndexList.resize(segnum);
    itsSegmentLength.resize(segnum);
    LINFO("read map size(%d,%d) scale %4.2f segnum %d",w,h,scale,segnum);

    // get the node coordinates
    BOOST_FOREACH(boost::property_tree::ptree::value_type &v, pt.get_child("USC.nodes"))
    {
      assert(v.first.empty()); // array elements have no names
      int nodeID = v.second.get("name",0);
      std::string comment = v.second.get("comment","");
      Point2D<int>pos(-1,-1); int i = 0;
      BOOST_FOREACH(boost::property_tree::ptree::value_type &v2, v.second.get_child("position"))
      {
        int p = atoi(v2.second.data().c_str());
        if(i++ == 0) pos.i = p; else pos.j = p;
      }
      LINFO("node[%d]: (%d,%d) comment %s", itsGraph->getNumNode(), pos.i, pos.j,comment.c_str());
      rutz::shared_ptr<Node> node(new Node(nodeID-1, pos,comment));
      itsGraph->addNode(node);
    }

    // get the edge information
    BOOST_FOREACH(boost::property_tree::ptree::value_type &v, pt.get_child("USC.edges"))
    {
      int src = v.second.get("src",-1);
      int dst = v.second.get("dst",-1);
      int seg = v.second.get("segment",-1);
      int edgeID = v.second.get("name",-1);
      Point2D<int> scoor = itsGraph->getNode(src-1)->getCoordinate();
      Point2D<int> ecoor = itsGraph->getNode(dst-1)->getCoordinate();
      double dist = scoor.distance(ecoor);
      LINFO("edge %d (%d,%d): %f",edgeID-1, src-1, dst-1, dist);
      
      int col_r = 127; int col_g = 127; int col_b = 127;  int i = 0;    
      if( v.second.get_child_optional("color") )
        {				
          BOOST_FOREACH(boost::property_tree::ptree::value_type &v2, 
                        v.second.get_child("color"))
            {
              int p = atoi(v2.second.data().c_str());
              LINFO("p = %d i = %d",p,i);
              if(i == 0) col_r = p; 
              else if(i == 1) col_g = p; 
              else if(i == 2) col_b = p; 
              i++;
            }
          LINFO("Found Color of Segment %d %d %d",col_r,col_g,col_b);
        }

      rutz::shared_ptr<Edge>
        edge(new Edge(itsGraph->getNode(src-1),
              itsGraph->getNode(dst-1), dist));
      edge->setDisplayColor(PixRGB<byte>(col_r,col_g,col_b));
      itsGraph->addEdge(edge);

      LINFO("seg[%d][%" ZU "]:%d",
          seg-1, itsSegmentList[seg-1].size(), edgeID-1);
      itsSegmentList[seg-1].push_back(itsGraph->getEdge(edgeID-1));
      itsSegmentEdgeIndexList[seg-1].push_back(edgeID-1);
    }

    // store the segment number for each edge
    uint numEdge = itsGraph->getNumEdge();
    itsEdgeSegmentNum.clear();
    itsEdgeSegmentNum.resize(numEdge);
    for(uint i = 0; i < itsSegmentEdgeIndexList.size(); i++)
      for(uint j = 0; j < itsSegmentEdgeIndexList[i].size(); j++) 
        itsEdgeSegmentNum[itsSegmentEdgeIndexList[i][j]] = i;

    // compute all the shortcuts for shortest-distance related operations
    // node-to-node at graph level
    itsGraph->computeAdjecencyList();
    itsGraph->computeDistances();

    // node-to-segment, and segment-to-segment at this level
    computeDistances();

    // segment length at this level
    setSegmentLength();

    return true;
  }
  catch (std::exception const& e)
  {
    std::cerr << e.what() << std::endl;
    LFATAL("JSON file: %s not properly read", fileName.c_str());
  }
  return EXIT_FAILURE;
}

// ######################################################################
bool TopologicalMap::read(std::string fileName)
{
  FILE *fp;  char inLine[200]; char comment[200];

  LINFO("file name: %s", fileName.c_str());
  if((fp = fopen(fileName.c_str(),"rb")) == NULL)
    {
      LINFO("not found");
      itsMapWidth = 0; itsMapHeight = 0;
      LINFO("map size: 0x0");

      itsMapScale = 1.0;
      LINFO("map scale: 1.0");

      itsSegmentList.resize(0);
      itsSegmentEdgeIndexList.resize(0);
      itsSegmentLength.resize(0);
      LINFO("segment size: %d", 0);

      return false;
    }

  // file creation info:
  //  user id, creation date, general map description

  // off-limit areas
  //   # of areas
  //   description of area (how to characterize the dimensions)

  // named places/position
  // need distance relationship wrt to area
  //   # of points, description, x,y,theta
  //   # of edges to connect the places, with expected distances

  // size of area covered
  if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");
  int w, h; sscanf(inLine, "%s %d %d", comment, &w, &h);
  itsMapWidth = w; itsMapHeight = h;
  LINFO("map size: %d %d", itsMapWidth, itsMapHeight);

  // size of area covered
  if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");
  float scale; sscanf(inLine, "%s %f", comment, &scale);
  itsMapScale = scale;
  LINFO("map scale: %f", itsMapScale);

  // segment size
  if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");
  int ssize;  sscanf(inLine, "%s %d", comment, &ssize);
  itsSegmentList.resize(ssize);
  itsSegmentEdgeIndexList.resize(ssize);
  itsSegmentLength.resize(ssize);
  LINFO("segment Num: %d", ssize);

  // skip the nodes section divider
  if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed");

  // get the node coordinates
  uint ncount = 0;
  while(fgets(inLine, 200, fp) != NULL && inLine[0] != '=')
  {
    int x, y; sscanf(inLine, "%d %d", &x, &y);
    LINFO("node[%d]: (%d,%d)", itsGraph->getNumNode(), x, y);

    rutz::shared_ptr<Node> node(new Node(ncount, Point2D<int>(x,y)));
    itsGraph->addNode(node);
    ncount++;
  }

  // get the edge information
  while((fgets(inLine, 200, fp) != NULL) && (inLine[0] != '='))
  {
    int s, e; sscanf(inLine, "%d %d", &s, &e);
    Point2D<int> scoor = itsGraph->getNode(s)->getCoordinate();
    Point2D<int> ecoor = itsGraph->getNode(e)->getCoordinate();
    double dist = scoor.distance(ecoor);
    LINFO("edge(%d,%d): %f", s, e, dist);

    rutz::shared_ptr<Edge>
      edge(new Edge(itsGraph->getNode(s),
                    itsGraph->getNode(e), dist));
    itsGraph->addEdge(edge);
  }

  // get the segment information
  uint scount = 0;
  while(fgets(inLine, 200, fp) != NULL)
  {
    std::string line(inLine);
    while(line.length() > 0)
      {
        uint cedge;
        std::string::size_type spos = line.find_first_of(' ');
        if(spos != std::string::npos)
          {
            cedge = uint(atoi(line.substr(0, spos).c_str()));
            line = line.substr(spos+1);
          }
        else
          {
            cedge = uint(atoi(line.c_str()));
            line = std::string("");
          }
        LINFO("seg[%d][%" ZU "]:%d",
              scount, itsSegmentList[scount].size(), cedge);
        itsSegmentList[scount].push_back(itsGraph->getEdge(cedge));
        itsSegmentEdgeIndexList[scount].push_back(cedge);
      }
    scount++;
  }

  // store the segment number for each edge
  uint numEdge = itsGraph->getNumEdge();
  itsEdgeSegmentNum.clear();
  itsEdgeSegmentNum.resize(numEdge);
  for(uint i = 0; i < itsSegmentEdgeIndexList.size(); i++)
    for(uint j = 0; j < itsSegmentEdgeIndexList[i].size(); j++) 
      itsEdgeSegmentNum[itsSegmentEdgeIndexList[i][j]] = i;

  fclose(fp);

  // compute all the shortcuts for shortest-distance related operations
  // node-to-node at graph level
  itsGraph->computeAdjecencyList();
  itsGraph->computeDistances();

  // node-to-segment, and segment-to-segment at this level
  computeDistances();

  // segment length at this level
  setSegmentLength();

  return true;
}

// ######################################################################
void TopologicalMap::computeDistances()
{
  // create segment number distance matrix
  uint nnode = itsGraph->getNumNode();
  uint nsegment = getSegmentNum();
  itsNodeSegmentDistanceMatrix.resize(nnode, nsegment);

  // for each node
  for(uint i = 0; i < nnode; i++)
    {
      // for each segment
      for(uint j = 0; j < nsegment; j++)
        {
          // find the closest distance
          double mindist;
          uint nsedge = itsSegmentList[j].size();
          if(nsedge > 0)
            {
              uint snode = itsSegmentList[j][0]->getSourceNode()->getIndex();
              mindist = itsGraph->getUndirectedDistance(i,snode);
            }
          else mindist = -1.0F;

          for(uint k = 0; k < nsedge; k++)
            {
              uint dnode = itsSegmentList[j][k]->getDestNode()->getIndex();
              double cdist = itsGraph->getUndirectedDistance(i, dnode);
              if(mindist > cdist) mindist = cdist;
            }

          LDEBUG("itsSegmentList[%d][%d]: %f",i,j, mindist);
          itsNodeSegmentDistanceMatrix.setVal(i, j, mindist);
        }
    }
}

// ######################################################################
bool TopologicalMap::write(std::string fileName)
{
  LFATAL("NOT YET implemented");
  return true;
}

// ######################################################################
Image<PixRGB<byte> > TopologicalMap::getMapImage(Image<PixRGB<byte> > backgroundImg,
double scale,Point2D<int> offset)
{
  //double min_scale = std::min(backgroundImg.getWidth()/itsMapWidth, 
  //                            backgroundImg.getHeight()/itsMapHeight);
  //LINFO("MinScale %f Scale %f bk(%d,%d) tmap(%d,%d)",min_scale,scale,
  //      backgroundImg.getWidth(),backgroundImg.getHeight(),itsMapWidth,itsMapHeight);

  Image<PixRGB<byte> > res = backgroundImg;
  
  //check offset is in the range
  if(!res.coordsOk(Point2D<int>(offset.i+itsMapWidth*scale,offset.j+itsMapHeight*scale)))
    offset = Point2D<int>(0,0);


  // draw the edges first
  for(uint i = 0; i < itsGraph->getNumEdge(); i++)
    {
      Point2D<int> sp  = itsGraph->getEdge(i)->getSourceNode()->getCoordinate();
      Point2D<int> ep  = itsGraph->getEdge(i)->getDestNode()->getCoordinate();
      PixRGB<byte> col = itsGraph->getEdge(i)->getDisplayColor();

      Point2D<int> p1 = Point2D<int>(sp.i*scale, sp.j*scale);
      Point2D<int> p2 = Point2D<int>(ep.i*scale, ep.j*scale);
      double dist = p1.distance(p2);
      int len = 20;
      int min_len = int(dist*.2); 
      if(min_len < 3) min_len = 3; 
      if(len > min_len) len = min_len;
      drawArrow(res, p1+offset, p2+offset,
                PixRGB<byte>(0,0,0),2, len);

      //drawLine(res, p1+offset, p2+offset, col,4, len);
      drawLine(res, p1+offset, p2+offset, PixRGB<byte>(0,0,0),2, len);
    }

  // draw the nodes
  for(uint i = 0; i < itsGraph->getNumNode(); i++)
    {
      Point2D<int> p = itsGraph->getNode(i)->getCoordinate();
      Point2D<int> pd(int(p.i * scale), int(p.j * scale));
      drawDisk(res, pd+offset, 8, PixRGB<byte>(0,0,0));
    }

  drawRect(res,res.getBounds(),PixRGB<byte>(255,255,255));

  return res;
}

// ######################################################################
Image<PixRGB<byte> > TopologicalMap::getMapImage(uint w, uint h)
{
  float scale = std::min(float(w)/float(itsMapWidth), 
                         float(h)/float(itsMapHeight));

  Image<PixRGB<byte> >
    res(int(itsMapWidth * scale), int(itsMapHeight * scale), ZEROS);

  // draw the edges first
  for(uint i = 0; i < itsGraph->getNumEdge(); i++)
    {
      Point2D<int> sp = 
        itsGraph->getEdge(i)->getSourceNode()->getCoordinate();
      Point2D<int> ep = 
        itsGraph->getEdge(i)->getDestNode()->getCoordinate();

      // LINFO("[%d %d][%d %d] [%d %d] [%d %d] scale: %f", 
      //       sp.i, sp.j, ep.i, ep.j,
      //       int(sp.i * scale), int(sp.j * scale),
      //       int(ep.i * scale), int(ep.j * scale), scale );

      drawArrow(res,
                Point2D<int>(int(sp.i * scale), int(sp.j * scale)),
                Point2D<int>(int(ep.i * scale), int(ep.j * scale)),
                PixRGB<byte>(0,0,0));
    }

  // draw the nodes
  for(uint i = 0; i < itsGraph->getNumNode(); i++)
    {
      Point2D<int> p = itsGraph->getNode(i)->getCoordinate();
      Point2D<int> pd(int(p.i * scale), int(p.j * scale));
      drawDisk(res, pd, 2, PixRGB<byte>(0,0,0));
    }

  drawRect(res,res.getBounds(),PixRGB<byte>(255,255,255));

  return res;
}

// ######################################################################
double TopologicalMap::getDistance
(uint asnum, double altrav, uint bsnum, double bltrav)
{
  ASSERT(asnum < itsSegmentList.size() && altrav <= 1.0F &&
         bsnum < itsSegmentList.size() && bltrav <= 1.0F    );

  // get the edges that the two points belongs to
  // and the distances to start and end nodes
  double dsa, dea, dsb, deb;
  rutz::shared_ptr<Edge> a = getEdge(asnum, altrav, dsa, dea);
  rutz::shared_ptr<Edge> b = getEdge(bsnum, bltrav, dsb, deb);

  // get the end node indexes
  uint sa = a->getSourceNode()->getIndex();
  uint ea = a->getDestNode()->getIndex();
  uint sb = b->getSourceNode()->getIndex();
  uint eb = b->getDestNode()->getIndex();
  LDEBUG("%d %d  -- %d %d", sa, ea, sb, eb);

  // same edge situation
  // in case we have a directed graph
  double dist = 0.0F;
  if(sa == sb && ea == eb) dist = fabs(dsa - dsb);
  else if(sa == eb && sb == ea) dist = fabs(dsa + dsb - a->getCost());
  else
    {
      // get the shortest distance in
      double dist1 = dsa + itsGraph->getUndirectedDistance(sa, sb) + dsb;
      double dist2 = dsa + itsGraph->getUndirectedDistance(sa, eb) + deb;
      double dist3 = dea + itsGraph->getUndirectedDistance(ea, sb) + dsb;
      double dist4 = dea + itsGraph->getUndirectedDistance(ea, eb) + deb;
      LDEBUG("%f %f %f %f", dist1,dist2,dist3,dist4);
      dist = std::min(std::min(dist1, dist2), std::min(dist3, dist4));
    }
  LDEBUG("-> %f",dist);
  return dist;
}

// ######################################################################
double TopologicalMap::getPath
(uint asnum, double altrav, uint bsnum, double bltrav,
 std::vector<int> &moves)
{
  // NOTE: assuming graph is undirected
  //       if want directed get path
  //       has to find eindexa and b for the opposite direction
  //       i.e. edge(ea,sa) and edge(eb,sb)

  ASSERT(asnum < itsSegmentList.size() && altrav <= 1.0F &&
         bsnum < itsSegmentList.size() && bltrav <= 1.0F    );
  moves.clear();

  // get the edges that the two points belongs to
  // and the distances to start and end nodes
  double dsa, dea, dsb, deb; uint eindexa, eindexb;
  rutz::shared_ptr<Edge> a = getEdge(asnum, altrav, dsa, dea, eindexa);
  rutz::shared_ptr<Edge> b = getEdge(bsnum, bltrav, dsb, deb, eindexb);

  // get the end node indexes
  uint sa = a->getSourceNode()->getIndex();
  uint ea = a->getDestNode()->getIndex();
  uint sb = b->getSourceNode()->getIndex();
  uint eb = b->getDestNode()->getIndex();
  LDEBUG("a[%d %f] -> b[%d %f]", asnum, altrav, bsnum, bltrav);
  LDEBUG("%d %d  -- %d %d", sa, ea, sb, eb);

  // same edge situation with same start and end
  double dist = 0.0;
  if(sa == sb && ea == eb)
    {
      dist = fabs(dsa - dsb);
      if(dsb > dsa) moves.push_back(TOPOMAP_MOVE_FORWARD);
      else if(dsb < dsa)
        {
          moves.push_back(TOPOMAP_TURN_AROUND);
          moves.push_back(TOPOMAP_MOVE_FORWARD);
        }
      // else already in goal
    }
  // same edge situation with flipped start and end
  else if(sa == eb && sb == ea)
    {
      dist = fabs(dsa + dsb - a->getCost());
      if(deb < dsa) moves.push_back(TOPOMAP_MOVE_FORWARD);
      else if(deb > dsa)
        {
          moves.push_back(TOPOMAP_TURN_AROUND);
          moves.push_back(TOPOMAP_MOVE_FORWARD);
        }
      // else already in goal
    }
  // goes through a few other edges
  else
    {
      double dsad = dsa;
      double dsbd = dsb;
      double dead = dea;
      double debd = deb;
      
      // get the shortest distance (after casting everything to double)
      std::vector<uint> edges1;
      double sasb = itsGraph->getUndirectedDistance(sa, sb, edges1);
      double sasbd = sasb;
      double dist1 = dsad + sasbd + dsbd;

      std::vector<uint> edges2;
      double saeb = itsGraph->getUndirectedDistance(sa, eb, edges2);
      double saebd = saeb;
      double dist2 = dsad + saebd + debd;

      std::vector<uint> edges3;
      double easb = itsGraph->getUndirectedDistance(ea, sb, edges3);
      double easbd = easb;
      double dist3 = dead + easbd + dsbd;

      std::vector<uint> edges4;
      double eaeb = itsGraph->getUndirectedDistance(ea, eb, edges4);
      double eaebd = eaeb;
      double dist4 = dead + eaebd + debd;

      dist = std::min(std::min(dist1, dist2), std::min(dist3, dist4));

      LDEBUG("%f %f %f %f: min: %f", dist1, dist2, dist3, dist4, dist);
      LDEBUG("dsa: %f dsb: %f dea: %f deb: %f", dsa,dsb,dea,deb);

      // get the edges
      if(dist1 == dist)
        {
          LDEBUG("E1size: %" ZU , edges1.size());
          if(dsa != 0.0F) moves.push_back(TOPOMAP_TURN_AROUND);
          moves.push_back(TOPOMAP_MOVE_FORWARD);
          for(uint i = 0; i < edges1.size(); i++)
            {
              moves.push_back(edges1[i]);
              LDEBUG("E1[%3d]: %d", i, edges1[i]);
            }
          moves.push_back(eindexb);
        }
      else if(dist2 == dist)
        {
          LDEBUG("E2size: %" ZU , edges2.size());
          if(dsa != 0.0F) moves.push_back(TOPOMAP_TURN_AROUND);
          moves.push_back(TOPOMAP_MOVE_FORWARD);
          for(uint i = 0; i < edges2.size(); i++)
            {
              moves.push_back(edges2[i]);
              LDEBUG("E2[%3d]: %d", i, edges2[i]);
            }
          moves.push_back(eindexb);

          // needs opposite edge eindexb for directed case
        }
      else if(dist3 == dist)
        {
          LDEBUG("E3size: %" ZU ,edges3.size());
          moves.push_back(TOPOMAP_MOVE_FORWARD);
          for(uint i = 0; i < edges3.size(); i++)
            {
              moves.push_back(edges3[i]);
              LDEBUG("E3[%3d]: %d", i, edges3[i]);
            }
          moves.push_back(eindexb);
        }
      else if(dist4 == dist)
        {
          LDEBUG("E4size: %" ZU , edges4.size());
          moves.push_back(TOPOMAP_MOVE_FORWARD);
          for(uint i = 0; i < edges4.size(); i++)
            {
              moves.push_back(edges4[i]);
              LDEBUG("E4[%3d]: %d", i, edges4[i]);
            }
          moves.push_back(eindexb);

          // needs opposite edge eindexb for directed case
        }

      for(uint i = 0; i < moves.size(); i++)
        LDEBUG("[%d]: %d", i, moves[i]);
    }
  return dist;
}


// ######################################################################
rutz::shared_ptr<Edge> TopologicalMap::getEdge
(uint cseg, double ltrav, double &sdist, double &edist)
{
  uint i;
  return getEdge(cseg, ltrav, sdist, edist, i);
}

// ######################################################################
rutz::shared_ptr<Edge> TopologicalMap::getEdge
(uint cseg, double ltrav, double &sdist, double &edist, uint &eindex)
{
  ASSERT((cseg < itsSegmentList.size()) && (ltrav <= 1.0F));

  // using the segment we can get its list of edges
  double slen = getSegmentLength(cseg);
  double altrav = ltrav * slen;
  LDEBUG("%d: %f * %f = altrav: %f", cseg, ltrav, slen, altrav);

  // iterate through the list of edges
  double cltrav = 0.0F;
  uint nsedge = itsSegmentList[cseg].size();
  if(nsedge == 0) return rutz::shared_ptr<Edge>();

  uint j = 0;
  while((j < nsedge) && (altrav >= cltrav))
    { cltrav += itsSegmentList[cseg][j]->getCost(); j++; }
  LDEBUG("nsedge: %d -> j: %d", nsedge, j);

  // get the distance to the next node
  edist = cltrav - altrav;
  sdist = itsSegmentList[cseg][j-1]->getCost() - edist;
  LDEBUG("sdist: %f, edist: %f", sdist, edist);
  eindex = itsSegmentEdgeIndexList[cseg][j-1];
  return itsSegmentList[cseg][j-1];
}

// ######################################################################
rutz::shared_ptr<Edge> TopologicalMap::getEdge
(Point2D<int> loc, double &sdist, double &edist)
{
  // can be used to implement conversion from Point2D<int> to (snum,ltrav)
  // draw the edges first
  int edgeIdx = 0;
  double minDist = -1;
  double ltrav = 0.0F;
  for(uint i = 0; i < itsGraph->getNumEdge(); i++)
    {
      Point2D<int> sp = itsGraph->getEdge(i)->getSourceNode()->getCoordinate();
      Point2D<int> ep = itsGraph->getEdge(i)->getDestNode()->getCoordinate();
      Point2D<int> midPt;
      double dist = pointDistOnLine(sp,ep,loc,midPt);
      if(dist < minDist ||minDist == -1) 
      {
  minDist = dist;
  edgeIdx = i;
  ltrav = lineDist(sp,midPt)/lineDist(sp,ep); 
      }
    }

  uint ii;
  return getEdge(edgeIdx, ltrav, sdist, edist, ii);
}

// Find nearest node label
// ######################################################################
std::string TopologicalMap::getNodeLabel (Point2D<int> loc)
{
  // can be used to implement conversion from Point2D<int> to (snum,ltrav)
  // draw the edges first
  double minDist = -1;
  std::string label = "";
  for(uint i = 0; i < itsGraph->getNumEdge(); i++)
    {
      Point2D<int> sp = itsGraph->getEdge(i)->getSourceNode()->getCoordinate();
      Point2D<int> ep = itsGraph->getEdge(i)->getDestNode()->getCoordinate();
      Point2D<int> midPt;
      double dist = pointDistOnLine(sp,ep,loc,midPt);
      if(dist < minDist ||minDist == -1) 
      {
        minDist = dist;
        //min pt to start is farer than end pt
        if(lineDist(sp,midPt) > lineDist(midPt,ep)) 
          label = std::string(itsGraph->getEdge(i)->getDestNode()->getLabel());
        else
          label = std::string(itsGraph->getEdge(i)->getSourceNode()->getLabel());
      }
    }

  return label; 
}
// ######################################################################
double TopologicalMap::getSegmentDistance
(uint sseg, double ltrav, uint dseg, uint &intIndex)
{
  ASSERT(sseg < itsSegmentList.size() &&
         dseg < itsSegmentList.size() && ltrav <= 1.0F);
  //LINFO(" [%d to %d]", sseg, dseg);

  // if the point is in the same segment
  // the closest node is an invalid value
  if(sseg == dseg) { intIndex = itsGraph->getNumNode(); return 0.0F; }

  // get the edge of the starting point
  double slen = getSegmentLength(sseg);
  double altrav = ltrav * slen;
  double clen = altrav; uint i = 0;
  //LINFO("slen: %f, altrav: %f",slen, altrav);

  while(clen > 0.0F)
    {
      double len = itsSegmentList[sseg][i]->getCost();
      if(len > clen)
        {
          uint sind = itsSegmentList[sseg][i]->getSourceNode()->getIndex();
          uint eind = itsSegmentList[sseg][i]->getDestNode()->getIndex();

          double sdist =
            itsNodeSegmentDistanceMatrix[Point2D<int>(sind,dseg)] + clen;
          double edist =
            itsNodeSegmentDistanceMatrix[Point2D<int>(eind,dseg)] + len - clen;

          //LINFO("2 sind: (%d  %f), eind: (%d %f)", sind, sdist, eind, edist);

          if(sdist > edist)
            { intIndex = eind; return edist; }
          else
            { intIndex = sind; return sdist; }
        }
      else if(len == clen)
        {
          uint eind = itsSegmentList[sseg][i]->getDestNode()->getIndex();
          double edist = itsNodeSegmentDistanceMatrix[Point2D<int>(eind,dseg)];

          //LINFO("1 eind: (%d %f)", eind, edist);
          intIndex = eind;
          return edist;
        }
      else{ clen -= len; i++; }
      //LINFO("len: %f, clen: %f", len, clen);
    }

  // the point is on the source node of the first edge on the list
  uint sind = itsSegmentList[sseg][0]->getSourceNode()->getIndex();
  double sdist = itsNodeSegmentDistanceMatrix[Point2D<int>(sind,dseg)];
  intIndex = sind;
  //LINFO("0 sind: (%d  %f)", sind, sdist);

  return sdist;
}

// ######################################################################
void TopologicalMap::setSegmentLength()
{
  // for each segment
  for(uint i = 0; i < itsSegmentList.size(); i++)
    {
      // add up the segment length (cost)
      itsSegmentLength[i] = 0.0F;
      uint nsedge = itsSegmentList[i].size();
      for(uint j = 0; j < nsedge; j++)
        itsSegmentLength[i] += itsSegmentList[i][j]->getCost();
    }
}

// ######################################################################
double TopologicalMap::getSegmentLength(uint index)
{
  ASSERT(index < itsSegmentLength.size());
  return itsSegmentLength[index];
}

// ######################################################################
double TopologicalMap::getNodeSegmentMaxDistance()
{
  double min, max;
  getMinMax(itsNodeSegmentDistanceMatrix, min, max);
  if(min == -1.0F) return -1.0F;
  else return max;
}

// ######################################################################
Point2D<float> TopologicalMap::getLocationFloat(uint cseg, double ltrav)
{
  LDEBUG("csg %d ssize %d, ltrav %f",cseg,(int)itsSegmentList.size(),ltrav);
  ASSERT((cseg < itsSegmentList.size()) &&
         (ltrav <= 1.0F) && (ltrav >= 0.0F));

  // using the segment we can get its list of edges
  double slen = getSegmentLength(cseg);
  double altrav = ltrav * slen;
  LDEBUG("%d: %f * %f = altrav: %f", cseg, ltrav, slen, altrav);

  // iterate through the list of edges
  double cltrav = 0.0F;
  uint nsedge = itsSegmentList[cseg].size();
  if(nsedge == 0) { return Point2D<float>(-1.0F,-1.0F); }
  uint j = 0;
  while((j < nsedge) && (altrav >= cltrav))
    { cltrav += itsSegmentList[cseg][j]->getCost(); j++; }
  LDEBUG("nsedge: %d -> j: %d", nsedge, j);

  // get the distance to the next node
  double rdist = cltrav - altrav;
  double dist = itsSegmentList[cseg][j-1]->getCost();
  Point2D<int> sloc =
    itsSegmentList[cseg][j-1]->getSourceNode()->getCoordinate();
  Point2D<int> eloc =
    itsSegmentList[cseg][j-1]->getDestNode()->getCoordinate();
  LDEBUG("%f/%f (%d,%d) -> (%d,%d)",
        rdist, dist, sloc.i,sloc.j, eloc.i, eloc.j);

  double x = eloc.i - rdist/dist*(eloc.i - sloc.i);
  double y = eloc.j - rdist/dist*(eloc.j - sloc.j);

  return Point2D<float>(x,y);
}

// ######################################################################
Point2D<int> TopologicalMap::getLocation(uint cseg, double ltrav)
{
  Point2D<float> p = getLocationFloat(cseg, ltrav);
  uint iloc = uint(p.i + 0.5F);
  uint jloc = uint(p.j + 0.5F);
  LDEBUG("(%d, %f) -> %d, %d", cseg, ltrav, iloc, jloc);
  return Point2D<int>(iloc, jloc);
}

// ######################################################################
void TopologicalMap::getLocation(Point2D<int> loc, uint &cseg, double &ltrav)
{
  // use get edge (also not yet implemented)
  // to implement conversion from Point2D<int> to (snum,ltrav)

  // check if the point is within the map
  int edgeIdx = 0;
  double minDist = -1;
  double edge_dist = 0.0;
  ltrav = 0.0F;
  for(uint i = 0; i < itsGraph->getNumEdge(); i++)
    {
      Point2D<int> sp = 
        itsGraph->getEdge(i)->getSourceNode()->getCoordinate();
      Point2D<int> ep = 
        itsGraph->getEdge(i)->getDestNode()->getCoordinate();
      Point2D<int> midPt;
      double dist = pointDistOnLine(sp,ep,loc,midPt);
      if(dist < minDist || minDist == -1) 
      {
        minDist = dist;
        edgeIdx = i;
        edge_dist = lineDist(sp,midPt); 
      }
    }
  //once we found the edges and midpoint, convert to seg and ltrav
  getEdgeStartLocation (edgeIdx, cseg, ltrav,edge_dist);
  
  return;

  //total length is the length before closest edge and portions of closest edge
  LINFO("minDist %f -> edge %2d, seg %2d,ltrav %4.2f",minDist,edgeIdx,cseg,ltrav);
}

// ######################################################################
uint TopologicalMap::getSegmentLocation(Point2D<int> loc)
{
  // use get edge (also not yet implemented)
  // to implement conversion from Point2D<int> to (snum,ltrav)

  LFATAL("NOT YET implemented");

  return 0;
}

// ######################################################################
double TopologicalMap::getSegmentLengthTraveled(Point2D<int> loc)
{
  // use get edge (also not yet implemented)
  // to implement conversion from Point2D<int> to (snum,ltrav)

  LFATAL("NOT YET implemented");
  return 0.0;
}

// ######################################################################
std::vector<rutz::shared_ptr<Node> >
TopologicalMap::getNodesInInterval(uint index, double fltrav, double lltrav)
{
  ASSERT(index < itsSegmentLength.size());
  ASSERT(fltrav >= 0.0 && fltrav <= 1.0);
  ASSERT(lltrav >= 0.0 && lltrav <= 1.0);

  double slen = getSegmentLength(index);
  std::vector<rutz::shared_ptr<Node> > res;
  double caltrav = 0.0;

  // go through the edges in the segment
  uint ssize = itsSegmentList[index].size();
  for(uint i = 0; i < ssize; i++)
    {
      // check if the source node is within the interval
      double ltrav = caltrav/slen;
      LDEBUG("%d: %f * %f = altrav: %f", index, ltrav, slen, caltrav);

      if(ltrav >= fltrav && ltrav <= lltrav)
        res.push_back(itsSegmentList[index][i]->getSourceNode());
      else if(ltrav > lltrav) return res;

      caltrav += itsSegmentList[index][i]->getCost();
    }

  if (lltrav == 1.0)
    res.push_back(itsSegmentList[index][ssize-1]->getDestNode());

  return res;
}

// ######################################################################
std::vector<std::pair<uint,float> >
TopologicalMap::getNodeLocationsInInterval
(uint index, double fltrav, double lltrav)
{
  ASSERT(index < itsSegmentLength.size());
  ASSERT(fltrav >= 0.0 && fltrav <= 1.0);
  ASSERT(lltrav >= 0.0 && lltrav <= 1.0);

  double slen = getSegmentLength(index);
  std::vector<std::pair<uint,float> > res;
  double caltrav = 0.0;

  // go through the edges in the segment
  uint ssize = itsSegmentList[index].size();
  for(uint i = 0; i < ssize; i++)
    {
      // check if the source node is within the interval
      double ltrav = caltrav/slen;
      LDEBUG("%d: %f * %f = caltrav: %f", index, ltrav, slen, caltrav);

      if(ltrav >= fltrav && ltrav <= lltrav)
        res.push_back(std::pair<uint,double>(index, ltrav));
      else if(ltrav > lltrav) return res;

      caltrav += itsSegmentList[index][i]->getCost();
    }

  if (lltrav == 1.0F)
    res.push_back(std::pair<uint,double>(index, 1.0F));

  return res;
}

// ######################################################################
double TopologicalMap::getAngle
(rutz::shared_ptr<Edge> e1, rutz::shared_ptr<Edge> e2)
{
  return itsGraph->getAngle(e1, e2);
}

// ######################################################################
void TopologicalMap::getEdgeStartLocation
(uint edge_index, uint &cseg, double &ltrav, double edge_dist)
{
  // get the segment number and length
  cseg = itsEdgeSegmentNum[edge_index];
  double slen = getSegmentLength(cseg);
  double caltrav = edge_dist;  ltrav = caltrav/slen;

  // go through the edges in the segment
  uint ssize = itsSegmentEdgeIndexList[cseg].size();

  LDEBUG("edge %d,seg %d,seg size %d,seglen %f,caltrav %f",
    edge_index,cseg,ssize,slen,caltrav);

  for(uint i = 0; i < ssize; i++)
    {
      uint e_index = itsSegmentEdgeIndexList[cseg][i];

      if(e_index == edge_index) { return; } 

      // accumulate length traveled      
      caltrav += itsSegmentList[cseg][i]->getCost();
      ltrav = caltrav/slen;

      //prevent 1.0000001 problem
      if(ltrav > 1.0F) ltrav = 1.0F;
      LDEBUG("%d: %f * %f = caltrav: %f", i, ltrav, slen, caltrav);
    }

  // this is 
  ltrav = -1.0F;
}
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
