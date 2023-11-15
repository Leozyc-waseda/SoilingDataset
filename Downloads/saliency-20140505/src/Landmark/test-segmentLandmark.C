/*! @file Landmark/test-segmentLandmark.C [put description here] */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   //
// by the University of Southern California (USC) and the iLab at USC.  //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; filed July 23, 2001, following provisional applications     //
// No. 60/274,674 filed March 8, 2001 and 60/288,724 filed May 4, 2001).//
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
// Primary maintainer for this file: Vidhya Navalpakkam <navalpak@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Landmark/test-segmentLandmark.C $
// $Id: test-segmentLandmark.C 15310 2012-06-01 02:29:24Z itti $
//


#include "Image/ColorOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/ImageSetOps.H"
#include "Image/MathOps.H"    // for binaryReverse()
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H" // for learningCoeff()
#include "Image/fancynorm.H"
#include "Landmark/Tree.H"
#include "Landmark/density.H"
#include "Raster/Raster.H"
#include "Util/Timer.H"
#include "Util/log.H"

#include <cstdio>
#include <iostream>
#include <map>
#include <vector>

int countClusters(Image<float> input, float thresh)
{
  // get image details
  int w = input.getWidth();
  int h = input.getHeight();

  int count = 0;
  for(int i = 0; i < w; i++)
    for(int j = 0; j < h; j++)
      {
        if(input.getVal(i,j) > thresh)
          {
            // flood starting at seed upto thresh
            Image<float> output;
            flood(input, output, Point2D<int>(i,j), thresh, 255.0f);
            input -= output;
            inplaceClamp(input, 0.0f, 255.0f);
            count++;
          }
      }
  return count;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int main (int argc, char **argv)
{
  if (argc != 3)
    {
      std::cerr << "usage: test-segmentLandmark <option> <weight>\n\n \
                    where <option> is f (flat) or h (heirarchy)\n \
                    where <weight> is y (weight maps) or n (use all maps)\n";
      exit(2);
    }
  std::map<int, Object*> list;              // list of all objects

  // to initialise the edges, invoke density once
  Image<byte> img = density("feature1.input", list);
  int global_edge[list.size()][list.size()];       // clique of objects
  for(uint i = 0; i < list.size(); i++)
    for(uint j = 0; j < list.size(); j++)
      global_edge[i][j] = 0;

  int edge[list.size()][list.size()][42];
  for(uint i = 0; i < list.size(); i++)
    for(uint j = 0; j < list.size(); j++)
      for(uint k = 0; k < 42; k++)
        edge[i][j][k] = 0;

  //################### FIND CLUSTERS IN ALL FEATURES ###################

  for(int features = 1; features < 42; features++)
    {
      char filename[256];
      sprintf(filename, "feature%d.input", features);
      list.clear();
      Image<byte> img = density(filename, list);
      Raster::WriteGray(img, sformat("density_%d.pgm",features));

      Image<float> input = (Image<float>)(img);
      input = lowPass9(input);
      PixRGB<float> color(1, 0, 0);
      Raster::WriteRGB( stain(input, color),"input.ppm");

      //inplaceNormalize(input, 0.0f, 255.0f);
      int targets = countClusters(input, 0.0f);
      LDEBUG("# of targets = %d", targets);

      // statistics of the input image
      const int w = input.getWidth();
      const int h = input.getHeight();
      float min, max;
      getMinMax(input, min, max);

      // analyze the options
      bool flat = false, heirarchy = false;
      if(strcmp(argv[1],"f") == 0)
        {
          flat = true;
          std::cout<<"just get the members of each cluster"<<std::endl;
        }
      else if(strcmp(argv[1], "h") == 0)
        {
          heirarchy = true;
          std::cout<<"get the complete heirarchy"<<std::endl;
        }
      else
        std::cout<<"unrecognized option!";

      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      // maintain the forest of trees or islands
      std::multimap<float, Tree*, std::greater<float> > forest;


      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      // find the local maxima and store in a map (val, locn)

      std::multimap<float, Point2D<int>, std::greater<float> > maxima;
      int lm_num = 0;  // num of local maxima or peaks

      // we want to detect quickly local maxes, but avoid getting local mins
      float thresh = (min + (max - min) / 10.0);
      // this is a 1d image
      for (int index = 1; index < w - 1; index ++)
        {
          float val = input.getVal(index, 0);
          if (val >= thresh &&
              val >= input.getVal(index - 1 , 0) &&
              val >= input.getVal(index + 1, 0))
            {
              // insert into map
              Point2D<int> p(index, 0);
              maxima.insert(std::pair<float, Point2D<int> >(val, p));
              lm_num ++;
            }
        }

      LDEBUG(" # of local maxima = %" ZU "", maxima.size());

      if(maxima.size() == 0)
        {
          LERROR(" there r no maxima in this image!");
          break;
        }
      // to consider every peak above the ground level, insert a new member
      std::multimap<float, Point2D<int>, std::greater<float> >::iterator
        min_peak = maxima.end();
      min_peak --;
      maxima.insert(std::pair<float, Point2D<int> >(min_peak->first / 2,
                                              Point2D<int>(0, 0)));


      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      // choose one local maxima at a time and flood upto the next local maxima

      std::multimap<float, Point2D<int>, std::greater<float> >::iterator itr;

      int iter = 0;
      Image< PixRGB<float> > cat;
      for( itr = maxima.begin(); itr != maxima.end(); iter++)
        {
          float thresh;

          if(heirarchy)
            {
              thresh = itr->first;
            }

          else if(flat)
            {
              // add all members to the forest except the last
              std::multimap<float, Point2D<int>, std::greater<float> >::iterator
                last = maxima.end();
              last --;
              while( itr != last)
                {
                  Tree* new_peak = new Tree(itr->second, itr->first);
                  forest.insert(std::pair<float, Tree*>(itr->first, new_peak));
                  itr++;
                }
              // set thresh to the height of the last member
              thresh = last->first;
            }

          else break;

          Image<float> output = highThresh( input, thresh, 255.0f);

          // consider the maxima that are equal;
          // advance to the next lowest maxima
          while (itr != maxima.end() && itr->first == thresh )
            {
              // this peak is the start of a potential island: add a new tree
              Tree* new_peak = new Tree(itr->second, itr->first);
              forest.insert(std::pair<float, Tree*>(thresh, new_peak));
              itr++;

            }

          // recompute clusters: brute force

          cat.clear(PixRGB<float>(0.0f)); // an image containing all clusters
          int count = 0;

          // ################## FIND CLUSTERS ############################
          // now find all the connected components in the forest

          std::multimap<float, Tree*, std::greater<float> >::iterator tree,
            curr_peak;
          for(tree = forest.begin(); tree != forest.end(); )
            {
              Tree* current = tree->second;

              Image<float> output;
              Point2D<int> seed = current->node->loc;
              LDEBUG("current peak ======== (%d, %d)",
                     seed.i, seed.j);
              flood(input, output, seed, thresh, -255.0f);

              // find all member peaks of the current cluster
              std::vector<Tree* > to_merge;
              std::multimap<float, Tree*, std::greater<float> >::iterator
                member = forest.begin();
              while(member != forest.end())
                {
                  for(member = forest.begin(); member != forest.end();
                      member++)
                    {
                      if(output.getVal(member->second->node->loc) == -255.0f &&
                         member->second->node->loc != seed)
                        {
                          // this peak belongs to the cluster
                          to_merge.push_back(member->second);
                          /* erase it from the forest since it will be part of
                             the new merged tree
                          */
                          LDEBUG(" --------- (%d, %d)",
                                 member->second->node->loc.i,
                                 member->second->node->loc.j);
                          /*
                            FIXME: why does forest.erase give problems?
                          */
                          forest.erase(member);
                          break;
                        }
                    }
                }

              if(to_merge.size() > 0)
                {
                  // merge the current tree with this member of the forest
                  to_merge.push_back(current);

                  Tree* new_tree = new Tree(seed, current->node->height);
                  new_tree->mergeTrees(to_merge);

                  // erase the old tree
                  float height = tree->first;
                  forest.erase(tree);

                  // add the new merged tree to the forest
                  current = new_tree;
                  tree = forest.insert(std::pair<float, Tree*>
                                       (height, current));
                  tree++; // advance to the next tree in the forest
                }

              else
                {
                  // i am a loner:(
                  to_merge.push_back(current);
                  tree++; // advance to the next tree in the forest
                }

              std::vector<Object*> siblings;
              // find objects that belong to this cluster
              for(uint l = 0; l < list.size(); l++)
                {
                  if(output.getVal((int)list[l]->mu1, 0) == -255.0f)
                    {
                      siblings.push_back(list[l]);
                      //LDEBUG("sibling %s",list[l]->name.c_str());
                    }

                }

              // set edge weights for siblings
              int size = siblings.size();
              for(int k = 0; k < size; k++)
                for(int l = 0; l < size; l++)
                  {
                    edge[siblings[k]->tag][siblings[l]->tag][features]  += 1;
                    global_edge[siblings[k]->tag][siblings[l]->tag] += 1;
                    //LDEBUG("%d-%d :edge = %d, global_edge = %d",
                    //     siblings[k]->tag,siblings[l]->tag,
                    //     edge[siblings[k]->tag][siblings[l]->tag],
                    //     global_edge[siblings[k]->tag][siblings[l]->tag]);
                  }
              // reset the output (flooded image) to its normal state
              for(int k = 0; k < w; k ++)
                for(int l = 0; l < h; l ++)
                  if(output.getVal(k,l) == -255.0f)
                    {
                      output.setVal(k, l, input.getVal(k, l));
                    }

              //LDEBUG(" size of cluster = %d", to_merge.size());

              // draw this cluster
              Image<PixRGB<float> > cluster;
              if(to_merge.size() > 0)
                {
                  count++;
                  // generate colors of clusters
                  float red = 0.0f, green = 0.0f, blue = 0.0f;

                  if(count % 3 == 1)
                    {
                      red = 1.0f ; // / count;
                      green = 0.0f;
                      blue = 0.0f;
                      //LDEBUG("red....");
                    }
                  else if(count % 3 == 2)
                    {
                      green = 1.0f ; // / (count - 1);
                      blue = 0.0f;
                      red = 0.0f;
                      //LDEBUG("green...");
                    }
                  else if(count % 3 == 0)
                    {
                      blue = 1.0f ; // / (count - 2);
                      green = 0.0f;
                      red = 0.0f;
                      //LDEBUG("blue....");
                    }

                  PixRGB<float> color(red, green, blue);
                  //LDEBUG("color = (%f, %f, %f)", red, green, blue);
                  cluster =  stain(output, color);

                  if(cluster.initialized())
                    Raster::WriteRGB(cluster,sformat("clusters_%d.ppm", count-1));
                  else
                    LDEBUG("cluster not initialized!");

                  if(!cat.initialized())
                    cat = cluster;
                  else
                    cat += cluster;
                }

              // FIXME
              //tree ++;
            }

          if(cat.initialized())
            Raster::WriteRGB((Image< PixRGB<byte> >)cat,
                             sformat("iter_%d_%d.ppm",
                                     iter,
                                     features));
          else
            LDEBUG("iter not initialized!");
          //LDEBUG(" # of clusters = %d", count);
          //LDEBUG("\n****************************************\n");
        }


      std::multimap<float, Tree*, std::greater<float> >::iterator tree;
      Image<PixRGB<byte> > plot = (Image< PixRGB<byte> >)cat;
      for(tree = forest.begin(); tree != forest.end(); tree++)
        tree->second->traverse(input, plot);

      Raster::WriteRGB(plot,"plot.ppm");


      // print out all the siblings
      // use law of transitivity :-)
      std::map<int, Object*>::iterator tr, jtr, start, stop;
      std::map<int, Object*> bak = list;
      for(tr = bak.begin(); tr != bak.end(); tr++)
        {
          start = tr;
          stop = bak.end();
          LINFO("%s", tr->second->name.c_str());
          for(jtr = tr ; jtr != stop; jtr ++)
            {
              if(edge[tr->second->tag][jtr->second->tag][features] == 1
                 && tr != jtr)
                {
                  LINFO("%s", jtr->second->name.c_str());
                  bak.erase(jtr);
                }
            }
          LINFO("---------------------------");
        }

    }
  if(strcmp(argv[2], "n") == 0)
    {
      // dont weight the maps; just add them uniformly
      std::map<int, Object*>::iterator tr, jtr, start, stop;
      for(int conf = 42; conf > 0; conf -- )
        {
          // evaluate the clusters with decreasing order of confidence
          LDEBUG("CONFIDENCE LEVEL = %d", conf);
          std::map<int, Object*> bak = list;
          for(tr = bak.begin(); tr != bak.end(); tr++)
            {
              start = tr;
              stop = bak.end();
              LINFO("%s", tr->second->name.c_str());
              for(jtr = tr ; jtr != stop; jtr ++)
                {
                  if(global_edge[tr->second->tag][jtr->second->tag] >= conf
                     && tr != jtr)
                    {
                      LINFO("%s", jtr->second->name.c_str());
                      bak.erase(jtr);
                    }
                }
              LINFO("---------------------------");
            }
          LINFO("################################################");
        }
    }
  else if(strcmp(argv[2], "y") == 0)
    {
      // weight the maps according to the channels
      for(uint i = 0; i < list.size(); i++)
        for(uint j = 0; j < list.size(); j++)
          {
            global_edge[i][j] = 0;
          }
      // use weighting
      for(uint i = 0; i < list.size(); i++)
        for(uint j = 0; j < list.size(); j++)
          {
            // color : 2 (rg and by)
            for(uint k = 0; k < 12; k++)
              global_edge[i][j] += edge[i][j][k];
            global_edge[i][j] /= 2;

            // intensity : 1
            for(uint k = 12; k < 18; k++)
              global_edge[i][j] += edge[i][j][k];

            // orientation: 4
            for(uint k = 18; k < 42; k++)
              global_edge[i][j] += edge[i][j][k];
            global_edge[i][j] /= 4;
          }

      // hence, max edge weight = 18
      std::map<int, Object*>::iterator tr, jtr, start, stop;
      for(int conf = 18; conf > 0; conf -- )
        {
          // evaluate the clusters with decreasing order of confidence
          LDEBUG("CONFIDENCE LEVEL = %d", conf);
          std::map<int, Object*> bak = list;
          for(tr = bak.begin(); tr != bak.end(); tr++)
            {
              start = tr;
              stop = bak.end();
              LINFO("%s", tr->second->name.c_str());
              for(jtr = tr ; jtr != stop; jtr ++)
                {
                  if(global_edge[tr->second->tag][jtr->second->tag] >= conf
                     && tr != jtr)
                    {
                      LINFO("%s", jtr->second->name.c_str());
                      bak.erase(jtr);
                    }
                }
              LINFO("---------------------------");
            }
          LINFO("################################################");
        }
    }

}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
