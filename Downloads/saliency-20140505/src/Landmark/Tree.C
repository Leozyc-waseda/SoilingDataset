/*! @file Landmark/Tree.C [put description here] */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Landmark/Tree.C $
// $Id: Tree.C 9412 2008-03-10 23:10:15Z farhan $

#include "Landmark/Tree.H"

#include "Image/DrawOps.H"
#include "Image/Pixels.H"

Tree::Tree(Point2D<int> loc, double height)
{
  node = new Node;

  node->parent = NULL;
  node->loc = loc;
  node->height = height;


}

//###############################################################

void Tree::insert(Point2D<int> loc, double height)
{
  Tree child(loc, height);
  node->children.push_back(child.node);

}

//###############################################################

void Tree:: mergeTrees(std::vector<Tree* > trees)
{
  std::vector<Node* > nodes;
  for(uint i = 0; i < trees.size(); i++)
    nodes.push_back(trees[i]->node);

  mergeNodes(nodes);
}

//###############################################################

void Tree::mergeNodes(std::vector<Node* > nodes)
{
  for(uint i = 0; i < nodes.size();  i++)
    node->children.push_back(nodes[i]);

}

//###############################################################

void Tree::traverse(Image<float>& input, Image<PixRGB<byte> >& output)
{
  //std::cout<<"\n"<< "root = (" << node->loc.i << "," << node->loc.j << ")";
  drawCircle(output, node->loc, 2, PixRGB<byte>(255, 255, 0), 3);
  int idx = 0;
  bfs(node, input, output, idx);
}

//###############################################################

void Tree::bfs(Node* q, Image<float>& input, Image<PixRGB<byte> >& output,
               int& idx)
{
  PixRGB<byte> col,
    red(255, 0, 0),
    blue(0, 0, 255),
    green(0, 255, 0),
    yellow(255, 255, 0);
  switch(idx % 3)
    {
    case 0:
      col = red;
      break;
    case 1:
      col = green;
      break;
    case 2:
      col = blue;
      break;
    }
  idx++;

  // print q's children
  int count = q->children.size();
  for (int i = 0; i < count; i++)
    {
      //std::cout<<"\t"<<"(" << q->children[i]->loc.i
      //               << "," << q->children[i]->loc.j << ")";
      drawCircle(output, q->children[i]->loc, 2, col, 1);
    }

  // recurse: do bfs on q's children
  for (int j = 0; j < count; j++)
    {
      drawCircle(output, node->loc, 3, yellow, 3);
      bfs(q->children[j], input, output, idx);
    }
}

//###############################################################
