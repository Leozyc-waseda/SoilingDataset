/*! @file Landmark/Tree.H [put description here] */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Landmark/Tree.H $
// $Id: Tree.H 9412 2008-03-10 23:10:15Z farhan $

// Program to implement a tree type structure

#include "Image/Image.H"
#include "Image/Point2D.H"

#include <vector>

template <class T> class Image;
template <class T> class PixRGB;

//###############################################################

class Tree
{

public:

  struct Node
  {
    Node* parent;
    Point2D<int> loc;
    double height;
    std::vector<Node*> children;
  }*node;

  //public:

  Tree(Point2D<int> loc, double height);
  void search(int n, int& found, Node* &parent);
  void insert(Point2D<int> loc, double height);
  void traverse(Image<float>& input, Image<PixRGB<byte> >& output);
  void mergeTrees(std::vector<Tree* > trees);
  void mergeNodes(std::vector<Node* > nodes);
  void bfs(Node* root, Image<float>& input, Image<PixRGB<byte> >& output,
           int& idx);
  void search(Point2D<int> loc);
};

//###############################################################
