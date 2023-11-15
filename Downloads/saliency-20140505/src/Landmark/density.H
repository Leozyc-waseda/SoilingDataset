/*! @file Landmark/density.H [put description here] */

// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Landmark/density.H $
// $Id: density.H 6410 2006-04-01 22:12:24Z rjpeters $

// find the density plot: treat each object as 1 pt in 2d space
// not considering sigma yet

#include <stdio.h>
#include <map>
#include <string>

#include "Image/Image.H"


class Object
{
public:

  std::string name;   // name of the object
  int tag;       // tag: order in list of objects
  double mu1;    // mean of finest scale
  double sigma1; // sigma of finest scale

  // constructor
  Object(std::string name, int tag, double mu1, double sigma)
  {
    this->name = name;
    this->tag = tag;
    this->mu1 = mu1;
    this->sigma1 = sigma1;

  }

};

//*****************************************************

Image<float> density(const char* filename, std::map<int, Object*>& list);
