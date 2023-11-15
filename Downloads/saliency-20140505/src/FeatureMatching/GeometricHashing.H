/*!@file SceneUnderstanding/GeometricHashing.H  Shape from shading */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/FeatureMatching/GeometricHashing.H $
// $Id: GeometricHashing.H 12962 2010-03-06 02:13:53Z irock $
//

#ifndef GeometricHashing_H_DEFINED
#define GeometricHashing_H_DEFINED

#include "Util/Types.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ImageSet.H"
#include "Image/Point2D.H"
#include "Image/Point3D.H"

#include <vector>

class GeometricHashing
{
public:

  struct Model
  {
    int id;
    std::vector<Point2D<int> > v;
    Point3D<float> rot;
  };

  struct ModelTableEntry
  {
    int modelId;
    Point3D<float> rot;
    Point2D<float> featureLoc;
    uint basisP1;
    uint basisP2;
  };

  struct Acc
  {
    uint P1;
    uint P2;
    uint inputP1;
    uint inputP2;
    int modelId;
    Point3D<float> rot;
    Point2D<float> center;
    int votes;
  };

  struct AccCmp
  {
    bool operator()(const Acc& c1, const Acc& c2)
    {
      return c1.votes > c2.votes;
    }
  };


  struct TableEntry
  {
    std::vector<ModelTableEntry> modelEntries;
  };



  GeometricHashing();

  //! Destructor
  virtual ~GeometricHashing();

  //! Add a model to search for
  void addModel(const Model& model);

  //! Change basis based on p1 and p2
  //First scale and rotate the model to p1p2
  //the center the model to the cetner of p1p2
  std::vector<Point2D<float> > changeBasis(
      const std::vector<Point2D<int> >& featureLoc,
      int p1, int p2);

  std::vector<Point2D<float> > changeBasis(
      const std::vector<Point2D<int> >& featureLoc,
      Point2D<int> tl, Point2D<int> br);

  //! Add a feature to hash table
  void insertToHashTable(const Point2D<float> loc,
      int p1, int p2, int modelId, Point3D<float> rot);

  //! Find features in the hash table
  Image<TableEntry>::iterator findInHash(const Point2D<float>& loc);

  //! Get votes for a given input
  std::vector<Acc> getVotes(std::vector<Point2D<int> >& input);


  //!Show the hash table
  Image<PixRGB<byte> > getHashTableImage();

  //! Write hash table to disk
  void writeTable(const char* filename);

  //! Read table from disk
  void readTable(const char* filename);

private:
  std::vector<Model> itsModels; //A list of models
  Image<TableEntry> itsHashTable;
  float itsTableWidth;
  int itsNumFeatures;
  int itsNumBinEntries;


};


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
