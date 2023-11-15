/*!@file FeatureMatching/PGH.H  Pairwise Geometric Histograms */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/FeatureMatching/PGH.H $
// $Id: PGH.H 12979 2010-03-07 17:10:23Z lior $
//

#ifndef PGH_H_DEFINED
#define PGH_H_DEFINED

#include "Util/Types.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ImageSet.H"
#include "Image/Point2D.H"
#include "Image/Point3D.H"

#include <vector>
#include <map>

class PGH
{
public:

  struct Line
  {
    Point2D<int> pos;
    double length;
    double ori;

    Line(const Point2D<int> p,const double l,const double o) :
      pos(p), length(l), ori(o)
    {}
  };

  struct ModelInfo
  {
    int id;
    std::vector<Line> lines;
    Image<float> histogram;
  };


  PGH();

  //! Destructor
  virtual ~PGH();

  //! Add a model to search for
  void addModel(const std::vector<Line>& lines, int id);

  //! Create a geometric historgram
  Image<float>  getGeometricHistogram(const std::vector<Line>& lines);
 
  //! Find the best match model
  int matchModel(const std::vector<Line>& lines);
  
  //! Compage two histograms
  double cmpHist(const Image<float>& hist1, const Image<float>& hist2);
  
private:

  std::vector<ModelInfo> itsModels;

};


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
