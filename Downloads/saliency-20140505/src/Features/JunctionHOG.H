/*!@file Features/JunctionHOG.H  */


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
// Primary maintainer for this file: Daniel Parks <danielfp@usc.edu>
// $HeadURL$
// $Id$
//

#ifndef JunctionHOG_H_DEFINED
#define JunctionHOG_H_DEFINED

#include "Util/Types.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ImageSet.H"
#include "Features/HistogramOfGradients.H"
#include "SIFT/FeatureVector.H"

#include <vector>
#include <string>


class JunctionHOG : public HistogramOfGradients
{
public:
  //! Default Constructor
  JunctionHOG();

  JunctionHOG(bool normalize, Dims cellDims = Dims(8,8), bool fixedCells=false, int numOrientations=9, bool signedOrient=false, int numContinuityBins=3, float neighborDistance=1.2F);

  //! Destructor
  virtual ~JunctionHOG();

  void initializeNeighbors(std::vector<Point2D<int> >& neighbors, std::vector<float>& perpAngles, std::vector<float>& parallelAngles, float angle);

  //! Calculate histogram using a gradient
  virtual std::vector<float> createHistogramFromGradient(const Image<float>& gradmag, const Image<float>& gradang);

  //! Calculate junction continuity histogram using the gradient
  std::vector<float> calculateJunctionHistogram(Image<float> gradmag, Image<float> gradang);

protected:
  int itsContinuityBins;
  float itsNeighborDistance;
private:
  // Parallel/Perpendicular Matching is treated as an unsigned process
  // Create fixed mask for each orientation
  std::vector<Point2D<int> > itsRelevantNeighbors[8];
  // Orientation angle and tolerance for a particular neighbor that are considered perpendicular
  std::vector<float> itsPerpAngles[8];
  // Orientation angle and tolerance for a particular neighbor that are considered parallel
  std::vector<float> itsParallelAngles[8]; 

};


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
