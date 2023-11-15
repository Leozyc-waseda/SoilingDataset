/*!@file MBARI/mbariFunctions.H a few functions used by MBARI programs
 */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   //
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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/mbariFunctions.H $
// $Id: mbariFunctions.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef MBARIFUNCTIONS_DEFINED
#define MBARIFUNCTIONS_DEFINED

#include "Util/Types.H"
#include <list>
class Token;
template <class T> class Point2D;
class MbariResultViewer;
class Brain;
class BitObject;
class Rectangle;
class SimEventQueue;
template <class T> class PixRGB;
template <class T> class Image;
namespace rutz { template <class T> class shared_ptr; }
namespace nub { template <class T> class soft_ref; }

//! extract a set of BitObjects from bitImg, which intersect region
/*! bitImg is flooded starting from each point within region,
  and each BitObject exceeding the minSize is stored in the
  list of BitObjects that is returned. */
std::list<BitObject> extractBitObjects(const Image<byte>& bitImg,
                                       Rectangle region,
                                       int minSize = 0);

std::list<BitObject> getLargestObjects(const Image<byte>& bImg,
                                       Rectangle region,
                                       int numObjects,
                                       int minSize);

//! returns the BitObjects at the most salient points
/*! @param brain the brain object used for the saliency computation
  @param img the image scanned for salient locations
  @param bitImg a binary image used for extracting BitObjects at the
  salient locations
  @param maxEvolveTime the maximum evolution time for the saliency map
  @param maxNumSalSpots the maximum number of salient locations extracted
  @param the minimum size of a BitObject to be considered*/
std::list<BitObject> getSalRegions(nub::soft_ref<Brain> brain,
                                   nub::soft_ref<SimEventQueue> q,
                                   const Image< PixRGB<byte> >& img,
                                   const Image<byte>& bitImg,
                                   float maxEvolveTime,
                                   int maxNumSalSpots,
                                   int minSize = 0);

//! all BitObjects in objs are drawninto the return image
Image<byte> showAllObjects(std::list<BitObject>& objs);

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
