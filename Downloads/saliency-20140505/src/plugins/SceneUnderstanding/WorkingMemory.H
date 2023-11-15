/*!@file SceneUnderstanding/WorkingMemory.H  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/plugins/SceneUnderstanding/WorkingMemory.H $
// $Id: WorkingMemory.H 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef WORKINGMEMORY_H_DEFINED
#define WORKINGMEMORY_H_DEFINED

#include "Util/Types.H" // for uint
#include "Image/Image.H" // for uint
#include "Image/Pixels.H" // for uint
#include <vector>
#include <string>

class WorkingMemory
{
public:

  WorkingMemory();

  //! Destructor
  ~WorkingMemory();

  //clear our memory and set the size of the memory for the size of the image
  void setImage(Image<PixRGB<byte> > &img);

  //Associate a position in the image (with a given radius) with a memory representation class
  //Return true if we overwrote
  bool setMemory(Point2D<int> &loc, int val, int radius=1);



private:

  Image<float> itsMemory;
  Point2D<int> itsCurrentFoveaLoc;     //The currentFovea Location


};

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
