/*!@file Beosub/BeeBrain/test-ForwardVision.C                           */
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/test-ForwardVision.C $
// $Id: test-ForwardVision.C 8623 2007-07-25 17:57:51Z rjpeters $

#include "BeoSub/BeeBrain/Globals.H"

#include "BeoSub/BeeBrain/ForwardVision.H"
#include "BeoSub/BeeBrain/OceanObject.H"

int main( int argc, const char* argv[] )
{
  ForwardVisionAgent fv = ForwardVisionAgent("ForwardVisionAgent");


  fv.msgFindAndTrackObject(1, OceanObject::CROSS, POSITION);

  std::cout<<"Num of Jobs: "<<fv.getNumJobs()<<std::endl;

  fv.msgFindAndTrackObject(2, OceanObject::BIN, POSITION);

  std::cout<<"Num of Jobs: "<<fv.getNumJobs()<<std::endl;

  fv.msgFindAndTrackObject(3, OceanObject::PIPE, POSITION);

  std::cout<<"Num of Jobs: "<<fv.getNumJobs()<<std::endl;

  fv.msgFindAndTrackObject(4, OceanObject::BUOY, FREQUENCY);

  std::cout<<"Num of Jobs: "<<fv.getNumJobs()<<std::endl;

  fv.msgFindAndTrackObject(4, OceanObject::BUOY, POSITION);

  std::cout<<"Num of Jobs: "<<fv.getNumJobs()<<std::endl;

  fv.msgFindAndTrackObject(4, OceanObject::BUOY, MASS);

  std::cout<<"Num of Jobs: "<<fv.getNumJobs()<<std::endl;

  uint schedulerCount = 0;
  while(schedulerCount++ < 3)
    {
      fv.pickAndExecuteAnAction();
    }

  fv.msgStopLookingForObject(1, POSITION);
  fv.msgStopLookingForObject(2, POSITION);
  fv.msgStopLookingForObject(3, POSITION);
  fv.msgStopLookingForObject(4, FREQUENCY);
  fv.msgStopLookingForObject(4, POSITION);
  fv.msgStopLookingForObject(4, MASS);

  //should not loop
  while(fv.pickAndExecuteAnAction());

  std::cout<<"Num of Jobs: "<<fv.getNumJobs()<<std::endl;

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

