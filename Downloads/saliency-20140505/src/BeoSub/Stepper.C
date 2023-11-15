/*!@file BeoSub/Stepper.C Basic Stepper Motor Advancing for Marker Dropper for the BeoSub */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
//
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/Stepper.C $
// $Id: Stepper.C 6414 2006-04-04 00:37:40Z ilab15 $

#ifdef HAVE_LINUX_PARPORT_H

#include "BeoSub/Stepper.H"


Stepper::Stepper(nub::soft_ref<ParPort> p)
{
/* TODO: WARNING! huge big assumption that the stepper motor is hardwired to pin 6/7 of parallel port!! */
  /* TODO: mask and gray code should be made constants! */
  mask = 0xe0;
  graystepper[0] = 0xa0;
  graystepper[1] = 0xe0;
  graystepper[2] = 0xc0;
  graystepper[3] = 0x80;
  step=0;

  itsStep = p;
}

Stepper::~Stepper()
{
}

void Stepper::Step(int totalSteps, long delay)
{
  if(totalSteps>0)
    {
      for(;totalSteps>0;totalSteps--) {
        itsStep->WriteData(mask,graystepper[step]);
        step = ((step == 0) ? 3 : step-1);
        usleep(delay);
      }
    } else if(totalSteps<0) {
      for(;totalSteps<0;totalSteps++) {
        itsStep->WriteData(mask,graystepper[step]);
        step = (step+1) % 4;
        usleep(delay);
      }
    }
}

#endif // HAVE_LINUX_PARPORT_H
