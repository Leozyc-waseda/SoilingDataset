/*!@file BeoSub/test-MarkerDropper.C to test the Stepper Motor for the BeoSub */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-MarkerDropper.C $
// $Id: test-MarkerDropper.C 6599 2006-05-17 21:53:39Z rjpeters $
//


#include <stdlib.h>
#include "BeoSub/Stepper.H"
#include "Component/ModelManager.H"

int main(int argc, char *argv[])
        {
#ifndef HAVE_LINUX_PARPORT_H
                LFATAL("you must have <linux/parport.h> to use this program");
#else
                //std::cout<<"Enter the  Arguments"<<std::endl;
                int steprot = atoi(argv[1]);
                ModelManager manager("test-MarkerDropper");
                nub::ref<ParPort> p(new ParPort(manager));
                Stepper itsPoint(p);
                itsPoint.Step(steprot , 65000);
                return 0;
#endif
     }
