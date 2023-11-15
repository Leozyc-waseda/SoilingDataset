/*!@file test-sc8000.C Test SC-8000 serial servo controller */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-sc8000.C $
// $Id: test-sc8000.C 7912 2007-02-14 21:12:44Z rjpeters $
//

#include "Component/ModelManager.H"
#include "Devices/sc8000.H"
#include "Util/log.H"

#include <stdio.h>

int main(int argc, const char **argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Test Model for SSC Class");

  // Instantiate our various ModelComponents:
  nub::soft_ref<SC8000> sc8000(new SC8000(manager));
  manager.addSubComponent(sc8000);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv,
                               "<servo> <position> <1 for calibrate movment>", 2, 3) == false) return(1);

  // let's get all our ModelComponent instances started:
  manager.start();

  // Let's get some of the option values, to configure our window:
  int sernum = manager.getExtraArgAs<int>(0);

  if (manager.numExtraArgs() > 2){
            //calibrate movment
          const char* pos = manager.getExtraArg(1).c_str();

          float serpos;
          if (*pos == 'N')
                  serpos = atof(pos+1)*-1;
          else
                  serpos = atof(pos);

          sc8000->calibrate(1, 13650, 10800, 16000);
          sc8000->calibrate(3, 14000, 12000, 16000);

          sc8000->move(sernum, serpos);
          LINFO("Moved servo %d to position %f", sernum, serpos);
          for (int i=1; i<3; i++)
                  LINFO("Servo %d is at position %f", i, sc8000->getPosition(i));
  } else {
          // command the ssc
          int serpos = manager.getExtraArgAs<int>(1);
          sc8000->moveRaw(sernum, serpos);
          LINFO("Moved servo %d to raw position %d", sernum, serpos);
  }


  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
