/*!@file AppDevices/test-ParPort.C */

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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-ParPort.C $
// $Id: test-ParPort.C 6795 2006-06-29 20:45:32Z rjpeters $
//

#ifndef APPDEVICES_TEST_PARPORT_C_DEFINED
#define APPDEVICES_TEST_PARPORT_C_DEFINED

#include "Component/ModelManager.H"
#include "Devices/ParPort.H"

int main(int argc, const char** argv)
{
#ifndef HAVE_LINUX_PARPORT_H
  LFATAL("you must have <linux/parport.h> to use this program");
#else
  ModelManager manager("test-ParPort");

  nub::ref<ParPort> pp(new ParPort(manager));
  manager.addSubComponent(pp);

  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false)
    return(1);

  manager.start();
  int flag = 0;
  while (true)
    {
      byte b = pp->ReadStatus();

      LINFO("b=%d %d",int(b), int(b & PARPORT_STATUS_PAPEROUT));
      LINFO("got PARPORT_STATUS_PAPEROUT");
      if ((b & PARPORT_STATUS_PAPEROUT) != 0)
        {
          LINFO("initial got PARPORT_STATUS_PAPEROUT");
      LINFO("b=%d %d",int(b), int(b & PARPORT_STATUS_PAPEROUT));
      flag = 1;
          while((b & PARPORT_STATUS_PAPEROUT) != 0)
            {
              b = pp->ReadStatus();

            }

          //for (int i = 0;i= 20; i++){
          // b = pp->ReadStatus();
          // LINFO("loop after %d   b=%d  %d",i, int(b), int(b & PARPORT_STATUS_PAPEROUT));}
          //break;
        }
          LINFO("after while got PARPORT_STATUS_PAPEROUT");
      LINFO("b=%d %d",int(b), int(b & PARPORT_STATUS_PAPEROUT));
      if (flag == 1 )
        break;
      // usleep(5000);

      // break;
      //usleep(5000);
    }
  //break;
  manager.stop();
#endif
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPDEVICES_TEST_PARPORT_C_DEFINED
