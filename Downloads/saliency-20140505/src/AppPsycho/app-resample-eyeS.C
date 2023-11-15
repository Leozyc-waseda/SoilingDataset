/*!@file AppPsycho/app-resample-eyeS.C Resample a .eyeS file to a new framerate */

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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppPsycho/app-resample-eyeS.C $
// $Id: app-resample-eyeS.C 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef APPPSYCHO_APP_RESAMPLE_EYES_C_DEFINED
#define APPPSYCHO_APP_RESAMPLE_EYES_C_DEFINED

#include "Component/ModelManager.H"
#include "Psycho/EyeSFile.H"

#include <fstream>

// This program resamples a .eyeS file to a new framerate. Currently,
// the new .eyeS file will have no auxiliary information, i.e., it
// will just have the resampled x and y values.

int main(int argc, char** argv)
{
  MYLOGVERB = LOG_INFO;

  ModelManager manager(argv[0]);

  nub::ref<EyeSFile> eyeS(new EyeSFile(manager));
  manager.addSubComponent(eyeS);

  if (manager.parseCommandLine(argc, argv, "newrate outfile", 2, 2) == false)
    return(1);

  const std::string newrate_string = manager.getExtraArg(0);
  const SimTime newrate = SimTime::fromString(newrate_string);

  const std::string newname = manager.getExtraArg(1);

  std::ofstream ofs(newname.c_str());

  if (!ofs.is_open())
    LFATAL("Couldn't open '%s' for writing", newname.c_str());

  SimTime t = newrate;
  int npts = 0;

  manager.start();

  while (1)
    {
      const Point2D<int> pt = eyeS->readUpTo(t);

      if (pt.i == -1 && pt.j == -1)
        break;

      ofs << pt.i << ' ' << pt.j << '\n';

      t += newrate;
      ++npts;
    }

  ofs.close();

  LINFO("wrote %d samples to %s", npts, newname.c_str());

  {
    std::ofstream f((newname + ".npts").c_str());
    if (!f.is_open())
      LFATAL("Couldn't open '%s.npts' for writing", newname.c_str());
    f << npts << '\n';
    f.close();
    LINFO("wrote %s.npts = %d", newname.c_str(), npts);
  }

  {
    std::ofstream f((newname + ".ntrash").c_str());
    if (!f.is_open())
      LFATAL("Couldn't open '%s.ntrash' for writing", newname.c_str());
    // we skipped over all the old trash, so there is no trash at the
    // beginning of the new .eyeS file:
    const int new_ntrash = 0;
    f << new_ntrash << '\n';
    f.close();
    LINFO("wrote %s.ntrash = %d", newname.c_str(), new_ntrash);
  }

  {
    std::ofstream f((newname + ".rate").c_str());
    if (!f.is_open())
      LFATAL("Couldn't open '%s.rate' for writing", newname.c_str());
    f << newrate_string << '\n';
    f.close();
    LINFO("wrote %s.rate = %s", newname.c_str(), newrate_string.c_str());
  }

  manager.stop();

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPPSYCHO_APP_RESAMPLE_EYES_C_DEFINED
