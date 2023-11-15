/*!@file BeoSub/test-BeoSubDB.C */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-BeoSubDB.C $
// $Id: test-BeoSubDB.C 6005 2005-11-29 18:49:14Z rjpeters $
//

#include "BeoSub/BeoSubDB.H"
#include <cstdlib>
#include <list>


//! A simple test for the BeoSubDB class

int main(int argc, char **argv)
{

  std::list<std::string> files;

  std::string destination;

  std::string temp;

  // check command-line args:
  if (argc < 3){
    LFATAL("USAGE: test-BeoSubDB <database destination filename> <source file 1> "
           "... <source file n>");
    return 1;
  }
  destination = argv[1];

  for (int i = 0; i < argc-2; i ++)
    {
      temp = argv[i+2];
      files.push_back(temp);
    }

  BeoSubDB* db = new BeoSubDB();
  db->initializeDatabase(files, destination);
  db->loadDatabase(destination); //just to test load
  printf("Saved database of size %d\n", db->getNumEntries());

  uint idx = temp.rfind('/'); temp = temp.substr((idx+1), temp.size());
  MappingData check = db->getMappingData(temp);

  printf("Position data for entry %s is-- x: %f Y: %f\n", temp.c_str(), check.itsXpos, check.itsYpos);

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
