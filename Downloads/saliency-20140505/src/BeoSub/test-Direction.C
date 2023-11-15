/*!@file BeoSub/test-Direction.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/test-Direction.C $
// $Id: test-Direction.C 6990 2006-08-11 18:13:51Z rjpeters $
//

#include "Util/Angle.H"
#include "BeoSub/Attitude.H"
#include "SIFT/VisualObjectDB.H"
#include "BeoSub/BeoSubDB.H"
#include "Raster/Raster.H"
#include <cstdlib>


//! A simple test for the BeoSubDB class

int main(int argc, char **argv)
{
  // check command-line args:
  if (argc < 5)
    LFATAL("USAGE: test-Direciton <visual object db> <BeoSub db> "
           "<image.png> <x position> <y position>");

  //load visual object db
  VisualObjectDB vdb;
  vdb.loadFrom(argv[1]);

  BeoSubDB db;
  db.loadDatabase(argv[2]); //just to test load

  //match image to db
  std::string name(argv[3]);
  uint idx = name.rfind('.'); if (idx > 0) name = name.substr(0, idx);
  Image< PixRGB<byte> > colim = Raster::ReadRGB(name, RASFMT_PNG);
  rutz::shared_ptr<VisualObject> vo(new VisualObject(name, name, colim));
  std::vector< rutz::shared_ptr<VisualObjectMatch> > matches;
  const uint nmatches = vdb.getObjectMatches(vo, matches, VOMA_KDTREEBBF);
  if (nmatches == 0U){
    printf("\n\n\nNo matches found.\n");
    return 1;
  }

  rutz::shared_ptr<VisualObjectMatch> vom = matches[0];
  rutz::shared_ptr<VisualObject> obj = vom->getVoTest();
  std::string foundName = obj->getImageFname();
  foundName.replace(foundName.length()-3, 3, "txt");
  idx = foundName.rfind('/'); foundName = foundName.substr((idx+1), foundName.size());

  printf("\n\n\nMatched to %s with a score of %f\n", foundName.c_str(), vom->getScore());
  //get matching object from BeoSubDB
  MappingData check = db.getMappingData(foundName);

  MappingData end;
  end.itsXpos = strtod(argv[4], NULL);
  end.itsYpos = strtod(argv[5], NULL);
  end.itsImgFilename = "TEST";

  //Get directions from BeoSubDB to input x and y
  Attitude tAtt;
  float tDist = 0.0;
  db.getDirections(check, end, tAtt, tDist);

  //output result
  printf("Directions from object position (%f, %f) to input position (%f, %f) is an angle of %f and distance of %f\n", check.itsXpos, check.itsYpos, end.itsXpos, end.itsYpos, tAtt.heading.getVal(), tDist);

  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
