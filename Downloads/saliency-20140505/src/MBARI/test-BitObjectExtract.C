/*!@file MBARI/test-BitObjectExtract.C test the routine for extracting BitObjects
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/test-BitObjectExtract.C $
// $Id: test-BitObjectExtract.C 15310 2012-06-01 02:29:24Z itti $
//

#include "GUI/XWinManaged.H"
#include "Image/Image.H"
#include "MBARI/BitObject.H"
#include "MBARI/mbariFunctions.H"
#include "Raster/Raster.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <list>

int main(const int argc, const char** argv)
{
  // get command line option
  if (argc != 2) LFATAL("usage: %s image.[ppm|pgm]",argv[0]);
  Image<byte> img = Raster::ReadGray(argv[1]);

  //new XWinManaged(img);

  Rectangle region = Rectangle::tlbrI(10,10,400,400);

  Timer timer;
  std::list<BitObject> objs = extractBitObjects(img,region);
  LINFO("found %" ZU " objects in %llx ms",objs.size(),timer.get());

  //for (std::list<BitObject>::iterator ob = objs.begin();
  //   ob != objs.end(); ++ob)
  //new XWinManaged(ob->getObjectMask(byte(1),BitObject::OBJECT));

  //while(true);
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
