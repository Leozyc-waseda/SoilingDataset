/*!@file Qt/test-BeoSubQt.cpp */

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
// Primary maintainer for this file
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Qt/test-BeoSubQt.cpp $
// $Id: test-BeoSubQt.cpp 5957 2005-11-16 18:10:52Z rjpeters $
//

#include <qapplication.h>
#include "Qt/ui/BeoSubQtMainForm.h"
#include "BeoSub/BeoSubOneBal.H"
#include "Component/ModelManager.H"
#include "Beowulf/Beowulf.H"
#include "BeoSub/ColorTracker.H"
#include "BeoSub/BeoSubTaskDecoder.H"
#include "QtUtil/Util.H"

// ######################################################################
int main(int argc, const  char ** argv)
{
  ModelManager manager("BeoSubOneBal Tester");

  //BeoSubOneBal
  nub::soft_ref<BeoSubOneBal> sub(new BeoSubOneBal(manager));
  manager.addSubComponent(sub);

  //BeoSubTaskDecoder
  nub::soft_ref<BeoSubTaskDecoder> decoder(new BeoSubTaskDecoder(manager));
  manager.addSubComponent(decoder);

  //ColorTracker 1
  nub::soft_ref<ColorTracker> tracker(new ColorTracker(manager));
  manager.addSubComponent(tracker);

  //ColorTracker 2
  nub::soft_ref<ColorTracker> tracker2(new ColorTracker(manager));
  manager.addSubComponent(tracker2);

  //BeoSubCanny
  nub::soft_ref<BeoSubCanny> detector(new BeoSubCanny(manager));
  manager.addSubComponent(detector);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // start the manager:
  manager.start();

  QApplication a( argc, argv2qt( argc, argv ) );
  BeoSubQtMainForm w;

  w.init(sub, nub::soft_ref<Beowulf>(), decoder, tracker, tracker2, detector);
  w.show();
  a.connect( &a, SIGNAL( lastWindowClosed() ), &a, SLOT( quit() ) );

  return a.exec();
}
