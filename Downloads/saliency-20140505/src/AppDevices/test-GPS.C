/*!@file AppDevices/test-GPS.C test GPS driver */

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
// Primary maintainer for this file: Nitin Dhavale <dhavale@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-GPS.C $
// $Id: test-GPS.C 13712 2010-07-28 21:00:40Z itti $
//

#include "Component/ModelManager.H"
#include "Devices/GPS.H"

//! GPS listener for our test application
class TestGPSlistener {
public:
  //! Constructor
  TestGPSlistener() { buf = new char[1000]; };

  //! Destructor
  virtual ~TestGPSlistener() { delete [] buf; };

  //! Called when data is received
  virtual void newData(const GPSdata& data)
  { data.toString(buf, 1000); LINFO("%s", buf); };

private:
  char *buf;
};


//! This program tests the GPS driver, using its listener hookup mode
int main(const int argc, const char **argv)
{
  // instantiate a model manager:
  ModelManager manager("GPS Driver Tester");

  // Instantiate our various ModelComponents:
  nub::soft_ref<GPS> gps(new GPS(manager));
  rutz::shared_ptr<TestGPSlistener> lis(new TestGPSlistener());
  rutz::shared_ptr<GPSlistener> lis2; lis2.dynCastFrom(lis);

  gps->setListener(lis2);

  manager.addSubComponent(gps);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "[device]", 0, 1) == false)
    return(1);

  // do post-command-line configs:
  if (manager.numExtraArgs() > 0)
    gps->setModelParamString("GPSserialDevName",
                             manager.getExtraArg(0), MC_RECURSE);

  // let's get all our ModelComponent instances started:
  manager.start();

  // our main loop is a no-op, the thread and event-based listener do
  // all the work:
  while(1) sleep(100);

  // end of story:
  manager.stop();
  return 0;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

