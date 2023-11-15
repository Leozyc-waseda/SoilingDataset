/*!@file AppDevices/test-HMR3300.C test the HMR3300 compass */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-HMR3300.C $
// $Id: test-HMR3300.C 7880 2007-02-09 02:34:07Z itti $
//

#include "Devices/HMR3300.H"
#include "Component/ModelManager.H"

class TestHMR3300Listener : public HMR3300Listener {
public:
  //! Destructor
  virtual ~TestHMR3300Listener() {};

  //! New data was received
  virtual void newData(const Angle heading, const Angle pitch,
                       const Angle roll)
  {
    LINFO("<Heading=%f Pitch=%f Roll=%f>", heading.getVal(),
          pitch.getVal(), roll.getVal());
  }
};

int main(const int argc, const char **argv)
{
  // get a manager going:
  ModelManager manager("Compass Manager");

  // instantiate our model components:
  nub::soft_ref<HMR3300> hmr(new HMR3300(manager) );
  manager.addSubComponent(hmr);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<serdev>", 1, 1) == false)
    return(1);

  // let's configure our serial device:
  hmr->setModelParamVal("HMR3300SerialPortDevName",
                        manager.getExtraArg(0), MC_RECURSE);

  // let's register our listener:
  rutz::shared_ptr<TestHMR3300Listener> lis(new TestHMR3300Listener);
  rutz::shared_ptr<HMR3300Listener> lis2; lis2.dynCastFrom(lis); // cast down
  hmr->setListener(lis2);

  // get started:
  manager.start();

  // this is completely event driven, so here we just sleep. When data
  // is received, it will trigger our listener:
  while(1) sleep(1000);

  // stop everything and exit:
  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
