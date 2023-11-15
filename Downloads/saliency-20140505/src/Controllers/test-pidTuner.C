/*!@file Controllers/test-pidTuner.C Test the pid tuner */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Controllers/test-pidTuner.C $
// $Id: test-pidTuner.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/DrawOps.H"
#include "Image/ColorOps.H"
#include "Image/VisualTracker.H"
#include "GUI/ImageDisplayStream.H"
#include "GUI/XWinManaged.H"
#include "Devices/Serial.H"
#include "Devices/WiiMote.H"
#include "Util/CpuTimer.H"
#include "Image/MathOps.H"
#include "Image/Layout.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "GUI/XWinManaged.H"
#include "GUI/DebugWin.H"

#define KEY_UP 98
#define KEY_DOWN 104
#define KEY_LEFT 100
#define KEY_RIGHT 102


int getKey(nub::ref<OutputFrameSeries> &ofs)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow("Output")
    : rutz::shared_ptr<XWinManaged>();
  return uiwin->getLastKeyPress();
}

Point2D<int> getMouseClick(nub::ref<OutputFrameSeries> &ofs, const char* wname)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow(wname)
    : rutz::shared_ptr<XWinManaged>();

  if (uiwin.is_valid())
    return uiwin->getLastMouseClick();
  else
    return Point2D<int>(-1,-1);
}

float simMassSpring(float force)
{

  static float acc = 0;
  static float vel = 0;
  static float pos = 0;

  float dt = 0.6;

  float pPos = pos;
  float pVel = vel;
  pos = (force - 10*vel - 1 * acc)/20;

  vel = (pPos-pos)/dt;
  acc = (pVel-vel)/dt;

  return pos;

}


int main(int argc, const char **argv)
{
  // Instantiate a ModelManager:
  ModelManager manager("Test wiimote");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  manager.start();

  CpuTimer timer;

  timer.reset();
  Image<PixRGB<byte> > simImg(512,512,ZEROS);

  //Simulate a mass,spring,damper

  float x = 0;
  Point2D<int> lastLoc(0,simImg.getWidth()/2);
  simMassSpring(1.0F);
  while(1)
  {
    //Point2D<int> trackLoc = getMouseClick(ofs, "Output");

    float y = simMassSpring(0.0F);
    LINFO("y=%f", y);
    Point2D<int> loc((int)x*7,simImg.getWidth()/2 + (int)(y*1000));

    if (simImg.coordsOk(loc))
    {
      drawLine(simImg, lastLoc, loc, PixRGB<byte>(255,0,0));
      lastLoc = loc;
    }

    ofs->writeRGB(simImg, "Output", FrameInfo("output", SRC_POS));
    ofs->updateNext();
    usleep(10000);

    x++;
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
