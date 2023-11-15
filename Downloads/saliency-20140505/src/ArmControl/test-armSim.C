/*!@file AppDevices/test-armSim.C Test the robot arm simulator */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/test-armSim.C $
// $Id: test-armSim.C 10794 2009-02-08 06:21:09Z itti $
//


#include "Image/Layout.H"
#include "Image/MatrixOps.H"
#include "GUI/ImageDisplayStream.H"
#include "GUI/XWinManaged.H"
#include "Component/ModelManager.H"
#include "Raster/GenericFrame.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Devices/JoyStick.H"
#include "Devices/Scorbot.H"
#include "ArmControl/ArmSim.H"
#include <stdio.h>
#include <stdlib.h>
#include <ode/ode.h>
#include <signal.h>

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
    ? ids->getWindow("ArmSiminput")
    : rutz::shared_ptr<XWinManaged>();
  return uiwin->getLastKeyPress();
}

Point2D<int> getClick(nub::ref<OutputFrameSeries> &ofs)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow("ArmSiminput")
    : rutz::shared_ptr<XWinManaged>();
  return uiwin->getLastMouseClick();
}


int main(int argc, char *argv[])
{
  // Instantiate a ModelManager:
  ModelManager manager("Test Model for armSim");

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::soft_ref<ArmSim> armSim ((new ArmSim(manager,"ArmSim","ArmSim",21,14,22.5+10,22.5+10)));// length in CM
  manager.addSubComponent(armSim);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // let's get all our ModelComponent instances started:
        manager.start();

  int speed = 75;
        while(1){
                armSim->simLoop();

                Image<PixRGB<byte> > armCam = flipVertic(armSim->getFrame(-1));

                ofs->writeRGB(armCam, "ArmSiminput", FrameInfo("ArmSiminput", SRC_POS));

    int key = getKey(ofs);
double pos[]={-0.3,0,0.1};
    if (key != -1)
    {
      switch(key)
      {
        case 10: //1
          armSim->setMotor(ArmSim::BASE, speed);
          break;
        case 24: //q
          armSim->setMotor(ArmSim::BASE, -1*speed);
          break;
        case 11: //2
          armSim->setMotor(ArmSim::SHOLDER, -1*speed);
          break;
        case 25: //w
          armSim->setMotor(ArmSim::SHOLDER, speed);
          break;
        case 12: //3
          armSim->setMotor(ArmSim::ELBOW, speed);
          break;
        case 26: //e
          armSim->setMotor(ArmSim::ELBOW, -1*speed);
          break;
        case 13: //4
          armSim->setMotor(ArmSim::WRIST1, speed);
          break;
        case 27: //r
          armSim->setMotor(ArmSim::WRIST1, -1*speed);
          break;
        case 14: //5
          armSim->setMotor(ArmSim::WRIST2, speed);
          break;
        case 28: //t
          armSim->setMotor(ArmSim::WRIST2, -1*speed);
          break;
        case 15: //6
          armSim->setMotor(ArmSim::GRIPPER, speed);
          break;
        case 29: //y
          armSim->setMotor(ArmSim::GRIPPER, -1*speed);
          break;
        case 65: //space
          armSim->stopAllMotors();
          break;
        case 33: //p
          //showEncoders = !showEncoders;
          //rawWrist = !rawWrist;
          break;
        case 58: //m
          //showMS = !showMS;
          break;
        case 43: //h
          //home(ofs);
          break;
        case 32://o Add an object
          armSim->addObject(ArmSim::BOX,pos);
          break;

        case KEY_UP: speed += 1; break;
        case KEY_DOWN: speed -= 1; break;

      }

      if (speed < 0) speed = 0;
      if (speed > 100) speed = 100;


      LINFO("Key = %i", key);
    }

    Point2D<int> clickLoc = getClick(ofs);
    if (clickLoc.isValid())
    {
      double loc[3];
      armSim->getObjLoc(clickLoc.i, clickLoc.j, loc);
      LINFO("Loc %f,%f,%f", loc[0], loc[1], loc[2]);
    }
        }
        return 0;

}
