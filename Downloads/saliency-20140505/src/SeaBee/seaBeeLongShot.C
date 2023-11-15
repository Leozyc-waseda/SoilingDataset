/*!@file SeaBee/seaBeeLongShot.C Main dispatcher  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/seaBeeLongShot.C $
// $Id: seaBeeLongShot.C 10794 2009-02-08 06:21:09Z itti $
//

//
#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Media/FrameSeries.H"
#include "Media/MediaOpts.H"
#include "Image/ShapeOps.H"
#include "SeaBee/SubController.H"
#include "SeaBee/SubGUI.H"

#include "Devices/DeviceOpts.H"
#include "GUI/XWinManaged.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "GUI/XWindow.H"
#include "Image/DrawOps.H"
#include "Image/CutPaste.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/MathOps.H"
#include "Neuro/EnvVisualCortex.H"
#include "Media/FrameSeries.H"
#include "Media/MediaOpts.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Util/Timer.H"
#include "Util/log.H"
#include "Util/MathFunctions.H"
#include "Neuro/EnvVisualCortex.H"

#include "Util/Angle.H"

int main(int argc, const char **argv)
{
        int compMode = false;
  bool simMode = true;

        // Instantiate a ModelManager:
        ModelManager *mgr = new ModelManager("USC BeoSub");

        // Instantiate our various ModelComponents:
  //nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(*mgr));
  //mgr->addSubComponent(ifs);

        nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
        mgr->addSubComponent(ofs);

        nub::soft_ref<SubController> subController(new SubController(*mgr, "SubController", "SubController", simMode));
        mgr->addSubComponent(subController);

        nub::soft_ref<SubGUI> subGUI(new SubGUI(*mgr));
        mgr->addSubComponent(subGUI);

        nub::ref<EnvVisualCortex> evc(new EnvVisualCortex(*mgr));
        mgr->addSubComponent(evc);

        mgr->exportOptions(MC_RECURSE);

        mgr->setOptionValString(&OPT_InputFrameSource, "V4L2");
        mgr->setOptionValString(&OPT_FrameGrabberMode, "YUYV");
        mgr->setOptionValString(&OPT_FrameGrabberDims, "1024x576");
        mgr->setOptionValString(&OPT_FrameGrabberByteSwap, "no");
        mgr->setOptionValString(&OPT_FrameGrabberFPS, "30");


        // Parse command-line:
        if (mgr->parseCommandLine(argc, argv, "<initialForward> <diveDown> <diveTime> <timeToGate> <xPropSalTrack> <leftTurn> <compmode>", 6, 7) == false) return(1);

        if (mgr->numExtraArgs() &&
                        mgr->getExtraArg(6) == "compMode")
                compMode = true;

        //parameters
        int initialForward = atoi(mgr->getExtraArg(0).c_str()); //the first forward to clear the bottom

        //the values required to bring the sub down
        int diveDown = atoi(mgr->getExtraArg(1).c_str());
        //the time required to reach depth
        int diveTime = atoi(mgr->getExtraArg(2).c_str());

        //the time required to reach gate
        int timeToGate = atoi(mgr->getExtraArg(3).c_str());

  //the proprotinal x and y track
  float xPropSalTrack = atof(mgr->getExtraArg(4).c_str());

  //int turnLeftAmount = atoi(mgr->getExtraArg(5).c_str());

        LINFO("Setting params to: %i %i %i %i %f",
                                                                        initialForward, diveDown, diveTime,
                                                                         timeToGate, xPropSalTrack);



        // do post-command-line configs:
        //Dims imageDims = ifs->peekDims();

        // let's get all our ModelComponent instances started:
        mgr->start();

        //start streaming
        //ifs->startStream();

        int smap_level = evc->getMapLevel();

        //setup gui for subController

        subController->setMotorsOn(false);
        subController->setDepth(-1); //kill depth motors
        if (!compMode)
          {
                subGUI->startThread(ofs);
                subGUI->setupGUI(subController.get(), true);
                subGUI->addMeter(subController->getIntPressurePtr(),
                                "Int Pressure", 500, PixRGB<byte>(255, 0, 0));
                subGUI->addMeter(subController->getHeadingPtr(),
                                "Heading", 360, PixRGB<byte>(192, 255, 0));
                subGUI->addMeter(subController->getPitchPtr(),
                                "Pitch", 256, PixRGB<byte>(192, 255, 0));
                subGUI->addMeter(subController->getRollPtr(),
                                "Roll", 256, PixRGB<byte>(192, 255, 0));
                subGUI->addMeter(subController->getDepthPtr(),
                                "Depth", 300, PixRGB<byte>(192, 255, 0));

                subGUI->addMeter(subController->getThruster_Up_Left_Ptr(),
                                "Motor_Up_Left", -100, PixRGB<byte>(0, 255, 0));
                subGUI->addMeter(subController->getThruster_Up_Right_Ptr(),
                                "Motor_Up_Right", -100, PixRGB<byte>(0, 255, 0));
                subGUI->addMeter(subController->getThruster_Up_Back_Ptr(),
                                "Motor_Up_Back", -100, PixRGB<byte>(0, 255, 0));
                subGUI->addMeter(subController->getThruster_Fwd_Left_Ptr(),
                                "Motor_Fwd_Left", -100, PixRGB<byte>(0, 255, 0));
                subGUI->addMeter(subController->getThruster_Fwd_Right_Ptr(),
                                "Motor_Fwd_Right", -100, PixRGB<byte>(0, 255, 0));


                subGUI->addImage(subController->getSubImagePtr());
                subGUI->addImage(subController->getPIDImagePtr());

        } else {
                //running in comp mode, wait for kill switch
    LINFO("Running in comp mode");
                while(!subController->getKillSwitch())
                {
                        sleep(1);
                        LINFO("Waiting for kill switch");
                }

                //start the RUN
    sleep(7); //Wait for the sub to be pointed to gate
        }

        //get heading by avraging the current direction
        subController->setSpeed(10);

        while(1)
        {
          //ifs->updateNext();

          //grab the images
          //GenericFrame input = ifs->readFrame();
          Point2D<int> maxPos(-1,-1); float maxVal = -1;
          Image<PixRGB<byte> > input = subController->getImage(2);

          if (input.initialized())
            {
              //look for the most salient point and go towerd it
              Image<PixRGB<byte> > frontImg = rescale(input, 320, 280);
              evc->input(frontImg);
              Image<float> vcxmap = evc->getVCXmap();
              findMax(vcxmap, maxPos, maxVal);

              if (maxVal > 50)
                {
                  if (!compMode)
                    drawCircle(frontImg,
                               Point2D<int>(maxPos.i<<smap_level,
                                            maxPos.j<<smap_level),
                               10, PixRGB<byte>(255,0,0));
                  int xerr = (maxPos.i-(vcxmap.getWidth()/2));
                  int desiredHeading = (int)((float)xerr*-1*xPropSalTrack);

                  subController->setHeading(subController->getHeading()
                                            + desiredHeading);
                }

              ofs->writeRGB(frontImg, "input", FrameInfo("Copy of input", SRC_POS));

            }

        }

        // stop all our ModelComponents
        mgr->stop();

        // all done!
        return 0;
}

