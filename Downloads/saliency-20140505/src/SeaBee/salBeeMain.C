/*!@file SeaBee/salBeeMain.C Main dispatcher  */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/salBeeMain.C $
// $Id: salBeeMain.C 10794 2009-02-08 06:21:09Z itti $
//

//
#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Media/FrameSeries.H"
#include "Media/MediaOpts.H"
#include "Image/ShapeOps.H"
#include "Image/MathOps.H"
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

        // Instantiate a ModelManager:
        ModelManager *mgr = new ModelManager("USC BeoSub");

        nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(*mgr));
        mgr->addSubComponent(ifs);

        nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
        mgr->addSubComponent(ofs);

        nub::soft_ref<SubController> subController(new SubController(*mgr));
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
        if (mgr->parseCommandLine(argc, argv, "<compMode>", 0, 1) == false) return(1);

        if (mgr->numExtraArgs() &&
                        mgr->getExtraArg(0) == "compMode")
                                        compMode = true;




        // do post-command-line configs:
        //Dims imageDims = ifs->peekDims();

        // let's get all our ModelComponent instances started:
        mgr->start();

        int smap_level = evc->getMapLevel();
        //start streaming
        ifs->startStream();


        //setup gui for subController

        subController->setMotorsOn(false);
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
        }

        subController->setSpeed(50);
        while(1)
        {
                const FrameState is = ifs->updateNext();
                if (is == FRAME_COMPLETE)
                        break;

                //grab the images
                GenericFrame input = ifs->readFrame();
                if (!input.initialized())
                        break;
                Image<PixRGB<byte> > frontImg = rescale(input.asRgb(), 320, 280);
                evc->input(frontImg);
                Image<float> vcxmap = evc->getVCXmap();
                Point2D<int> maxPos; float maxVal;
                findMax(vcxmap, maxPos, maxVal);
                LINFO("SalVal %f", maxVal);

                if (maxVal > 100)
                {

                        int xerr = (maxPos.i-(vcxmap.getWidth()/2));
                        int yerr = (maxPos.j-(vcxmap.getHeight()/2));

                        int desiredHeading = xerr*3;
                        int desiredDepth = yerr*3;

                        subController->setHeading(
                                        subController->getHeading()
                                               + desiredHeading);

                        subController->setDepth(
                                        subController->getDepth()
                                               + desiredDepth);

                }


                inplaceNormalize(vcxmap, 0.0F, 255.0F);
                Image<PixRGB<byte> > salDisp = rescale(vcxmap,frontImg.getDims()) ;
                drawCircle(frontImg,
                                       Point2D<int>(maxPos.i<<smap_level,
                                               maxPos.j<<smap_level),
                                30,
                                PixRGB<byte>(0,255,0));

                ofs->writeRGB(frontImg, "input", FrameInfo("Copy of input", SRC_POS));
                ofs->writeRGB(salDisp, "saliency", FrameInfo("Saliency", SRC_POS));

        }


        // stop all our ModelComponents
        mgr->stop();

        // all done!
        return 0;
}

