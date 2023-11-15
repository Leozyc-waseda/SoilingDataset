/*!@file SeaBee/SeaBeeMainA.C main 2007 competition code
  Run SeaBeeMainA at CPU_A                                              */
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/SeaBeeMain.C $
// $Id: SeaBeeMain.C 10794 2009-02-08 06:21:09Z itti $
//
//////////////////////////////////////////////////////////////////////////

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Raster/Raster.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "GUI/XWinManaged.H"
#include "GUI/ImageDisplayStream.H"
#include "Neuro/EnvVisualCortex.H"
#include "Image/ShapeOps.H"

#include "Media/MediaOpts.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/FrameGrabberFactory.H"
#include "Raster/GenericFrame.H"

#include "Image/CutPaste.H"

//#include "AgentManager.H"
#include "ForwardVision.H"
#include "MovementController.H"
#include "SubGUI.H"
#include "Globals.H"
#include "SubController.H"

#include <signal.h>

volatile bool goforever = false;


// ######################################################################
//! Signal handler (e.g., for control-C)
void terminate(int s)
{
  LERROR("*** INTERRUPT ***");
  goforever = false;
  exit(1);
}

const ModelOptionDef OPT_ALIASseabeeCam =
  { MODOPT_ALIAS, "ALIASseabeeCam", &MOC_ALIAS, OPTEXP_CORE,
    "Set parameters for the SeaBee's camera",
    "seabee-cam", '\0', "",
    //    "--in=v4l2 "
    //    "--framegrabber-dims=320x240 "
    "--framegrabber-mode=YUYV "
    "--framegrabber-bswap=false "
    "--out=display "
  };


static const ModelOptionDef OPT_RunAgents =
  { MODOPT_FLAG, "SeaBeeRunAgents", &MOC_OUTPUT, OPTEXP_CORE,
    "Whether to run seabee submarine agents ",
    "run-seabee-agents", '\0', "", "false" };

Point2D<int> getClick(nub::soft_ref<OutputFrameSeries> &ofs, const char* winName)
{
  const nub::soft_ref<ImageDisplayStream> ids =
    ofs->findFrameDestType<ImageDisplayStream>();

  const rutz::shared_ptr<XWinManaged> uiwin =
    ids.is_valid()
    ? ids->getWindow(winName)
    : rutz::shared_ptr<XWinManaged>();
  return uiwin->getLastMouseClick();
}


// ######################################################################
int main( int argc, const char* argv[] )
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("SeaBee 2008 Main");

  OModelParam<bool> runSubAgents(&OPT_RunAgents, &manager);

  // Instantiate our various ModelComponents:

  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::soft_ref<SubGUI> subGUI(new SubGUI(manager));
  manager.addSubComponent(subGUI);

  nub::soft_ref<SubController> subController(new SubController(manager, "SubController", "SubController"));
  manager.addSubComponent(subController);

  nub::soft_ref<MovementController> movementController(new MovementController(manager,
                                                                              subController));
  manager.addSubComponent(movementController);

  nub::soft_ref<EnvVisualCortex> evc(new EnvVisualCortex(manager));
  manager.addSubComponent(evc);

  // create an Agent Manager
//   nub::ref<AgentManager> agentManager(new AgentManager(subController,evc,manager));
//   manager.addSubComponent(agentManager);

  rutz::shared_ptr<ForwardVisionAgent> forwardVision(new ForwardVisionAgent(evc));
  //  manager.addSubComponent(forwardVision);

  manager.requestOptionAlias(&OPT_ALIASseabeeCam);

  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  Dims cameraDims = subController->peekDims();

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // let's do it!
  manager.start();

  //eventually make this a command-line param
  bool competitionMode = false;
  bool debugMovementController = true;
  bool debugGoThroughGate = false;
  bool debugGoForwardTurnAround = true;
  bool debugPipeFollowing = false;

  // Setup GUI if not in competition mode
  if(!competitionMode)
    {
      subGUI->startThread(ofs);
      subGUI->setupGUI(subController.get(), true);

      if(debugMovementController)
        subGUI->setupGUI(movementController.get(), true);

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
      //      subGUI->addImage(subController->getPIDImagePtr());
      //      subGUI->addImage(forwardVision->getDebugImagePtr());
    }


  // wait for sub to initialize
  sleep(2);

  goforever = true;  uint fnum = 0;

  /*  while(!subController->getKillSwitch())
    {
      sleep(1);
      LINFO("Waiting for kill switch");
      }*/

  subController->setHeading(subController->getHeading());


  while(goforever)
    {
      if(runSubAgents.getVal())
        {


        }
      else if(debugMovementController)
        {
          Image< PixRGB<byte> > fImg = subController->getImage(1);
          if(fImg.initialized())
            {
              ofs->writeRGB(fImg, "ForwardCam", FrameInfo("Frame",SRC_POS));
              ofs->updateNext();
            }

          Image< PixRGB<byte> > dImg = subController->getImage(2);
          if(dImg.initialized())
            {
              ofs->writeRGB(dImg, "DownwardCam", FrameInfo("Frame",SRC_POS));
              ofs->updateNext();
            }

          Image< PixRGB<byte> > debugImg(*forwardVision->getDebugImagePtr());

          if(debugGoThroughGate)
            {

              if(movementController->dive(movementController->setDiveValue.getVal(),
                                          true,
                                          movementController->setTimeout.getVal()))
                {
                  movementController->goStraight(movementController->setSpeedValue.getVal(),
                                                 movementController->setGoStraightTimeValue.getVal());
                }

            }
          else if(debugGoForwardTurnAround)
            {
 //              if(movementController->dive(movementController->setDiveValue.getVal(),
//                                           true,
//                                           movementController->setTimeout.getVal()))
              if(1)
                {
                  movementController->goStraight(movementController->setSpeedValue.getVal(), 0);
                                                 //                                                 movementController->setGoStraightTimeValue.getVal());
                 //  movementController->setHeading(180, true, movementController->setTimeout.getVal());

//                   movementController->goStraight(movementController->setSpeedValue.getVal(),
//                                                  movementController->setGoStraightTimeValue.getVal());

                  Point2D<int> buoyLoc = forwardVision->lookForBuoy(fImg);
                  int xerr = (buoyLoc.i-(320/2));
                  //                  int desiredHeading = (int)((float)xerr);

                  ofs->writeRGB(debugImg, "DebugCam", FrameInfo("Frame",SRC_POS));
                  ofs->updateNext();

                  movementController->setHeading(xerr,
                                                 true,
                                                 movementController->setTimeout.getVal());

                }

            }
          else if(debugPipeFollowing)
            {
              Image< PixRGB<byte> > img = subController->getImage(2);
              if(img.initialized())
                {
                  ofs->writeRGB(img, "DownwardCam", FrameInfo("Frame",SRC_POS));
                  ofs->updateNext();
                }

              Point2D<int> clickLoc = getClick(ofs, "DownwardCam");
              Point2D<int> center(320/2, 240/2);

              if(clickLoc.isValid())
                {
                  LINFO("Tracking point: %i, %i",clickLoc.i, clickLoc.j);
                  movementController->trackPipe(clickLoc, center);
                }
            }
        }
      else
        {
          Image< PixRGB<byte> > img = subController->getImage(1);
          if(img.initialized())
            {
              ofs->writeRGB(img, "ForwardCam", FrameInfo("Frame",SRC_POS));
              ofs->updateNext();
            }

          img = subController->getImage(2);
          if(img.initialized())
            {
              ofs->writeRGB(img, "DownwardCam", FrameInfo("Frame",SRC_POS));
              ofs->updateNext();
            }
        }

      fnum++;
    }

  // we are done
  manager.stop();
  return 0;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
