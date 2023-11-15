/*!@file ArmControl/app-ArmControl.C tests the multi-threaded salincy code
 * with the arm control to move the end effector towerd salient objects
 * */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/ArmControl/app-ArmControl.C $
// $Id: app-ArmControl.C 10982 2009-03-05 05:11:22Z itti $
//

#ifndef TESTARM_H_DEFINED
#define TESTARM_H_DEFINED

#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/ShapeOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/ColorOps.H"
#include "Image/Transforms.H"
#include "Image/MathOps.H"
#include "Image/CutPaste.H"     // for inplacePaste()
#include "Image/Pixels.H"
#include "Neuro/StdBrain.H"
#include "Neuro/VisualCortex.H"
#include "Neuro/VisualCortexConfigurator.H"
#include "Neuro/NeuroOpts.H"
#include "Channels/DescriptorVec.H"
#include "Channels/ComplexChannel.H"
#include "Channels/SubmapAlgorithmBiased.H"
#include "Simulation/SimEventQueue.H"
#include "Simulation/SimulationOpts.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "Learn/Bayes.H"
#include "GUI/DebugWin.H"
#include "ObjRec/BayesianBiaser.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "GUI/XWinManaged.H"
#include "Neuro/NeuroOpts.H"
#include "Neuro/SaccadeControllers.H"
#include "Neuro/SaccadeControllerConfigurator.H"
#include "Raster/Raster.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Devices/rt100.H"
#include "Controllers/PID.H"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#define UP_KEY 98
#define DOWN_KEY 104
#define LEFT_KEY 100
#define RIGHT_KEY 102

//! Number of frames over which average framerate is computed
#define NAVG 20

//! Factor to display the sm values as greyscale:

// UDP communications:
#define UDPHOST "192.168.0.8"
#define UDPPORT 5003

static bool goforever = true;  //!< Will turn false on interrupt signal

//! Signal handler (e.g., for control-C)
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; exit(1); }

enum STATE {INIT, REACH, GRASP, LIFT, PLACE_IN_BASKET, DROP, DONE};

struct ArmPosition
{
  int elbow;
  int sholder;
  int zed;
};

bool init(nub::soft_ref<RT100> rt100);
bool reach(nub::soft_ref<RT100> rt100, Point2D<int> fixation,
    int x, int y, int zedPos);
bool learnMovement(float xErr, float yErr, int *elbowPos, int *sholderPos);
bool grasp(nub::soft_ref<RT100> rt100);
bool lift(nub::soft_ref<RT100> rt100);
bool placeInBasket(nub::soft_ref<RT100> rt100);
bool drop(nub::soft_ref<RT100> rt100);
void updateDisplay(Image<PixRGB<byte> > &ima,
    Image<float> &sm,
    Point2D<int> &fixation);
void waitForMoveComplete(nub::soft_ref<RT100> rt100);

Point2D<int> evolveBrain(Image<PixRGB<byte> > &img, DescriptorVec& descVec,
    Image<float> &smap);

void biasVC(ComplexChannel &vc, Bayes &bayesNet, int objId, bool DoBias);


ModelManager *mgr;
nub::soft_ref<FrameIstream> gb;
XWinManaged *xwinPtr;
Image<PixRGB<byte> > *dispPtr;
Timer masterclock;                // master clock for simulations
int w=-1,h=-1;
int smlevel = -1;

ArmPosition armPosition;

// ######################################################################
int main(int argc, char **argv)
{
  MYLOGVERB = LOG_INFO;
  int sim = false;


  // instantiate a model manager (for camera input):
  ModelManager manager("Test Arm Control");

  mgr = &manager;
  // Instantiate our various ModelComponents:
  nub::ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  nub::soft_ref<SimEventQueueConfigurator>
    seqc(new SimEventQueueConfigurator(manager));
  manager.addSubComponent(seqc);

  //our brain
  nub::ref<StdBrain>  brain(new StdBrain(manager));
  manager.addSubComponent(brain);

  // out arm
  nub::soft_ref<RT100> rt100(new RT100(manager));
  manager.addSubComponent(rt100);

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  if (sim)
  {
    manager.addSubComponent(ifs);
  }

  // Set the appropriate defaults for our machine that is connected to
  manager.exportOptions(MC_RECURSE);

  manager.setOptionValString(&OPT_RawVisualCortexChans, "IOC");
  //manager.setOptionValString(&OPT_RawVisualCortexChans, "C");
  manager.setOptionValString(&OPT_SaliencyMapType, "Fast");
  manager.setOptionValString(&OPT_SMfastInputCoeff, "1");
  manager.setOptionValString(&OPT_WinnerTakeAllType, "Fast");
  manager.setOptionValString(&OPT_SimulationTimeStep, "0.2");

  manager.setModelParamVal("FOAradius", 20, MC_RECURSE);
  manager.setModelParamVal("FoveaRadius", 20, MC_RECURSE);

  //manager.setOptionValString(&OPT_IORtype, "Disc");
  manager.setOptionValString(&OPT_IORtype, "None");


  manager.setOptionValString(&OPT_FrameGrabberStreaming, "false");
  manager.setOptionValString(&OPT_FrameGrabberType, "V4L");
  manager.setOptionValString(&OPT_FrameGrabberChannel, "1");
  manager.setOptionValString(&OPT_FrameGrabberHue, "0");
  manager.setOptionValString(&OPT_FrameGrabberContrast, "16384");
  manager.setOptionValString(&OPT_FrameGrabberDims, "320x240");
  //manager.setOptionValString(&OPT_SaccadeControllerType, "Threshfric");
  manager.setOptionValString(&OPT_SCeyeMaxIdleSecs, "1000.0");
  manager.setOptionValString(&OPT_SCeyeThreshMinOvert, "4.0");
  manager.setOptionValString(&OPT_SCeyeThreshMaxCovert, "3.0");
  manager.setOptionValString(&OPT_SCeyeThreshMinNum, "2");
  //  manager.setOptionValString(&OPT_SCeyeSpringK, "1000000.0");

  //manager.setOptionValString(&OPT_EyeHeadControllerType, "Trivial");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);


  if (sim)
  {
    w = ifs->getWidth();
    h = ifs->getHeight();
  } else {
    // do post-command-line configs:
    gb = gbc->getFrameGrabber();
    if (gb.isInvalid())
      LFATAL("You need to select a frame grabber type via the "
          "--fg-type=XX command-line option for this program "
          "to be useful");
    w = gb->getWidth();
    h = gb->getHeight();
  }

  const int foa_size = std::min(w, h) / 12;
  manager.setModelParamVal("InputFrameDims", Dims(w, h),
                           MC_RECURSE | MC_IGNORE_MISSING);
  manager.setModelParamVal("SCeyeStartAtIP", true,
                           MC_RECURSE | MC_IGNORE_MISSING);
  manager.setModelParamVal("SCeyeInitialPosition", Point2D<int>(w/2,h/2),
                           MC_RECURSE | MC_IGNORE_MISSING);
  manager.setModelParamVal("FOAradius", foa_size,
                           MC_RECURSE | MC_IGNORE_MISSING);
  manager.setModelParamVal("FoveaRadius", foa_size,
                           MC_RECURSE | MC_IGNORE_MISSING);

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // let's do it!
  manager.start();

  ComplexChannel *cc = NULL;
  cc = &*dynCastWeak<ComplexChannel>(brain->getVC());
  const LevelSpec lspec = cc->getModelParamVal<LevelSpec>("LevelSpec");
  smlevel = lspec.mapLevel();

  Image<float> sm(w >> smlevel, h >> smlevel, ZEROS); // saliency map

  uint64 avgtime = 0; int avgn = 0; // for average framerate
  float fps = 0.0F;                 // to display framerate
  Timer tim;                        // for computation of framerate

  Point2D<int> fixation(-1, -1);         // coordinates of eye fixation


  STATE currentState = INIT;


  // image buffer for display:
  Image<PixRGB<byte> > disp(w * 2, h + 20, ZEROS);
  dispPtr = &disp;
  //disp += PixRGB<byte>(128);
  XWinManaged xwin(disp.getDims(), -1, -1, "RT100 arm control");
  xwinPtr = &xwin;

  char info[1000];  // general text buffer for various info messages

  Point2D<int> lastpointsent(w/2, h/2);

  static bool moveArm = false;
  // ######################################################################
  try {

    //Get a new descriptor vector
    DescriptorVec descVec(manager, "Descriptor Vector", "DecscriptorVec", cc);
    //Get  new classifier
    LINFO("size %i\n", descVec.getFVSize());

    Bayes bayesNet(descVec.getFVSize(), 10);
    Dims objSize(20, 20);
    descVec.setFoveaSize(objSize);

    //get command line options
   // const char *bayesNetFile = mgr->getExtraArg(0).c_str();
   // const char *imageSetFile = mgr->getExtraArg(1).c_str();
   // int objToBias = mgr->getExtraArgAs<int>(2)-1;

 //   bayesNet.load("test.net");


    initRandomNumbers();

    if (!sim)
    {
      // get the frame grabber to start streaming:
      gb->startStream();
    }

    // initialize the timers:
    tim.reset(); masterclock.reset();
    Timer timer(1000000); timer.reset();

    int frame = 0;
    unsigned int objID = 0;
    while(goforever)
    {
      // grab image:
      Image< PixRGB<byte> > ima;
      if (sim)
      {
        ifs->updateNext();
        ima = ifs->readRGB();
      } else {
        ima = gb->readRGB();
      }
      frame++;

      Point2D<int> loc = xwin.getLastMouseClick();

      //Arm control
      //int key = xwin.getLastKeyPress();
      if (loc.isValid()) moveArm = !moveArm; //toggole moveArm

      if (moveArm)
      {
        switch (currentState)
        {
          case INIT:
            if (init(rt100))
            {
              LINFO("INitalize complete");
              currentState = REACH;
              rt100->setJointParam(RT100::ZED, RT100::SPEED, 20);

              rt100->setJointParam(RT100::SHOLDER, RT100::SPEED, 60);
              rt100->setJointParam(RT100::ELBOW, RT100::SPEED, 60);
              //
              //rt100->setJointPosition(RT100::ZED, -2800);
              //
              // rt100->setJointPosition(RT100::SHOLDER, 1280);
              // rt100->setJointPosition(RT100::ELBOW, -492);
              printf("Init complete\n");
              printf("Fixation: %i %i\n", fixation.i, fixation.j);
             // rt100->moveArm();
              //moveArm=false;
              biasVC(*cc, bayesNet, -1, false);  //start with an unbiased smap
              frame = 0;
            }
            break;
          case REACH:
            {
              fixation = evolveBrain(ima, descVec, sm);
              printf("Fixation: %i %i\n", fixation.i, fixation.j);

              descVec.setFovea(fixation);
              descVec.buildRawDV();

              std::vector<double> FV = descVec.getFV();
             // double statSig = bayesNet.getStatSig(FV, (uint)0);

              bayesNet.learn(FV, objID);   //update the model

              if (!(frame%10))
                biasVC(*cc, bayesNet, objID, true);

              if (frame > 20) { //start reaching for the object
                if(reach(rt100, fixation,
                      ima.getWidth()/2,
                      (ima.getHeight()/2) + 70,
                      -2800)) //final zed position
                {
                  LINFO("Reached object");
                  currentState = GRASP;
                }
              }
            }
            break;
          case GRASP:
            if(grasp(rt100))
            {
              LINFO("Objected grasped");
              currentState = PLACE_IN_BASKET;
            } else {
              LINFO("Did not graped object. Trying again");
              rt100->setJointPosition(RT100::YAW, 600);
              rt100->setJointPosition(RT100::ZED, -2600);
              rt100->setJointPosition(RT100::GRIPPER, 1000);
              //rt100->moveArm(true);

              currentState = REACH;
            }
            break;

          case LIFT:
            if(lift(rt100))
            {
              LINFO("Objected lifted");
              currentState = INIT;
            }
            break;
          case PLACE_IN_BASKET:
            if(placeInBasket(rt100))
            {
              LINFO("Object placed in basket");
              currentState = INIT;
              objID++;
              if (objID == 5)
                currentState = DONE;
            }
            break;
          case DONE:
            init(rt100);
            goforever = false;
            break;


          default:
            break;
        }
      }

      updateDisplay(ima, sm, fixation);

      avgtime += tim.getReset(); avgn ++;
      if (avgn == NAVG)
      {
        fps = 1000.0F / float(avgtime) * float(avgn);
        avgtime = 0; avgn = 0;
      }

      // create an info string:
      sprintf(info, "%.1ffps Target - (%03d,%03d) %f                            ",
          fps, fixation.i-(ima.getWidth()/2),
          fixation.j-(ima.getHeight()/2)-75,
          masterclock.getSecs());

      writeText(disp, Point2D<int>(0, h), info,
          PixRGB<byte>(255), PixRGB<byte>(127));

      // ready for next frame:
    }

    // get ready to terminate:
    manager.stop();

  } catch ( ... ) { };

  return 0;
}


void biasVC(ComplexChannel &vc, Bayes &bayesNet, int objId, bool doBias)
{
  //Set mean and sigma to bias submap
  BayesianBiaser bb(bayesNet, objId, -1, doBias);
  vc.accept(bb);

  setSubmapAlgorithmBiased(vc);
}

void updateDisplay(Image<PixRGB<byte> > &ima,
    Image<float> &sm,
    Point2D<int> &fixation){

  static int frame = 0;
  char filename[255];
  const int foa_size = std::min(w, h) / 12;

  // display image
  inplacePaste(*dispPtr, ima, Point2D<int>(0, 0));

  if (sm.initialized())
  {
    Image<float> dispsm = quickInterpolate(sm, 1 << smlevel);
    inplaceNormalize(dispsm, 0.0f, 255.0f);

    inplacePaste(*dispPtr,
        (Image<PixRGB<byte> >)toRGB(dispsm),
        Point2D<int>(w, 0));

    Point2D<int> fix2(fixation); fix2.i += w;
    if (fixation.i >= 0)
    {
      drawDisk(*dispPtr, fixation, foa_size/6+2, PixRGB<byte>(20, 50, 255));
      drawDisk(*dispPtr, fixation, foa_size/6, PixRGB<byte>(255, 255, 20));
      drawDisk(*dispPtr, fix2, foa_size/6+2, PixRGB<byte>(20, 50, 255));
      drawDisk(*dispPtr, fix2, foa_size/6, PixRGB<byte>(255, 255, 20));
    }
  }

  printf("FrameInfo: %i %f\n", frame, masterclock.getSecs());
  sprintf(filename, "armControlFrames/pickup/frame%06d.ppm", frame++);
 // Raster::WriteRGB(*dispPtr, filename);

  xwinPtr->drawImage(*dispPtr);
}

bool init(nub::soft_ref<RT100> rt100)
{

  rt100->setJointParam(RT100::ZED, RT100::SPEED, 40);
  rt100->setJointParam(RT100::SHOLDER, RT100::SPEED, 40);
  rt100->setJointParam(RT100::ELBOW, RT100::SPEED, 40);

  rt100->setJointPosition(RT100::ZED, -1500);
  //rt100->setJointPosition(RT100::SHOLDER, 2600);
  rt100->setJointPosition(RT100::SHOLDER, 2300);
  //rt100->setJointPosition(RT100::ELBOW, -1500);
  rt100->setJointPosition(RT100::ELBOW, -2000);
  rt100->setJointPosition(RT100::YAW, 600);
  rt100->setJointPosition(RT100::TILT_WRIST, -700);
  rt100->setJointPosition(RT100::ROLL_WRIST, 2400);
  rt100->setJointPosition(RT100::GRIPPER, 1000);
  rt100->moveArm();
  waitForMoveComplete(rt100);

  short int val;
  LINFO("Joint Pos: ");
  rt100->getJointPosition(RT100::ZED, &val);
  armPosition.zed = val;
  printf("%i ", val);
  rt100->getJointPosition(RT100::SHOLDER, &val);
  armPosition.sholder = val;
  printf("%i ", val);
  rt100->getJointPosition(RT100::ELBOW, &val);
  armPosition.elbow = val;
  printf("%i ", val);
  rt100->getJointPosition(RT100::YAW, &val);
  printf("%i ", val);
  printf("\n");

  return true;
}

bool learnMovement(float xErr, float yErr, int *elbowPos, int *sholderPos)
{
  bool reachXPos = false, reachYPos = false;

  *elbowPos = 0;
  *sholderPos = 0;
  if (!(xErr < 15 && xErr > -15))
  {

    if (xErr > 0)
      *elbowPos = -30;
    else
      *elbowPos = 30;
    reachXPos = false;
  } else {
    reachXPos = true;
  }


  if (!(yErr < 15 && yErr > -15))
  {
    if (yErr > 0)
      *sholderPos = -30;
    else
      *sholderPos = 30;
    reachYPos = false;
  } else {
    reachYPos = true;
  }


  if (reachXPos && reachYPos)
    return true;
  else
    return false;

  return false;

}

bool reach(nub::soft_ref<RT100> rt100, Point2D<int> fixation, int x, int y, int zedPos)
{
  bool reachZedPos = false;
  std::vector<short int> movePos(rt100->getNumJoints(), 0);
  float xErr = fixation.i-x;
  float yErr = -1*(fixation.j-y);

  int elbowPos = 0, sholderPos = 0;
  bool reachedTarget = learnMovement(xErr, yErr, &elbowPos, &sholderPos);


 // movePos[RT100::ELBOW] = elbowPos;
 // movePos[RT100::SHOLDER] = sholderPos;

  armPosition.sholder += sholderPos;
  armPosition.elbow += elbowPos;
  armPosition.zed -= 25;
  if (armPosition.zed < zedPos) armPosition.zed = zedPos;

  if (armPosition.zed == zedPos)
    reachZedPos = true;

  LINFO("Err: %f:%f s:%i e:%i z:%i",
      xErr, yErr,
      armPosition.elbow,
      armPosition.sholder,
      armPosition.zed);

  if (reachedTarget && reachZedPos)
  {
    //set current positions
    short int sholderPos, elbowPos;
    rt100->getJointPosition(RT100::SHOLDER, &sholderPos);
    rt100->getJointPosition(RT100::ELBOW, &elbowPos);

    rt100->setJointPosition(RT100::SHOLDER, sholderPos);
    rt100->setJointPosition(RT100::ELBOW, elbowPos);

    short int val;
    printf("Reach complete\n");
    printf("Joint Pos: ");
    rt100->getJointPosition(RT100::ZED, &val);
    printf("%i ", val);
    rt100->getJointPosition(RT100::SHOLDER, &val);
    printf("%i ", val);
    rt100->getJointPosition(RT100::ELBOW, &val);
    printf("%i ", val);
    rt100->getJointPosition(RT100::YAW, &val);
    printf("%i\n", val);

    return true;
  } else {
   // rt100->interpolationMove(movePos);
    rt100->setJointPosition(RT100::SHOLDER, armPosition.sholder);
    rt100->setJointPosition(RT100::ELBOW, armPosition.elbow);
    rt100->setJointPosition(RT100::ZED, armPosition.zed);
    rt100->moveArm();

    return false;
  }

  return false;
}

void waitForMoveComplete(nub::soft_ref<RT100> rt100)
{
  bool moveDone = false;
  for(int i=0; i<1000 && !moveDone; i++) //timeout
  {
    moveDone = rt100->moveComplete();
    Image< PixRGB<byte> > ima = gb->readRGB();
    Image<float> sm;
    Point2D<int> fix;
    updateDisplay(ima, sm, fix);
  }
}


bool grasp(nub::soft_ref<RT100> rt100)
{

  short int elbowPos;
  rt100->getJointPosition(RT100::ELBOW, &elbowPos);

  //center the gripper in the image

  float yawPos = 0.3334*elbowPos+40.909; //the mapping between elbow and yaw pos
 // float yawPos = 0.3334*elbowPos+220;
  LINFO("Elbow pos %i: %i\n", elbowPos, (short int)yawPos);
  rt100->setJointPosition(RT100::YAW, (short int)yawPos);
  rt100->moveArm();
  waitForMoveComplete(rt100);
  //rt100->moveArm(true);

  LINFO("Zed change");
  rt100->setJointPosition(RT100::ZED, -2900);
  rt100->moveArm();
  waitForMoveComplete(rt100);

  LINFO("close gripper");
  rt100->setJointPosition(RT100::GRIPPER, -500);
  rt100->moveArm();
  waitForMoveComplete(rt100);

  short int gripperPos;
  rt100->getJointPosition(RT100::GRIPPER, &gripperPos);
  printf("Gripper pos %i", gripperPos);

  if (gripperPos <= 0)
    return false;
  else
    return true;
}

bool lift(nub::soft_ref<RT100> rt100)
{
  LINFO("Zed change");
  rt100->setJointPosition(RT100::ZED, -2500);
 // rt100->setJointPosition(RT100::SHOLDER, 2600);
 // rt100->setJointPosition(RT100::ELBOW, -1500);
  rt100->moveArm();
  waitForMoveComplete(rt100);

  LINFO("open gripper");
  rt100->setJointPosition(RT100::GRIPPER, 1000);
  rt100->moveArm();
  waitForMoveComplete(rt100);

  return true;
}

bool placeInBasket(nub::soft_ref<RT100> rt100)
{
  LINFO("Move to basket");
  rt100->setJointPosition(RT100::ZED, -2100);
  rt100->moveArm();
  waitForMoveComplete(rt100);

  rt100->setJointPosition(RT100::SHOLDER, 1890);
  rt100->setJointPosition(RT100::ELBOW, -120);
  rt100->setJointPosition(RT100::YAW, -226);
  rt100->moveArm();
  waitForMoveComplete(rt100);

  LINFO("prepare to drop");

  LINFO("open gripper");
  rt100->setJointPosition(RT100::GRIPPER, 1000);
  rt100->moveArm();
  waitForMoveComplete(rt100);

  return true;
}
bool drop(nub::soft_ref<RT100> rt100)
{


  return true;
}

Point2D<int> evolveBrain(Image<PixRGB<byte> > &img, DescriptorVec& descVec,
    Image<float> &smap)
{

  nub::ref<StdBrain>  brain = dynCastWeak<StdBrain>(mgr->subComponent("Brain"));
  nub::ref<SimEventQueueConfigurator> seqc =
    dynCastWeak<SimEventQueueConfigurator>(mgr->subComponent("SimEventQueueConfigurator"));
  nub::soft_ref<SimEventQueue> seq  = seqc->getQ();

  LINFO("Evolve Brain");

  if (mgr->started()){    //give the image to the brain

    if (img.initialized())
      {
        //place the image in the inputFrame queue
        rutz::shared_ptr<SimEventInputFrame>
          e(new SimEventInputFrame(brain.get(), GenericFrame(img), 0));
        seq->post(e);
       // brain->input(img, seq);
        descVec.setInputImg(img);
      }

    bool keep_going = true;
    while (keep_going){
      brain->evolve(*seq);
      seq->evolve();

      const SimStatus status = seq->evolve();
      if (status == SIM_BREAK) {
        LINFO("V %d\n", (int)(seq->now().msecs()) );
        keep_going = false;
      }
      if (brain->gotCovertShift()) // new attended location
        {

          const Point2D<int> winner = brain->getLastCovertPos();
          const float winV = brain->getLastCovertAgmV();

          LINFO("##### Winner (%d,%d) at %fms : %.4f #####\n",
                winner.i, winner.j, seq->now().msecs(), winV * 1000.0f);
          //Image<float> img = brain->getSM()->getV(false);
          smap = brain->getVC()->getOutput();
         // SHOWIMG(img);
          /* char filename[255];
             sprintf(filename, "SceneSMap%i.ppm", ii++);
             Raster::WriteRGB(img, filename);*/

          return winner;

          keep_going = false;

        }
      if (seq->now().secs() > 3.0) {
        LINFO("##### Time limit reached #####");
        keep_going = false;
      }
      LINFO("Evolve brain");
    }

  }

  return Point2D<int>();

}



#endif
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
