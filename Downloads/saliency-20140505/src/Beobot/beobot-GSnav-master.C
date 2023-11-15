/*!@file Beobot/beobot-GSnav-master.C Robot navigation using a
  combination saliency and gist.
  Run beobot-GSnav-master at CPU_A to run Gist-Saliency model
  Run beobot-GSnav        at CPU_B to run SIFT object recognition       */
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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/beobot-GSnav-master.C $
// $Id: beobot-GSnav-master.C 15464 2013-04-11 00:33:26Z kai $
//
//////////////////////////////////////////////////////////////////////////
// beobot-GSnav-Master.C  <operation mode> <environment file>
//                        [number of segments] [current Segment Number]
//
//
// This is an on-going project for biologically-plausible
// mobile-robotics navigation.
// It accepts any inputs: video  clip <input.mpg>, camera feed, frames.
//
// The system uses Gist to recognize places and saliency
// to get better localization within the place.
// The program also is made to be streamline for fast processing using
// parallel computation of the different the V1 feature channels
//
// It is able to recognize places through the use of gist features.
// The place classifier uses a neural networks,
// passed in a form of <input_train.txt> -
// the same file is used in the training phase by train-FFN.C.
//
// In parallel we use saliency to obtain salient objects for better
// spatial location accuracy.
// We use ShapeEstimator algorithm to segment out
// the sub-region to get a landmark and SIFT toidentify the object,
// create a database, etc.
//
// for localization, we use particle filter/Monte Carlo Localization
// for path planning we perform landmark-hopping to get to the destination

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Neuro/SaccadeController.H"
#include "Neuro/SaccadeControllers.H"
#include "Neuro/SaccadeControllerConfigurator.H"
#include "Neuro/NeuroOpts.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "GUI/XWinManaged.H"
#include "Media/MediaOpts.H"
#include "Util/Timer.H"
#include "Simulation/SimEventQueue.H"
#include "Simulation/SimEventQueueConfigurator.H"
#include "Devices/DeviceOpts.H"

#include "Beobot/BeobotBrainMT.H"
#include "Beobot/BeobotConfig.H"
#include "Beobot/BeobotControl.H"
#include "Beobot/BeobotBeoChipListener.H"

#include "RCBot/Motion/MotionEnergy.H"
#include "Controllers/PID.H"

#include "Image/MathOps.H"      // for findMax
#include "Image/CutPaste.H"     // for inplacePaste()
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/PyramidOps.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"

#include "Beobot/beobot-GSnav-def.H"
#include "Beobot/GSnavResult.H"

#include <signal.h>

#include <errno.h>

#define ERR_INTERVAL 5000

static bool goforever = true;

struct GroundTruth
{
  GroundTruth() { };

  GroundTruth(const int inFnum, const int inSegNum, const float inLtrav) :
    fnum(inFnum), snum(inSegNum), ltrav(inLtrav) {  }

  uint fnum;  int snum;  float ltrav;
};

std::vector<GroundTruth> groundTruth;
std::vector<float> groundTruthSegmentLength;

// ######################################################################
// ######################################################################
//! get the input prefix to save the intermdeiate results
std::string getInputPrefix(nub::soft_ref<InputFrameSeries> ifs,
                           int &inputType);

//! get the operation mode of the current run
int getOpMode(std::string opmodestr);

//! setup the BeoChip
void setupBeoChip(nub::soft_ref<BeoChip> b, BeobotConfig bbc);

//! process beoChip input
int beoChipProc
(rutz::shared_ptr<BeobotBeoChipListener> lis, nub::soft_ref<BeoChip> b);

//! get all the necessary information from the visual cortex
void getBbmtResults
( nub::ref<BeobotBrainMT> bbmt,
  Image<PixRGB<byte> > &currIma, Image<double> &cgist,
  Image<float> &currSalMap, ImageSet<float> &cmap,
  std::vector<Point2D<int> > &salpt, std::vector<std::vector<double> > &feat,
  std::vector<Rectangle> &objRect);

//! display results
void dispResults
( Image< PixRGB<byte> > disp, rutz::shared_ptr<XWinManaged> win,
  Image< PixRGB<byte> > ima, Image< PixRGB<byte> > prevIma,
  std::vector<Point2D<int> > clmpt, std::vector<Point2D<int> > nlmpt,
  Image<float> currSalMap,
  std::vector<Point2D<int> > salpt, std::vector<Rectangle> objRect);

//! check to see if the node have returned results
bool checkNode
( int opMode, nub::soft_ref<Beowulf> beo,
  int32 &rnode, TCPmessage &rmsg, int32 &raction, int32 &rframe);

//! check to see if the new input should be processed
bool checkInput(int opMode, bool resetNextLandmark,
                uint64 inputFrameRate, uint64 inputTime);

//! setup the packat containing the salient region information
//! to be recognized by the ventral module
void setupVentralPacket
( TCPmessage  &smsg, int rframe,
  bool resetVentralSearch, Image< PixRGB<byte> > currIma,
  Image<double> cgist, std::vector<Point2D<int> > salpt,
  std::vector<std::vector<double> > feat, std::vector<Rectangle> objRect,
  float dx, float dy, uint snumGT, float ltravGT);

//! setup the packet containing cmaps for tracking
void setupDorsalPacket
( TCPmessage  &smsg, int rframe, ImageSet<float> cmap,
  bool resetCurrLandmark, std::vector<Point2D<int> > clmpt,
  bool resetNextLandmark, std::vector<Point2D<int> > nlmpt);

//! get the proper motor command to run the robot
void processDorsalResult
( TCPmessage &rmsg, int32 raction, int32 rframe,
  std::vector<Point2D<int> > &clmpt, std::vector<Point2D<int> > &nlmpt,
  bool &resetNextLandmark, float &sp, float &st);

//! setup for the next search
void processVentralResult
( TCPmessage &rmsg, int32 raction, int32 rframe,
  std::vector<Point2D<int> >  &clmpt, std::vector<Point2D<int> > &nlmpt,
  bool &resetCurrLandmark, bool &resetNextLandmark);

//! get the odometry information
void getOdometry(float &dx, float &dy);

//! execute the motor command from Dorsal
void executeMotorCommand(int opMode, float st, float sp);

//! setup the ground truth
//! given an annotation file create a ground truth file
void setupGroundTruth(std::string gt_filename, bool writeGTfile = true);

//! feed in the current ground truth
void getGroundTruth
(uint fNum, uint &snumGT, float &ltravGT, float &dx, float &dy);

//! report the results of the current run
void reportResults(std::string resultPrefix, uint nsegment);

// ######################################################################
//! Signal handler (e.g., for control-C)
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; exit(1); }

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("beobot Navigation using Gist and Saliency - Master");

  // Instantiate our various ModelComponents:
  //BeobotConfig bbc;
  //nub::soft_ref<BeoChip> b(new BeoChip(manager));
  //manager.addSubComponent(b);

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::ref<BeobotBrainMT> bbmt(new BeobotBrainMT(manager));
  manager.addSubComponent(bbmt);

  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Master", "BeowulfMaster", true));
  manager.addSubComponent(beo);

  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  // if (manager.parseCommandLine
  //     (argc, argv,
  //      "<operation mode: train/trainX/test> <environment file> "
  //      "[TR: number of segments     |TX: NodeNumber        |TS: delay] "
  //      "[TR: current Segment Number |TX: startEdge,endEdge |TS: fstart] "
  //      "[test run file prefix] ",
  //      2, 5)
  if (manager.parseCommandLine(argc, argv,
                               "<operation mode> <environment file> "
                               "[TR: number of segments | TS: groundtruthfile]"
                               "[current Segment Number] "
                               "[test run file prefix] ",
                               2, 5)
      == false) return(1);

  // do post-command-line configs:

  // let's configure our serial device:
  //b->setModelParamVal("BeoChipDeviceName", std::string("/dev/ttyS0"));

  // let's register our listener:
  //rutz::shared_ptr<BeobotBeoChipListener> lis(new BeobotBeoChipListener(b));
  //rutz::shared_ptr<BeoChipListener> lis2; lis2.dynCastFrom(lis); // cast down
  //b->setListener(lis2);

  // inputs: number of segments and current segment number
  // is needed for initial training, otherwise it is reset later on
  int nSegment = 0, currSegNum  = -1;
  //int currNodeNum = -1, startEdge = -1, endEdge = -1;

  // get the environment file and folder to save
  std::string envFName = manager.getExtraArg(1);
  std::string saveFilePath = "";
  std::string::size_type lspos = envFName.find_last_of('/');
  int ldpos = envFName.find_last_of('.');
  std::string envPrefix;
  if(lspos != std::string::npos)
    {
      saveFilePath = envFName.substr(0, lspos+1);
      envPrefix =  envFName.substr(lspos+1, ldpos - lspos - 1);
    }
  else
    envPrefix =  envFName.substr(0, ldpos - 1);

  // get the the input type and prefix to name the salient objects, gistfile
  int inputType;
  std::string testRunFPrefix =
    std::string("results_") + getInputPrefix(ifs, inputType);
  if(manager.numExtraArgs() > 4)
    testRunFPrefix = manager.getExtraArgAs<std::string>(4);

  std::string testRunFolder =
    saveFilePath + testRunFPrefix + std::string("/");
  std::string resultPrefix = testRunFolder + envPrefix;

  // get the operation mode
  int opMode = getOpMode(manager.getExtraArg(0));

  // means next frame will be processed once Ventral is done
  uint64 inputFrameRate = 0;
  uint fstart = 0;

  // handle the command line inputs
  LINFO("opmode: %d", opMode);
  if(opMode == TRAIN_MODE)
    {
      if(manager.numExtraArgs() < 4)
        LFATAL("Training needs at least 4 params");
      nSegment = manager.getExtraArgAs<int>(2);
      currSegNum = manager.getExtraArgAs<int>(3);
      LINFO("number of segments: %d", nSegment);
      LINFO("current segment: %d", currSegNum);
    }
  // else if(opMode == TRAIN_X_MODE)
  //   {
  //     if(manager.numExtraArgs() < 4)
  //       LFATAL("Junction training needs at least 4 params");

  //     currNodeNum = manager.getExtraArgAs<int>(2);
  //     std::string tempRange = manager.getExtraArg(3);
  //     std::string::size_type lspos = tempRange.size()-1;
  //     std::string::size_type lcpos = tempRange.find_last_of(',');
  //     startEdge = atoi(tempRange.substr(1, lcpos-1).c_str());
  //     endEdge   = atoi(tempRange.substr(lcpos+1, lspos).c_str());

  //     LINFO("Junction Node: %d", currNodeNum);
  //     LINFO("Start Edge: %d",    startEdge);
  //     LINFO("End   Edge: %d",    endEdge);
  //   }
  else if(opMode == TEST_MODE)
    {
      setupGroundTruth(manager.getExtraArg(2), false);
      
      //if(manager.numExtraArgs() >  2)
      //  inputFrameRate = manager.getExtraArgAs<uint>(2);

      //if(manager.numExtraArgs() >  3)
      //  fstart = manager.getExtraArgAs<uint>(3);
    }
  else LFATAL("invalid opmode");

  LINFO("Environment file: %s", envFName.c_str());
  LINFO("Save file to this folder: %s", saveFilePath.c_str());
  LINFO("envPrefix: %s", envPrefix.c_str());
  LINFO("test run prefix: %s", testRunFPrefix.c_str());
  LINFO("test run folder: %s", testRunFolder.c_str());
  LINFO("result prefix: %s", resultPrefix.c_str());
  LINFO("frame frequency: %f ms/frame", inputFrameRate/1000.0F);

  // create the session result folder
  if (mkdir(testRunFolder.c_str(), 0777) == -1 && errno != EEXIST)
    {
      LFATAL("Cannot create log folder: %s", testRunFolder.c_str());
    }

  // HACK: results for paper
  //reportResults(resultPrefix, 9);  Raster::waitForKey();

  //int w = ifs->getWidth(),  h = ifs->getHeight();
  int w = 160,  h = 120;
  std::string dims = convertToString(Dims(w, h));
  LINFO("image size: [%dx%d]", w, h);
  manager.setOptionValString(&OPT_InputFrameDims, dims);

  manager.setModelParamVal("InputFrameDims", Dims(w, h),
                           MC_RECURSE | MC_IGNORE_MISSING);

  TCPmessage rmsg;     // buffer to receive messages
  TCPmessage smsg;     // buffer to send messages
  int32 rframe = 0, raction = 0, rnode = 0;

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // let's do it!
  manager.start();

  // send params to dorsal and ventral node to initialize contact:
  smsg.reset(0, INIT_COMM);
  smsg.addInt32(int32(fstart));
  smsg.addInt32(int32(w));
  smsg.addInt32(int32(h));
  smsg.addInt32(int32(opMode));
  smsg.addInt32(int32(nSegment));
  smsg.addInt32(int32(currSegNum));
  smsg.addString(envFName.c_str());
  smsg.addString(testRunFPrefix.c_str());
  smsg.addString(saveFilePath.c_str());
  smsg.addString(resultPrefix.c_str());

  // send the same initial values message to both nodes
  beo->send(DORSAL_NODE, smsg);
  beo->send(VENTRAL_NODE, smsg);

  // saliency map
  Image<float> currSalMap(w >> sml, h >> sml, ZEROS);

  // image buffer and window for display:
  Image<PixRGB<byte> > disp(w * 5, h, ZEROS);
  //rutz::shared_ptr<XWinManaged> mwin
  //  (new XWinManaged(disp.getDims(), w*2 + 10, 0, "Master window"));

  // setup the configuration of BeoChip
  //setupBeoChip(b,bbc);

  // if we are on training mode
  // the remote control can be used to steer
  //if(opMode == TRAIN_MODE) lis->moveServo = true;

  ImageSet<float> cmap(NUM_CHANNELS);
  Point2D<int> lastwin(-1,-1); Point2D<int> lastwinsc(-1,-1);

  // SYNCHRONIZATION: wait until the other board is ready
  LINFO("waiting until Ventral and Dorsal is ready to go");
  rnode = VENTRAL_NODE;
  while(!beo->receive(rnode, rmsg, rframe, raction, 5));
  LINFO("%d is ready", rnode);
  rnode = DORSAL_NODE;
  while(!beo->receive(rnode, rmsg, rframe, raction, 5));
  LINFO("%d is ready", rnode);
  rmsg.reset(rframe, raction);
  LINFO("Ventral and Dorsal are ready to go");
  Raster::waitForKey();

  // time result file
  std::string resFName = resultPrefix + sformat("_M_results.txt");

  // MAIN LOOP
  bool resetCurrLandmark = false;
  std::vector<Point2D<int> > clmpt;     // landmarks that are currently followed
  bool resetNextLandmark = true;
  std::vector<Point2D<int> > nlmpt;     // landmarks that are currently searched

  float sp = 0.0f; float st = 0.0f;

  Timer inputTimer(1000000); uint64 t[NAVG]; float frate = 0.0f;
  Timer ventralTimer(1000000); Timer dorsalTimer(1000000);
  float vTime = -1.0F; float dTime = -1.0F;

  // feed an initial image to get the ball rolling
  bool dorsalReplied  = false; std::vector<float> procTime;
  bool resetVentralSearch = false;
  bool stopSent = false;
  Image<PixRGB<byte> > ima; int32 fNum = fstart;
  Image<PixRGB<byte> > prevIma; inputTimer.reset();
  ifs->updateNext(); ima = ifs->readRGB();

  if(ima.getWidth() != 160 && ima.getHeight() != 120)
    ima = rescale(ima,160,120);

  if(!ima.initialized()) { goforever = false; }
  else { bbmt->input(ima);} //mwin->drawImage(ima,0,0); }
  while(goforever || !dorsalReplied)
    {
      // ADD GUI HANDLER HERE

      // beoChip procedures
      //int bcCom = beoChipProc(lis, b);
      //if(bcCom == BC_QUIT_SIGNAL) { goforever = false; break; }

//       // keyboard input
//       // FIX: figure out the mapping from just printing it
//       //      get the 1 - 0 - enter sequence
//       int tPress = mwin->getLastKeyPress();
//       if(tPress != -1 && tPress >= 10 && tPress <= 18)
//         currSegNum = tPress - 10;

      // if Dorsal computation is ready
      rnode = DORSAL_NODE; int32 rDframe;
      if(checkNode(opMode, beo, rnode, rmsg, raction, rDframe))
        {
          LDEBUG("IN DORSAL: %d", rDframe);
          processDorsalResult(rmsg, raction, rframe, clmpt, nlmpt,
                              resetNextLandmark, sp, st);
          dTime = dorsalTimer.get()/1000.0F;
          LDEBUG("Dorsal time[%d]: %f", rDframe, dTime);
          dorsalReplied = true;
        }

      // if ventral computation is ready
      //    & dorsal replied at least once
      rnode = VENTRAL_NODE; int32 rVframe;
      if(dorsalReplied &&
         checkNode(opMode, beo, rnode, rmsg, raction, rVframe))
        {
          processVentralResult(rmsg, raction, rframe, clmpt, nlmpt,
                               resetCurrLandmark, resetNextLandmark);
          LDEBUG("IN VENTRAL: %d ::: %d",  rVframe, resetNextLandmark);

          vTime = ventralTimer.get()/1000.0F;
          LDEBUG("Ventral time[%d]: %f", rVframe, vTime);
          stopSent = false;
        }

      // get odometry information
      float dx, dy; getOdometry(dx, dy);

      // execute the motor command
      executeMotorCommand(opMode, sp, st);

      // when the saliency and gist computation is ready
      // AND input timing is right
      uint64 inputTime = inputTimer.get();
      bool over10seconds = inputTime > 10000000;
      if(over10seconds) LINFO("time exceeds 10 seconds %fms", inputTime/1000.0f);


      if(checkInput(opMode, resetNextLandmark, inputFrameRate, inputTime)
         && bbmt->outputReady())
        {
          t[fNum % NAVG] = inputTime; inputTimer.reset();
          Timer t1(1000000); t1.reset();
          float bbmtTime = bbmt->getProcessTime();
          rframe = fNum;

          // get all the necesary information from V1:
          Image<PixRGB<byte> > currIma; Image<float> currSalMap;
          Image<double> cgist;
          std::vector<Point2D<int> > salpt;
          std::vector<std::vector<double> >feat;
          std::vector<Rectangle> objRect;
          getBbmtResults(bbmt, currIma, cgist, currSalMap,
                         cmap, salpt, feat, objRect);

          // get the next frame and start processing
          FrameState fState = ifs->updateNext(); ima = ifs->readRGB();
          if(!ima.initialized() || fState == FRAME_COMPLETE)
            { goforever = false; }
          else bbmt->input(ima);
          uint snumGT = 0; float ltravGT = -1.0F;

          // get ground truth
          if(opMode == TEST_MODE)
            getGroundTruth(fNum, snumGT, ltravGT, dx, dy);

          // if timer goes past the time limit and robot is moving
          // FIX: maybe we still discard search even if robot is stationary
          if(resetNextLandmark) ventralTimer.reset();
          if(inputFrameRate != 0 &&
             ventralTimer.get() > (inputFrameRate * SEARCH_TIME_LIMIT) &&
             !stopSent && sp != 0.0)
            {
              resetVentralSearch = true;
              LDEBUG("ventral time: %f ms. STOP", ventralTimer.get()/1000.0f);
              stopSent = true;
            }

          // set up the salient region packet and send to ventral
          setupVentralPacket
            (smsg, rframe, resetVentralSearch, currIma,
             cgist, salpt, feat, objRect, dx, dy, snumGT, ltravGT);
          beo->send(VENTRAL_NODE, smsg);
          resetVentralSearch = false;
          LINFO("send ventral done");

          // setup the tracking packet and send to dorsal
          setupDorsalPacket(smsg, rframe, cmap,
                            resetCurrLandmark, clmpt,
                            resetNextLandmark, salpt);
          beo->send(DORSAL_NODE, smsg);
          dorsalTimer.reset();
          dorsalReplied = false;

          // once landmarks are reset, we don't do it over again
          bool rCL = resetCurrLandmark;
          bool rNL = resetNextLandmark;
          if(resetCurrLandmark) resetCurrLandmark = false;
          if(resetNextLandmark) resetNextLandmark = false;

          // display the results
          //mwin->setTitle(sformat("fNum: %d",fNum).c_str());
          //dispResults(disp, mwin, currIma, prevIma, clmpt, nlmpt,
          //           currSalMap, salpt, objRect);
          prevIma = currIma;
          //Raster::waitForKey();

          // compute and show framerate over the last NAVG frames:
          float iTime = t[fNum % NAVG]/1000.0F;
          FILE *rFile = fopen(resFName.c_str(), "at");
          if (rFile != NULL)
            {
              // if ventral search has not returned
              float dispVTime = vTime;
              if(!rNL) dispVTime = ventralTimer.get()/1000.0F;

              std::string line =
                sformat("%5d %11.5f %d %d %11.5f %11.5f %11.5f",
                        fNum, bbmtTime, rCL, rNL, dTime, dispVTime, iTime);
              LINFO("%s", line.c_str());
              line += std::string("\n");
              fputs(line.c_str(), rFile);
              fclose (rFile);
            }
          else LFATAL("can't create result file: %s", resFName.c_str());

          fNum++;
          if (fNum % NAVG == 0 && fNum > 0)
            {
              uint64 sum = 0ULL; for(int i = 0; i < NAVG; i ++) sum += t[i];
              frate = 1000000.0F / float(sum) * float(NAVG);
              LINFO("Fr: %6.3f fps -> %8.3f ms/f\n", frate, 1000.0/frate);
            }
          uint64 ptime = t1.get(); float ptime2 = ptime/1000.0;
          procTime.push_back(ptime2);
          LDEBUG("total proc time: %f", ptime2);
          t1.reset();
        }
      //else{ usleep(10); }
    }

  // wait for the ventral response for one more time step
  uint64 inputTime = inputTimer.get();
  while(!checkInput(opMode, resetNextLandmark, inputFrameRate, inputTime))
    {
      rnode = VENTRAL_NODE; int32 rVframe;
      if(checkNode(opMode, beo, rnode, rmsg, raction, rVframe))
        {
          processVentralResult(rmsg, raction, rframe, clmpt, nlmpt,
                               resetCurrLandmark, resetNextLandmark);
          vTime = ventralTimer.get()/1000.0F;
          LDEBUG("Ventral time[%d]: %f", rVframe, vTime);
        }

      usleep(1000);
      inputTime = inputTimer.get();
    }

  // save last time step performance
  FILE *rFile = fopen(resFName.c_str(), "at");
  if (rFile != NULL)
    {
      float dispVTime = vTime;
      if(!resetNextLandmark) dispVTime = ventralTimer.get()/1000.0F;

      std::string line =
        sformat("%5d %11.5f %d %d %11.5f %11.5f %11.5f",
                fNum, -1.0F, resetCurrLandmark, resetNextLandmark,
                dTime, dispVTime, inputTime/1000.0);
      LINFO("%s", line.c_str());
      line += std::string("\n");
      fputs(line.c_str(), rFile);
      fclose (rFile);
    }
  else LFATAL("can't create result file: %s", resFName.c_str());

  // signal the other processors to stop working
  LINFO("sending ABORT to Dorsal");
  rframe = fNum;
  smsg.reset(rframe, ABORT);
  beo->send(DORSAL_NODE, smsg);
  //rnode = DORSAL_NODE;
  //while(!beo->receive(rnode, rmsg, rframe, raction, 5));
  //LINFO("rec Dorsal");
  //rmsg.reset(rframe,raction);

  LINFO("sending ABORT to ventral");
  beo->send(VENTRAL_NODE, smsg);
  rnode = VENTRAL_NODE;
  while(!beo->receive(rnode, rmsg, rframe, raction, 5));
  LINFO("rec Ventral");
  rmsg.reset(rframe,raction);

  // ending operations
  // this needs both XXX_M_results.txt and XXX_results.txt
  switch(opMode)
    {
    case TRAIN_MODE: break;

    case TEST_MODE:
      {
        // report some statistical data
        reportResults(resultPrefix, 9);

        float min = 0.0f, max = 0.0f;
        if(procTime.size() > 0){ min = procTime[0]; max = procTime[0]; }
        for(uint i = 1; i < procTime.size(); i++)
          {
            if (min > procTime[i]) min = procTime[i];
            if (max < procTime[i]) max = procTime[i];
          }
        LINFO("proc Time: %f - %f", min, max);
      }
      break;

    default: LERROR("Unknown operation mode");
    }

  LINFO("we can now exit");
  Raster::waitForKey();

  // get ready to terminate:
  manager.stop();
  return 0;
}

// ######################################################################
std::string getInputPrefix(nub::soft_ref<InputFrameSeries> ifs,
                           int &inputType)
{
  std::string input =
    ifs->getModelParamVal<std::string>("InputFrameSource");
  std::string prefix;

  // if it camera input
  std::string camInput("ieee1394");
  if(!input.compare(camInput))
    {
      inputType = CAMERA_INPUT;
      // use the time of day
      time_t rawtime; struct tm * timeinfo;
      time ( &rawtime );
      timeinfo = localtime ( &rawtime );
      std::string cTime(asctime (timeinfo));
      cTime = cTime.substr(4); // take out the day

      for(uint i = 0; i < cTime.length(); i++)
        if(cTime[i] == ' ' || cTime[i] == ':') cTime[i] = '_';
      prefix = std::string("cam_") + cTime;
      LINFO("Camera input, prefix: %s", prefix.c_str());
    }
  // if it is a frame series input
  else if(input.find_last_of('#') == (input.length() - 5))
    {
      inputType = FILE_INPUT;

      prefix = input.substr(0, input.length() - 5);
      std::string::size_type spos = prefix.find_last_of('/');
      if(spos != std::string::npos)
        prefix = prefix.substr(spos+1);

      // if the last part of the prefix is an '_'
      if(prefix.find_last_of('_') == prefix.length() - 1)
        {
          // take it out
          prefix = prefix.substr(0, prefix.length() - 1);
        }

      LINFO("Frame series input, prefix: %s", prefix.c_str());
    }
  // it is an mpeg
  else
    {
      inputType = FILE_INPUT;

      prefix = input.substr(0, input.length() - 4);
      std::string::size_type spos = prefix.find_last_of('/');
      if(spos != std::string::npos)
        prefix = prefix.substr(spos+1);

      LINFO("mpeg Input, prefix: %s", prefix.c_str());
    }

  return prefix;
}

// ######################################################################
int getOpMode(std::string opmodestr)
{
  // compare returns zero if they are equal
  std::string trainMode("train");  int op;
  //std::string trainXMode("trainX");
  if(!opmodestr.compare(trainMode)) op = TRAIN_MODE;
  //else if(!opmodestr.compare(trainXMode)) op = TRAIN_X_MODE;
  else op = TEST_MODE;
  return op;
}

// ######################################################################
void setupBeoChip(nub::soft_ref<BeoChip> b, BeobotConfig bbc)
{
  // reset the beochip:
  LINFO("Resetting BeoChip...");
  b->resetChip(); sleep(1);

  // calibrate the servos
  b->calibrateServo(bbc.steerServoNum, bbc.steerNeutralVal,
                    bbc.steerMinVal, bbc.steerMaxVal);
  b->calibrateServo(bbc.speedServoNum, bbc.speedNeutralVal,
                    bbc.speedMinVal, bbc.speedMaxVal);
  b->calibrateServo(bbc.gearServoNum, bbc.gearNeutralVal,
                    bbc.gearMinVal, bbc.gearMaxVal);

  // keep the gear at the lowest speed/highest Ntorque
  b->setServoRaw(bbc.gearServoNum, bbc.gearMinVal);

  // zero out the speed
  //b->setServoRaw(bbc.speedServoNum, bbc.speedNeutralVal);

  // turn on the keyboard
  b->debounceKeyboard(true);
  b->captureKeyboard(true);

  // calibrate the PWMs:
  b->calibratePulse(0,
                    bbc.pwm0NeutralVal,
                    bbc.pwm0MinVal,
                    bbc.pwm0MaxVal);
  b->calibratePulse(1,
                    bbc.pwm1NeutralVal,
                    bbc.pwm1MinVal,
                    bbc.pwm1MaxVal);
  b->capturePulse(0, true);
  b->capturePulse(1, true);

  // let's play with the LCD:
  b->lcdClear();   // 01234567890123456789
  b->lcdPrintf(0, 0, "collectFrames: 00000");
  b->lcdPrintf(0, 1, "STEER=XXX  SPEED=XXX");
  b->lcdPrintf(0, 2, "PWM0=0000  0000-0000");
  b->lcdPrintf(0, 3, "PWM1=0000  0000-0000");
}

// ######################################################################
void getOdometry(float &dx, float &dy)
{
  dx = 0.0F;
  dy = 0.0F;
}

// ######################################################################
void executeMotorCommand(int opMode, float st, float sp)
{
  switch(opMode)
    {
    case TRAIN_MODE:
    case TEST_MODE:
      // a remote controller to train the robot

      // have a point to pursue

      // or to close to it

      // execute action command
      //b->setServo(bbc.steerServoNum, st);
      //b->setServo(bbc.speedServoNum, sp);
      break;
    default:
      LERROR("Unknown operation mode");
    }
}

// ######################################################################
void dispResults
( Image< PixRGB<byte> > disp, rutz::shared_ptr<XWinManaged> win,
  Image< PixRGB<byte> > ima, Image< PixRGB<byte> > prevIma,
  std::vector<Point2D<int> > clmpt, std::vector<Point2D<int> > nlmpt,
  Image<float> currSalMap,
  std::vector<Point2D<int> > salpt, std::vector<Rectangle> objRect)
{
  int w = ima.getWidth();
  int h = ima.getHeight();
  const int foa_size = std::min(w, h) / 12;

  // display input image:
  inplacePaste(disp, ima, Point2D<int>(0, 0));

  // display the saliency map
  Image<float> dispsm = quickInterpolate(currSalMap * SMFAC, 1 << sml);
  inplaceNormalize(dispsm, 0.0f, 255.0f);
  Image<PixRGB<byte> > sacImg = Image<PixRGB<byte> >(toRGB(dispsm));
  for(uint i = 0; i < objRect.size(); i++)
    drawRect(sacImg,objRect[i],PixRGB<byte>(255,0,0));
  inplacePaste(disp, sacImg, Point2D<int>(w, 0));

  // draw coordinates of fixation for both input and sal image
  for(uint i = 0; i < salpt.size(); i++)
    {
      Point2D<int> salpt2(salpt[i].i + w, salpt[i].j);
      drawDisk(disp, salpt[i], foa_size/6+2, PixRGB<byte>(20, 50, 255));
      drawDisk(disp, salpt[i], foa_size/6,   PixRGB<byte>(255, 255, 20));
      drawDisk(disp, salpt2,   foa_size/6+2, PixRGB<byte>(20, 50, 255));
      drawDisk(disp, salpt2,   foa_size/6,   PixRGB<byte>(255, 255, 20));
    }

  // draw the SE bounding box
  Image< PixRGB<byte> > roiImg = ima;
  for(uint i = 0; i < objRect.size(); i++)
    {
      drawRect(roiImg, objRect[i], PixRGB<byte>(255,255,0));
      drawDisk(roiImg, salpt[i], 3, PixRGB<byte>(255,0,0));
      std::string ntext(sformat("%d", i));
      writeText(roiImg, salpt[i], ntext.c_str());
    }
  inplacePaste(disp, roiImg, Point2D<int>(w*2, 0));

  // if there is a previous image
  if(prevIma.initialized())
    {
      // display the current landmark tracked
      Image< PixRGB<byte> > clmDisp = prevIma;
        for(uint i = 0; i < clmpt.size(); i++)
          {
            if(clmpt[i].isValid())
              {
                drawDisk(clmDisp, clmpt[i], 3, PixRGB<byte>(0,255,0));
                std::string ntext(sformat("%d", i));
                writeText(clmDisp, clmpt[i], ntext.c_str());
              }
          }
        inplacePaste(disp, clmDisp, Point2D<int>(w*3, 0));

        // display the next landmark tracked
        Image< PixRGB<byte> > nlmDisp = prevIma;
        for(uint i = 0; i < nlmpt.size(); i++)
          {
            if(nlmpt[i].isValid())
              {
                drawDisk(nlmDisp, nlmpt[i], 3, PixRGB<byte>(0,255,255));
                std::string ntext(sformat("%d", i));
                writeText(nlmDisp, nlmpt[i], ntext.c_str());
              }
          }
        inplacePaste(disp, nlmDisp, Point2D<int>(w*4, 0));
    }

  // display the image
  win->drawImage(disp,0,0);
  //Raster::waitForKey();
}

// ######################################################################
int beoChipProc(rutz::shared_ptr<BeobotBeoChipListener> lis,
                nub::soft_ref<BeoChip> b)
{
  // print keyboard values:
  char kb[6]; kb[5] = '\0';
  for (int i = 0; i < 5; i ++) kb[i] = (lis->kbd>>(4-i))&1 ? '1':'0';

  // quit if both extreme keys pressed simultaneously:
  if (kb[0] == '0' && kb[4] == '0') {
    b->lcdPrintf(15, 0, "QUIT ");

    // return QUIT signal
    return BC_QUIT_SIGNAL;
  }

  return BC_NO_SIGNAL;
}

// ######################################################################
void getBbmtResults
( nub::ref<BeobotBrainMT> bbmt,
  Image<PixRGB<byte> > &currIma, Image<double> &cgist,
  Image<float> &currSalMap, ImageSet<float> &cmap,
  std::vector<Point2D<int> > &salpt,  std::vector<std::vector<double> > &feat,
  std::vector<Rectangle> &objRect)
{
  // current image, gist vector, and saliency map
  currIma    = bbmt->getCurrImage();
  cgist      = bbmt->getGist();
  currSalMap = bbmt->getSalMap();

  // current conspicuity maps
  for(uint i = 0; i < NUM_CHANNELS; i++) cmap[i] = bbmt->getCurrCMap(i);

  salpt.clear(); objRect.clear();
  uint numpt = bbmt->getNumSalPoint();
  for(uint i = 0; i < numpt; i++)
    {
      salpt.push_back(bbmt->getSalPoint(i));
      objRect.push_back(bbmt->getObjRect(i));

      std::vector<double> features; bbmt->getSalientFeatures(i, features);
      feat.push_back(features);

    }
}

// ######################################################################
bool checkNode
( int opMode, nub::soft_ref<Beowulf> beo,
  int32 &rnode, TCPmessage &rmsg, int32 &raction, int32 &rframe)
{
  // non-blocking call
  if(beo->receive(rnode, rmsg, rframe, raction))  return true;
  // && (rmsg.getSize() == 0));
  return false;
}

// ######################################################################
bool checkInput(int opMode, bool resetNextLandmark, uint64 inputFrameRate,
                uint64 inputTime)
{
  // if train mode: need to have a ventral reset signal
  if(opMode == TRAIN_MODE && resetNextLandmark) return true;

  // if test mode and using infinite time: need reset signal
  if(opMode == TEST_MODE && inputFrameRate == 0 &&
     resetNextLandmark) return true;

  // else test mode and need to get in after time is up
  if(opMode == TEST_MODE && inputFrameRate != 0 &&
     (inputFrameRate - ERR_INTERVAL) < inputTime) return true;

  return false;
}

// ######################################################################
void setupVentralPacket
( TCPmessage  &smsg, int rframe,
  bool resetVentralSearch, Image< PixRGB<byte> > currIma,
  Image<double> cgist, std::vector<Point2D<int> > salpt,
  std::vector<std::vector<double> > feat, std::vector<Rectangle> objRect,
  float dx, float dy, uint snumGT, float ltravGT)
{
  // FIX SEND TO VENTRAL
  // put a label that we are trying to track
  // and recognize the current tracked object
  // while we also want to track a new location for next

  // signal if we need to reset the landmark search
  smsg.reset(rframe, SEARCH_LM);
  smsg.addInt32(int32(resetVentralSearch));

  // send the full image (cropping done on the other side)
  smsg.addImage(currIma);

  // send the gist vector
  smsg.addInt32(int32(cgist.getSize()));
  Image<double>::iterator aptr = cgist.beginw();
  for(int i = 0; i < cgist.getSize(); i++) smsg.addDouble(*aptr++);

  // send the number of salient points included
  smsg.addInt32(int32(salpt.size()));
  for(uint i = 0; i < salpt.size(); i++)
    {
      // salient point
      smsg.addInt32(int32(salpt[i].i));
      smsg.addInt32(int32(salpt[i].j));

      // top, left, bottom, right
      smsg.addInt32(int32(objRect[i].top()));
      smsg.addInt32(int32(objRect[i].left()));
      smsg.addInt32(int32(objRect[i].bottomO()));
      smsg.addInt32(int32(objRect[i].rightO()));
      smsg.addInt32(int32(feat[i].size()));
      LDEBUG("[%u] salpt:(%s) r:[%s]", i,
             convertToString(salpt[i]).c_str(),
             convertToString(objRect[i]).c_str());
      for(uint j = 0; j < feat[i].size(); j++)
          smsg.addDouble(feat[i][j]);
    }

  // pass in the odometry information
  smsg.addFloat(dx);
  smsg.addFloat(dy);

  // pass in the ground truth
  smsg.addInt32(int32(snumGT));
  smsg.addFloat(ltravGT);

  LDEBUG("setup VENTRAL %d", rframe);
}

// ######################################################################
void setupDorsalPacket
( TCPmessage  &smsg, int rframe, ImageSet<float> cmap,
  bool resetCurrLandmark, std::vector<Point2D<int> > clmpt,
  bool resetNextLandmark, std::vector<Point2D<int> > nlmpt)
{
  // tracking packet: always have the cmap
  smsg.reset(rframe, TRACK_LM);
  smsg.addImageSet(cmap);
  smsg.addInt32(int32(resetCurrLandmark));
  smsg.addInt32(int32(resetNextLandmark));
  smsg.addInt32(int32(clmpt.size()));
  LDEBUG("[%4d] reset currLM? %d nextLM? %d",
         rframe, resetCurrLandmark, resetNextLandmark);

  for(uint i = 0; i < clmpt.size(); i++)
    {
      smsg.addInt32(int32(clmpt[i].i));
      smsg.addInt32(int32(clmpt[i].j));
      LDEBUG("curr[%d]: (%3d, %3d)", i, clmpt[i].i, clmpt[i].j);
    }

  smsg.addInt32(int32(nlmpt.size()));
  for(uint i = 0; i < nlmpt.size(); i++)
    {
      smsg.addInt32(int32(nlmpt[i].i));
      smsg.addInt32(int32(nlmpt[i].j));
      LDEBUG("next[%d]: (%3d, %3d)", i, nlmpt[i].i, nlmpt[i].j);
    }
}

// ######################################################################
void processVentralResult
( TCPmessage &rmsg, int32 raction, int32 rframe,
  std::vector<Point2D<int> > &clmpt, std::vector<Point2D<int> > &nlmpt,
  bool &resetCurrLandmark, bool &resetNextLandmark)
{
  if(raction == SEARCH_LM_RES)
    {
      LDEBUG("Packet size: %d", rmsg.getSize());
      bool isLocalized = true;
      int searchFrame = int(rmsg.getElementInt32());
      int sentAt      = int(rmsg.getElementInt32());
      rmsg.reset(rframe, raction);
      LDEBUG("Ventral processing[%d]: SEARCH_LM_RES: %d sent at %d\n",
             rframe, searchFrame, sentAt);

      // check the return type:
      // CURR_LM, NEXT_LM, BOTH_LM
      // CURR_LM: for feedback for current landmark dorsal tracking
      // NEXT_LM: for feedback for next landmark dorsal tracking
      // BOTH_LM: for feedback for both landmarks dorsal tracking

      // NEW FUNKY IDEA:
      // TRACKED_OBJ_W_BIAS, TRACKED_OBJECT_WO_BIAS
      // when the object is about to be passed
      // use the frame num of the DB

      // get the object setup the tracking:

      // if current landmark is identified
      //   we can keep tracking
      // -> note that it also returns the proper location in the memory
      //    that is, at which side of the corridor it was located & how far away
      //    we can use the getAffineDiff
      //  horizontal comp: - val: move right, + val: move left
      // otherwise tracking is lost

      // if next landmark is identified
      if(isLocalized)
        {
          // usually we make the new Landmark be the current one
          clmpt.clear();
          for(uint i = 0; i < nlmpt.size(); i++)
            clmpt.push_back(nlmpt[i]);
          resetCurrLandmark = true;
          // also try to bias to the next landmark
        }
      else
        {
          // otherwise, nothing to do
          //we'll just start a new search
        }
      resetNextLandmark = true;
    }
}

// ######################################################################
void processDorsalResult
(TCPmessage &rmsg,  int32 raction, int32 rframe,
 std::vector<Point2D<int> > &clmpt, std::vector<Point2D<int> > &nlmpt,
 bool &resetNextLandmark, float &sp, float &st)
{
  if(raction == TRACK_LM_RES)
    {
      LDEBUG("Dorsal TRACK_LM_RES[%4d]", rframe);
      uint nClmpt = (rmsg.getElementInt32());
      clmpt.clear();
      for(uint i = 0; i < nClmpt; i++)
        {
          uint cI = int(rmsg.getElementInt32());
          uint cJ = int(rmsg.getElementInt32());
          clmpt.push_back(Point2D<int>(cI,cJ));
          if(clmpt[i].isValid())
            LDEBUG("track currLM[%d]: (%3d, %3d)",
                   i, clmpt[i].i, clmpt[i].j);
          else
            LINFO("track currLM[%d]: (%3d, %3d) LOST",
                  i, clmpt[i].i, clmpt[i].j);
        }

      uint nNlmpt = (rmsg.getElementInt32());
      nlmpt.clear();
      for(uint i = 0; i < nNlmpt; i++)
        {
          uint nI = int(rmsg.getElementInt32());
          uint nJ = int(rmsg.getElementInt32());
          nlmpt.push_back(Point2D<int>(nI,nJ));
          if(nlmpt[i].isValid())
            LDEBUG("track nextLM[%d]: (%3d, %3d)",
                   i, nlmpt[i].i, nlmpt[i].j);
          else
            LINFO("track nextLM[%d]: (%3d, %3d) LOST",
                  i, nlmpt[i].i, nlmpt[i].j);
        }
      rmsg.reset(rframe, raction);

      //FIX: there is a time where tracking would fail

      // scenarios:
      // track currlm only: when the segment is about to end
      // track nextlm only: when starting a segment
      // track both: in between 2 landmarks

      // if current landmark is lost
      //if(false)//!clmptFound)
      //  {
          // stop motor:
          // have to wait for SEARCH_LM to return an answer
          // tell them to take all the time it needs
      //    sp = 0.0f; st = 0.0f;
      //  }

      // if next landmark is lost
      //if(!nlmptFound)
      //  {
          // stop DB search in Ventral
          // there's no use, it's already past
      //    resetNextLandmark = true;
      //    LINFO("stop Ventral Search");
      //  }

      // if both points are found
      //if(clmptFound && nlmptFound)
      //  {
          // process for best motor command
          // that keeps all points within sight

          // FIX: how do we home location in this case ??
      //  }

      // RESEARCH: if both tracked landmarks are lost:
      // add motion energy - run straight slowly
      // need special exploration stuff: scan in place 180 degrees
    }
}

// ######################################################################
//! setup the ground truth
//! given an annotation file create a ground truth file
void setupGroundTruth(std::string gt_filename, bool writeGTfile)
{
  //gt_filename = std::string("in.txt");

  // open the file
  LINFO("opening: %s", gt_filename.c_str());
  FILE *fp;  char inLine[200]; char comment[200];
  if((fp = fopen(gt_filename.c_str(),"rb")) == NULL)
    LFATAL("%s not found", gt_filename.c_str());

  // set the ground truth filename 
  //int ldpos = gt_filename.find_last_of('.');
  std::string output_gt_filename = std::string("out.txt");
    //gt_filename.substr(0, ldpos) + std::string("_gt.txt");

  // read out and discard the first line
  if (fgets(inLine, 200, fp) == NULL) LFATAL("fgets failed"); 
  
  FILE *oFile = fopen(output_gt_filename.c_str(), "at");
  if (oFile != NULL) 
    { if(writeGTfile) fputs(inLine, oFile); fclose (oFile); }

  while(fgets(inLine, 200, fp) != NULL)
    {
      int segnum; float dist;
      int ret = sscanf(inLine, "%d %f", &segnum, &dist);
      LINFO("RET: %d [%d %f]", ret, segnum, dist);

      groundTruthSegmentLength.push_back(dist);

      // check if we are still in segment length section
      if(ret < 2) break;
      
      // write in the image
      FILE *oFile = fopen(output_gt_filename.c_str(), "at");
      if (oFile != NULL)
        { if(writeGTfile) fputs(inLine, oFile); fclose (oFile); }
    }

  oFile = fopen(output_gt_filename.c_str(), "at");
  if (oFile != NULL) 
    { if(writeGTfile) fputs(inLine, oFile); fclose (oFile); }

  // read each line 
  int curr_seg = 0; int last_fnum = 0; float last_ldist = 0.0F;
  while(fgets(inLine, 200, fp) != NULL)
    {
      // get the three information, discard comments
      int snum; char t_iname[200]; float ldist;
      sscanf(inLine, "%s %d %f %s", t_iname, &snum, &ldist, comment);
      curr_seg = snum;

      // get the frame number
      std::string iname(t_iname);
      int ldpos = iname.find_last_of('.');
      int lupos = iname.find_last_of('_');
      std::string tnum = iname.substr(lupos+1, ldpos - lupos - 1);
      LDEBUG("fname: %s %s [%d %f]", t_iname, tnum.c_str(), snum, ldist);
      int fnum  = atoi(tnum.c_str());

      // if not at the start of a segment
      int start_fnum = last_fnum + 1; 
      if(ldist == 0.0)
        { start_fnum = fnum; last_fnum = fnum; last_ldist = ldist; } 

      float range = float(fnum-last_fnum);
      if(range == 0.0F) range = 1.0F;

      //LINFO("start_fnum: %d last_fnum: %d fnum: %d", start_fnum, last_fnum, fnum);

      // 
      for(int i = start_fnum; i <= fnum; i++)
        {
          //
          std::string cfname = 
            sformat("%s_%06d.ppm", iname.substr(0,lupos).c_str(), i );
          float c_ldist = 
            float(i - last_fnum)/range*(ldist-last_ldist) + last_ldist;
          
          // write in the image
          oFile = fopen(output_gt_filename.c_str(), "at");
          if (oFile != NULL)
            {
              //LINFO("appending session to %s", sessionFName.c_str());
              if(writeGTfile)
                {
                  std::string line =
                    sformat("%-40s %-10d %-20f\n",
                            cfname.c_str(), curr_seg, c_ldist);
                  fputs(line.c_str(), oFile);
                }
              fclose (oFile);
            }
          groundTruth.push_back(GroundTruth(i,curr_seg,c_ldist));
        }        

      last_fnum  = fnum;
      last_ldist = ldist;
    }

  LINFO("have a ground truth file of size: %d", int(groundTruth.size()));
}

// ######################################################################
void getGroundTruth
(uint fNum, uint &snumGT, float &ltravGT, float &dx, float &dy)
{
  uint gt_num = groundTruth.size();

  // check if fNum is above the range  
  if(fNum >= gt_num-1) 
    { 
      snumGT  = groundTruth[gt_num-1].snum; 
      ltravGT = groundTruth[gt_num-1].ltrav;
    }
  else
    {
      snumGT  = groundTruth[fNum].snum; 
      ltravGT = groundTruth[fNum].ltrav;
    }

  float slen = groundTruthSegmentLength[snumGT];
  LDEBUG("[%d]: %d %f/%f = %f", fNum, snumGT, ltravGT, slen, ltravGT/slen);
  ltravGT = ltravGT/slen;

  // add random error to the robot movement
  dx = 0.0f; dy = 0.0f;
  dy =+ 0.0F; //FIX
}

// the one that uses frame numbers
// // ######################################################################
// void getGroundTruth
// (uint fNum, uint &snumGT, float &ltravGT, float &dx, float &dy)
// {
//   int gt_num = groundTruth.size();

//   // check if fNum is below the range  
//   if(fNum < groundTruth[0].fnum) 
//     { 
//       snumGT  = groundTruth[0].snum; 
//       ltravGT = groundTruth[0].ltrav;
//     }

//   // check if fNum is above the range  
//   else if(gt_num > 0 && fNum > groundTruth[gt_num-1].fnum) 
//     { 
//       snumGT  = groundTruth[gt_num-1].snum; 
//       ltravGT = groundTruth[gt_num-1].ltrav;
//     }
  
//   else
//     {
//       snumGT  = groundTruth[fNum].snum; 
//       ltravGT = groundTruth[fNum].ltrav;
//     }

//   float slen = groundTruthSegmentLength[snumGT];
//   ltravGT = ltravGT/slen;

//   // add random error to the robot movement
//   dx = 0.0f; dy = 0.0f;
//   dy =+ 0.0F; //FIX
// }

// ######################################################################
//   // ACB IROS 07 test
//   uint sfnum [9] = {  377,  496,  541,  401,  304,  555,  486,  327,  307 };
//   uint ssfnum[9] = {    0,  377,  873, 1414, 1815, 2119, 2674, 3160, 3487 };

//   // AnFpark IROS 07 test
//   uint sfnum [9] = {  866,  569,  896,  496,  769, 1250,  588,  844,  919 };
//   uint ssfnum[9] = {    0,  866, 1435, 2331, 2827, 3596, 4846, 5434, 6278 };

//   // FDFpark IROS 07 test
//   uint sfnum [9] = {  782,  813,  869,  795,  857, 1434,  839, 1149,  749 };
//   uint ssfnum[9] = {    0,  782, 1595, 2464, 3259, 4116, 5550, 6389, 7538 };

  // =====================================================================
//   // ACB_1 IEEE-TR 07 test T_A
//   std::string stem("/lab/tmpib/u/siagian/PAMI07/ACB/gist/ACB");
//   uint sfnum [9] = {  387,  440,  465,  359,  307,  556,  438,  290,  341 };
//   uint ssfnum[9] = {    0,  387,  827, 1292, 1651, 1958, 2514, 2952, 3242 };

//   // ACB_2 IEEE-TR 07 test T_B
//   std::string stem("/lab/tmpib/u/siagian/PAMI07/ACB/gist/ACB");
//   uint sfnum [9] = {  410,  436,  485,  321,  337,  495,  445,  247,  373 };
//   uint ssfnum[9] = {    0,  410,  846, 1331, 1652, 1989, 2484, 2929, 3176 };

  // // ACB_3 IEEE-TR 07 test T_E
  // std::string stem("/lab/tmpib/u/siagian/PAMI07/ACB/gist/ACB");
  // uint sfnum [9] = {  388,  461,  463,  305,  321,  534,  398,  274,  313 };
  // uint ssfnum[9] = {    0,  388,  849, 1312, 1617, 1938, 2472, 2870, 3144 };

//   // ACB_4 IEEE-TR 07 test T_F
//   std::string stem("/lab/tmpib/u/siagian/PAMI07/ACB/gist/ACB");
//   uint sfnum [9] = {  411,  438,  474,  249,  319,  502,  400,  288,  296 };
//   uint ssfnum[9] = {    0,  411,  849, 1323, 1572, 1891, 2393, 2793, 3081 };

//   // AnFpark_1 IEEE-TR 07 test D
//   std::string stem("/lab/tmpib/u/siagian/PAMI07/AnFpark/gist/AnFpark");
//   uint sfnum [9] = {  698,  570,  865,  488,  617, 1001,  422,  598,  747 };
//   uint ssfnum[9] = {    0,  698, 1268, 2133, 2621, 3238, 4239, 4661, 5259 };

//   // AnFpark_2 IEEE-TR 07 test J
//   std::string stem("/lab/tmpib/u/siagian/PAMI07/AnFpark/gist/AnFpark");
//   uint sfnum [9] = {  802,  328,  977,  597,  770, 1122,  570,  692,  809 };
//   uint ssfnum[9] = {    0,  802, 1130, 2107, 2704, 3474, 4596, 5166, 5858 };

//   // AnFpark_3 IEEE-TR 07 test T_A
//   std::string stem("/lab/tmpib/u/siagian/PAMI07/AnFpark/gist/AnFpark");
//   uint sfnum [9] = {  891,  474,  968,  688,  774, 1003,  561,  797,  862 };
//   uint ssfnum[9] = {    0,  891, 1365, 2333, 3021, 3795, 4798, 5359, 6156 };

//   // AnFpark_4 IEEE-TR 07 test T_B
//   std::string stem("/lab/tmpib/u/siagian/PAMI07/AnFpark/gist/AnFpark");
//   uint sfnum [9] = {  746,  474,  963,  632,  777, 1098,  399,  768,  849 };
//   uint ssfnum[9] = {    0,  746, 1220, 2183, 2815, 3592, 4690, 5089, 5857 };

//   // FDFpark_1 IEEE-TR 07 test T_C
//   std::string stem("/lab/tmpib/u/siagian/PAMI07/FDFpark/gist/FDFpark");
//   uint sfnum [9] = {  881,  788,  858,  837,  831, 1680, 1037, 1172,  739 };
//   uint ssfnum[9] = {    0,  881, 1669, 2527, 3364, 4195, 5875, 6912, 8084 };

//   // FDFpark_2 IEEE-TR 07 test T_D
//   std::string stem("/lab/tmpib/u/siagian/PAMI07/FDFpark/gist/FDFpark");
//   uint sfnum [9] = {  670,  740,  696,  740,  748, 1565,  923, 1211,  825 };
//   uint ssfnum[9] = {    0,  670, 1410, 2106, 2846, 3594, 5159, 6082, 7293 };

//   // FDFpark_3 IEEE-TR 07 test T_H
//   std::string stem("/lab/tmpib/u/siagian/PAMI07/FDFpark/gist/FDFpark");
//   uint sfnum [9] = {  847,  797,  922,  837,  694, 1712,  857, 1355,  794 };
//   uint ssfnum[9] = {    0,  847, 1644, 2566, 3403, 4097, 5809, 6666, 8021 };

//   // FDFpark_4 IEEE-TR 07 test T_I
//   std::string stem("/lab/tmpib/u/siagian/PAMI07/FDFpark/gist/FDFpark");
//   uint sfnum [9] = {  953,  878,  870,  821,  854, 1672,  894, 1270,  743 };
//   uint ssfnum[9] = {    0,  953, 1831, 2701, 3522, 4376, 6048, 6942, 8212 };

  // =====================================================================
//   // ACB_1 ICRA 08 test T_A
//   std::string stem("/lab/tmpib/u/siagian/PAMI07/ACB/gist/ACB");
//   uint sfnum [9] = {  387,  440,  465,  359,  307,  556,  438,  290,  341 };
//   uint ssfnum[9] = {    0,  387,  827, 1292, 1651, 1958, 2514, 2952, 3242 };

//   // AnFpark_1 IEEE-TR 07 test D
//   std::string stem("/lab/tmpib/u/siagian/PAMI07/AnFpark/gist/AnFpark");
//   uint sfnum [9] = {  698,  570,  865,  488,  617, 1001,  422,  598,  747 };
//   uint ssfnum[9] = {    0,  698, 1268, 2133, 2621, 3238, 4239, 4661, 5259 };

//   // FDFpark_1 IEEE-TR 07 test T_C
//   std::string stem("/lab/tmpib/u/siagian/PAMI07/FDFpark/gist/FDFpark");
//   uint sfnum [9] = {  881,  788,  858,  837,  831, 1680, 1037, 1172,  739 };
//   uint ssfnum[9] = {    0,  881, 1669, 2527, 3364, 4195, 5875, 6912, 8084 };

  // // HNB testing for Loc & Nav
  // std::string stem("../data/HNB_T/gist/");
  // uint sfnum [4] = {9701,  653,  521,  353 };
  // uint ssfnum[4] = {   0,  655, 1308, 1829 };
  // //uint ssfnum[4] = { 21696, 22351, 23004, 23525 };

  // //FILE *gfp;
  // int afnum = 0;
  // //for(uint i = 0; i < 9; i++)
  // for(uint i = 0; i < 4; i++)
  //   {
  //     if((fNum >= ssfnum[i]) && (fNum < ssfnum[i]+sfnum[i]))
  //       {
  //         stem = stem + sformat("_T_%dE", i+1);
  //         afnum = fNum - ssfnum[i];
  //         dx = 1.0/float(sfnum[i]); dy = 0.0F;
  //         snumGT = i; ltravGT = float(afnum)/float(sfnum[i]);
  //         break;
  //       }
  //   }

//   ltravGT = fNum/866.0; dx = 1/866.0; afnum = fNum;
//   stem = sformat("/lab/tmpib/u/siagian/PAMI07/ACB/gist/ACB1B");

// ######################################################################
void reportResults(std::string resultPrefix, uint nsegment)
{
  // NOTE: for crash recovery
  //  resultPrefix = std::string("/lab/tmpib/u/siagian/PAMI07/FDFpark/envComb/RES/FDFpark_T_C_track/AAAI/FDFpark_500");
  //GSnavResult r1; r1.combine(resultPrefix);
  //r1.read(resultPrefix, nsegment);
  //r1.createSummaryResult();

  // read and print the timing information
  GSnav_M_Result r2;  r2.read(resultPrefix);

//   std::string resCfName = savePrefix + sformat("_comb_results.txt");
//   FILE *rFile = fopen(resCfName.c_str(), "at");
//   if (rFile == NULL) LFATAL("can't create res file: %s", resCfName.c_str());

  LINFO("bbmtTime: %f (%f - %f) std: %f",
        r2.meanBbmtTime, r2.minBbmtTime, r2.maxBbmtTime,
        r2.stdevBbmtTime);

  LINFO("Dorsal Time: %f (%f - %f) std: %f",
        r2.meanDorsalTime, r2.minDorsalTime, r2.maxDorsalTime,
        r2.stdevDorsalTime);

  LINFO("Ventral Time: %f (%f - %f) std: %f",
        r2.meanVentralSearchTime, r2.minVentralSearchTime,
        r2.maxVentralSearchTime, r2.stdevVentralSearchTime);

  LINFO("Input Time: %f (%f - %f) std: %f",
        r2.meanInputTime, r2.minInputTime, r2.maxInputTime,
        r2.stdevInputTime);
//   fputs(temp.c_str(), rFile);
//   fclose (rFile);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
