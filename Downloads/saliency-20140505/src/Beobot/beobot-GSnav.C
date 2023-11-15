/*!@file Beobot/beobot-GSnav.C Robot navigation using saliency and gist.
  Run beobot-GSnav-master at CPU_A to run Gist-Saliency model
  Run beobot-GSnav        at CPU_B to run SIFT                          */
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
// ///////////////////////////////////////////////////////////////////////
//
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/beobot-GSnav.C $
// $Id: beobot-GSnav.C 15454 2013-01-31 02:17:53Z siagian $
//
//////////////////////////////////////////////////////////////////////////
// beobot-Gist-Sal-Nav.C <input_train.txt>
//
//
// This is an on-going project for biologically-plausible
// mobile-robotics navigation.
// It accepts any inputs: video  clip <input.mpg>, camera feed, frames.
//
// The system uses Gist to recognize places and saliency
// to get better localization within the place
// The program also is made to be streamline for fast processing using
// parallel computation. That is the V1 features of different channels are
// computed in parallel
//
// Currently it is able to recognize places through the use of gist features.
// The place classifier is a neural networks,
// passed in a form of <input_train.txt> -
// the same file is used in the training phase by train-FFN.C.
// Related files of interest: GistEstimator.C (and .H) and
// GistEstimatorConfigurator.C (and .H) used by Brain.C to compute
// the gist features.
// test-Gist.C uses GistEstimator to extract gist features from an image.
//
// In parallel we use saliency to get a better spatial resolution
// as well as better place accuracy. The saliency model is used to obtain
// salient locations. We then use ShapeEstimator algorithm to segment out
// the sub-region to get a landmark. Using the pre-attentive features and
// SIFT we can identify the object, create a database, etc.
//
// for localization, path planning we perform landmark-hopping
// to get to the final destination
//
//
//

/*
maybe need to figure out how the shape of the object change on our movement

use scale
temporal ordering, frame number
winning channel, channel weights

can also play with biasing of where the next object could be
or bias away from (anywhere but) the salpt to see if there is another object.

benefit of temporal ordering:
-> figure out the best time the next LM should kick in
-> best way to turn at the end of a segment.

for current landmark tracking remember fnum of the previous match
so that you won't have to start over

Maybe we should be doing biased sal (as oppose to straight sal)
even in the training process

important landmarks to track:

guide you to the next landmark -> biased salient loc

currently in db processing

kept on getting salient hits -> plain salient loc

=============

scenarios:
have an object
we're done what next
we're almost done
we're following the point, any changes
we're lost what do we do now.

==============
*/

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Raster/Raster.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "GUI/XWinManaged.H"
#include "Util/Timer.H"

#include "Beobot/BeobotControl.H"

#include "SIFT/VisualObject.H"
#include "SIFT/VisualObjectDB.H"
#include "SIFT/Histogram.H"

#include "Image/MathOps.H"      // for inPlaceNormalize()
#include "Image/CutPaste.H"     // for inplacePaste()
#include "Image/DrawOps.H"      // for drawing
#include "Image/ShapeOps.H"     // for decXY()
#include "Image/MatrixOps.H"    // for matrixMult(), matrixInv(), etc

#include "Beobot/beobot-GSnav-def.H"
#include "Beobot/GSnavResult.H"
#include "Beobot/Environment.H"
#include "Beobot/GSlocalizer.H"

#include <signal.h>

#include <errno.h>

static bool goforever = true;  //!< Will turn false on interrupt signal

// ######################################################################
// ######################################################################

//! process the message passed by Visual Cortex
void getSearchCommand
( TCPmessage &rmsg, int32 rframe, int32 raction, bool &resetVentralSearch,
  Image<PixRGB<byte> > &ima, Image<double> &cgist,
  std::vector<rutz::shared_ptr<VisualObject> > &inputVO,
  std::vector<Point2D<int> > &objOffset,
  std::string saveFilePath, std::string testRunFPrefix,
  float &dx, float &dy, uint &snumGT, float &ltravGT);

//! display the salient objects passed by Visual Cortex program
Image<PixRGB<byte> > getSalImage
( Image<PixRGB<byte> > ima,
  std::vector<rutz::shared_ptr<VisualObject> > inputVO,
  std::vector<Point2D<int> > objOffset,  std::vector<bool> found);

//! process the returned search result
Image<PixRGB<byte> > processLocalizerResults
( nub::ref<GSlocalizer> gslocalizer, std::string savePrefix);

Image<PixRGB<byte> > getDisplayImage(nub::ref<GSlocalizer> gslocalizer);

Image<PixRGB<byte> > getFoundSalObjImage (nub::ref<GSlocalizer> gslocalizer);

//! save gist vector to a file
bool saveGist
( std::string saveFilePath, std::string testRunFPrefix,
  int currSegNum, int count, Image<double> cgist);

//! report the results of the trial run
void reportResults(std::string savePrefix, uint nsegment);

//! report the results of all the trial runs
//! it's a hack code
void reportTrialResults();

// ######################################################################
//! Signal handler (e.g., for control-C)
void terminate(int s) { LERROR("*** INTERRUPT ***"); goforever = false; }

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  // createGroundTruthFile
  //   (std::string("/lab/raid2/beobot2/logs/2012_12_10__16_05_53_Image_groundtruth.txt"));
  // Raster::waitForKey();

  // instantiate a model manager:
  ModelManager manager("beobot Navigation using Gist & Saliency - "
                       "Ventral");

  // Instantiate our various ModelComponents:
  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Slave", "BeowulfSlave", false));
  manager.addSubComponent(beo);

  nub::ref<GSlocalizer> gslocalizer(new GSlocalizer(manager));
  manager.addSubComponent(gslocalizer);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false)
    return(1);

  // setup signal handling:
  signal(SIGHUP,  terminate); signal(SIGINT,  terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);

  TCPmessage rmsg;            // message being received and to process
  TCPmessage smsg;            // message being sent
  int32 rframe, raction, rnode = -1;

  // let's get all our ModelComponent instances started:
  manager.start();

  // receive the image dimension first
  while(!beo->receive(rnode, rmsg, rframe, raction, 5));
  LINFO("Ventral size: %d", rmsg.getSize());
  const int fstart = int(rmsg.getElementInt32());
  const int w = int(rmsg.getElementInt32());
  const int h = int(rmsg.getElementInt32());
  const int opMode = int(rmsg.getElementInt32());
  const int nSegment = int(rmsg.getElementInt32());
  int currSegNum = int(rmsg.getElementInt32());
  std::string envFName = rmsg.getElementString();
  std::string testRunFPrefix = rmsg.getElementString();
  std::string saveFilePath = rmsg.getElementString();
  std::string resultPrefix = rmsg.getElementString();
  int ldpos = envFName.find_last_of('.');
  int lspos = envFName.find_last_of('/');
  std::string testRunEnvFName = envFName.substr(0, ldpos) +
    std::string("-") + testRunFPrefix + std::string(".env");
  std::string envPrefix =  envFName.substr(lspos+1, ldpos - lspos - 1);

  LINFO("fstart: %d", fstart);
  LINFO("envFName: %s", envFName.c_str());
  LINFO("Where we save data: %s", saveFilePath.c_str());
  LINFO("envPrefix: %s", envPrefix.c_str());
  LINFO("testRunEnvFName: %s", testRunEnvFName.c_str());
  LINFO("result prefix: %s", resultPrefix.c_str());
  LINFO("test run prefix: %s", testRunFPrefix.c_str());
  LINFO("Image dimension %d by %d", w,h);
  switch(opMode)
    {
    case TRAIN_MODE:
      LINFO("number of segments:  %d", nSegment);
      LINFO("current segment: %d", currSegNum);
      LINFO("TRAIN_MODE: build the landmark DB");
      break;
    case TEST_MODE:
      LINFO("TEST_MODE: let's roll");
      break;
    default: LERROR("Unknown operation mode");
    }
  rmsg.reset(rframe, raction);

  // create a gist folder
  std::string gistFolder =
    saveFilePath + std::string("gist/");
  if (mkdir(gistFolder.c_str(), 0777) == -1 && errno != EEXIST)
    {
      LFATAL("Cannot create gist folder: %s", gistFolder.c_str());
    }

  // create a gistTrain folder
  std::string gistTrainFolder =
    saveFilePath + std::string("gistTrain/");
  if (mkdir(gistTrainFolder.c_str(), 0777) == -1 && errno != EEXIST)
    {
      LFATAL("Cannot create gistTrain folder: %s", gistTrainFolder.c_str());
    }

  // HACK: this is for printing summary results
  //reportTrialResults();
  //reportResults(resultPrefix,9); Raster::waitForKey();

  rutz::shared_ptr<XWinManaged> inputWin;
  inputWin.reset(new XWinManaged(Dims(w, h), 2*w+ 20, 0, "Input Window" ));
  rutz::shared_ptr<XWinManaged> matchWin;
  matchWin.reset(new XWinManaged(Dims(2*w, 2*h), 0, 0, "Result Window" ));
  rutz::shared_ptr<XWinManaged> mainWin;

  // initialize an environment - even if the .env file does not exist
  // automatically create a blank environment, ready to be built
  rutz::shared_ptr<Environment> env(new Environment(envFName));
  env->setWindow(matchWin);

  // training: set the segment and current segment explicitly
  if(opMode == TRAIN_MODE)
    {
      // if the stored DB is already loaded,
      // don't reset it'll discard the stored DB
      if(env->getNumSegment() == 0) env->resetNumSegment(nSegment);

      env->setCurrentSegment(currSegNum);
      env->startBuild();
    }
  // else if(opMode == TRAIN_X_MODE)
  //   {
  //     env->startJunctionBuild();
  //   }
  else if(opMode == TEST_MODE)
    {
      // if environment object is blank then abort
      if(env->isBlank())
        LFATAL("empty environment .env file may not exist");

      // file prefix to save performance data
      gslocalizer->setSavePrefix(resultPrefix);

      // link the environment to the localizer
      gslocalizer->setEnvironment(env);

      // initialize the particles
      // in case we have a starting belief
      std::string sBelFName = resultPrefix +
        sformat("_bel_%07d.txt", fstart-1);
      LINFO("Starting belief file: %s", sBelFName.c_str());
      gslocalizer->initParticles(sBelFName);

      // set the beowulf access
      gslocalizer->setBeoWulf(beo);

      // set the window display as well
      gslocalizer->setWindow(matchWin);

      // this is the main window for display of test results
      mainWin.reset
        (new XWinManaged(Dims(5*w, 3*h), 0, 0, "Main Window" ));
    }
  else LFATAL("invalid opmode");

  // send a message of ready to go
  smsg.reset(rframe, INIT_DONE);
  beo->send(rnode, smsg);

  // MAIN LOOP
  LINFO("MAIN LOOP"); uint fNum = 0; std::vector<float> procTime;
  Timer tim(1000000); uint64 t[NAVG]; 
  uint snumGT; float ltravGT; // ground truth information
  while(goforever)
    {
      // if we get a new message from master
      // check the type of action to take with the data
      if(beo->receive(rnode, rmsg, rframe, raction, 2))
        {
          LDEBUG("[%d] message received: %d", rframe, raction);
          fNum = rframe;

          // abort: break out of the loop and finish the operation
          if(raction == ABORT)
            {
              LINFO("done: BREAK"); goforever = false;

              // if test mode and need to save last image
              // skip if this is the first input frame
              if(gslocalizer->getInputImage().initialized())
                {
                  // get whatever is available right now
                  // update the belief particles
                  gslocalizer->updateBelief();

                  // process localizer result
                  Image<PixRGB<byte> > dispIma =
                    processLocalizerResults(gslocalizer, resultPrefix);

                  uint lfnum = gslocalizer->getInputFnum();
                  mainWin->setTitle(sformat("Frame %d", lfnum).c_str());
                  mainWin->drawImage(dispIma,0,0);
                }
            }

          //FIX: maybe for inquiry if search is done
          // search not yet done
          //  smsg.reset(rframe, SEARCH_LM_RES);
          //  smsg.addInt32(int32(SEARCH_NOT_DONE));

          // Ventral track task
          // FIX: check with the prev landmark first

          // object received: recognize
          // train: build landmark db
          // test : search landmark db & localize
          if(raction == SEARCH_LM)
            {
              tim.reset();
              Image<PixRGB<byte> > ima; Image<double> cgist;
              std::vector<rutz::shared_ptr<VisualObject> > inputVO;
              std::vector<Point2D<int> > objOffset; std::vector<bool> found;
              float dx, dy; bool resetVentralSearch = false;
              getSearchCommand(rmsg, fNum, raction, resetVentralSearch,
                               ima, cgist, inputVO, objOffset,
                               saveFilePath, testRunFPrefix,
                               dx, dy, snumGT, ltravGT);
              for(uint i = 0; i < inputVO.size(); i++) found.push_back(false);
              LDEBUG("[%d] getSearchCommand: %f", fNum, tim.get()/1000.0);

              inputWin->drawImage
                (getSalImage(ima, inputVO, objOffset, found),0,0);
              inputWin->setTitle(sformat("fr: %d", fNum).c_str());

              // process based on the operation mode
              switch(opMode)
                {
                case TRAIN_MODE:
                  {
                    // create the visual object for the scene and salregs
                    std::string sName(sformat("scene%07d",fNum));
                    std::string sfName = sName + std::string(".png");
                    rutz::shared_ptr<VisualObject>
                      scene(new VisualObject(sName, sfName, ima));

                    for(uint i = 0; i < inputVO.size(); i++)
                      inputVO[i]->computeKeypoints();

                    // process input
                    env->build(inputVO, objOffset, fNum, scene);

                    // save the gist vector to the gist folder for training
                    saveGist(gistFolder, testRunFPrefix,
                             currSegNum, fNum, cgist);

                    smsg.reset(rframe, SEARCH_LM_RES);
                    smsg.addInt32(int32(fNum)); // always localized
                    smsg.addInt32(int32(fNum));
                    LINFO("DONE BUILD_LM: %d", fNum);
                    beo->send(rnode, smsg);
                    break;
                  }
                case TEST_MODE:
                  {
                    uint64 tSave = 0; uint64 t0;

                    // skip if this is the first input frame
                    if(gslocalizer->getInputImage().initialized())
                      {
                        // get whatever is available right now
                        // update the belief particles
                        gslocalizer->updateBelief();

                        // process localizer result
                        t0 = tim.get();
                        Image<PixRGB<byte> > dispIma =
                          processLocalizerResults
                          (gslocalizer, resultPrefix);
                        tSave = tim.get() - t0;

                        // DISPLAY ONCE EVERY 10 FRAMES
                        LDEBUG("\n\nfNum: %d",fNum);
                        //if(fNum%10 == 0)
                        //  {
                            uint lfnum =  gslocalizer->getInputFnum();
                            mainWin->setTitle
                              (sformat("Frame %d", lfnum).c_str());
                            mainWin->drawImage(dispIma,0,0);
                        //  }

                        // clear search and add a new set of tasks
                        if(resetVentralSearch)
                          {
                            // soft/non-blocking call
                            // to stop all search threads
                            LINFO("[%d] resetting ventral search", fNum);
                            gslocalizer->stopSearch2();
                          }
                      }

                    // FIX: if still searching, may want to append search
                    //      but now simply to take gist and (dx,dy)
                    t0 = tim.get();
                    gslocalizer->input
                      (ima, inputVO, objOffset, fNum, cgist, dx, dy);
                    gslocalizer->setGroundTruth(snumGT, ltravGT);
                    uint64 tInput = tim.get() - t0;

                    LDEBUG("tS = %f, tI = %f", tSave/1000.0, tInput/1000.0);
                  }
                break;
                default:
                  LERROR("Unknown operation mode");
                }

              // compute and show framerate over the last NAVG frames:
              t[fNum % NAVG] = tim.get();
              if (fNum % NAVG == 0 && fNum != 0)
                {
                  uint64 avg = 0ULL; for(int i=0; i<NAVG; i++) avg+=t[i];
                }
              uint64 ptime = tim.get(); float ptime2 = ptime/1000.0;
              LDEBUG("[%d] ventral proc time = %f", fNum, ptime2);
              procTime.push_back(ptime2);
            }
        }
    }

  // Received abort signal
  smsg.reset(rframe, raction);
  smsg.addInt32(int32(777));
  beo->send(rnode, smsg);
  LINFO("received ABORT signal");

  // ending operations
  switch(opMode)
    {
    case TRAIN_MODE:

      // finish up the training session
      env->finishBuild(envFName, testRunFPrefix, fNum);
      env->save(envFName, testRunEnvFName, envPrefix);
      break;

    case TEST_MODE:
      {
        // stop the gslocalizer - blocking code
        gslocalizer->stopSearch();

        // report some statistical data
        reportResults(resultPrefix, env->getNumSegment());

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

  // we got broken:
  manager.stop();
  Raster::waitForKey();
  return 0;
}

// ######################################################################
Image<PixRGB<byte> >  getSalImage
( Image<PixRGB<byte> > ima,
  std::vector<rutz::shared_ptr<VisualObject> > inputVO,
  std::vector<Point2D<int> > objOffset,
  std::vector<bool> found)
{
  int w = ima.getWidth();  int h = ima.getHeight();
  Image<PixRGB<byte> > dispIma(w,h,ZEROS);
  inplacePaste(dispIma,ima, Point2D<int>(0,0));

  // display the salient regions
  // and indicate if each is found or not
  LDEBUG("number of input objects: %" ZU , inputVO.size());
  for(uint i = 0; i < inputVO.size(); i++)
    {
      Rectangle r(objOffset[i], inputVO[i]->getImage().getDims());
      Point2D<int> salpt = objOffset[i] + inputVO[i]->getSalPoint();
      if(!found[i])
        {
          drawRect(dispIma,r,PixRGB<byte>(255,0,0));
          drawDisk(dispIma, salpt, 3, PixRGB<byte>(255,0,0));
        }
      else
        {
          drawRect(dispIma,r,PixRGB<byte>(0,255,0));
          drawDisk(dispIma, salpt, 3, PixRGB<byte>(0,255,0));
        }
      LDEBUG("found: %d", int(found[i]));

      std::string ntext(sformat("%d", i));
      writeText(dispIma, objOffset[i] + inputVO[i]->getSalPoint(),
                ntext.c_str());
    }

  return dispIma;
}


// ######################################################################
void getSearchCommand
( TCPmessage &rmsg, int32 rframe, int32 raction, bool &resetVentralSearch,
  Image<PixRGB<byte> > &ima, Image<double> &cgist,
  std::vector<rutz::shared_ptr<VisualObject> > &inputVO,
  std::vector<Point2D<int> > &objOffset,
  std::string saveFilePath, std::string testRunFPrefix,
  float &dx, float &dy, uint &snumGT, float &ltravGT)
{
  resetVentralSearch = (int(rmsg.getElementInt32()) > 0);

  // get the image
  ima = rmsg.getElementColByteIma();
  LDEBUG("Image size: %d %d", ima.getWidth(), ima.getHeight());

  // get the gist
  uint gsize = uint(rmsg.getElementInt32());
  cgist.resize(1, gsize, NO_INIT);
  Image<double>::iterator aptr = cgist.beginw();
  for(uint i = 0; i < gsize; i++) *aptr++ = rmsg.getElementDouble();

  // get all the visual objects
  inputVO.clear(); objOffset.clear();
  uint nvo = uint(rmsg.getElementInt32());
  for(uint i = 0; i < nvo; i++)
    {
      int si, sj;
      si = int(rmsg.getElementInt32());
      sj = int(rmsg.getElementInt32());
      Point2D<int> salpt(si,sj);

      int tin, lin, bin, rin;
      tin = int(rmsg.getElementInt32());
      lin = int(rmsg.getElementInt32());
      bin = int(rmsg.getElementInt32());
      rin = int(rmsg.getElementInt32());
      Rectangle rect = Rectangle::tlbrO(tin, lin, bin, rin);
      int fsize = int(rmsg.getElementInt32());
      LDEBUG("[%d] salpt:(%s) r:[%s]", i,
             convertToString(salpt).c_str(),
             convertToString(rect).c_str());
      Point2D<int> offset(rect.left(), rect.top());

      objOffset.push_back(offset);

      // get the pre-attentive feature vector
      std::vector<float> features;
      for(int j = 0; j < fsize; j++)
        features.push_back(rmsg.getElementDouble());

      // create a visual object for the salient region
      Image<PixRGB<byte> > objImg =  crop(ima, rect);
      std::string
        iName(sformat("%s_SAL_%07d_%02d",
                      testRunFPrefix.c_str(), rframe, i));
      std::string ifName = iName + std::string(".png");
      ifName = saveFilePath + ifName;
      rutz::shared_ptr<VisualObject>
        vo(new VisualObject
           (iName, ifName, objImg, salpt - offset, features,
            std::vector< rutz::shared_ptr<Keypoint> >(), false, false));
      inputVO.push_back(vo);

      LDEBUG("rframe[%d] image[%d]: %s sal:[%d,%d] objOffset:[%d,%d]",
            rframe, i, iName.c_str(),
            (salpt - offset).i, (salpt - offset).j,
            objOffset[i].i, objOffset[i].j);
    }

  // get the odometry information
  dx = rmsg.getElementFloat();
  dy = rmsg.getElementFloat();

  // get the ground truth information
  snumGT  = uint(rmsg.getElementInt32());
  ltravGT = rmsg.getElementFloat();

  rmsg.reset(rframe, raction);
}

// ######################################################################
Image<PixRGB<byte> > processLocalizerResults
( nub::ref<GSlocalizer> gslocalizer, std::string savePrefix)
{
  Timer t1(1000000); t1.reset();
  // get the results
  uint index       = gslocalizer->getInputFnum();
  uint64 tGetRes = t1.get();  t1.reset();

  // check the ground truth
  rutz::shared_ptr<Environment> env = gslocalizer->getEnvironment();
  float error = 0.0F;
  uint  snumRes = 0; float ltravRes = -1.0F;
  uint snumGT; float ltravGT; gslocalizer->getGroundTruth(snumGT, ltravGT);
  if(ltravGT != -1.0F)
    {
      //Point2D<float> p = env->getLocationFloat(snumGT, ltravGT);
      //float xGT = p.i; float yGT = p.j;
      snumRes  = gslocalizer->getSegmentLocation();
      ltravRes = gslocalizer->getSegmentLengthTraveled();
      error = env->getTopologicalMap()->
        getDistance(snumGT, ltravGT, snumRes, ltravRes);
      LDEBUG("Ground Truth [%d %f] vs [%d %f]: error: %f",
            snumGT, ltravGT, snumRes, ltravRes, error);
    }
  uint64 tGetGT = t1.get();

  Image<PixRGB<byte> > dispIma;
  Image<PixRGB<byte> > salObjIma;
  bool saveDisplay = true;
  if(saveDisplay)
    {
      dispIma   = getDisplayImage(gslocalizer);
      salObjIma = getFoundSalObjImage(gslocalizer);

      // save it to be mpeg encoded
      std::string saveFName =  savePrefix + sformat("_RES_%07d.ppm", index);
      LDEBUG("saving: %s",saveFName.c_str());
      Raster::WriteRGB(dispIma,saveFName);

      std::string saveSObjFName =  savePrefix + sformat("_SOBJ_%07d.ppm", index);
      LDEBUG("saving: %s",saveSObjFName.c_str());
      Raster::WriteRGB(salObjIma,saveSObjFName);

      // save the particle locations
      // in case we need to restart in the middle
      std::string belFName = savePrefix + sformat("_bel_%07d.txt", index);
      FILE *bfp; LDEBUG("belief file: %s", belFName.c_str());
      if((bfp = fopen(belFName.c_str(),"wt")) == NULL)LFATAL("not found");
      std::vector<LocParticle> belief = gslocalizer->getBeliefParticles();
      for(uint i = 0; i < belief.size(); i++)
        {
          std::string bel = sformat("%d %f ", belief[i].segnum,
                                    belief[i].lentrav);
          bel += std::string("\n");
          fputs(bel.c_str(), bfp);
        }
      fclose (bfp);
    }

  t1.reset();

  // save result in a file by appending to the file
  std::string resFName = savePrefix + sformat("_results.txt");
  FILE *rFile = fopen(resFName.c_str(), "at");
  if (rFile != NULL)
    {
      LDEBUG("saving result to %s", resFName.c_str());
      std::string line =
        sformat("%5d %3d %8.5f %3d %8.5f %10.6f %d",
                index, snumGT, ltravGT, snumRes, ltravRes, error, 0);

      LINFO("%s", line.c_str());
      line += std::string("\n");

      fputs(line.c_str(), rFile);
      fclose (rFile);
    }
  else LINFO("can't create file: %s", resFName.c_str());
  uint64 tSaveRes = t1.get();

  LDEBUG("tGR = %f, tGGT = %f, tSR = %f",
         tGetRes/1000.0, tGetGT/1000.0,tSaveRes/1000.0);

  return dispIma;
}

// ######################################################################
Image<PixRGB<byte> > getDisplayImage(nub::ref<GSlocalizer> gslocalizer)
{
  // if search is not done don't need to get those information
  Image<PixRGB<byte> > ima = gslocalizer->getInputImage();
  uint w = ima.getWidth(); uint h = ima.getHeight();
  Image<PixRGB<byte> > dispIma(5*w,3*h,ZEROS);

  uint ninput = gslocalizer->getNumInputObject();
  std::vector<bool> mfound(ninput);
  std::vector<uint> nObjSearch(ninput);
  std::vector<rutz::shared_ptr<VisualObject> > iObject(ninput);
  std::vector<Point2D<int> > iOffset(ninput);
  std::vector<rutz::shared_ptr<VisualObjectMatch> > cmatch(ninput);
  for(uint i = 0; i < ninput; i++)
    {
      mfound[i]     = gslocalizer->isMatchFound(i);
      nObjSearch[i] = gslocalizer->getNumObjectSearch(i);
      iObject[i]    = gslocalizer->getInputVO(i);
      iOffset[i]    = gslocalizer->getInputObjOffset(i);
      cmatch[i]     = gslocalizer->getVOmatch(i);
    }

  // display the results
  Image<PixRGB<byte> > salIma = getSalImage(ima, iObject, iOffset, mfound);
  inplacePaste(dispIma, zoomXY(salIma), Point2D<int>(0,0));

  // display the gist histogram
  Image<byte> gistHistImg =
    gslocalizer->getSegmentHistogram()->getHistogramImage(w*3, h, 0.0F, 1.0F);
  inplacePaste(dispIma, Image<PixRGB<byte> >(gistHistImg), Point2D<int>(0,2*h));

  // display the localization belief
  float scale;

  Image<PixRGB<byte> > beliefImg =
    gslocalizer->getBeliefImage(w*2, h*3, scale);

  // draw the ground truth
  rutz::shared_ptr<Environment> env = gslocalizer->getEnvironment();
  uint snumGT; float ltravGT; gslocalizer->getGroundTruth(snumGT, ltravGT);
  float xGT = -1.0F, yGT = -1.0F;
  Point2D<float> pgt = env->getLocationFloat(snumGT, ltravGT);
  xGT = pgt.i; yGT = pgt.j;

  if(ltravGT != -1.0F)
    {
      Point2D<int> loc(int(xGT*scale + .5), int(yGT*scale + .5));
      LDEBUG("Ground Truth disp %f %f -> %d %d", xGT, yGT, loc.i, loc.j);
      drawDisk(beliefImg,loc, 4, PixRGB<byte>(255,0,0));
    }

  // show where the objects indicate its position to be
  uint numObjectFound = 0;
  for(uint i = 0; i < ninput; i++)
    {
      if(mfound[i])
        {
          numObjectFound++;
          uint  snum  = gslocalizer->getSegmentNumberMatch(i);
          float ltrav = gslocalizer->getLengthTraveledMatch(i);
          Point2D<float> p = env->getLocationFloat(snum, ltrav);
          float x = p.i, y = p.j;
          Point2D<int> loc(int(x*scale + .5), int(y*scale + .5));
          LDEBUG("obj[%d] res: %f %f -> %d %d",i, x, y, loc.i, loc.j);
          drawDisk(beliefImg, loc, 3, PixRGB<byte>(255,255,0));
        }
    }
  inplacePaste(dispIma, beliefImg, Point2D<int>(3*w,0));

  // display a found object found
  uint fcount = 0;
  for(uint i = 0; i < ninput; i++)
    {
      if(mfound[i] && fcount == 0)
        {
          fcount++;

          //display the first object match found
          Image<PixRGB<byte> >matchIma =
            gslocalizer->getMatchImage(i, Dims(w,h));
          std::string ntext(sformat("object[%d]", i));
          writeText(matchIma, Point2D<int>(0,0), ntext.c_str());
          inplacePaste(dispIma, matchIma, Point2D<int>(2*w,0));
        }
    }
  if(fcount == 0) writeText(dispIma, Point2D<int>(2*w,0),"no objects found");

  return dispIma;
}

// ######################################################################
Image<PixRGB<byte> > getFoundSalObjImage
(nub::ref<GSlocalizer> gslocalizer)
{  
  Image<PixRGB<byte> > salObjIma;

  // if search is not done don't need to get those information
  Image<PixRGB<byte> > ima = gslocalizer->getInputImage();
  uint w = ima.getWidth(); uint h = ima.getHeight();

  uint ninput = gslocalizer->getNumInputObject();
  std::vector<bool> mfound(ninput);
  for(uint i = 0; i < ninput; i++)
    mfound[i]     = gslocalizer->isMatchFound(i);

  // display a found object found
  uint nfound = 0; 
  for(uint i = 0; i < ninput; i++) if(mfound[i]) nfound++; 
  if(nfound == 0) 
    {
      salObjIma = Image<PixRGB<byte> >(w,2*h,ZEROS);
      writeText(salObjIma, Point2D<int>(0,0),"no objects found");
      return salObjIma;
    }
  
  salObjIma = Image<PixRGB<byte> >(nfound*w,2*h,ZEROS);
  uint fcount = 0;
  for(uint i = 0; i < ninput; i++)
    {
      if(mfound[i])
        {
          Image<PixRGB<byte> >matchIma =
            gslocalizer->getMatchImage(i, Dims(w,h));
          std::string ntext(sformat("object[%d]", i));
          writeText(matchIma, Point2D<int>(0,0), ntext.c_str());
          inplacePaste(salObjIma, matchIma, Point2D<int>(fcount*w,0));
          fcount++;
        }
    }
  return salObjIma;
}

// ######################################################################
bool saveGist(std::string saveFilePath, std::string testRunFPrefix,
              int currSegNum, int count, Image<double> cgist)
{
  char gName[100];
  sprintf(gName,"%s_%03d_%06d.gist",
          (saveFilePath + testRunFPrefix).c_str(), currSegNum, count);
  LINFO("save gist file to: %s\n", gName);

  // write the data to the gist file
  FILE *gfp; if((gfp = fopen(gName,"wb")) != NULL)
    {
      Image<double>::iterator aptr = cgist.beginw(), stop = cgist.endw();
      while (aptr != stop)
        { double val = *aptr++; fwrite(&val, sizeof(double), 1, gfp); }
      fclose(gfp);
    }
  else { LINFO("can't save: %s",gName); return false; }

  return true;
}

// ######################################################################
void reportResults(std::string savePrefix, uint nsegment)
{
  GSnavResult r;

  // combine the *_GS_result.txt and *_result.txt files
  //r.combine(savePrefix);

  // read the result file
  r.read(savePrefix, nsegment);

  // create a result summary file
  r.createSummaryResult();
}

// ######################################################################
void reportTrialResults()
{
  // ===================================================================
  // RESULT Printing TR07
  // ===================================================================
//   uint nsegment =  9;
//   std::string savePrefix("/lab/tmpib/u/siagian/PAMI07/FDFpark/envComb/RES/FDFpark"),
//   float scale = 2.0;
//   // get the input off all the test files
//   // open the result file
//   uint nTrials = 4;
//   std::vector<std::string> inputFName(nTrials);
//   inputFName[0] = savePrefix + sformat("_T_A/ACB");
//   inputFName[1] = savePrefix + sformat("_T_B/ACB");
//   inputFName[2] = savePrefix + sformat("_T_E/ACB");
//   inputFName[3] = savePrefix + sformat("_T_F/ACB");
//   LINFO("%s", inputFName[0].c_str());

//   std::vector<GSnavResult> r(nTrials);
//   for(uint i = 0; i < nTrials; i++) r[i].read(inputFName[i], nsegment);

//   // trial segment result lines
//   for(uint j = 0; j < nsegment; j++)
//     {
//       // get segment total across trials
//       printf("%4d &", j+1);
//       float err = 0.0; uint ct = 0;
//       for(uint i = 0; i < nTrials; i++)
//         {
//           err += r[i].serror[j];
//           ct  += r[i].scount[j];
//           printf("%6d & %8.2f &",
//                  r[i].scount[j], scale * r[i].serror[j]/r[i].scount[j]);
//         }
//       printf(" %6d  & %8.2f \n", ct, scale *err/ct);
//     }

//   // trial total result lines
//   float terr = 0.0; uint tct = 0;
//   for(uint i = 0; i < nTrials; i++)
//     {
//       float err = 0.0; uint ct = 0;
//       for(uint j = 0; j < nsegment; j++)
//         { err += r[i].serror[j]; ct += r[i].scount[j]; }
//       printf("%6d & %8.2f & ",  ct, scale * err/ct);
//       terr += err; tct += ct;
//     }
//   printf(" %6d  & %8.2f \n", tct, scale * terr/tct);

  // ===================================================================
  // RESULT Printing ICRA07 + IJRR08
  // ===================================================================
  uint nsegment =  9;

  // ===================================================================
  // ACB
  std::string savePrefix
    ("/lab/tmpib/u/siagian/PAMI07/ACB/envComb/RES/ACB");
  float scale = 2.00;
  uint nSalReg = 29710;
  // 29710 82502 90660
  LINFO("%s has %d sal reg", savePrefix.c_str(), nSalReg);

  // get the input off all the result files
  uint nTrials = 6;
  std::vector<std::string> inputFName(nTrials);
  inputFName[0] = savePrefix + sformat("_T_A_rand2/ACB");
  inputFName[1] = savePrefix + sformat("_T_A_seg2/ACB");
  inputFName[2] = savePrefix + sformat("_T_A_sal2/ACB");
  inputFName[3] = savePrefix + sformat("_T_A_loc2/ACB");
  inputFName[4] = savePrefix + sformat("_T_A_nee/ACB");
  inputFName[5] = savePrefix + sformat("_T_A2/ACB");

  // supplement data for AR08
//   uint nTrials = 5;
//   std::vector<std::string> inputFName(nTrials);
//   inputFName[0] = savePrefix + sformat("_T_A_prioritize/ACB_10_20_33");
//   inputFName[1] = savePrefix + sformat("_T_A_prioritize/ACB_05_10_20");
//   inputFName[2] = savePrefix + sformat("_T_A_prioritize/ACB_03_06_10");
//   inputFName[3] = savePrefix + sformat("_T_A_prioritize/ACB_01_03_05");
//   inputFName[4] = savePrefix + sformat("_T_A_prioritize/ACB_01_02_03");

  // supplement data 2 for AR08
//   uint nTrials = 3;
//   std::vector<std::string> inputFName(nTrials);
//   inputFName[0] = savePrefix + sformat("_T_A_track/AAAI/ACB_inf_comb");
//   inputFName[1] = savePrefix + sformat("_T_A_track/AAAI/ACB_200_comb");
//   inputFName[2] = savePrefix + sformat("_T_A_track/AAAI/ACB_100_comb");


  // ===================================================================
  // AnF
//   std::string savePrefix
//     ("/lab/tmpib/u/siagian/PAMI07/AnFpark/envComb/RES/AnFpark");
//   float scale = 1.00;
//   uint nSalReg = 82502;
//   LINFO("%s has %d sal reg", savePrefix.c_str(), nSalReg);

  // get the input off all the result files
//   uint nTrials = 6;
//   std::vector<std::string> inputFName(nTrials);
//   inputFName[0] = savePrefix + sformat("_D_rand2/AnFpark");
//   inputFName[1] = savePrefix + sformat("_D_seg2/AnFpark");
//   inputFName[2] = savePrefix + sformat("_D_sal2/AnFpark");
//   inputFName[3] = savePrefix + sformat("_D_loc2/AnFpark");
//   inputFName[4] = savePrefix + sformat("_D_nee/AnFpark");
//   inputFName[5] = savePrefix + sformat("_D2/AnFpark");

  // supplement data for AR08
//   uint nTrials = 5;
//   std::vector<std::string> inputFName(nTrials);
//   inputFName[0] = savePrefix + sformat("_D_prioritize/AnFpark_10_20_33");
//   inputFName[1] = savePrefix + sformat("_D_prioritize/AnFpark_05_10_20");
//   inputFName[2] = savePrefix + sformat("_D_prioritize/AnFpark_03_06_10");
//   inputFName[3] = savePrefix + sformat("_D_prioritize/AnFpark_01_03_05");
//   inputFName[4] = savePrefix + sformat("_D_prioritize/AnFpark_01_02_03");

  // supplement data 2 for AR08
//   uint nTrials = 5;
//   std::vector<std::string> inputFName(nTrials);
//   inputFName[0] = savePrefix + sformat("_D_track/AAAI/AnFpark_inf_comb");
//   inputFName[1] = savePrefix + sformat("_D_track/AAAI/AnFpark_1000_comb");
//   inputFName[2] = savePrefix + sformat("_D_track/AAAI/AnFpark_500_comb");
//   inputFName[3] = savePrefix + sformat("_D_track/AAAI/AnFpark_200_comb");
//   inputFName[4] = savePrefix + sformat("_D_track/AAAI/AnFpark_100_comb");

  // ===================================================================
  // FDF
//   std::string savePrefix
//     ("/lab/tmpib/u/siagian/PAMI07/FDFpark/envComb/RES/FDFpark");
//   float scale = 3.75;
//   uint nSalReg =  90660;
//   LINFO("%s has %d sal reg", savePrefix.c_str(), nSalReg);

  // get the input off all the result files
//   uint nTrials = 6;
//   std::vector<std::string> inputFName(nTrials);
//   inputFName[0] = savePrefix + sformat("_T_C_rand2/AnFpark");
//   inputFName[1] = savePrefix + sformat("_T_C_seg2/AnFpark");
//   inputFName[2] = savePrefix + sformat("_T_C_sal2/AnFpark");
//   inputFName[3] = savePrefix + sformat("_T_C_loc2/AnFpark");
//   inputFName[4] = savePrefix + sformat("_T_C_nee/AnFpark");
//   inputFName[5] = savePrefix + sformat("_T_A2/AnFpark");

  // supplement data for IJRR08
  // NOTE: take out the "_comb" in the GSnavResult::read
//   uint nTrials = 5;
//   std::vector<std::string> inputFName(nTrials);
//   inputFName[0] = savePrefix + sformat("_T_C_prioritize/FDFpark_10_20_33");
//   inputFName[1] = savePrefix + sformat("_T_C_prioritize/FDFpark_05_10_20");
//   inputFName[2] = savePrefix + sformat("_T_C_prioritize/FDFpark_03_06_10");
//   inputFName[3] = savePrefix + sformat("_T_C_prioritize/FDFpark_01_03_05");
//   inputFName[4] = savePrefix + sformat("_T_C_prioritize/FDFpark_01_02_03");

  // supplement data 2 for IJRR08
//   uint nTrials = 5;
//   std::vector<std::string> inputFName(nTrials);
//   inputFName[0] = savePrefix + sformat("_T_C_track/AAAI/FDFpark_inf_comb");
//   inputFName[1] = savePrefix + sformat("_T_C_track/AAAI/FDFpark_1000_comb");
//   inputFName[2] = savePrefix + sformat("_T_C_track/AAAI/FDFpark_500_comb");
//   inputFName[3] = savePrefix + sformat("_T_C_track/AAAI/FDFpark_200_comb");
//   inputFName[4] = savePrefix + sformat("_T_C_track/AAAI/FDFpark_100_comb");

  std::vector<GSnavResult> r(nTrials);
  for(uint i = 0; i < nTrials; i++)
    {
      r[i].read(inputFName[i], nsegment);
      //r[i].createSummaryResult();
    }

  // for each episode
  for(uint i = 0; i < nTrials; i++)
    {
      LINFO("%6d %10d: %10.3f + %6d %10d: %10.3f = %6d %10d %10.3f| "
            "%6d %10.6f",
            r[i].tfobj,  r[i].tfsearch,  r[i].tfsearch/float(r[i].tfobj),
            r[i].tnfobj, r[i].tnfsearch, r[i].tnfsearch/float(r[i].tnfobj),
            r[i].tobj,   r[i].tsearch,   r[i].tsearch/float(r[i].tobj),
            r[i].tcount, r[i].terror/r[i].tcount*scale);
    }

  // for each episode
  for(uint i = 0; i < nTrials; i++)
    {
      printf("& %10.2f\\%% & $%6.2f \\pm %6.2f$ "
             "& %10.2f\\%% & $%6.2f \\pm %6.2f$ "
             "& %10.2f\\%% & $%6.2f \\pm %6.2f$ & $%6.2f \\pm %6.2f$\\\\ \n",
             r[i].tfsearch/float(r[i].tfobj)/nSalReg*100.0,
             float(r[i].tfobj)/r[i].tcount, r[i].stdevNObjectFound,
             r[i].tnfsearch/float(r[i].tnfobj)/nSalReg*100.0,
             float(r[i].tnfobj)/r[i].tcount, r[i].stdevNObjectNotFound,
             r[i].tsearch/float(r[i].tobj)/nSalReg*100.0,
             float(r[i].tobj)/r[i].tcount, r[i].stdevNObject,
             r[i].terror/r[i].tcount*scale, r[i].stdevError*scale);
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
