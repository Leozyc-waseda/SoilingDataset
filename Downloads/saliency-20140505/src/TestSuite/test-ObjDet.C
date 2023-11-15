/*!@file TestSuite/test-ObjDec.C Test Varius object detection code */

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
// Primary maintainer for this file: Lior Elazary
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/test-ObjDet.C $
// $Id: test-ObjDet.C 12962 2010-03-06 02:13:53Z irock $
//

#include "Component/GlobalOpts.H"
#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Component/ModelParam.H"
#include "Component/ModelParamBatch.H"
#include "GUI/XWindow.H"
#include "Image/Image.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/ShapeOps.H"
#include "Image/Rectangle.H"
#include "Image/MathOps.H"
#include "Image/MatrixOps.H"
#include "Image/Transforms.H"
#include "Image/Convolutions.H"
#include "Media/FrameSeries.H"
#include "Media/TestImages.H"
#include "nub/ref.h"
#include "Raster/GenericFrame.H"
#include "Transport/FrameInfo.H"
#include "Raster/Raster.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Util/Timer.H"
#include "TestSuite/ObjDetBrain.h"
#include "GUI/DebugWin.H"



//Other libs so that we link eginst them
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/fancynorm.H"
#include "Neuro/EnvVisualCortex.H"
#include "Neuro/getSaliency.H"
#include "nub/ref.h"
#include "Util/MathFunctions.H"




#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <unistd.h>
#include <cstdlib>
#include <cstdlib>
#include <dlfcn.h>

static const ModelOptionDef OPT_ObjDetTrainingMode =
  { MODOPT_FLAG, "ObjDetTrainingMode", &MOC_GENERAL, OPTEXP_CORE,
    "Whether to traing the classifier or detect ",
    "training-mode", '\0', "", "false" };

static const ModelOptionDef OPT_ObjDetFilterObject =
  { MODOPT_ARG_STRING, "ObjDetFilterObject", &MOC_GENERAL, OPTEXP_CORE,
    "Binary recognition. Is this object there or not. ",
    "filter-object", '\0', "<string>", "" };

static const ModelOptionDef OPT_ObjDetOutputROCFile =
  { MODOPT_ARG_STRING, "ObjDetOutputROCFile", &MOC_GENERAL, OPTEXP_CORE,
    "The file name to output the ROC data to. ",
    "roc-file", '\0', "<string>", "" };

static const ModelOptionDef OPT_ObjDetOutputTimingFile =
  { MODOPT_ARG_STRING, "ObjDetOutputTimingFile", &MOC_GENERAL, OPTEXP_CORE,
    "The file name to output timing information. ",
    "timing-file", '\0', "<string>", "" };

static const ModelOptionDef OPT_ObjDetOutputResultsFile =
  { MODOPT_ARG_STRING, "ObjDetOutputResultsFile", &MOC_GENERAL, OPTEXP_CORE,
    "The file name to output full results information to. "
    "This will include the frame number, the scene filename, which object we "
    "had, what we labeled it and the confidence. Only for recognition mode.",
    "results-file", '\0', "<string>", "" };

struct ResultData
{
  int frame;
  std::string objName;
  std::string labelName;
  float confidence;

  ResultData(int f, std::string& obj, std::string& label, float c) :
    frame(f),
    objName(obj),
    labelName(label),
    confidence(c)
  {}
};

bool ResultDataCmp(const ResultData& r1, const ResultData& r2)
{
  return r1.confidence > r2.confidence;
}

bool DetLocationCmp(const DetLocation& r1, const DetLocation& r2)
{
  return r1.val > r2.val;
}

int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager manager("Test Object Det");

  OModelParam<bool> optTrainingMode(&OPT_ObjDetTrainingMode, &manager);
  OModelParam<std::string> optFilterObject(&OPT_ObjDetFilterObject, &manager);
  OModelParam<std::string> optOutputROCFile(&OPT_ObjDetOutputROCFile, &manager);
  OModelParam<std::string> optOutputTimingFile(&OPT_ObjDetOutputTimingFile, &manager);
  OModelParam<std::string> optOutputResultsFile(&OPT_ObjDetOutputResultsFile, &manager);

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  manager.exportOptions(MC_RECURSE);


  //Get all the args for this module up to --
  int nModelArgs = 0;
  for(int i=0; i<argc; i++)
  {
    if (!strcmp(argv[i], "--"))
      break;
    nModelArgs++;
  }



  if (manager.parseCommandLine(
        (const int)nModelArgs, (const char**)argv, "<ObjDetBrainLib>", 1, 1) == false)
    return 1;

  std::string libFile = manager.getExtraArg(0);
  LDEBUG("Loading %s", libFile.c_str());
  void* brainLib = dlopen(libFile.c_str(), RTLD_LAZY );
  if (!brainLib)
    LFATAL("Can load library: %s (%s)", libFile.c_str(), dlerror());

  //Load the symbols
  dlerror(); //reset any errors
  CreateObjDetBrain* createBrain = (CreateObjDetBrain*) dlsym(brainLib, "createObjDetBrain");
  DestoryObjDetBrain* destoryBrain = (DestoryObjDetBrain*) dlsym(brainLib, "destoryObjDetBrain");

  if (!createBrain  || !destoryBrain)
    LFATAL("Can not find the create and destory symbols: %s", dlerror());

  int extraArgc = argc - nModelArgs;
  const char** extraArgv = &argv[nModelArgs];

  if (extraArgc == 0)
  {
    extraArgc = 1;
    extraArgv = argv;
  }

  ObjDetBrain* brain = createBrain(extraArgc, extraArgv);

  manager.start();

  ifs->startStream();

  Timer timer;
  FILE* timingFP = NULL;
  if (optOutputTimingFile.getVal().size() > 0)
  {
    timingFP = fopen(optOutputTimingFile.getVal().c_str(), "w");
    if (timingFP == NULL)
      LFATAL("Can not open timing file: %s",
          optOutputTimingFile.getVal().c_str());
  }


  timer.reset();
  if (optTrainingMode.getVal())
    brain->preTraining();
  else
    brain->preDetection();
  float preTime = timer.getSecs();

  if (timingFP)
    fprintf(timingFP, "%s %f\n",
        optTrainingMode.getVal() ? "PreTraining" : "PreDetection",
        preTime);

  FILE* resultsFP = NULL;
  if (optOutputResultsFile.getVal().size() > 0 && !optTrainingMode.getVal())
  {
    resultsFP = fopen(optOutputResultsFile.getVal().c_str(), "w");
    if (resultsFP == NULL)
      LFATAL("Can not open results file: %s",
          optOutputResultsFile.getVal().c_str());
  }
  std::vector<ResultData> results;

  double totalTime = 0;
  unsigned long totalNumFrames = 0;
  //unsigned long long totalNumFixations = 0;
  double firstObjectMeanFixations = 0;
  double firstObjectStvarFixations = 0;
  double allObjectsMeanFixations = 0;
  double allObjectsStvarFixations = 0;

  int numOfBins = 1000;
  std::vector<double>  totalTruePositive(numOfBins, 0);
  std::vector<double>  totalFalseNegative(numOfBins, 0);

  while(1)
  {
    Image< PixRGB<byte> > inputImg;
    const FrameState is = ifs->updateNext();
    if (is == FRAME_COMPLETE)
      break;

    //grab the images
    GenericFrame input = ifs->readFrame();
    if (!input.initialized())
      break;
    inputImg = input.asRgb();

    totalNumFrames++;

    //Get the metadata and find if we have the object name in the scene
    rutz::shared_ptr<GenericFrame::MetaData>
      metaData = input.getMetaData(std::string("SceneData"));
    if (metaData.get() != 0) {
      rutz::shared_ptr<TestImages::SceneData> sceneData;
      sceneData.dyn_cast_from(metaData);

      ObjectData labeledObj;

      if (optFilterObject.getVal().size() > 0)
      {
        labeledObj.name = "no_" + optFilterObject.getVal();
        labeledObj.confidence = -1;

        //Sech and see if we have this object in the scene
        for (uint i = 0; i < sceneData->objects.size(); i++) {
          TestImages::ObjData objData = sceneData->objects[i];
          if (optFilterObject.getVal() == objData.name)
            labeledObj.name = objData.name;
        }
      } else {
        //Take the first object
        for (uint i = 0; i < sceneData->objects.size() && i<1; i++) {
          TestImages::ObjData objData = sceneData->objects[i];
          labeledObj.name = objData.name;
        }
      }

      double frameTime = -1;
      if (optTrainingMode.getVal())
      {
        timer.reset();
        brain->onTraining(inputImg, labeledObj);
        frameTime = timer.getSecs();
      } else {
        timer.reset();
        std::vector<DetLocation> smap = brain->onDetection(inputImg);
        frameTime = timer.getSecs();

        //Sort the results
        std::sort(smap.begin(), smap.end(), DetLocationCmp);

        Image<float> objsMask(inputImg.getDims(), ZEROS);
        for (uint obj = 0; obj < sceneData->objects.size(); obj++) {
          TestImages::ObjData objData = sceneData->objects[obj];
          drawFilledPolygon(objsMask, objData.polygon, (float)(obj+1));
        }

        unsigned long totalObjects = 0;
        unsigned long totalNonObjects = 0;
        for(uint i=0; i<objsMask.size(); i++)
          if (objsMask[i] > 0)
            totalObjects++;
          else
            totalNonObjects++;

        //Check if this location has hit an object
        unsigned long numFixations = 0;
        //unsigned long numObjects = 0;

        unsigned long fixationsToFirstObject = 0;
        unsigned long fixationsToAllObjects = 0;

        std::vector<double>  truePositive(numOfBins, 0);
        std::vector<double>  falseNegative(numOfBins, 0);

        if (smap.size() != objsMask.size())
          LFATAL("Smap needs to be the same size as the image");
        int binNum = 0;
        double fixPerBin = double(smap.size())/double(numOfBins);
        for (uint i=0; i<smap.size(); i++)
        {
          Point2D<int> loc(smap[i].i, smap[i].j);
          float objID = objsMask.getVal(loc);

          numFixations++;
          if ((numFixations-1) > fixPerBin*(binNum+1))
          {
            //New bin number
            int prevBinNum = binNum;
            binNum++;
            if(binNum >= numOfBins)
              LFATAL("binNum(%i) >= numOfBins(%i)", binNum, numOfBins);

            truePositive[binNum] = truePositive[prevBinNum];
            falseNegative[binNum] = falseNegative[prevBinNum];
          }

          if (objID > 0)
            truePositive[binNum]++;
          else
            falseNegative[binNum]++;
        }

        //Aggragate values over frames
        for(int i=0; i<numOfBins; i++)
        {
          totalTruePositive[i] += (truePositive[i]/totalObjects);
          totalFalseNegative[i] += (falseNegative[i]/totalNonObjects);
        }

        //for(uint i=0; i<smap.size(); i++)
        //{
        //  Point2D<int> loc(smap[i].i, smap[i].j);
        //  float objID = objsMask.getVal(loc);
        //  numFixations++;
        //  totalNumFixations++;
        //  if (objID > 0)
        //  {
        //    numObjects++;
        //    //We got an object
        //    if (fixationsToFirstObject == 0)
        //      fixationsToFirstObject = numFixations;
        //    if (fixationsToAllObjects == 0 &&
        //        numObjects == totalObjects)
        //      fixationsToAllObjects = numFixations;
        //  }
        //}
        //printf("%lu %lu %lu %lu\n", numFixations, smap.size(), numObjects, totalObjects);

        //printf("%lu %lu %f %lu %lu %f\n",
        //    fixationsToFirstObject, numFixations,
        //    (double)fixationsToFirstObject/(double)numFixations,
        //    fixationsToAllObjects, totalObjects,
        //    (double)fixationsToAllObjects/(double)totalObjects);

        //fflush(stdout);
        //calculate the mean and std of the fixations online

        //Calc out of the total number of fixations, how many of these reached
        //the first object
        const double prevFirstMean = firstObjectMeanFixations;
        const double percFixationsToFirstObject =
          (double)fixationsToFirstObject/(double)numFixations;
        const double firstDelta = percFixationsToFirstObject - firstObjectMeanFixations;
        firstObjectMeanFixations += firstDelta/totalNumFrames;
        firstObjectStvarFixations += (
            ( percFixationsToFirstObject - prevFirstMean)*
            (percFixationsToFirstObject - firstObjectMeanFixations)
            );


        //Calc out of the total number of fixations needed to reach all the objects
        // how many of these reached all the objects
        const double prevAllMean = allObjectsMeanFixations;
        const double percFixationsToAllObjects =
          (double)fixationsToAllObjects/(double)totalObjects;
        const double allDelta = percFixationsToAllObjects - allObjectsMeanFixations;
        allObjectsMeanFixations += allDelta/totalNumFrames;
        allObjectsStvarFixations += (
            ( percFixationsToAllObjects - prevAllMean)*
            (percFixationsToAllObjects - allObjectsMeanFixations)
            );
      }

      if (timingFP)
        fprintf(timingFP, "%i %f\n",
            ifs->frame(), frameTime);
      totalTime += frameTime;

    }
    ofs->writeRGB(inputImg, "input", FrameInfo("input", SRC_POS));
    usleep(10000);
  }


  if (resultsFP)
    fclose(resultsFP);

  timer.reset();
  if (optTrainingMode.getVal())
    brain->postTraining();
  else
    brain->postDetection();
  float postTime = timer.getSecs();
  if (timingFP)
    fprintf(timingFP, "%s %f\n",
        optTrainingMode.getVal() ? "PostTraining" : "PostDetection",
        postTime);

  if (timingFP)
    fclose(timingFP);

  //Calculate ROC curve and AP
  if (!optTrainingMode.getVal())
  {

    //Normalize by number of frames
    for(int i=0; i<numOfBins; i++)
    {
      totalFalseNegative[i] /= totalNumFrames;
      totalTruePositive[i] /= totalNumFrames;
    }

    ////Output the roc curve
    FILE* rocFP = NULL;
    if (optOutputROCFile.getVal().size() > 0)
    {
      rocFP = fopen(optOutputROCFile.getVal().c_str(), "w");
      if (rocFP == NULL)
        LFATAL("Can not open roc file: %s",
            optOutputROCFile.getVal().c_str());
    }
    if (rocFP)
    {
      for(int i=0; i<numOfBins; i++)
        fprintf(rocFP, "%f %f\n", totalFalseNegative[i], totalTruePositive[i]);
      fclose(rocFP);
    }

    //Calculate the average true pos
    double ap=0;
    double step = 0.1;
    for(double t=0; t<=1; t+=step)
    {
      double maxPrec = 0;
      for(int i=0; i<numOfBins; i++)
      {
        if (totalFalseNegative[i] >= t)
          if (totalTruePositive[i] > maxPrec)
            maxPrec = totalTruePositive[i];
      }
      ap += (maxPrec / ((1/step)+1) ); //take the average
    }

    printf("Stats: Frames:%lu FPS:%f Ap:%f\n",
        totalNumFrames,
        (double)totalNumFrames/totalTime,
        ap);
    fflush(stdout);
  } else {
    printf("Stats: Frames:%lu FPS:%f \n",
        totalNumFrames, (double)totalNumFrames/totalTime);
  }



  destoryBrain(brain);

  //unload the library
  dlclose(brainLib);


  return 0;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
