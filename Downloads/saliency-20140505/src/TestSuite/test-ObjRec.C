/*!@file TestSuite/test-ObjRec.C Test Varius object rec code */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/test-ObjRec.C $
// $Id: test-ObjRec.C 12962 2010-03-06 02:13:53Z irock $
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
#include "TestSuite/ObjRecBrain.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <unistd.h>
#include <cstdlib>
#include <cstdlib>
#include <dlfcn.h>

static const ModelOptionDef OPT_ObjRecTrainingMode =
  { MODOPT_FLAG, "ObjRecTrainingMode", &MOC_GENERAL, OPTEXP_CORE,
    "Whether to traing the classifier or recognize ",
    "training-mode", '\0', "", "false" };

static const ModelOptionDef OPT_ObjRecFilterObject =
  { MODOPT_ARG_STRING, "ObjRecFilterObject", &MOC_GENERAL, OPTEXP_CORE,
    "Binary recognition. Is this object there or not. ",
    "filter-object", '\0', "<string>", "" };

static const ModelOptionDef OPT_ObjRecOutputROCFile =
  { MODOPT_ARG_STRING, "ObjRecOutputROCFile", &MOC_GENERAL, OPTEXP_CORE,
    "The file name to output the ROC data to. ",
    "roc-file", '\0', "<string>", "" };

static const ModelOptionDef OPT_ObjRecOutputTimingFile =
  { MODOPT_ARG_STRING, "ObjRecOutputTimingFile", &MOC_GENERAL, OPTEXP_CORE,
    "The file name to output timing information. ",
    "timing-file", '\0', "<string>", "" };

static const ModelOptionDef OPT_ObjRecOutputResultsFile =
  { MODOPT_ARG_STRING, "ObjRecOutputResultsFile", &MOC_GENERAL, OPTEXP_CORE,
    "The file name to output full results information to. "
    "This will include the frame number, the scene filename, which object we "
    "had, what we labeled it and the confidence. Only for recognition mode.",
    "results-file", '\0', "<string>", "" };

static const ModelOptionDef OPT_ObjRecObjectsDBFile =
  { MODOPT_ARG_STRING, "ObjRecObjectsDBFile", &MOC_GENERAL, OPTEXP_CORE,
    "The file name to use as the object database ",
    "objects-db-file", '\0', "<string>", "objects.dat" };


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



int main(const int argc, const char **argv)
{

  MYLOGVERB = LOG_INFO;
  ModelManager manager("Test Object Rec");

  OModelParam<bool> optTrainingMode(&OPT_ObjRecTrainingMode, &manager);
  OModelParam<std::string> optFilterObject(&OPT_ObjRecFilterObject, &manager);
  OModelParam<std::string> optOutputROCFile(&OPT_ObjRecOutputROCFile, &manager);
  OModelParam<std::string> optOutputTimingFile(&OPT_ObjRecOutputTimingFile, &manager);
  OModelParam<std::string> optOutputResultsFile(&OPT_ObjRecOutputResultsFile, &manager);
  OModelParam<std::string> optObjectsDBFile(&OPT_ObjRecObjectsDBFile, &manager);

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  manager.exportOptions(MC_RECURSE);

  if (manager.parseCommandLine(
        (const int)argc, (const char**)argv, "<ObjRecBrainLib>", 1, 1) == false)
    return 1;

  std::string libFile = manager.getExtraArg(0);
  LDEBUG("Loading %s", libFile.c_str());
  void* brainLib = dlopen(libFile.c_str(), RTLD_LAZY);
  if (!brainLib)
    LFATAL("Can load library: %s (%s)", libFile.c_str(), dlerror());

  //Load the symbols
  dlerror(); //reset any errors
  CreateObjRecBrain* createBrain = (CreateObjRecBrain*) dlsym(brainLib, "createObjRecBrain");
  DestoryObjRecBrain* destoryBrain = (DestoryObjRecBrain*) dlsym(brainLib, "destoryObjRecBrain");

  if (!createBrain  || !destoryBrain)
    LFATAL("Can not find the create and destory symbols: %s", dlerror());

  ObjRecBrain* brain = createBrain(optObjectsDBFile.getVal());

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
    brain->preRecognition();
  float preTime = timer.getSecs();

  if (timingFP)
    fprintf(timingFP, "%s %f\n",
        optTrainingMode.getVal() ? "PreTraining" : "PreRecognition",
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
        ObjectData obj = brain->onRecognition(inputImg);
        frameTime = timer.getSecs();

        float confidence = obj.confidence;

        //Invert the confidence since we are more confident that this is not the object
        //so we care less confident that this is the object
        if ( optFilterObject.getVal() != obj.name )
          confidence = 1/confidence;

        results.push_back(ResultData(ifs->frame(),
              labeledObj.name,
              obj.name,
              confidence));

        if (resultsFP)
          fprintf(resultsFP, "%i %s %s %s %f\n",
              ifs->frame(), sceneData->filename.c_str(),
              labeledObj.name.c_str(),
              obj.name.c_str(), confidence);

      }

      if (timingFP)
        fprintf(timingFP, "%i %f\n",
            ifs->frame(), frameTime);
      totalNumFrames++;
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
    brain->postRecognition();
  float postTime = timer.getSecs();
  if (timingFP)
    fprintf(timingFP, "%s %f\n",
        optTrainingMode.getVal() ? "PostTraining" : "PostRecognition",
        postTime);

  if (timingFP)
    fclose(timingFP);

  //Calculate ROC curve and AP
  if (!optTrainingMode.getVal())
  {
    std::sort(results.begin(), results.end(), ResultDataCmp);
    std::vector<float> tp;
    int numPosExamples = 0;
    for(uint i=0; i<results.size(); i++)
    {
      //Calculate true positive
      if (optFilterObject.getVal() == results[i].objName)
      {
        numPosExamples++;
        if (tp.size() > 0)
          tp.push_back(tp.at(i-1)+1);
        else
          tp.push_back(1);
      } else {
        if (tp.size() > 0)
          tp.push_back(tp.at(i-1));
        else
          tp.push_back(0);
      }
    }

    std::vector<float> rec;
    std::vector<float> prec;
    for(uint i=0; i<tp.size(); i++)
    {
      rec.push_back(tp[i]/numPosExamples);
      prec.push_back(tp[i]/(i+1));
    }

    ////Output the precision recall curve
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
      for(uint i=0; i<rec.size(); i++)
        fprintf(rocFP, "%f %f\n", rec[i], prec[i]);
      fclose(rocFP);
    }

    //Calculate the average precision
    double ap=0;
    double step = 0.1;

    for(double t=0; t<=1; t+=step)
    {
      double maxPrec = 0;
      for(uint i=0; i<rec.size(); i++)
        if (rec[i] >= t)
          if (prec[i] > maxPrec)
            maxPrec = prec[i];

      ap += (maxPrec / ((1/step)+1) ); //take the average
    }
    printf("Stats: Frames:%lu FPS:%f AP:%f\n",
        totalNumFrames, (double)totalNumFrames/totalTime, ap);
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
