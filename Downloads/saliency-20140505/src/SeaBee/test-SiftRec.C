/*!@file SeaBee/test-SiftRec.C test/train sift recognizer   */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/test-SiftRec.C $
// $Id: test-SiftRec.C 10794 2009-02-08 06:21:09Z itti $

#include "Component/GlobalOpts.H"
#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Component/ModelParam.H"
#include "Component/ModelParamBatch.H"

#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Raster/GenericFrame.H"
#include "Media/MediaOpts.H"
#include "Transport/FrameInfo.H"
#include "Neuro/EnvSegmenterConfigurator.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/DrawOps.H"
#include "Raster/Raster.H"

#include "SeaBee/SiftRec.H"

const ModelOptionCateg MOC_SIFTREC = {
  MOC_SORTPRI_2, "SiftRec Options" };

static const ModelOptionDef OPT_TrainingLabel =
  { MODOPT_ARG_STRING, "TrainingLabel", &MOC_SIFTREC, OPTEXP_CORE,
    "If this label is set, then the system goes into training mode, "
    "Traing all of the image with this label Whether to include an ",
    "training-label", '\0', "", "" };

static const ModelOptionDef OPT_TrainUnknown =
  { MODOPT_FLAG, "TrainUnknown", &MOC_SIFTREC, OPTEXP_CORE,
    "Wether to train all images with the label or just the unknown."
    "Note that with value of false, many more entries in the database will be entered.",
    "train-unknown", '\0', "", "true" };



int main(int argc, char* argv[])
{

  ModelManager mgr("SiftRec Tester");

  OModelParam<bool> trainUnknown(&OPT_TrainUnknown, &mgr);
  OModelParam<std::string> trainingLabel(&OPT_TrainingLabel, &mgr);

  nub::soft_ref<SiftRec> siftRec(new SiftRec(mgr));
  mgr.addSubComponent(siftRec);

  nub::ref<EnvSegmenterConfigurator> esec(new EnvSegmenterConfigurator(mgr));
  mgr.addSubComponent(esec);



  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(mgr));
  mgr.addSubComponent(ifs);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(mgr));
  mgr.addSubComponent(ofs);

  mgr.exportOptions(MC_RECURSE);

  if (mgr.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  mgr.start();

  siftRec->initVDB(); //initialize the database

  nub::soft_ref<EnvSegmenter> seg = esec->getSeg();

  while(true)
  {
    const FrameState is = ifs->updateNext();
    if (is == FRAME_COMPLETE) break;
    //grab the images
    GenericFrame input = ifs->readFrame();
    if (!input.initialized()) break;
    Image<PixRGB<byte> > img = input.asRgb();

    Image<byte> foamask;
    Image<PixRGB<byte> > segmentdisp;
    //Segment the image to get the card
    const Rectangle segRect = seg->getFoa(img, Point2D<int>(0,0),
                    &foamask, &segmentdisp);

    std::string objName;
    float score = 0;
    Rectangle matchRect;
    if (trainingLabel.getVal().size() > 0)
    {
      //Train the dataset
      if (trainUnknown.getVal()) //Train only unknown objects with this label
      {
        //Recognize
        objName = siftRec->matchObject(img, score,matchRect);
      }

      if (objName == "nomatch" || objName.size() == 0)
      {
        LINFO("Training with object name %s", trainingLabel.getVal().c_str());
        siftRec->trainObject(img, trainingLabel.getVal());
      }
    } else {
      //Recognize
      objName = siftRec->matchObject(img, score,matchRect);

    }

    //markup the image with the values
    const std::string txt =
      sformat("%s:%0.2f", objName.c_str(), score);
    writeText(img, Point2D<int>(0,0),
        txt.c_str(),
        PixRGB<byte>(255), PixRGB<byte>(0));

    if (matchRect.isValid() && img.rectangleOk(matchRect))
    {
      drawRect(img, matchRect, PixRGB<byte>(255, 255, 0), 1);
      drawCircle(img, matchRect.center(), 6, PixRGB<byte>(0,255,0));
    }

    if (segRect.isValid() && img.rectangleOk(segRect))
    {
        drawRect(img, segRect, PixRGB<byte>(0, 255, 0), 1);
    }

    ofs->writeRGB(img, "Input", FrameInfo("Input", SRC_POS));
    ofs->writeRGB(segmentdisp, "Seg", FrameInfo("Seg", SRC_POS));
    ofs->updateNext();
  }


  // get ready to terminate:
  mgr.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
