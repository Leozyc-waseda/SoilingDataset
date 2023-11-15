/*!@file plugins/SceneUnderstanding/GTEvaluator.C  */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
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
// $HeadURL: $
// $Id: $
//

#ifndef GTEvaluator_C_DEFINED
#define GTEvaluator_C_DEFINED

#include "Image/OpenCVUtil.H" //Needs to be first 

#include "plugins/SceneUnderstanding/GTEvaluator.H"

#include "Image/DrawOps.H"
#include "Raster/Raster.H"
#include "Simulation/SimEventQueue.H"
#include "GUI/DebugWin.H"

#include "Media/MediaSimEvents.H"
//#include "Neuro/NeuroSimEvents.H"
#include "Transport/FrameInfo.H"
#include "Transport/FrameOstream.H"


const ModelOptionCateg MOC_GTEvaluator = {
  MOC_SORTPRI_3,   "GTEvaluator-Related Options" };

// Used by: SimulationViewerEyeMvt
const ModelOptionDef OPT_GTEvaluatorShowDebug =
  { MODOPT_ARG(bool), "GTEvaluatorShowDebug", &MOC_GTEvaluator, OPTEXP_CORE,
    "Show debug img",
    "gtevaluator-debug", '\0', "<true|false>", "false" };

//Define the inst function name
SIMMODULEINSTFUNC(GTEvaluator);


// ######################################################################
GTEvaluator::GTEvaluator(OptionManager& mgr, const std::string& descrName,
    const std::string& tagName) :
  SimModule(mgr, descrName, tagName),
  SIMCALLBACK_INIT(SimEventLineMatchingOutput),
  SIMCALLBACK_INIT(SimEventSaveOutput),
  itsShowDebug(&OPT_GTEvaluatorShowDebug, this)
{
}

// ######################################################################
GTEvaluator::~GTEvaluator()
{
}


// ######################################################################
void GTEvaluator::onSimEventLineMatchingOutput(SimEventQueue& q,
                                  rutz::shared_ptr<SimEventLineMatchingOutput>& e)
{
  itsShapes = e->getShapes();

  //if (SeC<SimEventInputFrame> ee = q.check<SimEventInputFrame>(this,SEQ_UNMARKED,0))
  if (SeC<SimEventInputFrame> eframe = q.check<SimEventInputFrame>(this)) 
  {
    GenericFrame frame = eframe->frame();
    itsInImage = frame.asRgb();
    rutz::shared_ptr<GenericFrame::MetaData> metaData = frame.getMetaData(std::string("SceneData"));
    if (metaData.get() != 0)
      itsSceneData.dyn_cast_from(metaData);
  }

  evolve(q);
  
}

void GTEvaluator::evolve(SimEventQueue& q)
{
  return;

  std::vector<Rectangle> gtBB;

  //Get the ground truth and check
  if (itsSceneData.get() != 0)
  {
    for(uint obj=0; obj<itsSceneData->objects.size(); obj++)
    {
      TestImages::ObjData objData = itsSceneData->objects[obj];

      // find the object dimention from the polygon
      if (objData.polygon.size() > 0)
      {
        int minX=(int)objData.polygon[0].i;
        int minY=(int)objData.polygon[0].j;
        int maxX=minX;
        int maxY=minY;
        for(uint i=1;i<objData.polygon.size();i++)
        {
          if(objData.polygon[i].i < minX) minX = (int)objData.polygon[i].i;
          if(objData.polygon[i].j < minY) minY = (int)objData.polygon[i].j;
          if(objData.polygon[i].i > maxX) maxX = (int)objData.polygon[i].i;
          if(objData.polygon[i].j > maxY) maxY = (int)objData.polygon[i].j;
        }
        // Uses outer coordinates
        if(minX==maxX || minY==maxY)
          LFATAL("Invalid vertices");
        gtBB.push_back(Rectangle::tlbrI(minY,minX,maxY,maxX));


      }
    }
  }

  std::vector<bool> detected(gtBB.size(), false);

  for(uint shape=0; shape<itsShapes.size(); shape++)
  {

    //Find the max overlap for each polygon
    double maxOv = 0;
    int maxBBIdx = -1;
    for(uint i=0; i<gtBB.size(); i++)
    {
      drawRect(itsInImage, gtBB[i], PixRGB<byte>(0,255,0));

      //Find the shape with the best overlap that matches this bb
      Rectangle ovR = itsShapes[shape].bb.getOverlap(gtBB[i]);

      double ov = 0;
      if (ovR.isValid())
      {
        //compute overlap as area of intersection / area of union
        double ua = (itsShapes[shape].bb.area()+1) + (gtBB[i].area()+1) - ovR.area();
        ov = (double)ovR.area()/ua;

        if (ov > maxOv)
        {
          maxOv = ov;
          maxBBIdx = i;
        }
      }
    }

    if (maxOv >= 0.5)
    {
      if (maxBBIdx != -1)
      {
        //We found a valid BB

        if (!detected[maxBBIdx]) //Did we detect this shape already?
        {
          detected[maxBBIdx] = true;
          printf("Results: 1 %f %f\n", itsShapes[shape].score, maxOv); 


          drawRect(itsInImage, gtBB[maxBBIdx], PixRGB<byte>(0,255,0));
          drawRect(itsInImage, itsShapes[shape].bb, PixRGB<byte>(255,0,0));
          char msg[255];
          sprintf(msg, "ov: %0.2f", maxOv);
          writeText(itsInImage, itsShapes[shape].bb.center(), msg, PixRGB<byte>(255,255,255), PixRGB<byte>(0,0,0));


        } else {
          if (itsInImage.rectangleOk(itsShapes[shape].bb))
            drawRect(itsInImage, itsShapes[shape].bb, PixRGB<byte>(255,0,0));
          printf("Results: 0 %f %f\n", itsShapes[shape].score, maxOv); 
        }
      }
    } else {
      if (itsInImage.rectangleOk(itsShapes[shape].bb))
        drawRect(itsInImage, itsShapes[shape].bb, PixRGB<byte>(255,0,0));
        printf("Results: 0 %f %f\n", itsShapes[shape].score, maxOv); 
    }
  }
    SHOWIMG(itsInImage);


}
  

// ######################################################################
void GTEvaluator::onSimEventSaveOutput(SimEventQueue& q,
    rutz::shared_ptr<SimEventSaveOutput>& e)
{
  if (itsShowDebug.getVal())
    {
      // get the OFS to save to, assuming sinfo is of type
      // SimModuleSaveInfo (will throw a fatal exception otherwise):
      nub::ref<FrameOstream> ofs =
        dynamic_cast<const SimModuleSaveInfo&>(e->sinfo()).ofs;
      Layout<PixRGB<byte> > disp = getDebugImage();
      if (disp.initialized())
        ofs->writeRgbLayout(disp, "GTEvaluator", FrameInfo("GTEvaluator", SRC_POS));
    }
}

Layout<PixRGB<byte> > GTEvaluator::getDebugImage()
{

  Layout<PixRGB<byte> > disp;

  for(uint i=0; i<itsShapes.size() && i < 4; i++)
  {
    if (itsInImage.rectangleOk(itsShapes[i].bb))
      drawRect(itsInImage, itsShapes[i].bb, PixRGB<byte>(0,255,0));
  }
  disp = itsInImage;

  usleep(10000);
  return disp;
}

#endif
